import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import PIL.Image

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import get_im_matches, export_matches

import mast3r.utils.path_to_dust3r  # noqa: registers dust3r/croco in path (no-op when inside gim)
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import inference
import torchvision.transforms.functional as tvf


def load_mast3r_model(device='cuda', checkpoint=None):
    if checkpoint is None:
        checkpoint = os.path.join(os.path.dirname(__file__), '..', 'weights', 'mast3r', 'MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    model = AsymmetricMASt3R.from_pretrained(checkpoint).to(device).eval()
    return model


def _resize_to_maxdim(rgb_image, maxdim, patch_size):
    """Resize image so max(H,W) <= maxdim and both dims are multiples of patch_size."""
    W, H = rgb_image.size
    # Resize such that max side = maxdim
    scale = maxdim / max(H, W)
    new_W, new_H = int(W * scale), int(H * scale)
    # Round to multiples of patch_size
    new_W = (new_W // patch_size) * patch_size
    new_H = (new_H // patch_size) * patch_size
    new_W = max(new_W, patch_size)
    new_H = max(new_H, patch_size)

    rgb_tensor = ImgNorm(rgb_image)
    # Resize
    rgb_tensor = tvf.resize(rgb_tensor, [new_H, new_W])

    # Build to_orig: maps resized coords back to original pixel coords
    to_orig = np.array([
        [W / new_W, 0, 0],
        [0, H / new_H, 0],
        [0, 0, 1],
    ])
    return rgb_tensor, to_orig


def prepare_images(image_paths, root, maxdim=512, patch_size=16):
    """Resize and load images in the format expected by MASt3R inference."""
    images = []
    for idx in tqdm(range(len(image_paths)), desc="Loading images"):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert('RGB')
        H, W = rgb_image.size[1], rgb_image.size[0]

        rgb_tensor, to_orig = _resize_to_maxdim(rgb_image, maxdim, patch_size)

        images.append({
            'img': rgb_tensor.unsqueeze(0),
            'true_shape': np.int32([rgb_tensor.shape[1:]]),
            'to_orig': to_orig,
            'idx': idx,
            'instance': image_paths[idx],
            'orig_shape': np.int32([H, W]),
        })
    return images


def remove_duplicate_pairs(image_pairs, image_path_to_idx):
    """Deduplicate unordered image pairs."""
    pairs_added = set()
    deduped = []
    for name0, name1 in image_pairs:
        small, big = min(name0, name1), max(name0, name1)
        if (small, big) in pairs_added:
            continue
        pairs_added.add((small, big))
        deduped.append((name0, name1))
    return deduped


def run_mast3r_matching(
    model,
    image_dir,
    image_pairs,
    database_path,
    image_ids,
    device='cuda',
    maxdim=512,
    patch_size=16,
    conf_thr=1.001,
    pixel_tol=5,
    subsample=8,
    min_track_len=3,
    skip_geometric_verification=False,
):
    """
    Run MASt3R inference on image pairs and populate a COLMAP database.

    The database must already have `cameras` and `images` tables (e.g. created
    by HLOC's `create_empty_db` + `import_images`).  This function fills
    `keypoints`, `matches` and optionally `two_view_geometries`.

    Parameters
    ----------
    model : AsymmetricMASt3R
    image_dir : Path    root directory containing the images
    image_pairs : list of (str, str)    relative image paths in each pair
    database_path : Path    path to the COLMAP SQLite database
    image_ids : dict    {image_name: colmap_image_id}
    device : str
    maxdim : int    max image dimension for inference
    patch_size : int    ViT patch size (16)
    conf_thr : float    descriptor confidence threshold
    pixel_tol : int    tolerance for iterative NN refinement
    subsample : int    grid step for sparse matching
    min_track_len : int    minimum track length to keep a keypoint
    skip_geometric_verification : bool    if True, write identity two-view geometries

    Returns
    -------
    colmap_image_pairs : list of (str, str)    pairs that survived track filtering
    """
    from hloc.utils.database import COLMAPDatabase

    # Collect all unique image paths referenced by the pairs
    all_image_names = sorted(set(
        name for pair in image_pairs for name in pair
    ))
    image_path_to_idx = {name: i for i, name in enumerate(all_image_names)}

    # Load and resize images
    images = prepare_images(all_image_names, str(image_dir), maxdim, patch_size)

    # Deduplicate pairs
    pairs_unique = remove_duplicate_pairs(image_pairs, image_path_to_idx)
    matching_pairs = []
    for name0, name1 in pairs_unique:
        i0, i1 = image_path_to_idx[name0], image_path_to_idx[name1]
        matching_pairs.append((images[i0], images[i1]))

    # Build image_to_colmap mapping
    image_to_colmap = {}
    for name, idx in image_path_to_idx.items():
        image_to_colmap[idx] = {
            'colmap_imid': image_ids[name],
            'colmap_camid': 1,  # single camera after unique_camera_ids
        }

    im_keypoints = {idx: {} for idx in range(len(all_image_names))}

    # Run MASt3R inference in chunks of 4 pairs
    im_matches = {}
    for chunk_start in tqdm(range(0, len(matching_pairs), 4), desc="MASt3R inference"):
        chunk = matching_pairs[chunk_start:chunk_start + 4]
        output = inference(chunk, model, device, batch_size=1, verbose=False)
        pred1, pred2 = output['pred1'], output['pred2']

        chunk_matches = get_im_matches(
            pred1=pred1, pred2=pred2,
            pairs=chunk,
            image_to_colmap=image_to_colmap,
            im_keypoints=im_keypoints,
            conf_thr=conf_thr,
            is_sparse=True,
            subsample=subsample,
            pixel_tol=pixel_tol,
            device=device,
        )
        im_matches.update(chunk_matches)

    # Build tracks, filter, and write to database
    db = COLMAPDatabase.connect(database_path)
    colmap_image_pairs = export_matches(
        db, images, image_to_colmap, im_keypoints, im_matches,
        min_track_len, skip_geometric_verification,
    )
    db.commit()
    db.close()

    torch.cuda.empty_cache()
    return colmap_image_pairs


def filter_keypoints_by_masks(database_path, mask_dir, image_dir):
    """
    Remove keypoints that fall on mask regions (pixel value 255) from the
    COLMAP database.  Updates both the `keypoints` and `matches` tables so
    that indices stay consistent.

    Parameters
    ----------
    database_path : Path     path to the COLMAP SQLite database
    mask_dir : Path          directory containing `{image_stem}_mask.png` files
    image_dir : Path         root image directory, used to get image dimensions
    """
    import cv2
    from hloc.utils.database import (COLMAPDatabase, blob_to_array, array_to_blob,
                                     pair_id_to_image_ids)

    db = COLMAPDatabase.connect(database_path)

    # Image id → name
    image_rows = db.execute("SELECT image_id, name FROM images").fetchall()
    image_id_to_name = {row[0]: row[1] for row in image_rows}

    # Step 1: identify images that have masks, and build old→new index maps
    keep_masks = {}       # image_id → boolean array (N,)  True=keep
    old_to_new = {}       # image_id → int array (N,)  old_idx → new_idx, -1 if removed

    for image_id, name in image_rows:
        stem = Path(name).stem
        mask_path = mask_dir / f'{stem}_mask.png'
        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Read keypoints
        row = db.execute(
            "SELECT data FROM keypoints WHERE image_id=?", (image_id,)
        ).fetchone()
        if row is None:
            continue
        kpts = blob_to_array(row[0], np.float32, (-1, 2))

        # Get image dimensions from the original image
        img_path = image_dir / name
        if img_path.exists():
            H_img, W_img = cv2.imread(str(img_path)).shape[:2]
        else:
            H_img, W_img = mask.shape[:2]

        # Scale mask to match image dimensions if needed
        if mask.shape[0] != H_img or mask.shape[1] != W_img:
            mask = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        # Check each keypoint against the mask
        keep = np.ones(len(kpts), dtype=bool)
        for i, (x, y) in enumerate(kpts):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H_img and 0 <= xi < W_img:
                if mask[yi, xi] == 255:
                    keep[i] = False

        if keep.all():
            continue  # no keypoints removed for this image

        keep_masks[image_id] = keep
        # Build mapping: old→new
        mapping = np.full(len(kpts), -1, dtype=np.int32)
        mapping[keep] = np.arange(keep.sum(), dtype=np.int32)
        old_to_new[image_id] = mapping

        # Write back filtered keypoints
        filtered_kpts = kpts[keep]
        db.execute("DELETE FROM keypoints WHERE image_id=?", (image_id,))
        db.add_keypoints(image_id, filtered_kpts)
        print(f"[Mask] {name}: removed {np.sum(~keep)}/{len(kpts)} keypoints")

    # Step 2: update matches
    if not old_to_new:
        db.commit()
        db.close()
        return

    match_rows = db.execute(
        "SELECT pair_id, data FROM matches"
    ).fetchall()

    for pair_id, data in match_rows:
        id1, id2 = pair_id_to_image_ids(pair_id)
        if id1 not in old_to_new and id2 not in old_to_new:
            continue
        matches = blob_to_array(data, np.uint32, (-1, 2))
        M = len(matches)

        map1 = old_to_new.get(id1)
        map2 = old_to_new.get(id2)

        if map1 is not None:
            new0 = map1[matches[:, 0].astype(int)]
        else:
            new0 = matches[:, 0].astype(np.int32)

        if map2 is not None:
            new1 = map2[matches[:, 1].astype(int)]
        else:
            new1 = matches[:, 1].astype(np.int32)

        valid = (new0 >= 0) & (new1 >= 0)
        if not valid.any():
            # All matches removed for this pair
            db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
            db.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
            continue

        filtered = np.stack([new0[valid], new1[valid]], axis=-1).astype(np.uint32)
        db.execute("DELETE FROM matches WHERE pair_id=?", (pair_id,))
        db.add_matches(id1, id2, filtered)

        # Also update two_view_geometries if present
        geo_rows = db.execute(
            "SELECT rows FROM two_view_geometries WHERE pair_id=?", (pair_id,)
        ).fetchone()
        if geo_rows is not None:
            db.execute("DELETE FROM two_view_geometries WHERE pair_id=?", (pair_id,))
            db.add_two_view_geometry(id1, id2, filtered)

    db.commit()
    db.close()
    print(f"[Mask] Filtering complete.")
