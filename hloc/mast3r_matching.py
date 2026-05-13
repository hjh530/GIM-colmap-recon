import os
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
