"""
Test dataset reconstruction script.
- Terr_* (ground): sequential pairs window=20
- DSC* (aerial): sequential pairs window=20
- Cross-group: each DSC matched to top-5 Terr (excl. Terr_018–074) via NetVLAD
- LightGlue matching, full reconstruction with gim_lightglue
"""
import os
import re
import torch
import warnings
import numpy as np
import h5py
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

from hloc import extract_features, match_features, reconstruction
from hloc.utils.io import list_h5_names

from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def get_descriptors_subset(names, all_names, all_desc_tensor, name2idx):
    indices = [name2idx[n] for n in names]
    return all_desc_tensor[indices]


def match_netvlad_topk(query_names, db_names, query_desc, db_desc, topk, device,
                       sim_thresh=None):
    query_desc = query_desc.to(device)
    db_desc = db_desc.to(device)
    sim = torch.einsum("id,jd->ij", query_desc, db_desc)
    topk = min(topk, len(db_names))
    scores, indices = torch.topk(sim, topk, dim=1)
    pairs = []
    for i, q_name in enumerate(query_names):
        for j in range(topk):
            if sim_thresh is not None and scores[i, j].item() < sim_thresh:
                continue  # skip below threshold
            db_idx = indices[i, j].item()
            d_name = db_names[db_idx]
            if q_name != d_name:
                pairs.append((q_name, d_name))
    del sim, query_desc, db_desc
    torch.cuda.empty_cache()
    return pairs


def main(scene_name, version='gim_lightglue', stop_after_db=False, mask_dir=None,
         camera_model='PINHOLE'):
    images = Path('inputs') / scene_name / 'images'
    outputs = Path('outputs') / scene_name / version
    outputs.mkdir(parents=True, exist_ok=True)
    os.environ['GIMRECONSTRUCTION'] = str(outputs)
    sfm_dir = outputs / 'sparse'
    # Reuse pairs from gim_lightglue if it already exists
    shared_pairs = Path('outputs') / scene_name / 'gim_lightglue' / 'pairs-cross.txt'
    if shared_pairs.exists() and version != 'gim_lightglue':
        pairs_file = shared_pairs
    else:
        pairs_file = outputs / 'pairs-cross.txt'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- identify image groups ----
    all_images = sorted([
        p.name for p in images.iterdir()
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ], key=natural_sort_key)

    terr_images = [n for n in all_images if n.startswith('Terr')]
    dsc_images = [n for n in all_images if n.startswith('DSC')]
    terr_sorted = sorted(terr_images, key=natural_sort_key)
    dsc_sorted = sorted(dsc_images, key=natural_sort_key)

    print(f"Found {len(terr_sorted)} Terr (ground) + {len(dsc_sorted)} DSC (aerial)")

    # ---- Step 1: generate pairs ----
    if not pairs_file.exists():
        print("Generating pairs...")
        pairs_set = set()

        # Terr sequential (cyclic), window=10
        Nt = len(terr_sorted)
        for i in range(Nt):
            for offset in range(1, 11):
                j = (i + offset) % Nt  # wrap around for loop closure
                if i != j:
                    a, b = terr_sorted[i], terr_sorted[j]
                    pairs_set.add((a, b) if a < b else (b, a))
        print(f"  Terr sequential (cyclic): {Nt} images, window=10")

        # NetVLAD for cross-group matching
        print("  Extracting NetVLAD for cross-group matching...")
        netvlad_conf = extract_features.confs['netvlad']
        netvlad_outputs = Path('outputs') / scene_name / 'gim_lightglue'
        netvlad_outputs.mkdir(parents=True, exist_ok=True)
        netvlad_path = extract_features.main(netvlad_conf, images, netvlad_outputs)

        all_feat_names = list_h5_names(netvlad_path)
        name2idx = {n: i for i, n in enumerate(all_feat_names)}
        with h5py.File(str(netvlad_path), 'r', libver='latest') as fd:
            all_desc = [fd[n]['global_descriptor'].__array__() for n in all_feat_names]
        all_desc_tensor = torch.from_numpy(np.stack(all_desc, 0)).float()

        # DSC→DSC (aerial self-match via NetVLAD): top-20, threshold 0.15
        dsc_in_feats = [n for n in dsc_sorted if n in all_feat_names]
        terr_in_feats = [n for n in terr_sorted if n in all_feat_names]
        if len(dsc_in_feats) > 1:
            dsc_desc = get_descriptors_subset(
                dsc_in_feats, all_feat_names, all_desc_tensor, name2idx)
            dsc2dsc = match_netvlad_topk(
                dsc_in_feats, dsc_in_feats, dsc_desc, dsc_desc,
                topk=20, device=device, sim_thresh=0.15)
            pairs_set.update(dsc2dsc)
            print(f"  DSC→DSC (aerial self-match NetVLAD): top-20, thresh=0.15, "
                  f"added {len(dsc2dsc)} pairs")

        # Terr→DSC (ground anchors aerial, excl Terr_018–Terr_074): top-5, threshold 0.15
        terr_for_cross = [
            n for n in terr_sorted
            if n in all_feat_names
            and not (18 <= int(re.search(r'Terr_(\d+)', n).group(1)) <= 74)
        ]
        if terr_for_cross and dsc_in_feats:
            query_desc = get_descriptors_subset(
                terr_for_cross, all_feat_names, all_desc_tensor, name2idx)
            db_desc = get_descriptors_subset(
                dsc_in_feats, all_feat_names, all_desc_tensor, name2idx)
            terr2dsc = match_netvlad_topk(
                terr_for_cross, dsc_in_feats, query_desc, db_desc,
                topk=5, device=device, sim_thresh=0.0)
            pairs_set.update(terr2dsc)
            print(f"  Terr→DSC (ground anchors aerial, excl 18-74): top-5, thresh=0.15, "
                  f"added {len(terr2dsc)} pairs")

        with open(pairs_file, 'w') as f:
            f.write('\n'.join(f'{a} {b}' for a, b in sorted(pairs_set)))
        print(f"  Total unique pairs: {len(pairs_set)}")
    else:
        print(f"Using existing pairs: {pairs_file}")

    # ---- Step 2: Matching (lightglue or mast3r) ----
    import pycolmap

    if version == 'mast3r':
        from hloc.mast3r_matching import load_mast3r_model, run_mast3r_matching
        from hloc.reconstruction import (create_empty_db, import_images, get_image_ids,
                                         estimation_and_geometric_verification, run_reconstruction)

        print(f"\nStep 2: MASt3R matching...")
        mast3r_model = load_mast3r_model(device)

        database_path = sfm_dir / 'database.db'
        database_path.parent.mkdir(parents=True, exist_ok=True)
        create_empty_db(database_path)

        with open(pairs_file) as f:
            pairs_list = [line.split() for line in f if line.strip()]

        import_images(images, database_path, camera_mode=pycolmap.CameraMode.AUTO,
                      image_list=all_images,
                      options={'camera_model': camera_model})
        image_ids = get_image_ids(database_path)

        colmap_pairs = run_mast3r_matching(
            model=mast3r_model, image_dir=images, image_pairs=pairs_list,
            database_path=database_path, image_ids=image_ids, device=device,
            maxdim=512, conf_thr=1.001, pixel_tol=5, subsample=8,
            min_track_len=2, max_keypoints=16384)

        del mast3r_model
        torch.cuda.empty_cache()

        filtered_pairs = outputs / 'pairs-mast3r-filtered.txt'
        with open(filtered_pairs, 'w') as f:
            f.write('\n'.join(f'{a} {b}' for a, b in colmap_pairs))
        estimation_and_geometric_verification(database_path, filtered_pairs)

        if not stop_after_db:
            print(f"\nStep 3: Sparse reconstruction...")
            from hloc.reconstruction import unique_camera_ids
            unique_camera_ids(database_path)
            rec = run_reconstruction(sfm_dir, database_path, images, verbose=False,
                                     options={'camera_model': camera_model})
            if rec:
                print(f"\nReconstruction complete: {rec.summary()}")
        else:
            print("Stopping after database generation.")
        return

    # ---- LightGlue path ----
    print(f"\nStep 2: Feature extraction + LightGlue matching...")
    feature_conf = extract_features.confs['gim_superpoint']
    matcher_conf = match_features.confs['gim_lightglue']

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        checkpoints_path = 'weights/gim_lightglue_100h.ckpt'
        detector = SuperPoint({
            'max_num_keypoints': 8192, 'force_num_keypoints': True,
            'detection_threshold': 0.0, 'nms_radius': 3, 'trainable': False,
        })
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        detector.load_state_dict(state_dict)

        model = LightGlue({'filter_threshold': 0.1, 'flash': True, 'checkpointed': True})
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'):
                state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        feature_path = extract_features.main(
            feature_conf, images, outputs, model=detector, mask_dir=mask_dir)
        match_path = match_features.main(
            matcher_conf, pairs_file, feature_conf['output'], outputs, model=model)

    # ---- Step 3: Reconstruction ----
    print(f"\nStep 3: Sparse reconstruction...")
    reconstruction.main(
        sfm_dir, images, pairs_file, feature_path, match_path,
        camera_mode=pycolmap.CameraMode.AUTO,
        image_options=dict(camera_model=camera_model),
        stop_after_db=stop_after_db)

    if not stop_after_db:
        rec = pycolmap.Reconstruction(str(sfm_dir))
        print(f"\nReconstruction complete: {rec.summary()}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--stop_after_db', action='store_true')
    parser.add_argument('--mask_dir', type=str, default=None)
    parser.add_argument('--version', type=str, default='gim_lightglue',
                        choices=['gim_lightglue', 'mast3r'])
    parser.add_argument('--camera_model', type=str, default='PINHOLE',
                        choices=['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'OPENCV'])
    args = parser.parse_args()
    main(args.scene_name, args.version, args.stop_after_db, mask_dir=args.mask_dir,
         camera_model=args.camera_model)
