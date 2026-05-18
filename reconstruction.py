import os
import re
import torch
import warnings
import numpy as np
import h5py
import cv2
import re
from tqdm import tqdm
from os.path import join
from pathlib import Path
from argparse import ArgumentParser

# HLOC imports
import pycolmap
from hloc import extract_features, match_features, match_dense, reconstruction
from hloc.utils import segment
from hloc.utils.io import read_image, list_h5_names
from hloc.match_dense import ImagePairDataset

# GIM / LightGlue imports
from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue
from networks.mit_semseg.models import ModelBuilder, SegmentationModule

# =========================================================
# 辅助函数
# =========================================================
def natural_sort_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]  
    
def get_descriptors_subset(names, all_names, all_desc_tensor, name2idx):
    """从全量描述子中提取子集"""
    indices = [name2idx[n] for n in names]
    return all_desc_tensor[indices]

def match_groups(query_names, db_names, query_desc, db_desc, num_matched, device, thresh=None):
    """
    使用 NetVLAD 描述子进行相似度匹配，返回匹配对列表。
    如果 thresh 不为 None，则只保留相似度 >= thresh 的对。
    """
    query_desc = query_desc.to(device)
    db_desc = db_desc.to(device)

    sim = torch.einsum("id,jd->ij", query_desc, db_desc)
    if query_names == db_names:
        sim.fill_diagonal_(float('-inf'))

    k = min(num_matched, len(db_names))
    topk = torch.topk(sim, k, dim=1)
    scores = topk.values.cpu().numpy()
    indices = topk.indices.cpu().numpy()

    pairs = []
    for i in range(len(query_names)):
        for j_idx in range(k):
            score = scores[i, j_idx]
            if thresh is not None and score < thresh:
                continue
            db_idx = indices[i, j_idx]
            q_name = query_names[i]
            d_name = db_names[db_idx]
            if q_name == d_name:
                continue
            pair = tuple(sorted((q_name, d_name)))
            pairs.append(pair)

    del sim, query_desc, db_desc
    torch.cuda.empty_cache()
    return pairs

def generate_sequential_pairs_with_netvlad(descriptors_path, output_path, images_dir, window=20, sim_thresh=0.15):
    """
    使用 NetVLAD 描述子生成环形序列匹配对：每张图像与后续 window 张图像匹配（循环）。
    """
    print(f"Generating cyclic sequential pairs (window={window}) with NetVLAD filtering (thresh={sim_thresh})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_names = list_h5_names(descriptors_path)
    name2idx = {n: i for i, n in enumerate(all_names)}

    with h5py.File(str(descriptors_path), "r", libver="latest") as fd:
        all_desc = [fd[n]["global_descriptor"].__array__() for n in all_names]

    all_desc_tensor = torch.from_numpy(np.stack(all_desc, 0)).float()

    # 从图像目录生成相对路径，仅保留存在于描述子中的文件
    images_dir = Path(images_dir)
    image_rels = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        for p in images_dir.rglob(ext):
            rel = p.relative_to(images_dir).as_posix()
            if rel in all_names:
                image_rels.append(rel)
    sorted_names = sorted(image_rels, key=natural_sort_key)

    N = len(sorted_names)
    final_pairs = set()

    print("Step 1: Cyclic sequential matching...")
    for i in range(N):
        q_name = sorted_names[i]
        q_desc = get_descriptors_subset([q_name], all_names, all_desc_tensor, name2idx)

        db_indices = [(i + offset) % N for offset in range(1, window + 1)]
        db_names = [sorted_names[idx] for idx in db_indices]
        db_desc = get_descriptors_subset(db_names, all_names, all_desc_tensor, name2idx)

        pairs = match_groups([q_name], db_names, q_desc, db_desc,
                             num_matched=len(db_names), device=device, thresh=sim_thresh)
        final_pairs.update(pairs)

    print(f"Total unique pairs after filtering: {len(final_pairs)}")
    with open(output_path, "w") as f:
        f.write("\n".join(" ".join(p) for p in final_pairs))

def segmentation(images, segment_root, matcher_conf):
    # initial device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # initial segmentation model
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='weights/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='weights/decoder_epoch_20.pth',
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module = segmentation_module.to(device).eval()

    dataset = ImagePairDataset(None, matcher_conf["preprocessing"], None)
    # 修复：正确处理 Path 对象
    image_list = [p.name for p in Path(images).iterdir() if p.is_file()]

    with torch.no_grad():
        for img in tqdm(image_list):
            segment_path = join(segment_root, '{}.npy'.format(img[:-4]))
            if not os.path.exists(segment_path):
                rgb = read_image(images / img, dataset.conf.grayscale)
                mask = segment(rgb, 1920, device, segmentation_module)
                np.save(segment_path, mask)




# =========================================================
# 主流程
# =========================================================

def main(scene_name, version, stop_after_db, mask_dir=None,
         mast3r_maxdim=512, mast3r_conf_thr=1.001, mast3r_pixel_tol=5,
         mast3r_subsample=8, mast3r_min_track_len=3,
         mast3r_max_keypoints=8192, dkm_max_keypoints=8192,
         camera_model='PINHOLE'):
    # 路径设置
    images = Path('inputs') / scene_name / 'images'
    outputs = Path('outputs') / scene_name / version
    outputs.mkdir(parents=True, exist_ok=True)

    os.environ['GIMRECONSTRUCTION'] = str(outputs)

    segment_root = Path('outputs') / scene_name / 'segment'
    segment_root.mkdir(parents=True, exist_ok=True)

    # ---------- 处理 mask_dir ----------
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        if not mask_dir.is_dir():
            print(f"[Mask] Provided mask_dir does not exist: {mask_dir}, ignoring.")
            mask_dir = None
        else:
            print(f"[Mask] Using provided mask directory: {mask_dir}")
    else:
        auto_mask = images.parent / 'masks'          # inputs/<scene>/masks
        mask_dir = auto_mask if auto_mask.is_dir() else None
        if mask_dir:
            print(f"[Mask] Auto-detected masks at {mask_dir}")
        else:
            print("[Mask] No masks directory found, skip filtering.")


    sfm_dir = outputs / 'sparse'
    database_path = sfm_dir / 'database.db'
    # mast3r uses raw simple window, gim_* use NetVLAD-filtered subset
    raw_pairs = outputs / 'pairs-raw.txt'
    image_pairs = outputs / (f'pairs-netvlad.txt' if version != 'mast3r' else 'pairs-raw.txt')

    # 根据 version 选择特征提取和匹配配置
    feature_conf = matcher_conf = None
    if version == 'gim_dkm':
        feature_conf = None
        matcher_conf = match_dense.confs[version]
    elif version == 'gim_lightglue':
        feature_conf = extract_features.confs['gim_superpoint']
        matcher_conf = match_features.confs[version]

    # Step 1: 生成简单滑动窗口 pairs（所有版本共享基准）
    if not raw_pairs.exists():
        print("Step 1: Generating raw sequential pairs (simple window)...")
        image_list = sorted([
            p.relative_to(images).as_posix()
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
            for p in images.rglob(ext)
        ], key=natural_sort_key)
        N = len(image_list)
        window = 20
        pairs_list = []
        for i in range(N):
            for offset in range(1, min(window + 1, N - i)):
                pairs_list.append((image_list[i], image_list[i + offset]))
        with open(raw_pairs, 'w') as f:
            f.write('\n'.join(f'{a} {b}' for a, b in pairs_list))
        print(f"  Generated {len(pairs_list)} pairs (window={window}).")

    # Step 2: gim_* 版本用 NetVLAD 从 raw pairs 中筛选
    if version != 'mast3r' and not image_pairs.exists():
        print("Step 2: Filtering pairs with NetVLAD...")
        netvlad_conf = extract_features.confs['netvlad']
        netvlad_path = extract_features.main(netvlad_conf, images, outputs)
        generate_sequential_pairs_with_netvlad(
            netvlad_path, image_pairs, images_dir=images,
            window=20, sim_thresh=0.20
        )

    if version != 'mast3r':
        # Step 3: 语义分割（仅 gim_* 版本需要）
        segmentation(images, segment_root, matcher_conf)

    # Step 4: 特征提取与匹配
    print(f"Step 3: Running Feature Extraction & Matching ({version})...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if version == 'gim_lightglue':
            # LightGlue + SuperPoint
            checkpoints_path = join('weights', 'gim_lightglue_100h.ckpt')

            detector = SuperPoint({
                'max_num_keypoints': 12000,
                'force_num_keypoints': False,
                'detection_threshold': 0.0,
                'nms_radius': 3,
                'trainable': False,
            })
            state_dict = torch.load(checkpoints_path, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('model.'):
                    state_dict.pop(k)
                if k.startswith('superpoint.'):
                    state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
            detector.load_state_dict(state_dict)

            model = LightGlue({
                'filter_threshold': 0.2,
                'flash': True,
                'checkpointed': True,
            })
            state_dict = torch.load(checkpoints_path, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('superpoint.'):
                    state_dict.pop(k)
                if k.startswith('model.'):
                    state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            model.load_state_dict(state_dict)

            feature_path = extract_features.main(feature_conf, images, outputs,
                                                 model=detector,
                                                 mask_dir=mask_dir)
            match_path = match_features.main(matcher_conf, image_pairs,
                                             feature_conf['output'], outputs,
                                             model=model)

        elif version == 'gim_dkm':
            # DKM dense matching: 为使用 mask，分步提取特征
            dense_feat_conf = dict(extract_features.confs['gim_superpoint'])
            dense_feat_conf['model'] = dict(dense_feat_conf['model'])
            dense_feat_conf['model']['max_num_keypoints'] = dkm_max_keypoints
            feature_path = extract_features.main(dense_feat_conf, images, outputs,
                                                 mask_dir=mask_dir)
            feature_path, match_path = match_dense.main(
                matcher_conf, image_pairs, images,
                export_dir=outputs,
                features=dense_feat_conf['output'],
                max_kps=dkm_max_keypoints)

        elif version == 'mast3r':
            # MASt3R: single-pass dense matching → COLMAP database
            from hloc.mast3r_matching import load_mast3r_model, run_mast3r_matching
            from hloc.reconstruction import (
                create_empty_db, import_images, get_image_ids,
                estimation_and_geometric_verification, run_reconstruction,
            )

            # Step 3a: 加载 MASt3R 模型
            print("Loading MASt3R model...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            mast3r_model = load_mast3r_model(device)

            # Step 3b: 初始化 COLMAP 数据库
            database_path = sfm_dir / 'database.db'
            database_path.parent.mkdir(parents=True, exist_ok=True)
            create_empty_db(database_path)
            # 使用 images/ 下所有图片
            image_list = sorted([
                p.relative_to(images).as_posix()
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
                for p in images.rglob(ext)
            ], key=natural_sort_key)
            import_images(images, database_path, camera_mode=pycolmap.CameraMode.AUTO,
                          image_list=image_list,
                          options={'camera_model': camera_model})
            image_ids = get_image_ids(database_path)

            with open(image_pairs) as f:
                pairs_list = [line.split() for line in f if line.strip()]

            # Step 3d: MASt3R 推理 + 匹配 → 写入数据库
            print("Running MASt3R matching...")
            colmap_pairs = run_mast3r_matching(
                model=mast3r_model,
                image_dir=images,
                image_pairs=pairs_list,
                database_path=database_path,
                image_ids=image_ids,
                device=device,
                # mast3r options
                maxdim=mast3r_maxdim,
                conf_thr=mast3r_conf_thr,
                pixel_tol=mast3r_pixel_tol,
                subsample=mast3r_subsample,
                min_track_len=mast3r_min_track_len,
                max_keypoints=mast3r_max_keypoints,
                skip_geometric_verification=False,
            )

            # 释放模型显存
            del mast3r_model
            torch.cuda.empty_cache()

            if not colmap_pairs:
                raise RuntimeError("MASt3R matching produced no valid pairs.")

            # Step 3d2: 掩码过滤（如果有 mask_dir）
            if mask_dir is not None:
                from hloc.mast3r_matching import filter_keypoints_by_masks
                filter_keypoints_by_masks(database_path, mask_dir, images)

            # Step 3e: 几何验证
            # 更新 pairs 文件为过滤后的结果
            filtered_pairs = outputs / 'pairs-mast3r-filtered.txt'
            with open(filtered_pairs, 'w') as f:
                f.write('\n'.join(f'{a} {b}' for a, b in colmap_pairs))
            estimation_and_geometric_verification(database_path, filtered_pairs)

            # Step 3f: 重建
            if not stop_after_db:
                print("Running sparse reconstruction...")
                from hloc.reconstruction import unique_camera_ids
                unique_camera_ids(database_path)
                reconstruction_result = run_reconstruction(
                    sfm_dir, database_path, images,
                    verbose=False,
                    options={'camera_model': camera_model},
                )
                print(f"Reconstruction complete: {reconstruction_result.summary()}")
            else:
                print("Stopping after database generation as requested.")
            return

    if version != 'mast3r':
        # Step 5: 稀疏重建
        print("Step 4: Running Sparse Reconstruction...")
        opts = dict(camera_model=camera_model)
        reconstruction.main(sfm_dir, images, image_pairs, feature_path, match_path,
                            image_options=opts, stop_after_db=stop_after_db)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--version', type=str,
                        choices={'gim_dkm', 'gim_lightglue', 'mast3r'},
                        default='gim_dkm')
    parser.add_argument('--stop_after_db', action='store_true',
                        help='Stop after generating COLMAP database, skip reconstruction.')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing binary masks (PNG, e.g., image_mask.png). '
                             'If not set, defaults to inputs/<scene>/masks if it exists.')
    # MASt3R options
    parser.add_argument('--mast3r_maxdim', type=int, default=512,
                        help='MASt3R: max image dimension for inference.')
    parser.add_argument('--mast3r_conf_thr', type=float, default=1.001,
                        help='MASt3R: descriptor confidence threshold.')
    parser.add_argument('--mast3r_pixel_tol', type=int, default=5,
                        help='MASt3R: tolerance for iterative NN refinement.')
    parser.add_argument('--mast3r_subsample', type=int, default=8,
                        help='MASt3R: grid step for sparse matching.')
    parser.add_argument('--mast3r_min_track_len', type=int, default=3,
                        help='MASt3R: minimum track length to keep a keypoint.')
    parser.add_argument('--mast3r_max_keypoints', type=int, default=8192,
                        help='MASt3R: max keypoints per image (keep top by match count).')
    parser.add_argument('--dkm_max_keypoints', type=int, default=8192,
                        help='DKM: max keypoints per image for SuperPoint + dense aggregation.')
    parser.add_argument('--camera_model', type=str, default='PINHOLE',
                        choices=['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'OPENCV'],
                        help='COLMAP camera model.')
    args = parser.parse_args()
    main(args.scene_name, args.version, args.stop_after_db, mask_dir=args.mask_dir,
         mast3r_maxdim=args.mast3r_maxdim,
         mast3r_conf_thr=args.mast3r_conf_thr,
         mast3r_pixel_tol=args.mast3r_pixel_tol,
         mast3r_subsample=args.mast3r_subsample,
         mast3r_min_track_len=args.mast3r_min_track_len,
         mast3r_max_keypoints=args.mast3r_max_keypoints,
         dkm_max_keypoints=args.dkm_max_keypoints,
         camera_model=args.camera_model)
