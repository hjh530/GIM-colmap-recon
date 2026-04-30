import os
import re
import torch
import warnings
import numpy as np
import h5py
from tqdm import tqdm
from os.path import join
from pathlib import Path
from argparse import ArgumentParser

# HLOC imports
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

def generate_sequential_pairs_with_netvlad(descriptors_path, output_path, window=20, sim_thresh=0.15):
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
    image_list = os.listdir(images_path)

    # 只保留 descriptor 中存在的
    sorted_names = [n for n in image_list if n in all_names]
    

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
    # initial segmentation mode
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
    image_list = os.listdir(images)

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

def main(scene_name, version, stop_after_db, mask_dir):
    # 路径设置
    images = Path('inputs') / scene_name / 'images'
    outputs = Path('outputs') / scene_name / version
    outputs.mkdir(parents=True, exist_ok=True)

    os.environ['GIMRECONSTRUCTION'] = str(outputs)
    
    segment_root = Path('outputs') / scene_name / 'segment'
    segment_root.mkdir(parents=True, exist_ok=True)

    # ---------- 自动检测 masks 目录 ---------- 
    if mask_dir is not None:
        mask_dir = Path(mask_dir)
        if not mask_dir.is_dir():
            print(f"[Mask] Provided mask_dir does not exist: {mask_dir}, ignoring.")
            mask_dir = None
        else:
            print(f"[Mask] Using provided mask directory: {mask_dir}")
    else:
        auto_mask = images.parent / 'masks'
        mask_dir = auto_mask if auto_mask.is_dir() else None
        if mask_dir:
            print(f"[Mask] Auto-detected masks at {mask_dir}")
        else:
            print("[Mask] No masks directory found, skip filtering.")
           


    sfm_dir = outputs / 'sparse'
    database_path = sfm_dir / 'database.db'
    image_pairs = outputs / 'pairs-sequential.txt'
    
    # 根据 version 选择特征提取和匹配配置
    feature_conf = matcher_conf = None
    if version == 'gim_dkm':
        feature_conf = None
        matcher_conf = match_dense.confs[version]
    elif version == 'gim_lightglue':
        feature_conf = extract_features.confs['gim_superpoint']
        matcher_conf = match_features.confs[version]
    
    # Step 1: 提取 NetVLAD 全局描述子（用于生成匹配对）
    netvlad_conf = extract_features.confs['netvlad']
    netvlad_out = outputs / 'global-feats-netvlad.h5'
    if not netvlad_out.exists():
        print("Step 1: Extracting NetVLAD global features...")
        netvlad_path = extract_features.main(netvlad_conf, images, outputs)
    else:
        netvlad_path = netvlad_out
        print(f"Using existing NetVLAD features: {netvlad_path}")
    
    # Step 2: 生成序列匹配对（带 NetVLAD 筛选）
    if not image_pairs.exists():
        print("Step 2: Generating sequential pairs with NetVLAD filtering...")
        generate_sequential_pairs_with_netvlad(
            netvlad_path,
            image_pairs,
            window=90,
            sim_thresh=0.20
        )
    else:
        print(f"Pairs file {image_pairs} already exists. Using existing pairs.")
    
    # Step 3: 语义分割（保留原逻辑，与掩膜过滤相互独立）
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
                                                 mask_dir=mask_dir)   # <-- 传入 mask_dir
            match_path = match_features.main(matcher_conf, image_pairs,
                                             feature_conf['output'], outputs,
                                             model=model)

        elif version == 'gim_dkm':
            # DKM dense matching: 若要使用 mask，需分步提取特征
            dense_feat_conf = extract_features.confs['gim_superpoint']
            feature_path = extract_features.main(dense_feat_conf, images, outputs,
                                                 mask_dir=mask_dir)   # <-- 传入 mask_dir
            match_path = match_dense.main(matcher_conf, image_pairs,
                                          dense_feat_conf['output'], outputs)
    
    # Step 5: 稀疏重建
    print("Step 4: Running Sparse Reconstruction...")
    opts = dict(camera_model='PINHOLE')
    reconstruction.main(sfm_dir, images, image_pairs, feature_path, match_path,
                        image_options=opts, stop_after_db=stop_after_db)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--version', type=str,
                        choices={'gim_dkm', 'gim_lightglue'},
                        default='gim_dkm')
    parser.add_argument('--stop_after_db', action='store_true',
                        help='Stop after generating COLMAP database, skip reconstruction.')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing binary masks (PNG/NPY) to filter dynamic keypoints. '
                             'White (255) regions will be removed. If not set, defaults to inputs/<scene>/masks '
                             'if it exists.')
    args = parser.parse_args()
    main(args.scene_name, args.version, stop_after_db=args.stop_after_db, mask_dir=args.mask_dir)
