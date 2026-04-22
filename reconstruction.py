# -*- coding: utf-8 -*-
import os
import torch
import warnings
import numpy as np
import h5py
import re
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
# 1. 辅助函数
# =========================================================

def extract_image_number(name):
    """从文件名中提取数字"""
    match = re.search(r'(\d+)', name)
    if match:
        return int(match.group(1))
    return -1

def get_descriptors_subset(names, all_names, all_desc_tensor, name2idx):
    indices = [name2idx[n] for n in names]
    return all_desc_tensor[indices]

def match_groups(query_names, db_names, query_desc, db_desc, num_matched, device, thresh=None):
    """NetVLAD 相似度匹配辅助函数"""
    print(f"  - Matching {len(query_names)} queries vs {len(db_names)} db images (Top-{num_matched})...")
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
            if q_name == d_name: continue
            pair = tuple(sorted((q_name, d_name)))
            pairs.append(pair)
            
    del sim, query_desc, db_desc
    torch.cuda.empty_cache()
    return pairs

def get_sequential_pairs(image_names, overlap=30):
    """
    序列匹配：构建刚性骨架
    image_names: 已排序的图片名称列表
    overlap: 向后连接的窗口大小
    """
    pairs = set()
    # 再次确保排序，虽然外面排过了
    image_names = sorted(image_names)
    N = len(image_names)
    
    print(f"  - Generating PURE sequential pairs for {N} images (Window={overlap})...")
    
    for i in range(N):
        # 让第 i 张图去连接后面 [i+1, i+1+overlap] 张图
        for j in range(i + 1, min(i + 1 + overlap, N)):
            pair = tuple(sorted((image_names[i], image_names[j])))
            pairs.add(pair)
            
    return list(pairs)

def generate_custom_mixed_pairs(descriptors_path, output_path):
    print(f"Generating custom mixed pairs using NetVLAD features from {descriptors_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    all_names = list_h5_names(descriptors_path)
    name2idx = {n: i for i, n in enumerate(all_names)}
    
    with h5py.File(str(descriptors_path), "r", libver="latest") as fd:
        all_desc = [fd[n]["global_descriptor"].__array__() for n in all_names]
    
    all_desc_tensor = torch.from_numpy(np.stack(all_desc, 0)).float()
    
    ground_names = sorted([n for n in all_names if n.startswith("Terr_")])
    aerial_names = [n for n in all_names if not n.startswith("Terr_")]
    
    print(f"Found {len(ground_names)} Ground images and {len(aerial_names)} Aerial images.")
    
    ground_desc = get_descriptors_subset(ground_names, all_names, all_desc_tensor, name2idx)
    aerial_desc = get_descriptors_subset(aerial_names, all_names, all_desc_tensor, name2idx)
    
    final_pairs = set()
    
    # ---------------------------------------------------------
    # Task A: 地面重建 (【关键修改】纯序列模式)
    # ---------------------------------------------------------
    if len(ground_names) > 0:
        print("Task A: Ground-to-Ground Pure Sequential Matching...")
        
        # 1. 纯序列匹配 (Pure Sequence)
        # 将 overlap 增加到 30，形成极粗的链条，防止中间断开
        # 彻底放弃 NetVLAD 全局搜索，防止"前门连后门"
        seq_pairs = get_sequential_pairs(ground_names, overlap=10)
        final_pairs.update(seq_pairs)
        
        print(f"    [Task A] Added {len(seq_pairs)} rigid sequential pairs. (Global matching REMOVED)")

    # ---------------------------------------------------------
    # Task B: 空中重建 (保持 Top-20)
    # ---------------------------------------------------------
    if len(aerial_names) > 0:
        print("Task B: Aerial-to-Aerial Matching (Top-20)...")
        pairs_aa = match_groups(aerial_names, aerial_names, aerial_desc, aerial_desc, 20, device)
        final_pairs.update(pairs_aa)
        
    # ---------------------------------------------------------
    # Task C: 空地融合 (保持人工范围 [2639, 2836] 锚定正面)
    # ---------------------------------------------------------
    if len(ground_names) > 0 and len(aerial_names) > 0:
        # 使用你之前的人工范围
        target_start = 1000
        target_end = 5000 # 或者是你刚才设置的 3036，请确认一下你的文件名
        print(f"Task C: Manual Range Constraint Matching [{target_start}, {target_end}]...")
        
        target_ground_names = []
        for name in ground_names:
            num = extract_image_number(name)
            if target_start <= num <= target_end:
                target_ground_names.append(name)
        
        print(f"    [Filter] Found {len(target_ground_names)} ANCHOR ground images.")
        
        if len(target_ground_names) > 0:
            target_ground_desc = get_descriptors_subset(target_ground_names, all_names, all_desc_tensor, name2idx)
            
            # 地面锚点 -> 空中 (Top-20)
            pairs_ga = match_groups(
                target_ground_names, 
                aerial_names, 
                target_ground_desc, 
                aerial_desc, 
                num_matched=5, 
                device=device,
                thresh=0.15
            )
            final_pairs.update(pairs_ga)
            print(f"    [Task C] Added {len(pairs_ga)} constrained pairs.")
        else:
            print("    [Warning] No ground images found in specified range!")

    print(f"Total unique pairs generated: {len(final_pairs)}")
    with open(output_path, "w") as f:
        f.write("\n".join(" ".join(p) for p in final_pairs))


def segmentation(images, segment_root, matcher_conf):
    print("Skipping segmentation for LightGlue pipeline.")
    return

# =========================================================
# 2. 主流程
# =========================================================

def main(scene_name, version):
    # Setup
    images = Path('inputs') / scene_name / 'images'
    outputs = Path('outputs') / scene_name / version
    outputs.mkdir(parents=True, exist_ok=True)
    os.environ['GIMRECONSTRUCTION'] = str(outputs)
    
    segment_root = outputs / 'segment'
    segment_root.mkdir(parents=True, exist_ok=True)
    
    sfm_dir = outputs / 'sparse'
    image_pairs = outputs / 'pairs-custom-mixed.txt'
    
    feature_conf = matcher_conf = None
    if version == 'gim_lightglue':
        feature_conf = extract_features.confs['gim_superpoint']
        matcher_conf = match_features.confs[version]

    # Step 1: 生成匹配对
    if not image_pairs.exists():
        print("Step 1: Running NetVLAD to generate custom pairs...")
        netvlad_conf = extract_features.confs['netvlad']
        netvlad_out = outputs / 'global-feats-netvlad.h5'
        if not netvlad_out.exists():
            netvlad_path = extract_features.main(netvlad_conf, images, outputs)
        else:
            netvlad_path = netvlad_out
        generate_custom_mixed_pairs(netvlad_path, image_pairs)
    else:
        print(f"Pairs file {image_pairs} already exists. Using existing pairs.")

    # Step 2: 语义分割 (跳过)
    segmentation(images, segment_root, matcher_conf)

    # Step 3: LightGlue
    print(f"Step 3: Running Feature Extraction & Matching ({version})...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        checkpoints_path = join('weights', 'gim_lightglue_100h.ckpt')
        
        # SuperPoint
        print("Loading SuperPoint...")
        detector = SuperPoint({
            'max_num_keypoints': 8192, 'force_num_keypoints': True, 
            'detection_threshold': 0.0, 'nms_radius': 3, 'trainable': False
        })
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'): state_dict.pop(k)
            if k.startswith('superpoint.'):
                state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
        detector.load_state_dict(state_dict)
        feature_path = extract_features.main(feature_conf, images, outputs, model=detector)

        # LightGlue
        print("Loading LightGlue...")
        model = LightGlue({
            'filter_threshold': 0.1, 'flash': False, 'checkpointed': True
        })
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('superpoint.'): state_dict.pop(k)
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        match_path = match_features.main(
            matcher_conf, image_pairs, feature_conf['output'], outputs, model=model
        )

    # Step 4: 稀疏重建
    print("Step 4: Running Sparse Reconstruction...")
    
    # 坚持 PINHOLE
    opts = dict(camera_model='PINHOLE') 
    
    # LightGlue 很准，可以稍微放松
    mapper_opts = dict(min_num_matches=10) 

    reconstruction.main(
        sfm_dir, images, image_pairs, feature_path, match_path,
        image_options=opts, mapper_options=mapper_opts
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--version', type=str, default='gim_lightglue')
    args = parser.parse_args()
    
    main(args.scene_name, args.version)






