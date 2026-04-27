import os
import re
import torch
import warnings
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from os.path import join
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image

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
    indices = [name2idx[n] for n in names]
    return all_desc_tensor[indices]


def match_groups(query_names, db_names, query_desc, db_desc, num_matched, device, thresh=None):
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


def generate_sequential_pairs_with_netvlad(
    descriptors_path, output_path, images_path,
    window=20, sim_thresh=0.15
):
    print(f"Generating sequential pairs (folder order) with NetVLAD filtering...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====== 读取 descriptor ======
    all_names = list_h5_names(descriptors_path)
    name2idx = {n: i for i, n in enumerate(all_names)}

    with h5py.File(str(descriptors_path), "r", libver="latest") as fd:
        all_desc = [fd[n]["global_descriptor"].__array__() for n in all_names]

    all_desc_tensor = torch.from_numpy(np.stack(all_desc, 0)).float()

    # ====== 使用文件夹默认顺序（不排序） ======
    image_list = os.listdir(images_path)

    # 只保留 descriptor 中存在的
    sorted_names = [n for n in image_list if n in all_names]

    N = len(sorted_names)
    final_pairs = set()

    print("Step 1: Sequential matching (folder order)...")

    for i in range(N):
        q_name = sorted_names[i]
        q_desc = get_descriptors_subset([q_name], all_names, all_desc_tensor, name2idx)

        db_indices = [(i + offset) % N for offset in range(1, window + 1)]
        db_names = [sorted_names[idx] for idx in db_indices]
        db_desc = get_descriptors_subset(db_names, all_names, all_desc_tensor, name2idx)

        pairs = match_groups(
            [q_name], db_names,
            q_desc, db_desc,
            num_matched=len(db_names),
            device=device,
            thresh=sim_thresh
        )

        final_pairs.update(pairs)

    print(f"Total unique pairs: {len(final_pairs)}")

    with open(output_path, "w") as f:
        f.write("\n".join(" ".join(p) for p in final_pairs))


def segmentation(images, segment_root, matcher_conf):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            segment_path = join(segment_root, f'{img[:-4]}.npy')
            if not os.path.exists(segment_path):
                rgb = read_image(images / img, dataset.conf.grayscale)
                mask = segment(rgb, 1920, device, segmentation_module)
                np.save(segment_path, mask)


def main(scene_name, version, stop_after_db, mask_dir):
    images = Path('inputs') / scene_name / 'images'
    outputs = Path('outputs') / scene_name / version
    outputs.mkdir(parents=True, exist_ok=True)

    os.environ['GIMRECONSTRUCTION'] = str(outputs)

    segment_root = Path('outputs') / scene_name / 'segment'
    segment_root.mkdir(parents=True, exist_ok=True)

    sfm_dir = outputs / 'sparse'
    database_path = sfm_dir / 'database.db'
    image_pairs = outputs / 'pairs-sequential.txt'

    feature_conf = matcher_conf = None

    if version == 'gim_dkm':
        matcher_conf = match_dense.confs[version]
    elif version == 'gim_lightglue':
        feature_conf = extract_features.confs['gim_superpoint']
        matcher_conf = match_features.confs[version]

    # ===== NetVLAD =====
    netvlad_conf = extract_features.confs['netvlad']
    netvlad_out = outputs / 'global-feats-netvlad.h5'

    if not netvlad_out.exists():
        print("Step 1: Extracting NetVLAD...")
        netvlad_path = extract_features.main(netvlad_conf, images, outputs)
    else:
        netvlad_path = netvlad_out

    # ===== 生成匹配对 =====
    if not image_pairs.exists():
        print("Step 2: Generating pairs...")
        generate_sequential_pairs_with_netvlad(
            netvlad_path,
            image_pairs,
            images,
            window=90,
            sim_thresh=0.20
        )

    # ===== segmentation（原流程，可保留） =====
    segmentation(images, segment_root, matcher_conf)

    # ===== 特征匹配（动态 mask 注入） =====
    print("Step 3: Feature matching...")

    import hloc.utils.io as hloc_io
    original_read_image = hloc_io.read_image

    # 决定 mask 目录
    mask_dir_path = None
    if mask_dir is None:
        default_mask_dir = Path('inputs') / scene_name / 'masks'
        if default_mask_dir.exists():
            mask_dir_path = default_mask_dir
    elif Path(mask_dir).exists():
        mask_dir_path = Path(mask_dir)

    # 如果存在 mask 目录，替换 read_image
    if mask_dir_path is not None:
        print(f"[Dynamic Mask] Applying masks from {mask_dir_path} during feature extraction")

        def masked_read_image(path, grayscale=False):
            # 读取原始图像 (numpy array)
            img = original_read_image(path, grayscale=grayscale)

            # 查找对应的 mask 文件
            img_path = Path(path)
            stem = img_path.stem
            mask_files = list(mask_dir_path.glob(f"{stem}_mask.*")) + list(
                mask_dir_path.glob(f"{stem}.*"))
            if mask_files:
                mask_file = mask_files[0]
                mask = np.array(Image.open(mask_file).convert('L'))  # 0-255

                # 确保 mask 尺寸与图像匹配
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)

                # 白色区域 (>128) 视为要忽略的区域，涂白
                white_mask = mask > 128
                if grayscale:
                    img[white_mask] = 255
                else:
                    img[white_mask] = (255, 255, 255)  # RGB 白色
            return img

        hloc_io.read_image = masked_read_image

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            if version == 'gim_dkm':
                feature_path, match_path = match_dense.main(
                    matcher_conf, image_pairs, images, outputs)

            elif version == 'gim_lightglue':
                checkpoints_path = join('weights', 'gim_lightglue_100h.ckpt')

                detector = SuperPoint({
                    'max_num_keypoints': 12000,
                    'force_num_keypoints': True,
                    'detection_threshold': 0.0,
                    'nms_radius': 3,
                    'trainable': False,
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

                model = LightGlue({
                    'filter_threshold': 0.2,
                    'flash': False,
                    'checkpointed': True,
                })

                state_dict = torch.load(checkpoints_path, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']

                for k in list(state_dict.keys()):
                    if k.startswith('superpoint.'):
                        state_dict.pop(k)
                    if k.startswith('model.'):
                        state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)

                model.load_state_dict(state_dict)

                feature_path = extract_features.main(feature_conf, images, outputs, model=detector)
                match_path = match_features.main(
                    matcher_conf, image_pairs,
                    feature_conf['output'], outputs,
                    model=model
                )
    finally:
        # 恢复原始 read_image
        if mask_dir_path is not None:
            hloc_io.read_image = original_read_image

    # ===== 重建 =====
    print("Step 4: Reconstruction...")
    opts = dict(camera_model='PINHOLE')

    reconstruction.main(
        sfm_dir,
        images,
        image_pairs,
        feature_path,
        match_path,
        image_options=opts,
        stop_after_db=stop_after_db
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene_name', type=str, required=True)
    parser.add_argument('--version', type=str,
                        choices={'gim_dkm', 'gim_lightglue'},
                        default='gim_dkm')
    parser.add_argument('--stop_after_db', action='store_true')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing binary masks (PNG/NPY). '
                             'White (255) regions will be painted white in memory during feature extraction. '
                             'If not set, defaults to inputs/<scene>/masks if it exists.')

    args = parser.parse_args()

    main(args.scene_name, args.version, stop_after_db=args.stop_after_db, mask_dir=args.mask_dir)
