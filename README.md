# GIM-MASt3R: Robust SfM Matching & COLMAP Reconstruction

[English](#english) | [中文](#chinese)

---

<a id="english"></a>

## English

### Overview

GIM-MASt3R generates **COLMAP-compatible databases** for sparse 3D reconstruction in challenging indoor/outdoor scenarios. It integrates three state-of-the-art matching methods:

| Method | `--version` | Description |
|--------|-------------|-------------|
| **MASt3R** | `mast3r` | Single-pass dense descriptor matching via [MASt3R](https://github.com/naver/mast3r/tree/mast3r_sfm). Per-pixel 3D points + 24-dim dense descriptors. Encoder caching avoids re-encoding images across pairs. Best for extreme viewpoint changes and low-texture scenes. |
| DKM | `gim_dkm` | Dense kernelized matching from [GIM](https://github.com/xuelunshen/gim). |
| SuperPoint + LightGlue | `gim_lightglue` | Sparse learned keypoints + transformer matcher from [GIM](https://github.com/xuelunshen/gim). |

All three versions share the same **NetVLAD-filtered sequential pair generation** (window=20, sim_thresh=0.20).

Output database is directly usable in:
- COLMAP GUI or CLI for triangulation and bundle adjustment (BA)
- [GLOMAP](https://github.com/colmap/glomap) for global sparse reconstruction

---

### Installation

```bash
# 1. Clone
git clone --recursive https://github.com/hjh530/GIM-colmap-recon.git
cd GIM-colmap-recon

# 2. Create environment
conda create -n gim-MASt3R python=3.9
conda activate gim-MASt3R

# 3. Install PyTorch 2.x (adjust CUDA version for your GPU)
pip install torch>=2.0 torchvision>=0.15 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt
```

#### MASt3R Checkpoint

The MASt3R checkpoint (~2.3 GB) must be placed under `weights/mast3r/`:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                  local_dir='weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
"
```

> `weights/` is git-ignored. The checkpoint stays local.

#### RoPE CUDA Kernel (optional, ~10-20% speedup)

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../..
```

---

### Usage

#### 1. Run full pipeline

```bash
python reconstruction.py --scene_name <scene_name> --version <version>
```

Place images in `inputs/<scene_name>/images/`. Output goes to `outputs/<scene_name>/<version>/`.

#### 2. Generate database only

```bash
python reconstruction.py --scene_name <scene_name> --version <version> --stop_after_db
```

#### 3. MASt3R options

| Flag | Default | Description |
|------|---------|-------------|
| `--mast3r_maxdim` | 512 | Max image dimension (lower = faster, e.g. 384) |
| `--mast3r_conf_thr` | 1.001 | Descriptor confidence threshold |
| `--mast3r_pixel_tol` | 5 | Tolerance for iterative NN refinement |
| `--mast3r_subsample` | 8 | Grid step for sparse matching (higher = faster, e.g. 12) |
| `--mast3r_min_track_len` | 3 | Minimum track length to retain a keypoint |

#### 4. Mask filtering (all versions)

Binary masks (255 = ignored, 0 = keep) are auto-loaded from `inputs/<scene_name>/masks/` if present. Use `--mask_dir` to specify a custom path. Supports both `gim_*` (pre-extraction filtering) and `mast3r` (post-matching filtering).

#### 5. External reconstruction

```bash
colmap mapper --database_path database.db --image_path <images> --output_path <output>
# or
colmap global_mapper --database_path database.db --image_path <images> --output_path <output>
```

---

### Benchmarks

Tested on RTX 4090, PyTorch 2.5.1, CUDA 12.1, 70 images (picture scene):

| Method | Registered | Points3D | Observations | Notes |
|--------|-----------|----------|--------------|-------|
| **mast3r** | **70 / 70** | 175,490 | 1,075,023 | RoPE CUDA + encoder cache |
| gim_dkm | 54 / 70 | 40,210 | — | 8192 kp/image |
| gim_lightglue | 52 / 70 | 27,948 | — | LightGlue matching |

JG scene (550 images, mast3r): 1,524,493 tracks, ~58 min, 273 MB database.

---

### Acknowledgements

- [MASt3R](https://github.com/naver/mast3r)
- [GIM](https://github.com/xuelunshen/gim)
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [GLOMAP](https://github.com/colmap/glomap)

---

<a id="chinese"></a>

## 中文

### 项目简介

GIM-MASt3R 为室内外挑战性场景生成 **COLMAP 兼容的匹配数据库**，集成三种前沿匹配方法：

| 方法 | `--version` | 说明 |
|--------|-------------|------|
| **MASt3R** | `mast3r` | 单次推理密集描述子匹配。每像素 3D 点 + 24 维密集描述子，编码器缓存避免重复编码。最适合大视角差和弱纹理场景。 |
| DKM | `gim_dkm` | 密集核化匹配 |
| SuperPoint + LightGlue | `gim_lightglue` | 稀疏学习关键点 + Transformer 匹配器 |

所有版本共用 **NetVLAD 筛选的序列匹配对**（window=20，sim_thresh=0.20）。

生成的数据库可直接用于 COLMAP 或 GLOMAP 进行稀疏重建。

---

### 安装步骤

```bash
# 1. 克隆仓库
git clone --recursive https://github.com/hjh530/GIM-colmap-recon.git
cd GIM-colmap-recon

# 2. 创建环境
conda create -n gim-MASt3R python=3.9
conda activate gim-MASt3R

# 3. 安装 PyTorch 2.x（根据 GPU 调整 CUDA 版本）
pip install torch>=2.0 torchvision>=0.15 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其余依赖
pip install -r requirements.txt
```

#### MASt3R 模型权重

MASt3R 权重（~2.3 GB）需放置于 `weights/mast3r/` 目录：

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                  local_dir='weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
"
```

> `weights/` 已被 git 忽略，权重不会上传到仓库。

#### RoPE CUDA 算子（可选，约 10-20% 加速）

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../..
```

---

### 使用方法

#### 1. 运行完整流程

```bash
python reconstruction.py --scene_name <场景名> --version <方法>
```

将图片放入 `inputs/<场景名>/images/`。结果输出到 `outputs/<场景名>/<方法>/`。

#### 2. 仅生成数据库

```bash
python reconstruction.py --scene_name <场景名> --version <方法> --stop_after_db
```

#### 3. MASt3R 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mast3r_maxdim` | 512 | 推理最大图像边长（越小越快，如 384） |
| `--mast3r_conf_thr` | 1.001 | 描述子置信度阈值 |
| `--mast3r_pixel_tol` | 5 | 迭代最近邻细化容差 |
| `--mast3r_subsample` | 8 | 稀疏匹配采样步长（越大越快，如 12） |
| `--mast3r_min_track_len` | 3 | 保留关键点的最小 track 长度 |

#### 4. 掩码过滤（所有版本）

若 `inputs/<场景名>/masks/` 存在，会自动加载二值掩码（255=忽略，0=保留）。`gim_*` 版本在特征提取前过滤，`mast3r` 在匹配后过滤。使用 `--mask_dir` 可自定义路径。

#### 5. 外部重建

```bash
colmap mapper --database_path database.db --image_path <images> --output_path <output>
# 或
colmap global_mapper --database_path database.db --image_path <images> --output_path <output>
```

---

### 测试数据

RTX 4090, PyTorch 2.5.1, CUDA 12.1, picture 场景（70 张图）：

| 方法 | 注册数 | 3D 点数 | 观测数 | 备注 |
|------|--------|---------|--------|------|
| **mast3r** | **70 / 70** | 175,490 | 1,075,023 | RoPE CUDA + 编码器缓存 |
| gim_dkm | 54 / 70 | 40,210 | — | 8192 关键点/图 |
| gim_lightglue | 52 / 70 | 27,948 | — | LightGlue 匹配 |

JG 场景（550 张，mast3r）：152 万 tracks，约 58 分钟，数据库 273 MB。

---

### 致谢

- [MASt3R](https://github.com/naver/mast3r)
- [GIM](https://github.com/xuelunshen/gim)
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [GLOMAP](https://github.com/colmap/glomap)
