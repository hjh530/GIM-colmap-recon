# GIM-MASt3R: Robust SfM Matching & COLMAP Reconstruction

[English](#english) | [中文](#chinese)

---

<a id="english"></a>

## English

### Overview

GIM-MASt3R generates **COLMAP-compatible databases** for sparse 3D reconstruction in challenging indoor/outdoor scenarios. It integrates three state-of-the-art matching methods:

| Method | `--version` | Description |
|--------|-------------|-------------|
| **MASt3R** | `mast3r` | Single-pass dense descriptor matching via [MASt3R](https://github.com/naver/mast3r/tree/mast3r_sfm). Produces per-pixel 3D point maps + 24-dim dense descriptors. One forward pass per image pair, no separate feature extractor needed. Best for extreme viewpoint changes and low-texture surfaces. |
| DKM | `gim_dkm` | Dense kernelized matching from [GIM](https://github.com/xuelunshen/gim). |
| SuperPoint + LightGlue | `gim_lightglue` | Sparse learned keypoints + transformer matcher from [GIM](https://github.com/xuelunshen/gim). |

The generated database can be directly used in:
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

# 3. Install PyTorch (CUDA 11.8 for RTX 4090; adjust for your GPU)
pip install torch>=2.0 torchvision>=0.15 --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt
```

#### MASt3R Checkpoint

The MASt3R model checkpoint (~2.3 GB) must be placed under `weights/mast3r/`:

```bash
# Option A: Download from HuggingFace Hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                  local_dir='weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
"

# Option B: Manual download
# Download from https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
# Place config.json and model.safetensors in:
#   weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/
```

> `weights/` is git-ignored. The checkpoint stays local.

---

### Usage

#### 1. Run full pipeline

```bash
python reconstruction.py --scene_name <scene_name> --version <version>
```

Place images in `inputs/<scene_name>/images/`. Output goes to `outputs/<scene_name>/<version>/`.

#### 2. Generate database only (skip reconstruction)

```bash
python reconstruction.py --scene_name <scene_name> --version mast3r --stop_after_db
```

#### 3. MASt3R options

| Flag | Default | Description |
|------|---------|-------------|
| `--mast3r_maxdim` | 512 | Max image dimension for inference |
| `--mast3r_conf_thr` | 1.001 | Descriptor confidence threshold |
| `--mast3r_pixel_tol` | 5 | Tolerance for iterative NN refinement |
| `--mast3r_subsample` | 8 | Grid step for sparse matching (dense: set to 1) |
| `--mast3r_min_track_len` | 3 | Minimum track length to retain a keypoint |

#### 4. Mask filtering (gim_dkm / gim_lightglue only)

If `inputs/<scene_name>/masks/` exists, binary masks (255 = ignored, 0 = background) are automatically loaded to filter keypoints on dynamic objects. Use `--mask_dir` to specify a custom path.

#### 5. Pair selection strategies

For `gim_*` modes, the pair strategy can be adjusted in `reconstruction.py`:
- **Sequential** — ordered sequences (video frames), default
- **Exhaustive** — small image sets
- **NetVLAD retrieval** — large unordered collections

For `mast3r` mode, sequential pairs with configurable window size are used.

#### 6. External reconstruction

After generating `database.db`, you can run:

```bash
# COLMAP incremental mapper
colmap mapper --database_path database.db --image_path <images> --output_path <output>

# GLOMAP global mapper
colmap global_mapper --database_path database.db --image_path <images> --output_path <output>
```

---

### Benchmarks

Tested on RTX 4090 with PyTorch 2.5.1, CUDA 12.1:

| Scene | Images | Pairs | Tracks | Database Time | DB Size |
|-------|--------|-------|--------|---------------|---------|
| picture | 70 | 690 | 174K | ~3 min | 38 MB |
| JG | 550 | 8,050 | 1,524K | ~58 min | 273 MB |

---

### Acknowledgements

- [MASt3R](https://github.com/naver/mast3r) — Grounding Image Matching in 3D with MASt3R
- [GIM](https://github.com/xuelunshen/gim) — Geometric Image Matching
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
| **MASt3R** | `mast3r` | 单次推理密集描述子匹配。每像素输出 3D 点云 + 24 维密集描述子，无需单独的特征提取器。最适合大视角差和弱纹理场景。 |
| DKM | `gim_dkm` | 密集核化匹配 |
| SuperPoint + LightGlue | `gim_lightglue` | 稀疏学习关键点 + Transformer 匹配器 |

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

# 3. 安装 PyTorch（根据 GPU 调整 CUDA 版本）
pip install torch>=2.0 torchvision>=0.15 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其余依赖
pip install -r requirements.txt
```

#### MASt3R 模型权重

MASt3R 权重（~2.3 GB）需放置于 `weights/mast3r/` 目录：

```bash
# 方式 A：从 HuggingFace Hub 自动下载
python -c "
from huggingface_hub import snapshot_download
snapshot_download('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                  local_dir='weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
"

# 方式 B：手动下载
# 从 https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric 下载
# 将 config.json 和 model.safetensors 放入：
#   weights/mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/
```

> `weights/` 已被 git 忽略，权重不会上传到仓库。

---

### 使用方法

#### 1. 运行完整流程

```bash
python reconstruction.py --scene_name <场景名> --version <方法>
```

将图片放入 `inputs/<场景名>/images/`。结果输出到 `outputs/<场景名>/<方法>/`。

#### 2. 仅生成数据库

```bash
python reconstruction.py --scene_name <场景名> --version mast3r --stop_after_db
```

#### 3. MASt3R 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mast3r_maxdim` | 512 | 推理最大图像边长 |
| `--mast3r_conf_thr` | 1.001 | 描述子置信度阈值 |
| `--mast3r_pixel_tol` | 5 | 迭代最近邻细化容差 |
| `--mast3r_subsample` | 8 | 稀疏匹配采样步长（密集匹配设为 1） |
| `--mast3r_min_track_len` | 3 | 保留关键点的最小 track 长度 |

#### 4. 掩码过滤（仅 gim_dkm / gim_lightglue）

若 `inputs/<场景名>/masks/` 存在，会自动加载二值掩码（255=忽略，0=背景）过滤动态物体上的关键点。使用 `--mask_dir` 可自定义路径。

#### 5. 外部重建

生成 `database.db` 后：

```bash
# COLMAP 增量式
colmap mapper --database_path database.db --image_path <images> --output_path <output>

# GLOMAP 全局式
colmap global_mapper --database_path database.db --image_path <images> --output_path <output>
```

---

### 测试数据

RTX 4090, PyTorch 2.5.1, CUDA 12.1：

| 场景 | 图片数 | 匹配对数 | Tracks | 数据库耗时 | 数据库大小 |
|------|--------|----------|--------|------------|------------|
| picture | 70 | 690 | 174K | ~3 分钟 | 38 MB |
| JG | 550 | 8,050 | 1,524K | ~58 分钟 | 273 MB |

---

### 致谢

- [MASt3R](https://github.com/naver/mast3r)
- [GIM](https://github.com/xuelunshen/gim)
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [GLOMAP](https://github.com/colmap/glomap)
