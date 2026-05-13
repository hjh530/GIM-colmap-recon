# GIM-COLMAP-Recon

[English](#english) | [中文](#chinese)

---

<a id="english"></a>

## 🇬🇧 English

### Overview

This project is a modified version of [GIM](https://github.com/xuelunshen/gim), specifically tailored for generating **COLMAP-compatible feature and match databases** in challenging indoor/outdoor scenarios. It is designed to handle extreme conditions such as:
- Large viewpoint changes
- Low-texture or texture-less surfaces
- Repetitive patterns

By leveraging robust learned matchers like **SuperPoint + LightGlue / DKM**, the pipeline produces reliable keypoints and correspondences that significantly improve the success rate of subsequent 3D reconstruction.

The generated database can be directly used in:
- COLMAP GUI or command-line interface for triangulation and bundle adjustment (BA)
- [GLOMAP](https://github.com/colmap/glomap) for global sparse reconstruction

---

### Installation

Please follow the environment setup instructions from the original [GIM repository](https://github.com/xuelunshen/gim). Ensure all dependencies (PyTorch, pycolmap, hloc, etc.) are correctly installed.
```bash
git clone --recursive https://github.com/hjh530/GIM-colmap-recon.git
cd GIM-colmap-recon
pip install -r requirements.txt
```

#### MASt3R matching (optional)

For the `--version mast3r` pipeline, install additional dependencies:
```bash
conda create --name gim-MASt3R --clone gim
conda activate gim-MASt3R
pip install -r requirements-mast3r.txt
```

The MASt3R checkpoint will be downloaded automatically from HuggingFace Hub on first use, or place it manually under `weights/mast3r/`.


---

### Usage

#### 1. Run full pipeline
```bash
sh reconstruction.sh
```
This command executes the complete workflow: feature extraction, matching, and sparse reconstruction.

**Available matching backends (`--version`):**
- `gim_dkm` (default) — DKM dense matching
- `gim_lightglue` — SuperPoint + LightGlue
- `mast3r` — [MASt3R](https://github.com/naver/mast3r/tree/mast3r_sfm) single-pass dense matching (requires `gim-MASt3R` environment)

Example with MASt3R:
```bash
conda activate gim-MASt3R
python reconstruction.py --scene_name my_scene --version mast3r --stop_after_db
python reconstruction.py --scene_name my_scene --version mast3r  # full reconstruction
```

**MASt3R options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mast3r_maxdim` | 512 | Max image dimension for inference |
| `--mast3r_conf_thr` | 1.001 | Descriptor confidence threshold |
| `--mast3r_pixel_tol` | 5 | Tolerance for iterative NN refinement |
| `--mast3r_subsample` | 8 | Grid step for sparse matching |
| `--mast3r_min_track_len` | 3 | Minimum track length |

**Mask filtering:** By default, if the directory `inputs/<scene_name>/masks` exists, the pipeline will automatically load binary masks (255 = ignored regions, 0 = background) to filter keypoints on dynamic objects. Set `--mask_dir` to use a custom path.

#### 2. Generate database only

In `reconstruction.sh` add the `--stop_after_db` flag to stop after database creation:
```bash
python reconstruction.py --scene_name ${scene_name} --version ${version} --stop_after_db
```


This produces a `database.db` file ready for external reconstruction tools.

#### 3. Reconstruction


You can adjust the pair selection strategy in `reconstruction.py` according to your data characteristics:
- **Exhaustive matching** – suitable for small image sets.
- **Sequential matching** – ideal for ordered sequences (e.g., video frames).
- **NetVLAD retrieval** – recommended for large, unordered collections.

After generating the database, you have two options for sparse reconstruction:

- **GLOMAP (global SfM)** – Integrated in recent COLMAP releases. It is significantly faster than incremental SfM while achieving comparable accuracy. Example command:
```bash
colmap global_mapper --database_path H:\JG2\database2.db --image_path H:\JG2\images --output_path H:\JG2\sparse5
```
- **Incremental SfM (COLMAP GUI)** – More beginner-friendly. Simply open `database.db` in the COLMAP GUI and follow the standard reconstruction steps (feature matching is already completed).
Robust feature extraction and matching are often the key to successful reconstruction—especially when default COLMAP methods fail. Unless the input data is extremely poor, this pipeline will likely produce a usable model.
---

### Acknowledgements

- [GIM](https://github.com/xuelunshen/gim)
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [GLOMAP](https://github.com/colmap/glomap)

---

<a id="chinese"></a>

## 🇨🇳 中文

### 项目简介

本项目基于 [GIM](https://github.com/xuelunshen/gim) 修改而来，专门用于生成 **COLMAP 格式的特征与匹配数据库**，以应对室内外极端场景下的三维重建挑战，包括：
- 大视角差
- 弱纹理或缺乏纹理区域
- 重复纹理结构

通过集成 **SuperPoint + LightGlue / DKM** 等鲁棒性更强的特征匹配方法，本工具能够在困难条件下获得更可靠的匹配结果，从而提升后续重建的成功率。

生成的数据库可直接用于：
- COLMAP 图形界面或命令行进行三角测量与光束法平差 (BA)
- [GLOMAP](https://github.com/colmap/glomap) 全局式稀疏重建

---

### 安装步骤

请参照原项目 [GIM](https://github.com/xuelunshen/gim) 的环境配置指南，确保已正确安装 PyTorch、pycolmap、hloc 等依赖。
```bash
git clone --recursive https://github.com/hjh530/GIM-colmap-recon.git
cd GIM-colmap-recon
pip install -r requirements.txt
```

#### MASt3R 匹配（可选）

使用 `--version mast3r` 需要额外安装：
```bash
conda create --name gim-MASt3R --clone gim
conda activate gim-MASt3R
pip install -r requirements-mast3r.txt
```

MASt3R 模型权重默认从 HuggingFace Hub 自动下载，也可手动放置于 `weights/mast3r/` 目录。


---

### 使用方法

#### 1. 运行完整流程
```bash
sh reconstruction.sh
```
这段命令执行完整工作流程：特征提取、匹配和稀疏重建。

**可用的匹配后端（`--version`）：**
- `gim_dkm`（默认）— DKM 密集匹配
- `gim_lightglue` — SuperPoint + LightGlue
- `mast3r` — [MASt3R](https://github.com/naver/mast3r/tree/mast3r_sfm) 单次推理密集匹配（需要 `gim-MASt3R` 环境）

MASt3R 示例：
```bash
conda activate gim-MASt3R
python reconstruction.py --scene_name my_scene --version mast3r --stop_after_db
python reconstruction.py --scene_name my_scene --version mast3r  # 完整重建
```

**掩码过滤：** 默认情况下，如果目录 `inputs/<scene_name>/masks` 存在，流程会自动加载二值掩码（255 = 忽略区域，0 = 背景），过滤动态物体上的关键点。使用 `--mask_dir` 可自定义路径。

#### 2. 仅生成数据库（跳过重建）

在 `reconstruction.sh` 中添加 `--stop_after_db` 标志即可在生成数据库后停止：
```bash
python reconstruction.py --scene_name ${scene_name} --version ${version} --stop_after_db
```


此时程序会在生成 `database.db` 文件后退出，方便您后续使用外部工具进行重建。

#### 3. 完成重建

您可以根据数据特点在 `reconstruction.py` 中调整匹配对选择策略：
- **穷举匹配** – 适合小规模图像集。
- **顺序匹配** – 适合有序序列（如视频帧）。
- **NetVLAD 检索** – 推荐用于大规模无序图像集。

生成数据库后，有两种稀疏重建方式可选：

- **GLOMAP 全局式重建** – 已集成在最新版 COLMAP 中。相比增量式重建速度更快，精度相当。示例命令：
```bash
colmap global_mapper --database_path H:\JG2\database2.db --image_path H:\JG2\images --output_path H:\JG2\sparse5
```
- **增量式重建（COLMAP GUI）** – 更适合新手。在 COLMAP 图形界面中打开 `database.db`，按常规步骤操作即可（特征匹配已完成，可直接跳至三角测量）。

更稳健的特征点提取与匹配往往能生成更准确的相机位姿。当您使用 COLMAP 默认算法无法重建时，不妨尝试本工具——除非数据质量极差，否则大概率能够成功。

---

### 致谢

- [GIM](https://github.com/xuelunshen/gim)
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [GLOMAP](https://github.com/colmap/glomap)
