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


---

### Usage

#### 1. Run full pipeline
```bash
sh reconstruction.sh
```


This script processes input images, extracts features, performs matching, and (by default) runs sparse reconstruction. Modify the script arguments as needed.

#### 2. Generate database only

In `reconstruction.sh` add the `--stop_after_db` flag to stop after database creation:
```bash
python reconstruction1.py --scene_name ${scene_name} --version ${version} --stop_after_db
```


This produces a `database.db` file ready for external reconstruction tools.

#### 3. Reconstruction

- **COLMAP**: Open `database.db` and proceed directly to triangulation.
- **GLOMAP**: Follow [GLOMAP instructions](https://github.com/colmap/glomap) for global SfM.

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


---

### 使用方法

#### 1. 运行完整流程
```bash
sh reconstruction.sh
```


该脚本将处理输入图像、提取特征、执行匹配，默认还会运行稀疏重建。您可以通过修改脚本参数来控制具体行为。

#### 2. 仅生成数据库（跳过重建）

在 `reconstruction.sh` 中添加 `--stop_after_db` 标志即可在生成数据库后停止：
```bash
python reconstruction1.py --scene_name ${scene_name} --version ${version} --stop_after_db
```


此时程序会在生成 `database.db` 文件后退出，方便您后续使用外部工具进行重建。

#### 3. 完成重建

- **COLMAP**：打开生成的 `database.db`，由于特征与匹配已完成，可直接跳转至三角测量步骤。
- **GLOMAP**：按照 [GLOMAP 官方指南](https://github.com/colmap/glomap) 使用相同数据库进行全局式运动恢复结构。

---

### 致谢

- [GIM](https://github.com/xuelunshen/gim)
- [COLMAP](https://colmap.github.io/)
- [HLOC](https://github.com/cvg/Hierarchical-Localization)
- [GLOMAP](https://github.com/colmap/glomap)
