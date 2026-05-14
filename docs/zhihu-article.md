# GIM + COLMAP 做三维重建：更适合复杂场景的图像匹配方案

> 最近在做三维重建和 3D Gaussian Splatting 数据预处理时，我把 MASt3R 集成到了 COLMAP 的匹配管线中，得到了远超默认方法的注册率。这篇文章记录一下整个方案的设计思路和实测结果。

---

## 一、COLMAP 的匹配为什么总出问题？

COLMAP 本身已经是 SfM/MVS 领域非常成熟的工具。但在实际使用中，很多失败案例并不是"重建器不行"，而是前面的**图像匹配**没有做好。

传统 COLMAP 默认使用 SIFT 特征，稳定、经典，但在这些场景里容易翻车：

- 室内弱纹理：大面积白墙、天花板
- 重复纹理：瓷砖、窗格、栏杆
- 大视角差：环绕拍摄时相邻帧之间旋转过大
- 光照变化、运动模糊

相机位姿一旦估错，后面的点云、mesh、3DGS 训练都会跟着出问题。

---

## 二、解决方案：三种匹配后端，一个统一管线

GIM-colmap-recon 不是一个全新的重建系统，而是把三种前沿的**学习式图像匹配方法**接入 COLMAP，用更强的匹配前端来增强重建流程。

| 方法 | 说明 |
|------|------|
| **MASt3R** | 单次推理同时输出 3D 点云 + 24 维密集描述子，无需单独的特征提取器。最强，适合极端场景。 |
| DKM | 密集核化匹配，像素级匹配密度高。 |
| SuperPoint + LightGlue | 稀疏关键点 + Transformer 匹配，速度最快。 |

基本流程：

```
输入图片 → 学习式图像匹配 → COLMAP 几何验证 + SfM → 输出 sparse 模型
```

所有方法共用 NetVLAD 全局描述子筛选的序列匹配对，保证公平对比的同时减少无效匹配。

---

## 三、为什么 MASt3R 比 SIFT / SuperPoint 更强？

传统的匹配流程是两阶段：先提取特征点（SIFT/SuperPoint），再匹配特征（NN/LightGlue）。MASt3R 把这两个步骤合并了——**一次前向传播，同时输出每个像素的 3D 坐标和描述子**。

这意味着：
- 不需要"检测到"关键点才能匹配——弱纹理区域也能产生有效匹配
- 描述子和 3D 信息联合学习，匹配更可靠
- 对旋转、尺度变化更鲁棒

但代价是模型更大（ViT-L 编码器），推理更慢。所以我们做了一系列优化。

---

## 四、工程优化

为了在 RTX 4090 上高效运行 MASt3R，做了以下优化。

### 编码器缓存

MASt3R 的 Encoder（ViT-L）单独处理每张图，Decoder（ViT-B）联合处理图像对。传统方式每对都重新编码：

```
Pair(A,B): encode(A) + encode(B) → decode  ← A 被重复编码几十次
```

优化后：所有图只编码一次，解码时直接取缓存。**编码次数从 2×对数 降到 图片数**，550 张图场景下节省约 40 倍编码计算。

### Flash Attention

将手写的 QK^T attention 替换为 PyTorch 2.x 的 `scaled_dot_product_attention`，RTX 4090 自动启用 Flash Attention 内核，Transformer 层加速 15-25%。

### RoPE CUDA

编译了 croco 自带的 CUDA 位置编码算子，避免 PyTorch 慢速 fallback。

### 关键点数量控制

MASt3R 默认每张图产生约 26,000 个关键点，DKM 因配置 bug 产生了 75 万+。统一加上 `max_keypoints` 限制，默认 8192，按置信度保留 Top-K。

### 掩码过滤

支持用二值掩码过滤动态物体（人、车等）区域的关键点，三种方法均支持。

---

## 五、实测结果

测试环境：RTX 4090，PyTorch 2.5.1，CUDA 12.1。

### Picture 场景（70 张室内图，大视角差）

| 方法 | 注册数 | 3D 点数 | 耗时 | 峰值显存 |
|------|--------|---------|------|----------|
| **MASt3R** | **70 / 70** | 137,128 | 13m36s | 6.9 GB |
| gim_dkm | 52 / 70 | 49,399 | 13m12s | 14 GB |
| gim_lightglue | 52 / 70 | 26,336 | 3m42s | 8 GB |

MASt3R 是唯一实现** 100% 注册**的方法。

### JG 场景（550 张室外大场景）

| 方法 | 注册数 | 3D 点数 | 耗时 | 峰值显存 |
|------|--------|---------|------|----------|
| **MASt3R** | **550 / 550** | 1,292,695 | 2h43m | 13.6 GB |
| gim_lightglue | 550 / 550 | 529,410 | 1h22m | 9.5 GB |
| gim_dkm | 57 / 550 | 21,171 | 2h32m | 16 GB |

mast3r 和 gim_lightglue 均实现 100% 注册，mast3r 的点数多 2.4 倍。gim_dkm 大场景几乎全灭。


---

## 六、使用

安装和详细文档见 GitHub：

**[https://github.com/hjh530/GIM-colmap-recon](https://github.com/hjh530/GIM-colmap-recon)**

核心命令：

```bash
# 图片放入 inputs/<场景名>/images/
python reconstruction.py --scene_name <场景名> --version mast3r
```

主要参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--version` | gim_dkm | mast3r / gim_dkm / gim_lightglue |
| `--stop_after_db` | — | 仅生成数据库，不重建 |
| `--mast3r_maxdim` | 512 | 推理边长（越小越快） |
| `--mast3r_max_keypoints` | 8192 | 每图最大关键点数 |
| `--mast3r_subsample` | 8 | 匹配采样步长 |

---

## 七、适合谁用？

- **3DGS / NeRF 数据预处理**：如果默认 COLMAP 跑不出稳定的 sparse 模型，换个匹配前端可能就通了
- **室内或复杂场景重建**：弱纹理、重复纹理、大视角差
- **研究图像匹配和 SfM 管线**：学习式匹配器如何接入经典几何重建的工程参考

---

## 八、局限

- 环境配置比原生 COLMAP 复杂，需要 GPU + PyTorch 2.x
- 学习式匹配需要显存（MASt3R 约 6GB），图片多时用时长
- MASt3R 模型较大（ViT-L），单帧编码约 1.5 秒，虽然做了缓存优化，处理上千张图仍需数小时

---

## 九、总结

**MASt3R 两个场景均 100% 注册，gim_lightglue 也在 JG 场景达到 100%**。MASt3R 点数是 gim_lightglue 的 2.4 倍，质量更优但速度慢 2 倍。推荐追求质量的场景用 mast3r，速度优先用 gim_lightglue。

这个项目不是什么颠覆性工作，而是把当前最好的开源图像匹配模型接入到成熟的 COLMAP 管线中，配合工程优化，让它在实际场景中可落地。

如果你也在做三维重建相关的工作，欢迎试用、提 Issue 或 PR。

---

*GitHub: [https://github.com/hjh530/GIM-colmap-recon](https://github.com/hjh530/GIM-colmap-recon)*

*MASt3R: [https://github.com/naver/mast3r](https://github.com/naver/mast3r/tree/mast3r_sfm)*
