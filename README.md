# 🚀 3DGS Optimization Platform (3D高斯压缩算法优化平台)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2Bcu118-ee4c2c.svg)

本项目是一个针对 3D Gaussian Splatting (3DGS) 及其前沿压缩算法（Scaffold-GS, HAC, CompactGS, TamingGS）的**自动化基准测试与优化平台**。旨在消除底层 CUDA 编译壁垒，提供一站式、高并发的训练与多维评估闭环。

## ✨ 核心特性
- **环境硬隔离**：利用 Conda 沙盒化解各算法间严重的 C++ ABI 编译冲突。
- **异步非阻塞调度**：基于 FastAPI + WebSocket 实现任务全双工派发与毫秒级日志捕获。
- **防 OOM 与 DOM 卡死**：动态扩展显存池解决 20GB+ 渲染碎片；独创 200ms 前端渲染节流阀。
- **自动化多维大屏**：一键聚合体积、PSNR、SSIM、LPIPS，生成 Trade-off 散点图与雷达图。
- **云端资产一键回传**：自动搜寻训练产物（.ply, .json, bitstreams）并流式打包下载。

## 🛠️ 快速开始 (Quick Start)
1. 克隆本仓库
2. 启动 FastAPI 后端调度器: `uvicorn main:app --host 0.0.0.0 --port 8000`
3. 访问 Web 控制台开始标准基准测试
