import torch
import math
# 需要后续在 3dgs_base 环境中安装: pip install lpips torchmetrics
import lpips 
from torchmetrics.image import StructuralSimilarityIndexMeasure

class UnifiedEvaluator:
    def __init__(self, device="cuda"):
        """
        初始化统一评估管道，加载必要的深度学习感知模型
        """
        self.device = device
        # 实例化 LPIPS 模型，默认使用 VGG 网络提取深度特征 
        print("[Evaluator] 正在加载 LPIPS (VGG) 预训练权重...")
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_vgg.eval() # 开启评估模式，冻结梯度
        
        # 实例化 SSIM 计算器，设置数据范围为 0-1
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def calculate_psnr(self, img_pred: torch.Tensor, img_gt: torch.Tensor) -> float:
        """
        计算信号能量保真度 (PSNR) 
        img_pred, img_gt: 形状为 [C, H, W] 的张量，值域应归一化到 [0, 1]
        """
        # 计算均方误差 (MSE)
        mse = torch.mean((img_pred - img_gt) ** 2)
        if mse == 0:
            return float('inf')
        # 基于 MSE 计算以分贝(dB)为单位的 PSNR 
        psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
        return psnr

    def calculate_ssim(self, img_pred: torch.Tensor, img_gt: torch.Tensor) -> float:
        """
        计算结构相似度指数 (SSIM)
        从亮度、对比度、结构三个维度综合评估 
        """
        # torchmetrics 要求输入形状带有 Batch 维度，即 [B, C, H, W]
        img_pred_b = img_pred.unsqueeze(0)
        img_gt_b = img_gt.unsqueeze(0)
        
        ssim_val = self.ssim_metric(img_pred_b, img_gt_b)
        return ssim_val.item()

    def calculate_lpips(self, img_pred: torch.Tensor, img_gt: torch.Tensor) -> float:
        """
        计算学习感知图像块相似度 (LPIPS)
        利用 VGG 提取空间距离，进行严苛的视觉异常惩罚 
        """
        # LPIPS 库通常期望输入范围在 [-1, 1] 之间，且带 Batch 维度
        img_pred_b = (img_pred.unsqueeze(0) * 2.0) - 1.0
        img_gt_b = (img_gt.unsqueeze(0) * 2.0) - 1.0
        
        with torch.no_grad():
            lpips_val = self.lpips_vgg(img_pred_b, img_gt_b)
        return lpips_val.item()

    def evaluate_image_pair(self, img_pred: torch.Tensor, img_gt: torch.Tensor) -> dict:
        """
        全量评估接口：一次性返回三大维度的综合报告
        """
        # 确保数据都在指定的设备上
        img_pred = img_pred.to(self.device)
        img_gt = img_gt.to(self.device)

        return {
            "PSNR": self.calculate_psnr(img_pred, img_gt),
            "SSIM": self.calculate_ssim(img_pred, img_gt),
            "LPIPS": self.calculate_lpips(img_pred, img_gt)
        }

# ================= 测试探针 =================
if __name__ == "__main__":
    # 模拟两张 800x800 的随机渲染图和真实 Ground Truth (3通道, RGB)
    print("启动统一评估管道测试...")
    dummy_pred = torch.rand(3, 800, 800)
    dummy_gt = torch.rand(3, 800, 800)
    
    evaluator = UnifiedEvaluator(device="cpu") # 测试时先用 CPU
    results = evaluator.evaluate_image_pair(dummy_pred, dummy_gt)
    
    print("\n--- 图像质量多维评估报告 ---")
    print(f"信号保真度 (PSNR): {results['PSNR']:.4f} dB")
    print(f"结构相似度 (SSIM): {results['SSIM']:.4f}")
    print(f"感知扭曲度 (LPIPS): {results['LPIPS']:.4f}")