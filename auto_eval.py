import os
import torch
import glob
from torchvision.io import read_image
from evaluator import UnifiedEvaluator  # 导入你刚刚写的完美代码

def evaluate_folder(render_dir: str, gt_dir: str, device="cuda"):
    """
    自动遍历文件夹中的渲染图和真实图，并输出平台级综合评估报告
    """
    print(f"🔗 开始联动评估...\n渲染图路径: {render_dir}\n真实图路径: {gt_dir}")
    
    # 1. 抓取所有图片路径并排序，确保视角一一对应
    render_paths = sorted(glob.glob(os.path.join(render_dir, "*.png")))
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))

    if not render_paths or len(render_paths) != len(gt_paths):
        raise ValueError(f"⚠️ 图片数量不匹配！渲染图: {len(render_paths)} 张, 真实图: {len(gt_paths)} 张")

    # 2. 实例化你写的评估器
    evaluator = UnifiedEvaluator(device=device)
    
    total_metrics = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}
    num_imgs = len(render_paths)

    # 3. 循环比对每一张图
    print(f" 正在逐帧对比 {num_imgs} 张图像...")
    for r_path, g_path in zip(render_paths, gt_paths):
        # 读取图像，torchvision read_image 返回的是 [C, H, W] 的 uint8 张量 (0-255)
        # 必须除以 255.0 归一化到你要求的 [0, 1] 范围
        img_render = read_image(r_path).float() / 255.0
        img_gt = read_image(g_path).float() / 255.0
        
        # 调用你的全量评估接口
        metrics = evaluator.evaluate_image_pair(img_render, img_gt)
        
        total_metrics["PSNR"] += metrics["PSNR"]
        total_metrics["SSIM"] += metrics["SSIM"]
        total_metrics["LPIPS"] += metrics["LPIPS"]

    # 4. 计算平均分
    avg_psnr = total_metrics["PSNR"] / num_imgs
    avg_ssim = total_metrics["SSIM"] / num_imgs
    avg_lpips = total_metrics["LPIPS"] / num_imgs

    print("\n" + "="*40)
    print("📊 3DGS 平台测试报告 (测试集)")
    print("="*40)
    print(f"✅ 平均 PSNR  : {avg_psnr:.4f} dB (越大越好)")
    print(f"✅ 平均 SSIM  : {avg_ssim:.4f} (越接近1越好)")
    print(f"✅ 平均 LPIPS : {avg_lpips:.4f} (越小越好)")
    print("="*40)

if __name__ == "__main__":
    real_render_path = "output/cdfd86f4-a/train/ours_30000/renders"
    real_gt_path = "output/cdfd86f4-a/train/ours_30000/gt"
    
    evaluate_folder(real_render_path, real_gt_path)