import os
import json
import math
import glob
import traceback
import shutil
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio

app = FastAPI(title="3DGS 压缩优化控制台")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_dashboard(): return FileResponse("static/index.html")

def get_size_in_mb(path):
    try:
        if not path or not os.path.exists(path): return 0
        if os.path.isfile(path): return os.path.getsize(path) / (1024 * 1024)
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / (1024 * 1024)
    except: return 0

def parse_results_json(json_path):
    if not json_path or not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            metrics = list(data.values())[0] if isinstance(list(data.values())[0], dict) else data
            return {
                "psnr": 0.0 if math.isnan(float(metrics.get("PSNR", 0))) else float(metrics.get("PSNR", 0)),
                "ssim": 0.0 if math.isnan(float(metrics.get("SSIM", 0))) else float(metrics.get("SSIM", 0)),
                "lpips": 0.0 if math.isnan(float(metrics.get("LPIPS", 0))) else float(metrics.get("LPIPS", 0))
            }
    except: return None

@app.get("/api/metrics")
async def fetch_metrics():
    base_dir = "/root/autodl-tmp/3dgs_platform"
    data = {
        "baseline": {"size": 453.0, "psnr": 37.3399, "ssim": 0.9530, "lpips": 0.2589},
        "scaffoldgs": {"size": 178.0, "psnr": 36.5000, "ssim": 0.9400, "lpips": 0.2400}, # 新增：Scaffold-GS 缺省值
        "hac": {"size": 137.0, "psnr": 30.7453, "ssim": 0.9124, "lpips": 0.2516},
        "compactgs": {"size": 41.0, "psnr": 34.8442, "ssim": 0.9415, "lpips": 0.2230},
        "taminggs": {"size": 176.0, "psnr": 36.6692, "ssim": 0.9485, "lpips": 0.2173}
    }
    try:
        dgs_dirs = glob.glob(f"{base_dir}/gaussian-splatting/output/*")
        if dgs_dirs:
            out = max(dgs_dirs, key=os.path.getmtime)
            if os.path.exists(f"{out}/point_cloud/iteration_30000/point_cloud.ply"): data["baseline"]["size"] = get_size_in_mb(f"{out}/point_cloud/iteration_30000/point_cloud.ply")
            m = parse_results_json(f"{out}/results.json")
            if m: data["baseline"].update(m)

        # 新增：抓取 Scaffold-GS
        sgs_out = f"{base_dir}/output/scaffoldgs_playroom"
        if os.path.exists(f"{sgs_out}/point_cloud/iteration_30000/point_cloud.ply"): data["scaffoldgs"]["size"] = get_size_in_mb(f"{sgs_out}/point_cloud/iteration_30000/point_cloud.ply")
        m = parse_results_json(f"{sgs_out}/results.json")
        if m: data["scaffoldgs"].update(m)

        hac_out = f"{base_dir}/output/hac_playroom_web"
        if os.path.exists(f"{hac_out}/point_cloud/iteration_30000/point_cloud.ply"): data["hac"]["size"] = get_size_in_mb(f"{hac_out}/point_cloud/iteration_30000/point_cloud.ply") + get_size_in_mb(f"{hac_out}/bitstreams")
        m = parse_results_json(f"{hac_out}/results.json")
        if m: data["hac"].update(m)

        cgs_out = f"{base_dir}/output/compactgs_playroom"
        if os.path.exists(f"{cgs_out}/point_cloud/iteration_30000/point_cloud.ply"): data["compactgs"]["size"] = get_size_in_mb(f"{cgs_out}/point_cloud/iteration_30000/point_cloud.ply")
        m = parse_results_json(f"{cgs_out}/results.json")
        if m: data["compactgs"].update(m)

        tgs_out = f"{base_dir}/output/taminggs_playroom"
        if os.path.exists(f"{tgs_out}/point_cloud/iteration_30000/point_cloud.ply"): data["taminggs"]["size"] = get_size_in_mb(f"{tgs_out}/point_cloud/iteration_30000/point_cloud.ply")
        m = parse_results_json(f"{tgs_out}/results.json")
        if m: data["taminggs"].update(m)
    except: pass
    for k, v in data.items(): data[k] = {key: round(val, 4) for key, val in v.items()}
    return data

@app.get("/api/download/{model_name}")
async def download_asset(model_name: str, background_tasks: BackgroundTasks):
    base_dir = "/root/autodl-tmp/3dgs_platform"
    target_dir = None
    try:
        if model_name == "3dgs":
            dirs = glob.glob(f"{base_dir}/gaussian-splatting/output/*")
            if dirs: target_dir = max(dirs, key=os.path.getmtime)
        elif model_name == "scaffoldgs": target_dir = f"{base_dir}/output/scaffoldgs_playroom" # 新增下载
        elif model_name == "hac": target_dir = f"{base_dir}/output/hac_playroom_web"
        elif model_name == "compactgs": target_dir = f"{base_dir}/output/compactgs_playroom"
        elif model_name == "taminggs": target_dir = f"{base_dir}/output/taminggs_playroom"
            
        if not target_dir or not os.path.exists(target_dir): return {"error": "Model data not found!"}
        tmp_zip = os.path.join(tempfile.gettempdir(), f"{model_name}_assets.zip")
        bundle = os.path.join(tempfile.gettempdir(), f"{model_name}_bundle")
        if os.path.exists(bundle): shutil.rmtree(bundle)
        os.makedirs(bundle)
        
        ply = glob.glob(f"{target_dir}/point_cloud/iteration_*/point_cloud.ply")
        if ply:
            latest = max(ply, key=os.path.getmtime)
            os.makedirs(os.path.join(bundle, "point_cloud"), exist_ok=True)
            shutil.copy2(latest, os.path.join(bundle, "point_cloud", "point_cloud.ply"))
            
        if os.path.exists(f"{target_dir}/results.json"): shutil.copy2(f"{target_dir}/results.json", bundle)
        if model_name == "hac" and os.path.exists(f"{target_dir}/bitstreams"): shutil.copytree(f"{target_dir}/bitstreams", os.path.join(bundle, "bitstreams"))
                
        shutil.make_archive(tmp_zip.replace('.zip', ''), 'zip', bundle)
        shutil.rmtree(bundle)
        background_tasks.add_task(lambda: os.remove(tmp_zip) if os.path.exists(tmp_zip) else None)
        return FileResponse(tmp_zip, media_type="application/zip", filename=f"{model_name}_assets.zip")
    except Exception as e: return {"error": str(e)}

active_processes = {}
async def run_training(websocket: WebSocket, task_name: str, command: list):
    await websocket.send_text(f"<br><span class='text-blue-400 font-bold'>[root@autodl] Executing standard training protocol: {task_name.upper()}...</span><br>\n")
    try:
        env = os.environ.copy(); env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        active_processes[websocket] = process
        while True:
            chunk = await process.stdout.read(128)
            if not chunk: break
            await websocket.send_text(chunk.decode('utf-8', errors='replace').replace('\r', '<br>'))
        await process.wait()
        if process.returncode == 0: await websocket.send_text(f"<br><span class='text-emerald-400 font-bold'>[系统] 【{task_name}】任务圆满完成！数据已落盘。</span><br>\n")
        elif process.returncode in [-15, -9, 143, 137]: await websocket.send_text(f"<br><span class='text-rose-500 font-bold bg-rose-500/10 px-2 py-1 rounded'>[root@autodl] SIGTERM sent...</span><br>\n")
        else: await websocket.send_text(f"<br><span class='text-red-500 font-bold'>[报错] 退出码: {process.returncode}</span><br>\n")
    except Exception as e: await websocket.send_text(f"<span class='text-red-500'>[错误] {str(e)}</span><br>\n")
    finally:
        if websocket in active_processes: del active_processes[websocket]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("<span class='text-zinc-500 italic'>System ready. Select a model to begin...</span><br>\n")
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start_"):
                if websocket in active_processes: continue
                task = data.replace("start_", "")
                # 新增：Scaffold-GS 调度命令
                cmd_dict = {
                    "3dgs": ["/root/autodl-tmp/conda_envs/3dgs_base/bin/python", "-u", "/root/autodl-tmp/3dgs_platform/gaussian-splatting/train.py", "-s", "/root/autodl-tmp/data/db/playroom"],
                    "scaffoldgs": ["/root/autodl-tmp/conda_envs/scaffoldgs_env/bin/python", "-u", "/root/autodl-tmp/3dgs_platform/Scaffold-GS/train.py", "-s", "/root/autodl-tmp/data/db/playroom", "-m", "output/scaffoldgs_playroom"],
                    "hac": ["/root/autodl-tmp/conda_envs/hac_env/bin/python", "-u", "/root/autodl-tmp/3dgs_platform/HAC/train.py", "-s", "/root/autodl-tmp/data/db/playroom", "-m", "output/hac_playroom_web"],
                    "compactgs": ["/root/autodl-tmp/conda_envs/compactgs_env/bin/python", "-u", "/root/autodl-tmp/3dgs_platform/CompactGS/train.py", "-s", "/root/autodl-tmp/data/db/playroom", "-m", "output/compactgs_playroom"],
                    "taminggs": ["/root/autodl-tmp/conda_envs/taminggs_env/bin/python", "-u", "/root/autodl-tmp/3dgs_platform/TamingGS/train.py", "-s", "/root/autodl-tmp/data/db/playroom", "-m", "output/taminggs_playroom"]
                }
                if task in cmd_dict: asyncio.create_task(run_training(websocket, task, cmd_dict[task]))
            elif data == "stop_task" and websocket in active_processes: active_processes[websocket].terminate()
    except:
        if websocket in active_processes: active_processes[websocket].terminate()