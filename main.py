from fastapi import FastAPI, BackgroundTasks
import subprocess
import time

app = FastAPI(title="3D Gaussian 优化与测试平台")

@app.get("/")
def read_root():
    return {"message": "指挥中心已就绪！欢迎来到 3DGS 测试平台"}

# 这是一个模拟后台运行模型的任务函数
def run_model_task(model_name: str):
    print(f"后台开始运行模型: {model_name}...")
    # 这里未来会替换成咱们上一轮聊过的 subprocess.Popen('conda run -n...')
    time.sleep(5) 
    print(f"模型 {model_name} 运行结束！")

@app.post("/start_training/{model_name}")
def start_training(model_name: str, background_tasks: BackgroundTasks):
    # 将模型训练扔到后台执行，这样前端点击按钮后能瞬间得到回复，不用傻等
    background_tasks.add_task(run_model_task, model_name)
    return {"status": "success", "info": f"指令已下达，{model_name} 正在后台启动！"}