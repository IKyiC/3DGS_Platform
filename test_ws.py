import asyncio
import websockets

async def test_training_monitor():
    # 连接到我们刚才在 main.py 里写的 WebSocket 路由
    uri = "ws://127.0.0.1:8000/ws/training_log/truck_model"
    try:
        async with websockets.connect(uri) as websocket:
            print("🚀 已成功连接到 3DGS 指挥中心监控大屏！\n")
            while True:
                # 实时接收后台传来的每一行训练日志
                log_line = await websocket.recv()
                print(log_line, end="")
    except Exception as e:
        print(f"连接断开: {e}")

if __name__ == "__main__":
    asyncio.run(test_training_monitor())