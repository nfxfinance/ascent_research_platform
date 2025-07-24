#!/usr/bin/env python3

import os
import sys
import subprocess
import logging

def main():
    """启动量化回测平台"""
    print("🚀 启动量化回测平台...")

    # 设置环境变量
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"

    # 配置日志级别以减少警告
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('streamlit').setLevel(logging.WARNING)

    try:
        # 启动 Streamlit 应用
        print("📱 启动 Streamlit 服务器...")
        print("🌐 应用将在 http://localhost:8501 启动")
        print("💡 刷新页面不会再跳转到登录界面了！")
        print("---")

        cmd = [
            sys.executable, "-m", "streamlit", "run", "main_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "true"
        ]

        subprocess.run(cmd)

    except KeyboardInterrupt:
        print("\n🛑 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保已安装所有依赖：pip install streamlit pandas numpy plotly")

if __name__ == "__main__":
    main()
