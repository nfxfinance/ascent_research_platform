FROM continuumio/miniconda3:latest

WORKDIR /app

# 复制环境配置和项目文件
COPY environment.yml /app/environment.yml
COPY . /app

# 创建 conda 环境并激活
RUN conda env create -f /app/environment.yml && \
    conda clean -afy

# 设置默认激活的环境（假设你的环境名为 ascent_platform）
ENV PATH /opt/conda/envs/ascent_platform/bin:$PATH

ENV PYTHONPATH=/app/AscentQuantMaster/python/lib:/app/AscentQuantMaster/python:/app/AscentQuantMaster

# 推荐用 conda run
CMD ["conda", "run", "-n", "ascent_platform", "streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
