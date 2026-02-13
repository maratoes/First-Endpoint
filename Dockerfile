FROM runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir vllm==0.16.0 runpod==1.6.2 huggingface-hub pydantic

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
COPY config.yaml .

ENV MODEL_NAME="Qwen/Qwen3-14B-Instruct"
ENV MAX_MODEL_LEN=8192
ENV QUANTIZATION="awq"
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.95

CMD ["python", "-u", "handler.py"]
