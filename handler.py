import os
from typing import Any, Dict

import runpod
from vllm import LLM, SamplingParams

model = None


def initialize_model() -> LLM:
    global model
    if model is not None:
        return model
    model = LLM(
        model=os.getenv("MODEL_NAME", "Qwen/Qwen3-14B-Instruct"),
        trust_remote_code=True,
        max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
        quantization=os.getenv("QUANTIZATION", "awq"),
        tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
    )
    return model


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        data = job.get("input", {})
        prompt = data.get("prompt", "")
        sampling = SamplingParams(
            max_tokens=data.get("max_tokens", 150),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.95),
            top_k=data.get("top_k", 50),
        )
        outputs = model.generate([prompt], sampling)
        return {"output": outputs[0].outputs[0].text, "status": "success"}
    except Exception as exc:
        return {"error": str(exc), "status": "error"}


initialize_model()
runpod.serverless.start({"handler": handler})
