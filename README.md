# First Endpoint (Comments, vLLM)

RunPod serverless endpoint for `Qwen/Qwen3-14B-Instruct` using vLLM.

## Build

```bash
./scripts/build.sh
```

## Push

```bash
./scripts/push.sh
```

## RunPod settings

- GPU: NVIDIA A40
- Workers: min `0`, max `3`
- Idle timeout: `30s`
- Env:
  - `MODEL_NAME=Qwen/Qwen3-14B-Instruct`
  - `MAX_MODEL_LEN=8192`
  - `QUANTIZATION=awq`
  - `GPU_MEMORY_UTILIZATION=0.95`
  - `HF_TOKEN=<token>`
