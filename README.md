# Running AI Models on Apple M4

This directory contains scripts to help you get started running AI models locally on your Mac.

Apple Silicon provides two main accelerators for AI:
1. **The GPU (Metal):** Usually the fastest for Large Language Models (LLMs) due to much higher memory bandwidth.
2. **The Neural Engine (ANE):** Has high compute power (38 TOPS on M4) but lower memory bandwidth, making it great for smaller models like image generation or embeddings.

## Recommended Approach for LLMs (Uses GPU)

Apple's official open-source framework, **MLX**, is the best way to run models on Apple Silicon. It optimizes for unified memory and Metal.

```bash
bash setup_mlx.sh
```

## Approach for Neural Engine (Uses Core ML)

If you strictly want to run models on the Apple Neural Engine, you must convert them into Apple's `.mlpackage` format and execute using Core ML.

```bash
bash setup_coreml.sh
```
# ANE---Apple-Neural-Engine-AI-Studio
