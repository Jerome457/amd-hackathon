# General Purpose AI Agent

## Docker Commands

### Build
```bash
docker build -t amd-agent .
```

### Run
```bash
docker run --env-file .env -v $(pwd)/input:/input -v $(pwd)/output:/output amd-agent
```

## Install Local LLM
- Download `.gguf` from `https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/blob/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf`
- Place it in `models/`

## Classifier
```bash
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip install scikit-learn
pip install transformers
python router/train_router.py
```
Then run the testing file (with `.` in input/output path for virtual env and without for docker)
