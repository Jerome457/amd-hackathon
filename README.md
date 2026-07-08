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