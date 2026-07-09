FROM python:3.12-slim

WORKDIR /app

COPY requirements-testing.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch
RUN pip install --no-cache-dir -r requirements-testing.txt

COPY . .

CMD ["python", "testing.py"]
