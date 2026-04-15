FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py transcribe.py ./
COPY templates/ ./templates/
COPY static/ ./static/

RUN mkdir -p uploads_temp web_output

CMD ["python3", "app.py"]
