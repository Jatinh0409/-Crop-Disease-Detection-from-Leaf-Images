FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py ./app/app.py
COPY models/gradcam.py ./models/gradcam.py
COPY models/model.pt ./models/model.pt
COPY models/metrics.json ./models/metrics.json

ENV MODEL_PATH=./models/model.pt
ENV METRICS_PATH=./models/metrics.json

WORKDIR /code/app
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
