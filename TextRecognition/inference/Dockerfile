FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_WEIGHTS_PATH=./gi_model_artifacts/model_weights.pth
ENV TOKENIZER_PATH=./gi_model_artifacts/tokenizer.pkl

EXPOSE 8000
CMD ["uvicorn", "model_inf:app", "--host", "0.0.0.0", "--port", "8000"]