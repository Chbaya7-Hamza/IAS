FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash", "-c", "uvicorn heatguard_cloud_api:app --host 0.0.0.0 --port 8000 & python main.py"]
main.py