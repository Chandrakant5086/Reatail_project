FROM python:3.10-slim

WORKDIR /api

ENV PYTHONPATH=/api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "api/main.py"]
