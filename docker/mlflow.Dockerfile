FROM python:3.12-slim

RUN pip install \
    --no-cache-dir \
    mlflow==3.3.2 \
    boto3==1.40.18 \
    psycopg2-binary==2.9.10

EXPOSE 5000
