FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

RUN pip install tensorflow==2.9.1

COPY ./sentiment_model /sentiment_model/
COPY ./app /app/

WORKDIR /app

EXPOSE 8000

CMD ["python", "main.py"]

