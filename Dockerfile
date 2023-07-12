FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . .