# app/Dockerfile
FROM python:3.10-alpine

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git

RUN git clone https://github.com/miladtavakoli/chat-with-pdf.git .
COPY .env .env
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
