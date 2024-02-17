# app/Dockerfile
FROM python:3.10-slim

WORKDIR /app
RUN echo "nameserver 178.22.122.100" > /etc/resolv.conf && \
    echo "nameserver 185.51.200.2" >> /etc/resolv.conf

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/miladtavakoli/chat-with-pdf.git .
COPY .env .env
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
