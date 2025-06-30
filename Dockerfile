# Dockerfile Final Version - The Real One

FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set all necessary environment variables
ENV NUMBA_CACHE_DIR=/tmp
ENV TTS_HOME=/tmp/.tts
# ⭐ AUTO-AGREE TO COQUI TERMS OF SERVICE ⭐
ENV COQUI_TOS_AGREED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["gunicorn", "--worker-class", "gevent", "--workers", "1", "--bind", "0.0.0.0:7860", "--timeout", "600", "app:app"]