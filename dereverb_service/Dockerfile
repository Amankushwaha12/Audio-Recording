# GPU-ready container for Unsupervised Vocal Dereverberation (Saito et al., ICASSP 2023)
# Build: docker build -t diffusion-dereverb ./dereverb_service
# Run (GPU):
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/checkpoints:/models:ro \
#     -e DEREVERB_CHECKPOINT_PATH=/models/latest.pt \
#     -e ALLOWED_ORIGINS="http://localhost:3000,https://your-app.vercel.app" \
#     diffusion-dereverb

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MAX_UPLOAD_MB=100

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Run as non-root user for container hardening
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
