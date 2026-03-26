FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for Whisper audio conversion, curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for layer caching
COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# Install core Python deps explicitly (in case requirements.txt is minimal)
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    discord.py \
    qdrant-client \
    httpx \
    python-dotenv \
    nats-py \
    psutil \
    pypdf \
    ebooklib html2text \
    pyttsx3 \
    openai-whisper \
    google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

# Copy project
COPY . .

# Ensure the hermes-agent submodule is installed if present
RUN if [ -f integrations/hermes-agent/setup.py ]; then \
      pip install --no-cache-dir -e integrations/hermes-agent; \
    fi

# Plugins directory
RUN mkdir -p plugins

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["python", "buzlock_bot.py"]
