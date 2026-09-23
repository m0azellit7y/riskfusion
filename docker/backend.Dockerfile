FROM python:3.12-slim
# ffmpeg is needed from Phase 2 (audio extraction from recordings)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY backend ./backend
COPY migrations ./migrations
COPY configs ./configs
COPY contracts ./contracts
COPY docs ./docs
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps .
ENV RISKFUSION_STORAGE_ROOT=/var/lib/riskfusion/storage RISKFUSION_DATA_ROOT=/app/data
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn riskfusion_api.main:app --host 0.0.0.0 --port 8000"]
