FROM python:3.13.5-slim

# Copy the 'uv' and 'uvx' executables from the latest uv image into /bin/ in this image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Python optimization flags
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

# Sync strictly the main dependencies (ignores 'dev' and 'train' groups)
RUN uv sync --frozen --no-dev --no-group train --no-install-project

COPY model_artifacts/ /app/model_artifacts/

COPY src/app.py src/inference.py src/model.py src/config.py ./

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]