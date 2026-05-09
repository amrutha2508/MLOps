FROM python:3.11.5-slim

RUN pip install -U pip && pip install poetry

WORKDIR /app

# Copy only the dependency files first (for better Docker caching)
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install only production dependencies and skip the root project installation
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

# Copy the rest of the application
COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]