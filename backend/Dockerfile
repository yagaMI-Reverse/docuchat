# DocuChat FastAPI backend — production-parity container.
# Python pinned to 3.12 so scikit-learn/numpy install from prebuilt wheels
# (no compiler needed), exactly like the Render deploy.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY . .

EXPOSE 8000

# Same start command as the Render blueprint.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
