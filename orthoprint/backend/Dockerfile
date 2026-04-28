FROM python:3.11-slim
WORKDIR /app

# gcc         — needed by passlib bcrypt C extension and trimesh deps
# libgeos-dev — needed by shapely
# libffi-dev  — needed by python-jose cryptography
# cargo/rustc — needed by cryptography wheel on some platforms
RUN apt-get update && apt-get install -y \
    gcc \
    libgeos-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install cryptography separately first to ensure it builds correctly
RUN pip install --no-cache-dir cryptography
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
