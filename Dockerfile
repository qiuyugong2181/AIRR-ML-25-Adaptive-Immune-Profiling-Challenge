FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN python3.11 -m pip install --no-cache-dir \
    numpy \
    pandas \
    tqdm \
    scikit-learn \
    xgboost \
    optuna

WORKDIR /app
COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python3.11", "-c", "print('AIRR ML container ready')"]
