FROM python:3.14-slim

WORKDIR /app

# git is only needed to install twikit from its pinned commit.
COPY requirements.lock ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir -r requirements.lock \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY config.py main.py ./

RUN useradd --create-home --uid 10001 app
USER app

CMD ["python", "main.py"]
