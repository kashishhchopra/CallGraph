FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only PyTorch first (default wheels pull multi-GB CUDA libraries the containers never use)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# Dependencies depend only on pyproject.toml, so code edits do not invalidate this layer
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/req.txt \
    && pip install --no-cache-dir --default-timeout=120 --retries 10 -r /tmp/req.txt && python -m spacy download en_core_web_sm
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
COPY . .
ENV CALLGRAPH_ROOT=/app \
    PYTHONPATH=/app
