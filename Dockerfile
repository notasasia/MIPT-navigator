FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 NAV_STATE=/app/state
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-osd libgomp1 build-essential cmake ninja-build && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
ENV CMAKE_BUILD_PARALLEL_LEVEL=4 CMAKE_ARGS="-DGGML_NATIVE=OFF -DLLAMA_CURL=OFF"
RUN pip install --no-cache-dir -r requirements.txt && pip check
RUN useradd --create-home --uid 10001 navigator && mkdir -p /app/state && chown navigator /app/state
COPY . .
USER navigator
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["python","-m","streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
