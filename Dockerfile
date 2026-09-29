FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY results/ results/
COPY data/ data/

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
