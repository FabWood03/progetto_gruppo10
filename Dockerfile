# Immagine Python stabile
FROM python:3.12-slim

# Variabili di ambiente utili
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Directory di lavoro dentro il container
WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il progetto (escluso ciò che è in .dockerignore)
COPY . .

# Comando di avvio: main.py è dentro src/
CMD ["python", "src/main.py"]
