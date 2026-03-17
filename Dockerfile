# 1. Basis-Image: Wir nehmen ein leichtes, offizielles Python-Image
FROM python:3.10-slim

# 2. Arbeitsverzeichnis im Container festlegen
WORKDIR /app

# 3. WICHTIG für OpenCV: System-Bibliotheken installieren, die auf einem nackten OS oft fehlen
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Zuerst nur die requirements.txt kopieren (spart Zeit beim erneuten Bauen durch Caching)
COPY requirements.txt .

# 5. Alle Python-Pakete installieren
RUN pip install --no-cache-dir -r requirements.txt

# 6. Jetzt den gesamten restlichen Code (api.py, index.html, modules, dataset) kopieren
COPY . .

# 7. Dem Container sagen, dass er Port 8000 freigeben soll
EXPOSE 8000

# 8. Der Befehl, der ausgeführt wird, wenn der Container startet
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]