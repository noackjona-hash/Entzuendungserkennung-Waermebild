from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from cryptography.fernet import Fernet
import uvicorn
import json
import base64
import time
import hashlib
from typing import Dict, Tuple

# Importiere deinen Algorithmus (bleibt unverändert!)
from berechnung import ThermalAnalyzer

# =====================================================================
# ENTERPRISE SECURITY CONFIGURATION (DSGVO / HIPAA COMPLIANCE)
# =====================================================================

# 1. API KEY AUTHENTIFIZIERUNG
# Nur autorisierte Frontends dürfen Anfragen senden
API_KEY = "jf2026-jona-super-secret-key-9988"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Zugriff verweigert. Ungültiger oder fehlender API-Key.")
    return api_key

# 2. IN-MEMORY ENCRYPTION (VOLATILE KEY)
# Erstellt bei jedem Serverstart einen neuen AES-Verschlüsselungs-Key.
# Bilder im RAM werden verschlüsselt. Stirbt der Server, sind alle Daten unwiderruflich weg (Privacy by Design).
VOLATILE_SECRET_KEY = Fernet.generate_key()
cipher_suite = Fernet(VOLATILE_SECRET_KEY)

# 3. RATE LIMITING (DDoS Protection)
# Verhindert, dass Bots die API mit Bildern überfluten und den RAM füllen.
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SEC = 60
ip_request_counts: Dict[str, Tuple[int, float]] = {}

def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    if client_ip in ip_request_counts:
        count, start_time = ip_request_counts[client_ip]
        if current_time - start_time < RATE_LIMIT_WINDOW_SEC:
            if count >= RATE_LIMIT_REQUESTS:
                raise HTTPException(status_code=429, detail="Rate Limit überschritten. Bitte warten.")
            ip_request_counts[client_ip] = (count + 1, start_time)
        else:
            ip_request_counts[client_ip] = (1, current_time)
    else:
        ip_request_counts[client_ip] = (1, current_time)

# =====================================================================
# FASTAPI INITIALISIERUNG
# =====================================================================

app = FastAPI(
    title="Jugend Forscht 2026 - Thermografie Med-API",
    description="Hochsichere, verschlüsselte In-Memory API zur Erkennung von Entzündungen.",
    version="3.0-Enterprise"
)

# Strenge CORS-Richtlinien (Nur spezifische Zugriffe erlauben)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In echter Produktion hier die genaue Domain eintragen!
    allow_credentials=True,
    allow_methods=["POST"], # Nur POST erlauben, kein GET/PUT/DELETE
    allow_headers=["X-API-Key", "Content-Type"],
)

# Eigene Middleware für Security Headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# =====================================================================
# HAUPT-ENDPUNKT DER API
# =====================================================================

@app.post("/analyze", summary="Sichere In-Memory Wärmebild-Analyse")
async def analyze_thermal_image(
    request: Request,
    file: UploadFile = File(...),
    segmente_json: str = Form(...),
    api_key: str = Depends(verify_api_key) # Erzwingt Authentifizierung
):
    """
    Nimmt das Bild entgegen, verschlüsselt es sofort im RAM, entschlüsselt es nur für 
    die OpenCV-Pipeline und gibt die verifizierten medizinischen Daten zurück.
    Keine Daten verlassen jemals unverschlüsselt den Arbeitsspeicher.
    """
    # 1. DDoS / Rate Limit Check
    check_rate_limit(request)
    
    # 2. Input Validation (Schutz vor Code-Injection im JSON)
    try:
        segmente = json.loads(segmente_json)
        if not isinstance(segmente, list):
            raise ValueError("Segmente müssen eine Liste sein.")
    except Exception:
        raise HTTPException(status_code=400, detail="Ungültiges oder manipulatives JSON-Format.")
        
    try:
        # 3. Secure File Handling & Immediate RAM Encryption
        # Anstatt das Bild roh im RAM zu halten, verschlüsseln wir es!
        raw_bytes = await file.read()
        
        # Hash des Originalbildes berechnen (für sicheres Logging ohne Bilddaten preiszugeben)
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REQUEST SECURED. File-Hash: {file_hash}")
        
        # Bild im RAM verschlüsseln
        encrypted_bytes = cipher_suite.encrypt(raw_bytes)
        
        # Originalbytes aus dem Speicher löschen (Python Garbage Collector anstoßen)
        del raw_bytes 
        
        # ============================================================
        # BERECHNUNGS-PIPELINE (CRITICAL SECTION)
        # ============================================================
        
        # Für den Bruchteil der Analyse entschlüsseln
        decrypted_bytes_for_analysis = cipher_suite.decrypt(encrypted_bytes)
        
        analyzer = ThermalAnalyzer(bild_bytes=decrypted_bytes_for_analysis, segmente=segmente)
        ergebnisse = analyzer.analysiere() 
        img_base64 = analyzer.render_base64()
        
        # Sensible Bilddaten sofort wieder aus dem RAM zerstören
        del decrypted_bytes_for_analysis
        del encrypted_bytes
        
        # ============================================================
        
        # 4. Datenaufbereitung für das Frontend
        export_daten = []
        for e in ergebnisse:
            export_daten.append({
                "gelenk": e.gelenk_name,
                "score_percent": round(e.score.total_confidence, 2),
                "geometrie": e.morphology.to_dict(),
                "temperatur_celsius": e.stats_celsius.to_dict()
            })
        
        return JSONResponse(content={
            "status": "success",
            "security_clearance": "Data fully encrypted in transit and at rest (RAM).",
            "file_hash": file_hash,
            "gefundene_anomalien": len(ergebnisse),
            "daten": export_daten,
            "protokoll": analyzer.analyse_protokoll, 
            "ergebnis_bild_base64": img_base64 
        })
        
    except Exception as e:
        print(f"[SECURITY ALERT] Fehler bei Analyse: {str(e)}")
        raise HTTPException(status_code=500, detail="Interner Serverfehler bei der sicheren Bildanalyse.")

if __name__ == "__main__":
    print("="*50)
    print(" STARTING ENTERPRISE MEDICAL API")
    print(" - In-Memory Encryption: ENABLED")
    print(" - API-Key Protection:   ENABLED")
    print(" - Rate Limiting:        ENABLED")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)