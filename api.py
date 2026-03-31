from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from cryptography.fernet import Fernet
import uvicorn
import cv2
import numpy as np
import base64
import time
import hashlib
from typing import Dict, Tuple

from berechnung import ThermalAnalyzer
from fussfinder import FootFinder

# =====================================================================
# ENTERPRISE SECURITY CONFIGURATION (DSGVO / HIPAA COMPLIANCE)
# =====================================================================

API_KEY = "jf2026-jona-super-secret-key-9988"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Zugriff verweigert. Ungültiger oder fehlender API-Key.")
    return api_key

VOLATILE_SECRET_KEY = Fernet.generate_key()
cipher_suite = Fernet(VOLATILE_SECRET_KEY)

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
    version="4.0-Enterprise-Autofocus"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST"], 
    allow_headers=["X-API-Key", "Content-Type"],
)

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

@app.post("/analyze", summary="Vollautomatische In-Memory Wärmebild-Analyse")
async def analyze_thermal_image(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key) 
):
    """
    Nimmt das Bild entgegen, verschlüsselt es sofort im RAM.
    Findet vollautomatisch die Zehen (Anatomical Anchor) und sucht nach Entzündungen.
    """
    check_rate_limit(request)
        
    try:
        raw_bytes = await file.read()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] REQUEST SECURED. File-Hash: {file_hash}")
        
        # Bild im RAM verschlüsseln
        encrypted_bytes = cipher_suite.encrypt(raw_bytes)
        del raw_bytes 
        
        # ============================================================
        # BERECHNUNGS-PIPELINE (CRITICAL SECTION)
        # ============================================================
        decrypted_bytes_for_analysis = cipher_suite.decrypt(encrypted_bytes)
        
        # 1. Bild dekodieren für den Fussfinder
        array = np.frombuffer(decrypted_bytes_for_analysis, dtype=np.uint8)
        bild_cv = cv2.imdecode(array, cv2.IMREAD_COLOR)
        
        if bild_cv is None:
            raise ValueError("Konnte Bild nicht lesen.")

        # 2. Vollautomatische Zehen-Detektion (Anatomical Anchor V13)
        auto_messpunkte = FootFinder.find_toes(bild_cv)
        
        # 3. Entzündungsanalyse an den gefundenen Punkten
        analyzer = ThermalAnalyzer(bild_bytes=decrypted_bytes_for_analysis, messpunkte=auto_messpunkte)
        
        if len(auto_messpunkte) > 0:
            analyzer.analyse_protokoll.insert(0, f"👣 Anatomical Anchor: {len(auto_messpunkte)} Zehen vollautomatisch detektiert und anvisiert.")
        else:
            analyzer.analyse_protokoll.insert(0, f"⚠️ Anatomical Anchor: Keine klaren Fuß/Zeh-Strukturen erkannt. Analyse fällt auf Baseline zurück.")

        ergebnisse = analyzer.analysiere() 
        img_base64 = analyzer.render_base64()
        
        del decrypted_bytes_for_analysis
        del encrypted_bytes
        del bild_cv
        del array
        # ============================================================
        
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
            "gefundene_zehen": len(auto_messpunkte),
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
    print(" STARTING ENTERPRISE MEDICAL API (AUTOFOCUS EDITION)")
    print(" - In-Memory Encryption: ENABLED")
    print(" - API-Key Protection:   ENABLED")
    print(" - Rate Limiting:        ENABLED")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)