from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from cryptography.fernet import Fernet
import uvicorn
import cv2
import numpy as np
import time
import hashlib
import json
from typing import Dict, Tuple

from berechnung import ThermalAnalyzer
from fussfinder import FootFinder

# =====================================================================
# ENTERPRISE SECURITY CONFIGURATION
# =====================================================================
API_KEY = "jf2026-jona-super-secret-key-9988"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Zugriff verweigert.")
    return api_key

VOLATILE_SECRET_KEY = Fernet.generate_key()
cipher_suite = Fernet(VOLATILE_SECRET_KEY)

RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW_SEC = 60
ip_request_counts: Dict[str, Tuple[int, float]] = {}

def check_rate_limit(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    if client_ip in ip_request_counts:
        count, start_time = ip_request_counts[client_ip]
        if current_time - start_time < RATE_LIMIT_WINDOW_SEC:
            if count >= RATE_LIMIT_REQUESTS: raise HTTPException(status_code=429, detail="Rate Limit überschritten.")
            ip_request_counts[client_ip] = (count + 1, start_time)
        else:
            ip_request_counts[client_ip] = (1, current_time)
    else:
        ip_request_counts[client_ip] = (1, current_time)

# =====================================================================
# FASTAPI INITIALISIERUNG
# =====================================================================
app = FastAPI(title="JF 2026 - Thermografie API", version="5.0-Universal")

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
    return response

@app.post("/analyze")
async def analyze_thermal_image(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),          
    manual_points: str = Form("[]"),   
    api_key: str = Depends(verify_api_key) 
):
    check_rate_limit(request)
        
    try:
        raw_bytes = await file.read()
        file_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
        
        # In-Memory Encryption
        encrypted_bytes = cipher_suite.encrypt(raw_bytes)
        del raw_bytes 
        
        decrypted_bytes_for_analysis = cipher_suite.decrypt(encrypted_bytes)
        array = np.frombuffer(decrypted_bytes_for_analysis, dtype=np.uint8)
        bild_cv = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bild_cv is None: raise ValueError("Konnte Bild nicht lesen.")

        # ============================================================
        # MODUS ENTSCHEIDUNG (Dynamischer Suchradius)
        # ============================================================
        if mode == "manual":
            raw_points = json.loads(manual_points)
            messpunkte = [{"name": p["name"], "punkt": (int(p["x"]), int(p["y"]))} for p in raw_points]
            protokoll_start = f"🎯 Universal Modus: {len(messpunkte)} Gelenke manuell erfasst."
            suchradius = max(30, int(bild_cv.shape[1] * 0.05)) # Adaptiv: Kleinerer Radius für Hand/Knie
        else:
            messpunkte = FootFinder.find_toes(bild_cv)
            protokoll_start = f"👣 Anatomical Anchor: {len(messpunkte)} Gelenk-Anker automatisch detektiert."
            suchradius = max(60, int(bild_cv.shape[1] * 0.12)) # Adaptiv: Größerer Radius für Füße

        # Core Engine
        analyzer = ThermalAnalyzer(bild_bytes=decrypted_bytes_for_analysis, messpunkte=messpunkte, suchradius=suchradius)
        analyzer.analyse_protokoll.insert(0, protokoll_start)

        ergebnisse = analyzer.analysiere() 
        
        # RAM Cleanup
        del decrypted_bytes_for_analysis, encrypted_bytes, bild_cv, array
        
        export_daten = []
        for e in ergebnisse:
            konturen_dict = {ebene: k.reshape(-1, 2).tolist() for ebene, k in e.konturen_ebenen.items()}
            export_daten.append({
                "gelenk": e.gelenk_name,
                "score_percent": round(e.score.total_confidence, 2),
                "severity": e.score.severity.name,
                "zentrum": {"x": int(e.zentrum[0]), "y": int(e.zentrum[1])},
                "bounding_box": {"x": int(e.bounding_box[0]), "y": int(e.bounding_box[1]), "w": int(e.bounding_box[2]), "h": int(e.bounding_box[3])},
                "konturen": konturen_dict,
                "geometrie": e.morphology.to_dict(),
                "temperatur_celsius": e.stats_celsius.to_dict()
            })
            
        messpunkte_export = [{"name": m["name"], "x": int(m["punkt"][0]), "y": int(m["punkt"][1])} for m in messpunkte]
        
        return JSONResponse(content={
            "status": "success",
            "security_clearance": "Data fully encrypted at rest (RAM).",
            "file_hash": file_hash,
            "gefundene_zehen": len(messpunkte),
            "gefundene_anomalien": len(ergebnisse),
            "messpunkte": messpunkte_export,
            "daten": export_daten,
            "protokoll": analyzer.analyse_protokoll
        })
        
    except Exception as e:
        print(f"[ERROR] Analyse fehlgeschlagen: {str(e)}")
        raise HTTPException(status_code=500, detail="Interner Serverfehler.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)