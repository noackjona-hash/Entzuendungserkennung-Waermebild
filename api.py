from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import cv2
import numpy as np
import json
from typing import Optional

from berechnung import ThermalAnalyzer, TrendManager
from fussfinder import FootFinder
from universal_finder import UniversalFinder

# =============================================================================
# ENTERPRISE API SETUP
# =============================================================================
app = FastAPI(
    title="ThermoAI Vision Clinical API", 
    version="16.1",
    description="Backend für die automatisierte Entzündungserkennung"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "jf2026-jona-super-secret-key-9988"

async def verify_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unbefugter Zugriff: API-Key ungültig.")

# =============================================================================
# ENDPUNKTE
# =============================================================================

@app.get("/")
@app.get("/health")
async def root():
    return {"status": "online", "version": "16.1", "engine": "ThermoAI-Core-V16"}

def process_image_sync(img_array: np.ndarray, patient_id: str):
    # Bild dekodieren
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Bilddaten konnten nicht dekodiert werden. Bitte ein gültiges Bild hochladen."

    # 1. Versuch: Füße
    messpunkte = FootFinder.find_toes(img)
    
    # 2. Versuch: Universal (Fallback)
    if not messpunkte:
        messpunkte = UniversalFinder.find_hotspots(img)
    
    if not messpunkte:
        return None, "Bildqualität zu schlecht. Keine relevanten thermischen Regionen gefunden."

    analyzer = ThermalAnalyzer(img, messpunkte)
    befunde = analyzer.analysiere()
    history = TrendManager.save_scan(patient_id, befunde)

    # WICHTIGER FIX: Konvertierung von Numpy-Datentypen in native Python-Typen
    # Andernfalls stürzt FastAPI bei der JSON-Umwandlung ab!
    daten_liste = []
    for b in befunde:
        x, y, w, h = b.bounding_box
        daten_liste.append({
            "gelenk": str(b.gelenk_name),
            "score": float(round(b.score_total, 1)),
            "severity": str(b.severity.name),
            "temp": b.stats_celsius.to_dict(), 
            "symmetrie_alarm": bool(b.symmetrie_alarm),
            "zentrum": {"x": int(b.zentrum[0]), "y": int(b.zentrum[1])},
            "bbox": [int(x), int(y), int(w), int(h)],
            "konturen": {str(k): v.reshape(-1, 2).tolist() for k, v in b.konturen_ebenen.items()}
        })

    return {
        "status": "success",
        "daten": daten_liste,
        "protokoll": analyzer.analyse_protokoll,
        "history_trend": history
    }, None

@app.post("/analyze", dependencies=[Depends(verify_key)])
async def analyze(
    file: UploadFile = File(...),
    patient_id: str = Form("ANONYMOUS")
):
    try:
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        
        # In Threadpool ausführen, damit die API nicht blockiert
        result_data, error_msg = await run_in_threadpool(process_image_sync, nparr, patient_id)
        
        if error_msg:
            # Sende sauberen 422 Fehlercode ans Frontend
            return JSONResponse(status_code=422, content={"detail": error_msg})

        return result_data

    except Exception as e:
        import traceback
        traceback.print_exc() # Schreibt den genauen Fehler auf Render.com in die Logs
        raise HTTPException(status_code=500, detail=f"Interner Server Fehler: {str(e)}")

@app.get("/history/{patient_id}", dependencies=[Depends(verify_key)])
async def get_history(patient_id: str):
    history_db = TrendManager.load_history()
    if patient_id in history_db:
        return {"patient_id": patient_id, "scans": history_db[patient_id]}
    return JSONResponse(status_code=404, content={"detail": "Patient nicht gefunden."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)