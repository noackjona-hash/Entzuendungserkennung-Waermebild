from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse # GEFIXT: Dieser Import hat gefehlt
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool # VERBESSERUNG: Für performante Background-Tasks
import cv2
import numpy as np
import json
import time
from typing import Optional

from berechnung import ThermalAnalyzer, TrendManager
from fussfinder import FootFinder

# =============================================================================
# ENTERPRISE API SETUP
# =============================================================================
app = FastAPI(
    title="ThermoAI Vision Clinical API", 
    version="16.0", # Version auf 16.0 angehoben
    description="Backend für die automatisierte Entzündungserkennung (Jugend Forscht 2026)"
)

# Erlaubt alle Origins für die Web-Vorschau und lokale Tests
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
@app.get("/health") # GEFIXT: Das Dashboard pingt /health, nicht /
async def root():
    """Health-Check Endpoint zum Aufwecken des Servers."""
    return {"status": "online", "version": "16.0", "engine": "ThermoAI-Core-V16"}

def process_image_sync(img_array: np.ndarray, patient_id: str):
    """
    VERBESSERUNG: Die eigentliche Analyse ist rechenintensiv.
    Sie wurde in diese Funktion ausgelagert, damit sie im Threadpool laufen kann.
    """
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Konnte Bilddaten nicht dekodieren.")

    messpunkte = FootFinder.find_toes(img)
    
    if not messpunkte:
        return None, "Keine anatomischen Anker gefunden. Bitte Bildqualität prüfen."

    analyzer = ThermalAnalyzer(img, messpunkte)
    befunde = analyzer.analysiere()
    history = TrendManager.save_scan(patient_id, befunde)

    return {
        "status": "success",
        "daten": [
            {
                "gelenk": b.gelenk_name,
                "score": round(b.score_total, 1),
                "severity": b.severity.name,
                "temp": b.stats_celsius.to_dict(),
                "symmetrie_alarm": b.symmetrie_alarm,
                "zentrum": {"x": b.zentrum[0], "y": b.zentrum[1]},
                "bbox": b.bounding_box,
                "konturen": {k: v.reshape(-1, 2).tolist() for k, v in b.konturen_ebenen.items()}
            } for b in befunde
        ],
        "protokoll": analyzer.protokoll,
        "history_trend": history
    }, None

@app.post("/analyze", dependencies=[Depends(verify_key)])
async def analyze(
    file: UploadFile = File(...),
    patient_id: str = Form("ANONYMOUS"),
    manual_points: str = Form("[]")
):
    try:
        # VERBESSERUNG: Validierung, ob überhaupt ein Bild gesendet wurde
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"detail": "Nur Bilddateien sind erlaubt."})

        # Bild in den RAM laden
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        
        # VERBESSERUNG: Ausführung der Analyse in einem separaten Thread,
        # damit die FastAPI nicht während der Berechnung blockiert.
        result_data, error_msg = await run_in_threadpool(process_image_sync, nparr, patient_id)
        
        if error_msg:
            return JSONResponse(status_code=422, content={"detail": error_msg})

        return result_data

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# NEUE ERWEITERUNG
# =============================================================================
@app.get("/history/{patient_id}", dependencies=[Depends(verify_key)])
async def get_history(patient_id: str):
    """Gibt den historischen Temperatur-Verlauf eines bestimmten Patienten zurück."""
    history_db = TrendManager.load_history()
    if patient_id in history_db:
        return {"patient_id": patient_id, "scans": history_db[patient_id]}
    return JSONResponse(status_code=404, content={"detail": "Patient nicht gefunden."})

if __name__ == "__main__":
    import uvicorn
    # Wichtig für Render: Port muss über Umgebungsvariable oder default 8000 kommen
    uvicorn.run(app, host="0.0.0.0", port=8000)