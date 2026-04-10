from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.middleware.cors import CORSMiddleware
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
    version="15.5",
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
async def root():
    """Health-Check Endpoint zum Aufwecken des Servers."""
    return {"status": "online", "version": "15.5", "engine": "ThermoAI-Core-V15"}

@app.post("/analyze", dependencies=[Depends(verify_key)])
async def analyze(
    file: UploadFile = File(...),
    patient_id: str = Form("ANONYMOUS"),
    manual_points: str = Form("[]")
):
    try:
        # Bild einlesen
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Konnte Bilddaten nicht dekodieren.")

        # Analyse-Logik
        messpunkte = FootFinder.find_toes(img)
        
        if not messpunkte:
            # Fallback falls Auto-Erkennung versagt
            return JSONResponse(
                status_code=422, 
                content={"detail": "Keine anatomischen Anker gefunden. Bitte Bildqualität prüfen."}
            )

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
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Wichtig für Render: Port muss über Umgebungsvariable oder default 8000 kommen
    uvicorn.run(app, host="0.0.0.0", port=8000)