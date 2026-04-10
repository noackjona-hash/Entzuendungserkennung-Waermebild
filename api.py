from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import hashlib
import json
from typing import Optional

from berechnung import ThermalAnalyzer, TrendManager
from fussfinder import FootFinder

# =============================================================================
# API SETUP & SECURITY
# =============================================================================
API_KEY = "jf2026-jona-super-secret-key-9988"
app = FastAPI(title="ThermoAI Vision Clinical API", version="15.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

async def verify_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unbefugter Zugriff (Invalid Key)")

# =============================================================================
# ANALYSE ENDPUNKT
# =============================================================================
@app.post("/analyze", dependencies=[Depends(verify_key)])
async def analyze(
    file: UploadFile = File(...),
    rgb_file: Optional[UploadFile] = File(None),
    mode: str = Form("auto"),
    patient_id: str = Form("ANONYMOUS"),
    manual_points: str = Form("[]")
):
    try:
        # 1. Bild in Memory laden und decodieren
        thermal_bytes = await file.read()
        nparr = np.frombuffer(thermal_bytes, np.uint8)
        img_thermal = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_thermal is None: raise ValueError("Ungültiges Bildformat.")

        # 2. RGB Overlay (Optional) registrieren
        img_rgb = None
        if rgb_file:
            rgb_bytes = await rgb_file.read()
            img_rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.resize(img_rgb, (img_thermal.shape[1], img_thermal.shape[0]))

        # 3. Messpunkte ermitteln
        if mode == "manual":
            pts = json.loads(manual_points)
            messpunkte = [{"name": p["name"], "punkt": (int(p["x"]), int(p["y"]))} for p in pts]
            radius = 40
        else:
            messpunkte = FootFinder.find_toes(img_thermal)
            radius = 80

        # 4. Analyse Engine starten
        analyzer = ThermalAnalyzer(img_thermal, messpunkte, suchradius=radius)
        befunde = analyzer.analysiere()
        
        # 5. Therapie-Trend speichern
        # Nutzen eines Hashes wenn kein Patient angegeben wurde
        pid = patient_id if patient_id != "ANONYMOUS" else hashlib.sha256(thermal_bytes).hexdigest()[:8]
        history = TrendManager.save_scan(pid, befunde)

        # 6. JSON Export aufbereiten
        return {
            "status": "success",
            "patient_id": pid,
            "anomalien_count": len(befunde),
            "protokoll": analyzer.analyse_protokoll,
            "history_trend": history,
            "daten": [
                {
                    "gelenk": b.gelenk_name,
                    "score": round(b.score_total, 1),
                    "severity": b.severity.name,
                    "temp": b.stats_celsius.to_dict(),
                    "symmetrie_alarm": b.symmetrie_alarm,
                    "delta_t": round(b.delta_t_gegenseite, 1),
                    "zentrum": {"x": b.zentrum[0], "y": b.zentrum[1]},
                    "bbox": b.bounding_box,
                    "konturen": {k: v.reshape(-1, 2).tolist() for k, v in b.konturen_ebenen.items()}
                } for b in befunde
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)