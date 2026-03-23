from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import shutil
import base64
import uuid
from berechnung import ThermalAnalyzer

app = FastAPI(
    title="Jugend Forscht 2026 - Thermografie API",
    description="API zur automatischen Erkennung von Entzündungen auf Wärmebildern.",
    version="2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "api_temp_runs"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze", summary="Analysiert ein Wärmebild auf Entzündungen")
async def analyze_thermal_image(
    file: UploadFile = File(...),
    segmente_json: str = Form(...)
):
    try:
        segmente = json.loads(segmente_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Ungültiges JSON-Format.")
        
    req_id = str(uuid.uuid4())
    req_dir = os.path.join(TEMP_DIR, req_id)
    os.makedirs(req_dir, exist_ok=True)
    
    file_path = os.path.join(req_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        analyzer = ThermalAnalyzer(bild_pfad=file_path, segmente=segmente)
        ergebnisse = analyzer.analysiere() 
        
        output_img_path = os.path.join(req_dir, f"result_{file.filename}")
        analyzer.render_output(output_img_path)
        
        with open(output_img_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            
        export_daten = []
        for e in ergebnisse:
            export_daten.append({
                "gelenk": e.gelenk_name,
                "score_percent": round(e.score.total_confidence, 2),
                "geometrie": e.morphology.to_dict(),
                "temperatur_celsius": e.stats_celsius.to_dict()
            })
        
        # NEU: Das detaillierte Analyse-Protokoll wird an das Frontend geschickt!
        return JSONResponse(content={
            "status": "success",
            "request_id": req_id,
            "gefundene_anomalien": len(ergebnisse),
            "daten": export_daten,
            "protokoll": analyzer.analyse_protokoll, 
            "ergebnis_bild_base64": img_base64 
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildanalyse: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)