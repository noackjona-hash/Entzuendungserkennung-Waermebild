from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64
from berechnung import ThermalAnalyzer

app = FastAPI(
    title="Jugend Forscht 2026 - Thermografie API",
    description="API zur automatischen Erkennung von Entzündungen auf Wärmebildern. (In-Memory Version)",
    version="2.6"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", summary="Analysiert ein Wärmebild komplett im Arbeitsspeicher")
async def analyze_thermal_image(
    file: UploadFile = File(...),
    segmente_json: str = Form(...)
):
    try:
        segmente = json.loads(segmente_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Ungültiges JSON-Format.")
        
    try:
        # 1. Bild rein in den Arbeitsspeicher laden - KEIN Festplattenzugriff!
        bild_bytes = await file.read()
        
        # 2. Pipeline aus Bytes starten
        analyzer = ThermalAnalyzer(bild_bytes=bild_bytes, segmente=segmente)
        ergebnisse = analyzer.analysiere() 
        
        # 3. Das gerenderte Bild direkt als Base64 Text aus dem RAM abholen
        img_base64 = analyzer.render_base64()
            
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
            "gefundene_anomalien": len(ergebnisse),
            "daten": export_daten,
            "protokoll": analyzer.analyse_protokoll, 
            "ergebnis_bild_base64": img_base64 
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildanalyse: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)