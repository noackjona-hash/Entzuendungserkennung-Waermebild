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

# Initialisiere die FastAPI App (Das wird deine API)
app = FastAPI(
    title="Jugend Forscht 2026 - Thermografie API",
    description="API zur automatischen Erkennung von Entzündungen auf Wärmebildern.",
    version="2.0"
)

# CORS (Cross-Origin Resource Sharing) erlauben, damit spätere Web-Frontends darauf zugreifen dürfen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In Produktion sollte hier die URL deiner Website stehen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ordner für temporäre API-Verarbeitungen erstellen
TEMP_DIR = "api_temp_runs"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/analyze", summary="Analysiert ein Wärmebild auf Entzündungen")
async def analyze_thermal_image(
    file: UploadFile = File(..., description="Das Wärmebild (PNG, JPG)"),
    segmente_json: str = Form(..., description='JSON-String der Segmente, z.B. [{"name": "Zeh 1", "start": [100, 200], "end": [150, 250]}]')
):
    """
    Nimmt ein Bild und die Markierungs-Koordinaten entgegen, jagt sie durch die
    ThermalAnalyzer-Pipeline und gibt die Ergebnisse + Bild als JSON zurück.
    """
    # 1. JSON String zu Python-Objekt parsen
    try:
        segmente = json.loads(segmente_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Ungültiges JSON-Format für 'segmente_json'.")
        
    # 2. Einzigartigen Ordner für diesen API-Call erstellen (verhindert Überschneidungen)
    req_id = str(uuid.uuid4())
    req_dir = os.path.join(TEMP_DIR, req_id)
    os.makedirs(req_dir, exist_ok=True)
    
    # 3. Hochgeladenes Bild temporär speichern
    file_path = os.path.join(req_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 4. Deine magische Pipeline ausführen
        analyzer = ThermalAnalyzer(bild_pfad=file_path, segmente=segmente)
        ergebnisse = analyzer.analysiere() # Nutzt die automatischen Thresholds
        
        # 5. Ergebnisbild rendern
        output_img_path = os.path.join(req_dir, f"result_{file.filename}")
        analyzer.render_output(output_img_path)
        
        # 6. Bild in Base64 umwandeln (damit wir es per API ohne File-Download zurückschicken können)
        with open(output_img_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            
        # 7. Die von berechnung.py generierte JSON-Datei einlesen für den Response
        basis_dateiname = os.path.splitext(file.filename)[0]
        json_export_pfad = os.path.join(req_dir, f"{basis_dateiname}_daten.json")
        
        export_daten = []
        if os.path.exists(json_export_pfad):
            with open(json_export_pfad, "r", encoding="utf-8") as f:
                export_daten = json.load(f)
        
        # 8. Saubere API Response zurückgeben
        return JSONResponse(content={
            "status": "success",
            "request_id": req_id,
            "gefundene_anomalien": len(ergebnisse),
            "daten": export_daten,
            "ergebnis_bild_base64": img_base64 # Das Frontend kann das direkt als src="data:image/png;base64,..." nutzen!
        })
        
    except Exception as e:
        # Falls in der Pipeline was crasht, geben wir einen sauberen 500er Error zurück
        raise HTTPException(status_code=500, detail=f"Fehler bei der Bildanalyse: {str(e)}")

if __name__ == "__main__":
    # Startet den Server auf Port 8000
    print("Starte Jugend Forscht API Server...")
    print("-> Swagger UI Dokumentation unter: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)