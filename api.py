import time
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from modules.geometry import find_both_feet, extract_toes_with_ai
from modules.analysis import perform_deep_analysis, render_diagnostics

app = FastAPI(
    title="IGNITE Diagnostics API", 
    version="2.0.0", 
    description="Hochleistungs-Backend Service für thermografische Bildanalyse"
)

# CORS-Middleware hinzufügen, damit lokale Webseiten/Apps mit der API kommunizieren dürfen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SERVER EINSTELLUNGEN ---
# Alle klinischen Schwellenwerte liegen nun sicher auf dem Server
SERVER_SETTINGS = {
    "warn_threshold_tdi": 8.0,    # Ab 8% Asymmetrie: Verdacht (Orange)
    "severe_threshold_tdi": 15.0, # Ab 15% Asymmetrie: Schwerer Befund (Rot)
    "jpeg_quality": 85            # Kompressionsrate für das zurückgesendete Bild
}

# --- DATENMODELLE ---
class AnalysisResponse(BaseModel):
    global_metrics: Dict[str, float]
    regional_metrics: List[Dict[str, Any]]
    processing_time_ms: float
    processed_image_base64: Optional[str] = None
    error: Optional[str] = None

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_thermogram(file: UploadFile = File(...)):
    """
    Empfängt ein Infrarot-Thermogramm, analysiert es via Machine Learning
    und retourniert klinische Metriken, Berechnungszeit sowie das visuelle Overlay-Bild.
    """
    start_time = time.time()
    
    try:
        # 1. Bild einlesen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Konnte Bilddatei nicht dekodieren.")
            
        img_copy = img_bgr.copy() # Kopie für den späteren Render-Prozess
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. Segmentierung der Füße
        left_contour, right_contour, foot_mask = find_both_feet(gray_img)
        
        if left_contour is None or right_contour is None:
            raise ValueError("Füße konnten nicht segmentiert werden. Bildkontrast zu gering.")
            
        # 3. KI-basierte Zehen-Extraktion
        left_toes = extract_toes_with_ai(left_contour, gray_img, foot_mask)
        right_toes = extract_toes_with_ai(right_contour, gray_img, foot_mask)
        
        # 4. Tiefe statistische Analyse (Nutzt Server-Settings)
        warn_th = SERVER_SETTINGS["warn_threshold_tdi"]
        severe_th = SERVER_SETTINGS["severe_threshold_tdi"]
        
        analysis_payload = perform_deep_analysis(
            left_toes, right_toes, gray_img, foot_mask, warn_th, severe_th
        )
        
        if "error" in analysis_payload:
            raise ValueError(analysis_payload["error"])
            
        # 5. Visuelles Diagnose-Overlay generieren
        final_img = render_diagnostics(img_copy, analysis_payload)
        
        # 6. Bild in Base64 konvertieren für den Web-Transport
        _, buffer = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), SERVER_SETTINGS["jpeg_quality"]])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Berechnungszeit stoppen
        calc_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return AnalysisResponse(
            global_metrics=analysis_payload.get("global_metrics", {}),
            regional_metrics=analysis_payload.get("regional_metrics", []),
            processing_time_ms=calc_time_ms,
            processed_image_base64=f"data:image/jpeg;base64,{img_base64}",
            error=None
        )
        
    except Exception as e:
        calc_time_ms = round((time.time() - start_time) * 1000, 2)
        return AnalysisResponse(
            global_metrics={},
            regional_metrics=[],
            processing_time_ms=calc_time_ms,
            processed_image_base64=None,
            error=str(e)
        )

if __name__ == "__main__":
    # Startet den Server direkt für das Localhost-Testing
    print("\n" + "="*50)
    print("🚀 IGNITE DIAGNOSTICS API SERVER WIRD GESTARTET...")
    print("Einstellungen:", SERVER_SETTINGS)
    print("="*50 + "\n")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)