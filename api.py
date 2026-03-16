from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

from modules.geometry import find_both_feet, extract_toes_with_ai
from modules.analysis import perform_deep_analysis

app = FastAPI(title="IGNITE Diagnostics API", version="1.0.0", description="Backend Service für thermografische Bildanalyse")

# CORS-Middleware hinzufügen, damit die lokale Webseite mit der API kommunizieren darf
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Im Produktivbetrieb sollte hier nur die Domain der Webseite stehen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    global_metrics: Dict[str, float]
    regional_metrics: List[Dict[str, Any]]
    error: str = None

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_thermogram(file: UploadFile = File(...), warn_th: float = 8.0, severe_th: float = 15.0):
    """
    Empfängt ein Infrarot-Thermogramm und retourniert die extrahierten klinischen und statistischen Metriken im JSON-Format.
    Dies ermöglicht zukünftig die Anbindung von Web-Apps oder mobilen Dashboards.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        left_contour, right_contour, foot_mask = find_both_feet(gray_img)
        
        if left_contour is None or right_contour is None:
            return {"global_metrics": {}, "regional_metrics": [], "error": "Segmentation failure. Bildkontrast zu gering."}
            
        left_toes = extract_toes_with_ai(left_contour, gray_img, foot_mask)
        right_toes = extract_toes_with_ai(right_contour, gray_img, foot_mask)
        
        # Aufruf der API-ready Analyse-Engine
        analysis_payload = perform_deep_analysis(left_toes, right_toes, gray_img, foot_mask, warn_th, severe_th)
        
        return analysis_payload
        
    except Exception as e:
         return {"global_metrics": {}, "regional_metrics": [], "error": str(e)}

if __name__ == "__main__":
    # Startet den Server direkt, wenn die Datei ausgeführt wird
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)