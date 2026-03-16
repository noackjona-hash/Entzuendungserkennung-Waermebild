import time
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from modules.geometry import find_both_feet, extract_toes_with_ai
from modules.analysis import perform_deep_analysis, render_diagnostics

app = FastAPI(
    title="IGNITE Analytics Core", 
    version="4.0.0", 
    description="Vollständiges Backend für die IGNITE Web-App"
)

# CORS erlaubt es unserer lokalen index.html, mit dieser API zu quatschen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    global_metrics: Dict[str, Any]
    regional_metrics: List[Dict[str, Any]]
    processing_time_ms: float
    processed_image_base64: Optional[str] = None

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def process_thermogram(
    file: UploadFile = File(...), 
    warn_th: float = Form(8.0), 
    severe_th: float = Form(15.0)
):
    """
    Nimmt das Bild und die Schieberegler-Werte von der Webseite an, 
    analysiert alles und schickt Daten + gemaltes Bild zurück.
    """
    start_time = time.perf_counter()
    
    try:
        # 1. Bild laden
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Bild konnte nicht gelesen werden.")
            
        img_copy = img_bgr.copy()
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 2. KI Segmentierung
        left_contour, right_contour, foot_mask = find_both_feet(gray_img)
        if left_contour is None or right_contour is None:
            raise HTTPException(status_code=422, detail="Füße nicht gefunden. Kontrast zu schlecht.")
            
        left_toes = extract_toes_with_ai(left_contour, gray_img, foot_mask)
        right_toes = extract_toes_with_ai(right_contour, gray_img, foot_mask)
        
        # 3. Deep Analysis
        analysis_payload = perform_deep_analysis(
            left_toes, right_toes, gray_img, foot_mask, warn_th, severe_th
        )
        
        if "error" in analysis_payload and analysis_payload["error"]:
            raise HTTPException(status_code=500, detail=analysis_payload["error"])
            
        # 4. Bild rendern (Rote/Orange Boxen malen)
        final_img = render_diagnostics(img_copy, analysis_payload)
        
        # 5. Bild für die Webseite verpacken (Base64)
        _, buffer = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        calc_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        return AnalysisResponse(
            global_metrics=analysis_payload.get("global_metrics", {}),
            regional_metrics=analysis_payload.get("regional_metrics", []),
            processing_time_ms=calc_time_ms,
            processed_image_base64=f"data:image/jpeg;base64,{img_base64}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 IGNITE API SERVER LÄUFT! (Drücke STRG+C zum Beenden)")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info")