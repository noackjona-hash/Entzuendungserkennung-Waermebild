import time
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# render_diagnostics wird nicht mehr importiert, da es jetzt Teil der API ist
from modules.geometry import find_both_feet, extract_toes_with_ai
from modules.analysis import perform_deep_analysis

app = FastAPI(
    title="IGNITE Analytics Core", 
    version="4.0.1", 
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

# --- BILD-RENDER FUNKTIONEN (Jetzt exklusiv für die API) ---
def render_diagnostics(image, analysis_payload):
    """Malt die Diagnose-Boxen in das Bild basierend auf den tiefen Metriken."""
    if "error" in analysis_payload and analysis_payload["error"]:
        return image

    overlay = image.copy()
    regional_data = analysis_payload.get("regional_metrics", [])

    for data in regional_data:
        pt_l = data["left_hemisphere"]["coordinates"]
        pt_r = data["right_hemisphere"]["coordinates"]
        status = data["status"]
        tdi = data["bilateral_comparisons"]["tdi_percentage"]
        is_left_hotter = data["bilateral_comparisons"]["is_left_dominant"]
        
        temp_l = data["left_hemisphere"]["metrics"]["thermo_statistics"]["max_temp"]
        temp_r = data["right_hemisphere"]["metrics"]["thermo_statistics"]["max_temp"]

        # Fadenkreuze zeichnen
        cv2.drawMarker(overlay, pt_l, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.circle(overlay, pt_l, 2, (255, 255, 255), -1)
        cv2.drawMarker(overlay, pt_r, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.circle(overlay, pt_r, 2, (255, 255, 255), -1)

        hot_pt = pt_l if is_left_hotter else pt_r
        normal_pt = pt_r if is_left_hotter else pt_l
        normal_temp = temp_r if is_left_hotter else temp_l

        if status == "PATHOLOGISCH_SCHWER":
            _draw_bounding_box(overlay, hot_pt, (0, 0, 255), "SCHWER", tdi)
            _draw_normal_marker(overlay, normal_pt, normal_temp)
        elif status == "PATHOLOGISCH_VERDACHT":
            _draw_bounding_box(overlay, hot_pt, (0, 140, 255), "VERDACHT", tdi)
            _draw_normal_marker(overlay, normal_pt, normal_temp)
        else:
            _draw_normal_marker(overlay, pt_l, temp_l)
            _draw_normal_marker(overlay, pt_r, temp_r)

    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image

def _draw_bounding_box(img, pt, color, status, tdi):
    box_w, box_h = 45, 45
    start_pt = (pt[0]-box_w, pt[1]-box_h)
    end_pt = (pt[0]+box_w, pt[1]+box_h)
    cv2.rectangle(img, start_pt, end_pt, color, cv2.FILLED)
    cv2.rectangle(img, start_pt, end_pt, (255,255,255), 2)
    cv2.putText(img, f"{status} ({tdi}%)", (start_pt[0], start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def _draw_normal_marker(img, pt, temp):
    cv2.circle(img, pt, 6, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (pt[0]-15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)


# --- API ROUTEN ---
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
        
        # 3. Deep Analysis (Nutzt die neue geschwollene Methode)
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