import time
import base64
import math
import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from modules.geometry import find_both_feet, extract_toes_with_ai
from modules.analysis import perform_deep_analysis

app = FastAPI(title="IGNITE Analytics Core", version="6.2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AnalysisResponse(BaseModel):
    global_metrics: Dict[str, Any]
    regional_metrics: List[Dict[str, Any]]
    processing_time_ms: float
    processed_image_base64: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "IGNITE Analytics Core"}

@app.get("/")
async def serve_frontend():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"error": "Dashboard (static/index.html) nicht gefunden!"}

def check_anatomical_plausibility(toes_data):
    if not toes_data or len(toes_data) != 5: return False
    pts = [t["sensor"] for t in toes_data]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = math.sqrt((pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2)
            if dist < 8.0: return False
    return True

def render_diagnostics(image, payload):
    if "error" in payload and payload["error"]: return image
    overlay = image.copy()
    for data in payload.get("regional_metrics", []):
        pt_l = data["left_hemisphere"]["coordinates"]
        pt_r = data["right_hemisphere"]["coordinates"]
        status = data["status"]
        tdi = data["bilateral_comparisons"]["tdi_percentage"]
        is_left_hotter = data["bilateral_comparisons"]["is_left_dominant"]
        temp_l = data["left_hemisphere"]["metrics"]["thermo_statistics"]["max_temp"]
        temp_r = data["right_hemisphere"]["metrics"]["thermo_statistics"]["max_temp"]

        cv2.drawMarker(overlay, pt_l, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.circle(overlay, pt_l, 2, (255, 255, 255), -1)
        cv2.drawMarker(overlay, pt_r, (255, 255, 255), cv2.MARKER_CROSS, 8, 1)
        cv2.circle(overlay, pt_r, 2, (255, 255, 255), -1)

        hot_pt, normal_pt, normal_temp = (pt_l, pt_r, temp_r) if is_left_hotter else (pt_r, pt_l, temp_l)

        if status == "PATHOLOGISCH_SCHWER":
            _draw_box(overlay, hot_pt, (0, 0, 255), "SCHWER", tdi)
            _draw_marker(overlay, normal_pt, normal_temp)
        elif status == "PATHOLOGISCH_VERDACHT":
            _draw_box(overlay, hot_pt, (0, 140, 255), "VERDACHT", tdi)
            _draw_marker(overlay, normal_pt, normal_temp)
        else:
            _draw_marker(overlay, pt_l, temp_l)
            _draw_marker(overlay, pt_r, temp_r)

    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    return image

def _draw_box(img, pt, color, status, tdi):
    start_pt, end_pt = (pt[0]-45, pt[1]-45), (pt[0]+45, pt[1]+45)
    cv2.rectangle(img, start_pt, end_pt, color, cv2.FILLED)
    cv2.rectangle(img, start_pt, end_pt, (255,255,255), 2)
    cv2.putText(img, f"{status} ({tdi}%)", (start_pt[0], start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

def _draw_marker(img, pt, temp):
    cv2.circle(img, pt, 6, (0, 255, 0), -1)
    cv2.putText(img, str(temp), (pt[0]-15, pt[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def process_thermogram(file: UploadFile = File(...), warn_th: float = Form(8.0), severe_th: float = Form(15.0)):
    start_time = time.perf_counter()
    try:
        contents = await file.read()
        img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None: raise HTTPException(status_code=400, detail="Bild fehlerhaft.")
            
        img_copy = img_bgr.copy()
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clean_gray = cv2.bilateralFilter(gray_img, d=9, sigmaColor=75, sigmaSpace=75)
        
        left_contour, right_contour, foot_mask = find_both_feet(clean_gray)
        if left_contour is None or right_contour is None:
            raise HTTPException(status_code=422, detail="Fuesse nicht gefunden.")
            
        left_toes = extract_toes_with_ai(left_contour, clean_gray, foot_mask)
        right_toes = extract_toes_with_ai(right_contour, clean_gray, foot_mask)
        
        if not check_anatomical_plausibility(left_toes) or not check_anatomical_plausibility(right_toes):
            raise HTTPException(status_code=422, detail="KI Plausibilitaets-Check fehlgeschlagen.")
        
        analysis_payload = perform_deep_analysis(left_toes, right_toes, clean_gray, foot_mask, warn_th, severe_th)
        if "error" in analysis_payload and analysis_payload["error"]: raise HTTPException(status_code=500, detail=analysis_payload["error"])
            
        final_img = render_diagnostics(img_copy, analysis_payload)
        _, buffer = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        
        return AnalysisResponse(
            global_metrics=analysis_payload.get("global_metrics", {}),
            regional_metrics=analysis_payload.get("regional_metrics", []),
            processing_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            processed_image_base64=f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")