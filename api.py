from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import uvicorn
import os
import traceback

# Hier importieren wir unsere Analyse-Funktionen
from modules.analysis import perform_deep_analysis

# SICHERHEITSNETZ: Falls modules/analysis.py noch die alte Version ist
try:
    from modules.analysis import render_diagnostics
except ImportError:
    print("⚠️ WARNUNG: 'render_diagnostics' fehlt in modules/analysis.py!")
    print("⚠️ Der Server startet trotzdem, aber überprüfe bitte, ob du die Datei gespeichert hast.")
    
    # Fallback-Funktion, die einfach das Originalbild zurückgibt, damit nichts abstürzt
    def render_diagnostics(image, analysis_results):
        return image.copy()

app = FastAPI(
    title="Ignite - Entzündungserkennung",
    description="Jugend forscht 2026 - API zur Auswertung von Wärmebildern",
    version="1.0.0"
)

# Wichtig, damit ein Web-Frontend mit der API kommunizieren darf
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wir binden den Ordner "static" ein, damit CSS/JS Bilder geladen werden können
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    """
    Diese Funktion wird aufgerufen, wenn du http://localhost:8000 im Browser öffnest.
    Sie liefert deine index.html Datei aus dem static-Ordner zurück.
    """
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "API läuft super! Aber die Datei static/index.html wurde nicht gefunden."}


@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Nimmt ein hochgeladenes Wärmebild entgegen und analysiert es auf Entzündungen.
    """
    try:
        # 1. Bilddaten einlesen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            raise ValueError("Das hochgeladene Bild konnte nicht decodiert werden. Ist es eine gültige Bilddatei?")

        print(f"\n📸 [API] Bild erfolgreich vom Frontend empfangen! Dimensionen: {img.shape}")

        # 2. KI-Analyse durchführen
        try:
            results = perform_deep_analysis(img)
        except cv2.error as cv_err:
            if "empty()" in str(cv_err):
                print("\n" + "!"*60)
                print("❌ FEHLER IN DEINEM ANALYSE-MODUL (modules/analysis.py)!")
                print("OpenCV meldet, dass das Bild leer (None) ist, wenn es umgewandelt werden soll.")
                print("Tipp: Überprüfe, ob du die Variable 'image' überschreibst oder versuchst,")
                print("das Bild per cv2.imread() neu von der Festplatte zu laden. Die API")
                print("übergibt das Bild bereits fertig an deine Funktion!")
                print("!"*60 + "\n")
            raise cv_err # Wirft den Fehler trotzdem, damit er im Traceback steht

        # 3. (Optional) Diagnose-Bild erstellen
        # diagnostic_image = render_diagnostics(img, results)
        # cv2.imwrite("diagnostic_output.jpg", diagnostic_image)

        # 4. Ergebnis als JSON zurückgeben
        return JSONResponse(content={
            "status": "success",
            "data": results
        })

    except Exception as e:
        print("\n" + "="*50)
        print("🚨 FEHLER BEI DER BILDANALYSE:")
        traceback.print_exc()
        print("="*50 + "\n")
        
        # Fängt Fehler ab, damit der Server nicht komplett abstürzt
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Startet den Server auf Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)