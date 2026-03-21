from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uvicorn

# Hier importieren wir unsere Analyse-Funktionen
from modules.analysis import perform_deep_analysis, render_diagnostics

app = FastAPI(
    title="Ignite - Entzündungserkennung",
    description="Jugend forscht 2026 - API zur Auswertung von Wärmebildern",
    version="1.0.0"
)

# Wichtig, damit ein Web-Frontend (z.B. deine index.html) mit der API kommunizieren darf
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Nimmt ein hochgeladenes Wärmebild entgegen und analysiert es auf Entzündungen.
    """
    try:
        # 1. Bilddaten einlesen
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Das hochgeladene Bild konnte nicht gelesen werden.")

        # 2. KI-Analyse durchführen
        results = perform_deep_analysis(img)

        # 3. (Optional) Diagnose-Bild erstellen, falls du es speichern/anzeigen willst
        # diagnostic_image = render_diagnostics(img, results)
        # cv2.imwrite("diagnostic_output.jpg", diagnostic_image)

        # 4. Ergebnis als JSON zurückgeben
        return JSONResponse(content={
            "status": "success",
            "data": results
        })

    except Exception as e:
        # Fängt Fehler ab, damit der Server nicht abstürzt, sondern eine saubere Fehlermeldung schickt
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Startet den Server auf Port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)