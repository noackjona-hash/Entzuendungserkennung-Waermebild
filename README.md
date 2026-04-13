# ThermoAI Vision 🔬🌡️
**Jugend Forscht 2026**

ThermoAI Vision ist ein fortschrittliches System zur automatisierten Erkennung von Entzündungen in medizinischen Wärmebildern. Es kombiniert hochsichere In-Memory-Bildverarbeitung mit intelligenten Computer-Vision-Algorithmen, um pathologische Hitze-Signaturen zu detektieren und zu klassifizieren.

**Lead Researcher & Entwickler:** Jona Noack

---

## ✨ Features

* **Automatische anatomische Verankerung (V16):** Erkennt Zehen und Fußstrukturen vollautomatisch. Wenn dies fehlschlägt, greift der intelligente `UniversalFinder` als Fallback für andere Körperteile.
* **Hierarchische Multilevel-Segmentierung:** Nutzt Bilateral-Filter, CLAHE und morphologische Analysen zur präzisen Isolierung von Entzündungsherden.
* **Enterprise Security:** DSGVO/HIPAA-konformes FastAPI-Backend mit Pydantic-Validierung. Bilder werden im RAM via AES-256 verschlüsselt und nach der Analyse restlos gelöscht.
* **Zwei Benutzeroberflächen:**
  * **Web-Dashboard:** Modernes Dark-Mode Interface (Tailwind CSS) mit XAI-Log und interaktiver Visualisierung.
  * **Desktop GUI:** Lokales Tool (Tkinter) für die schnelle lokale Berichterstellung (PDF-Report via ReportLab).

## 📂 Projektstruktur

* `api.py` - Sicheres FastAPI-Backend mit Rate-Limiting, asynchronem Threadpool und In-Memory-Verarbeitung.
* `berechnung.py` - Core-Engine (`ThermalAnalyzer`) zur Bildvorverarbeitung und Pydantic-Validierung.
* `fussfinder.py` - Algorithmus zur Detektion der Zehen über Kontur- und Signalverarbeitung.
* `universal_finder.py` - Generische Hotspot-Erkennung als Fallback.
* `gui.py` - Lokale Tkinter-Anwendung für Analysen und PDF-Reporting.
* `Web/index.html` - Das Frontend-Dashboard.
* `requirements.txt` - Alle Python-Abhängigkeiten.

## 🚀 Installation

1. Repository klonen.
2. Virtuelle Umgebung erstellen und aktivieren:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3. Abhängigkeiten installieren:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Nutzung

**1. Web-API Backend starten:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload