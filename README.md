# ThermoAI Vision 🔬🌡️
**Jugend Forscht 2026**

ThermoAI Vision ist ein fortschrittliches System zur automatisierten Erkennung von Entzündungen in medizinischen Wärmebildern. Es kombiniert hochsichere In-Memory-Bildverarbeitung mit intelligenten Computer-Vision-Algorithmen, um pathologische Hitze-Signaturen – speziell im Bereich der Füße und Zehen – zu detektieren und zu klassifizieren.

**Lead Researcher & Entwickler:** Jona Noack

---

## ✨ Features

* **Automatische anatomische Verankerung (V13):** Erkennt Zehen und Fußstrukturen in Wärmebildern vollautomatisch, ohne manuelle Markierungen.
* **Hierarchische Multilevel-Segmentierung:** Nutzt Bilateral-Filter, CLAHE und morphologische Analysen zur präzisen Isolierung von Entzündungsherden.
* **Enterprise Security:** DSGVO/HIPAA-konformes FastAPI-Backend. Bilder werden im RAM via AES-256 verschlüsselt und nach der Analyse restlos gelöscht.
* **Zwei Benutzeroberflächen:**
  * **Web-Dashboard:** Modernes Dark-Mode Interface (Tailwind CSS) mit XAI-Log (Explainable AI) und interaktiver Visualisierung.
  * **Desktop GUI:** Lokales Tool (Tkinter) für die manuelle Markierung und schnelle lokale Berichterstellung (HTML-Report).

## 📂 Projektstruktur

* `api.py` - Sicheres FastAPI-Backend mit Rate-Limiting und In-Memory-Verschlüsselung.
* `berechnung.py` - Core-Engine (`ThermalAnalyzer`) zur Bildvorverarbeitung, statistischen Auswertung und Heuristik-Bewertung.
* `fussfinder.py` - Algorithmus zur vollautomatischen Detektion der Zehen über Kontur- und Signalverarbeitung.
* `gui.py` - Lokale Tkinter-Anwendung für manuelle Analysen und PDF/HTML-Reporting.
* `Web/index.html` - Das Frontend-Dashboard zur Kommunikation mit der API.
* `requirements.txt` - Alle Python-Abhängigkeiten.

## 🚀 Installation

1. Repository klonen oder herunterladen.
2. Virtuelle Umgebung erstellen und aktivieren:
    ```python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate```
3. Abhöngigkeiten installieren:
    ```pip install -r requirements.txt
    ```

## 💻 Nutzung

1. Web-API Backend starte
Um das Backend für das Web-Dashboard bereitzustellen:

```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload```

Anschließend die Datei `Web/index.html` im Browser öffnen.
(Hinweis: Für die lokale Nutzung in der index.html die API_URL ggf. auf http://localhost:8000/analyze anpassen).

2. Lokale Desktop-GUI starten
Für die manuelle, lokale Auswertung ohne Web-Server:
```python gui.py
```

## 🔒 Security & Datenschutz
Dieses Projekt ist Privacy-First entwickelt. Hochgeladene Bilder in der API werden nicht auf der Festplatte gespeichert. Sie werden durch das cryptography-Modul sofort im Arbeitsspeicher verschlüsselt, analysiert und unmittelbar danach via Garbage Collection entfernt.