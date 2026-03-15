# Ignite - Thermografische Entzündungserkennung

**Ein Jugend Forscht Projekt (2026)** *Entwickelt von Jona Noack*

## 🔍 Das Problem
Entzündungen an Händen und Füßen (z. B. nach Sportverletzungen oder bei chronischen Erkrankungen) sind mit dem bloßen Auge oft erst spät erkennbar. Herkömmliche Wärmebild-Analysen leiden oft unter Messfehlern durch schwankende Raumtemperaturen oder individuelle biologische Unterschiede.

## 💡 Die Lösung: Bilaterale Symmetrie-Analyse
Ignite nutzt einen wissenschaftlichen Ansatz aus der medizinischen Thermografie: den Seitenvergleich. Da der menschliche Körper thermisch weitgehend symmetrisch ist, deutet eine signifikante Abweichung ($\Delta T$) zwischen dem linken und rechten Zeh/Finger auf eine lokale Entzündung hin.

### Kern-Features:
- **Computer Vision Pipeline:** Automatische Segmentierung von Händen und Füßen mittels OpenCV.
- **Deep-Sensor Technologie:** Intelligente Suche nach dem thermischen Maximum innerhalb der Gewebestruktur (verhindert kalte Messwerte an den Außenkanten).
- **Quantitative Analyse:** Berechnung von Hitze-Clustern, Pixel-Fläche und Schweregrad-Index (Mild, Moderat, Schwer).
- **Hybrid-Modus:** Automatische Erkennung mit interaktiver manueller Korrekturmöglichkeit.
- **PDF-Reporting:** Export eines detaillierten Diagnose-Berichts für die ärztliche Dokumentation.

## 🛠 Installation & Start

1. Repository klonen
2. Virtuelle Umgebung erstellen: `python -m venv .venv`
3. Aktivieren: `.venv\Scripts\activate`
4. Abhängigkeiten installieren: `pip install -r requirements.txt`
5. Starten: `python main.py`

## 📊 Wissenschaftliche Grundlage
Das System nutzt klinische Schwellenwerte:
- **$\Delta T > 20$:** Verdacht auf leichte Entzündung (Gelb/Orange)
- **$\Delta T > 35$:** Eindeutiger pathologischer Befund (Rot)s