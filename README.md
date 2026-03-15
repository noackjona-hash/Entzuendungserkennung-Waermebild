🔬 IGNITE - Clinical Thermography Pro

Jugend Forscht 2026 | Fachgebiet: Mathematik / Informatik / Technik

Entwickelt von Jona Noack (15 Jahre)

📑 Inhaltsverzeichnis

Projektübersicht (Abstract)

Problemstellung & Motivation

Wissenschaftlicher Hintergrund

Klinische Metriken & Indizes

Systemarchitektur & Algorithmen

Die Machine Learning Pipeline

Software-Features (UI/UX)

Installation & Setup

Bedienungsanleitung

Projektstruktur

Zukünftige Entwicklungen (Ausblick)

1. 💡 Projektübersicht (Abstract)

Ignite ist ein KI-gestütztes, computerunterstütztes Diagnosesystem (CAD) zur Erkennung, Quantifizierung und Dokumentation von Entzündungen an den Extremitäten (Füße/Hände) mittels Infrarot-Thermografie.

Das System nutzt fortschrittliche Computer Vision (OpenCV) zur Bildsegmentierung und ein selbst trainiertes Machine-Learning-Modell (Random Forest Regressor) zur anatomischen Feature-Erkennung. Anstatt sich auf fehleranfällige absolute Temperaturwerte zu verlassen, berechnet Ignite die bilaterale Symmetrie des Körpers und quantifiziert Entzündungsherde über standardisierte, selbst entwickelte Metriken wie den Thermal Divergence Index (TDI). Die Ergebnisse werden in einer modernen Benutzeroberfläche visualisiert und können als professioneller klinischer PDF-Report exportiert werden.

2. 🚨 Problemstellung & Motivation

Entzündungen, rheumatische Erkrankungen, Sportverletzungen oder Durchblutungsstörungen verändern den lokalen Stoffwechsel und damit die Wärmeabstrahlung der Haut. Herkömmliche Wärmebildkameras machen dies zwar sichtbar, jedoch ist die menschliche (optische) Auswertung stark fehleranfällig:

Umgebungseinflüsse: Raumtemperatur und Kamera-Kalibrierung verfälschen absolute Messwerte (z.B. 32°C).

Physiologische Varianz: Manche Patienten haben von Natur aus kältere Extremitäten als andere.

Bildrauschen: Unpräzise Sensoren erzeugen "Ghosting" oder Rauschen an den Rändern der Gliedmaßen.

Die Lösung: Eine automatisierte, referenzbasierte Auswertung, die den gesunden Fuß/die gesunde Hand als absolute Baseline für den erkrankten Bereich nutzt.

3. 🧬 Wissenschaftlicher Hintergrund

Der menschliche Körper weist im gesunden Zustand eine bemerkenswerte thermische Symmetrie auf. Das bedeutet, dass der linke und rechte Körperteil (z.B. der linke und rechte große Zeh) unter gleichen Umgebungsbedingungen nahezu identische Temperaturen aufweisen.

Ignite nutzt das Prinzip der Bilateralsymmetrischen Thermografie. Tritt eine lokale Entzündung auf, steigt die Durchblutung (Hyperämie) und Gewebeaktivität an. Dies führt zu einer messbaren asymmetrischen Temperaturdifferenz ($\Delta T$). Die Software identifiziert diese Asymmetrien mathematisch und filtert natürliche Varianzen heraus.

4. 📊 Klinische Metriken & Indizes

Um Ärzten und Diagnostikern objektive Daten zu liefern, berechnet das System in Echtzeit vier Kernmetriken:

Thermal Divergence Index (TDI)

Der TDI standardisiert die Temperaturdifferenz unabhängig vom Kameratyp oder der gewählten Farbpalette. Er gibt die prozentuale Abweichung zwischen der linken und rechten Seite an.

Formel: $TDI = \left( \frac{|T_{left} - T_{right}|}{T_{max\_sensor}} \right) \times 100$

Klinische Schwelle: > 8% (Verdacht), > 15% (Schwerer Befund). Einstellbar in der GUI.

Foot Asymmetry Index (FAI)

Während der TDI einzelne Zehen betrachtet, bewertet der FAI die systemische Durchblutung des gesamten Fußes, indem er den Durchschnitt aller Extremitätenpaare bildet. Ein hoher FAI (> 10%) deutet eher auf eine generelle Gefäßerkrankung (z.B. pAVK) als auf eine lokale Entzündung hin.

Hotspot Area (Fläche)

Berechnet die exakte Pixelanzahl des Entzündungsherdes. Hierbei werden durch adaptive Binarisierung nur jene Pixel gezählt, die sich innerhalb der obersten 10% des lokalen Temperaturmaximums befinden.

Thermischer Gradient ($\nabla T$)

Der Gradient beschreibt, wie abrupt die Temperatur vom Zentrum der Entzündung zum gesunden Gewebe hin abfällt.

Bedeutung: Ein steiler Abfall deutet auf eine akute, scharf umrissene Entzündung hin (z.B. ein eingewachsener Nagel). Ein flacher Abfall weist auf eine großflächige, chronische Entzündung oder Durchblutungsstörung hin.

5. ⚙️ Systemarchitektur & Algorithmen

Die Software-Architektur ist streng modular aufgebaut, um maximale Performance und Wartbarkeit zu garantieren.

Preprocessing (loader.py): Sicheres Einlesen der Thermogramme über NumPy-Byte-Streams, um Pfad-Probleme (z.B. Windows-Umlaute) in OpenCV zu umgehen. Konvertierung in Graustufen-Wärmemaps.

Segmentierung (geometry.py): * Anwendung eines Gaussian Blur zur Rauschunterdrückung.

Adaptive Otsu-Binarisierung zur vollautomatischen Ermittlung des optimalen Schwellenwerts für die Hintergrundtrennung.

Morphologische Operationen (Open/Close) zum Schließen von Maskenlücken.

Deep Sensor Algorithmus: Anstatt blind die Kanten der Füße abzutasten, legt das System dynamische Region of Interests (ROIs) um die anatomischen Zentren und sucht maskiert nach dem absoluten thermischen Spitzenwert im tiefen Gewebe.

6. 🧠 Die Machine Learning Pipeline

Um die 10 Zehen/Finger in Bildern mit hoher Varianz (aneinanderliegende Zehen, invertierte Fußstellung) robust zu erkennen, wurde eine vollständige Supervised Learning Pipeline entwickelt:

1. Rapid Annotation (annotate_dataset.py)

Ein eigens entwickeltes Tool, um in Sekunden große Mengen an Wärmebildern manuell zu annotieren und die Koordinaten in einer labels.csv zu speichern.

2. Data Augmentation (augment_data.py)

Um Overfitting zu vermeiden und die Robustheit zu steigern, wird das initiale Datenset algorithmisch um ein Vielfaches vergrößert:

Affine Transformationen: Rotation der Bilder (-15° bis +15°) inklusive matrix-basierter Neuberechnung der Annotationen.

Intensitätsverschiebungen: Simulation von Umgebungstemperaturschwankungen durch Helligkeits- und Kontrastanpassungen.

3. Feature Extraktion & Training (train_ai.py)

Die Formen der Füße werden durch Hu-Moments (7 translations- und rotationsinvariante Momente) beschrieben. Ein Random Forest Regressor (100 Estimators) wird trainiert, um von diesen globalen Form-Features auf die relativen Koordinaten der 5 Extremitäten zu schließen.

4. KI-Inferenz & Fallback

In der Hauptanwendung sagt das Modell die Koordinaten in Millisekunden voraus. Schlägt die Inferenz aufgrund von zu starker Bildabweichung fehl, springt das System in den manuellen Fallback-Modus.

7. 🖥️ Software-Features (UI/UX)

Das Frontend wurde komplett in CustomTkinter realisiert, um modernste UX-Standards zu erfüllen:

Dark-Mode UI: Ermüdungsfreies Arbeiten in medizinischen Umgebungen.

Live-Parameter Tuning: Echtzeit-Anpassung der TDI-Schwellenwerte via Slider. Das Bild und die Tabellen aktualisieren sich in Millisekunden.

Tab-System: Saubere Trennung zwischen Bild-Visualisierung, rohen Daten-Metriken und Einstellungen.

Klinischer PDF-Report (fpdf): Mit einem Klick generiert Ignite einen mehrseitigen, druckfertigen Diagnosebericht inkl. Metriken, FAI und visualisiertem Wärmebild zur Ablage in der Patientenakte.

8. 🛠️ Installation & Setup

Systemanforderungen

OS: Windows 10/11, macOS, oder Linux

Python: Version 3.10 oder höher

Schritt-für-Schritt Anleitung

Repository klonen

git clone [https://github.com/noackjona-hash/Entzuendungserkennung-Waermebild.git](https://github.com/noackjona-hash/Entzuendungserkennung-Waermebild.git)
cd Entzuendungserkennung-Waermebild


Virtuelle Umgebung erstellen & aktivieren

Windows:

python -m venv .venv
.venv\Scripts\activate


MacOS / Linux:

python3 -m venv .venv
source .venv/bin/activate


Abhängigkeiten installieren

pip install -r requirements.txt


Software starten

python main.py


9. 📖 Bedienungsanleitung

Wärmebild laden: Klicke in der Seitenleiste auf Bild laden und wähle dein Thermogramm (.jpg, .png).

Deep Scan (KI): Klicke auf Auto-Analyse. Das ML-Modell identifiziert die Messpunkte, und der Deep Sensor analysiert das Gewebe.

Ergebnisse prüfen: Im Reiter Visualisierung siehst du farbcodierte Bounding-Boxen (Rot = Schwer, Orange = Verdacht). Im Reiter Detail-Analyse findest du alle rohen Messwerte und Gradienten.

Parameter anpassen: Im Reiter Einstellungen können die Toleranzgrenzen des Algorithmus stufenlos reguliert werden.

Report exportieren: Klicke unten links auf PDF Bericht, um die Sitzung zu dokumentieren.

Hinweis zum KI-Training: Eigene Modelle können trainiert werden, indem Bilder über annotate_dataset.py gelabelt, mit augment_data.py erweitert und durch train_ai.py in das System kompiliert werden.

10. 📂 Projektstruktur

Ignite-Thermography/
├── main.py                   # Hauptapplikation & CustomTkinter GUI
├── annotate_dataset.py       # Tool zur schnellen manuellen Datenerfassung
├── augment_data.py           # Skript zur Datensatz-Vermehrung (Rotation/Helligkeit)
├── train_ai.py               # ML-Training (Random Forest) Skript
├── requirements.txt          # Python-Abhängigkeiten
├── dataset/
│   ├── ignite_ai_model.pkl   # Trainiertes Machine-Learning Modell
│   ├── labels.csv            # Annotations-Datenbank
│   └── images/               # Ordner für Trainingsbilder
└── modules/
    ├── __init__.py
    ├── loader.py             # Sicheres Image-Loading (NumPy-Bypass)
    ├── geometry.py           # Computer Vision, Otsu, Hu-Moments & ML-Inferenz
    └── analysis.py           # Berechnung von TDI, Area, Gradient & Rendering


11. 🔮 Zukünftige Entwicklungen (Ausblick)

DICOM-Support: Nativer Support für medizinische Bilddatenformate (Digital Imaging and Communications in Medicine).

Deep Learning Segmentierung: Integration eines U-Net oder Mask R-CNN Architektur zur pixelgenauen Erkennung von Fuß- und Handgeometrien anstelle von Bounding Box Regression.

Live-Video-Feed: Echtzeit-Analyse direkt über via USB/WLAN verbundene FLIR-Wärmebildkameras (Real-Time Tracking).

Jugend Forscht 2026 - Innovation in der digitalen Medizintechnik. © Jona Noack