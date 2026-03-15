# Wissenschaftliche Abhandlung über das System IGNITE: Eine hochspezialisierte Instrumentierung für die klinische Thermografie

**Wissenschaftliche Einreichung im Rahmen des Wettbewerbs Jugend Forscht 2026**
**Fachdisziplin:** Mathematik / Informatik / Technik
**Konzeptionelle und technische Realisierung:** Jona Noack (15 Jahre)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-darkblue.svg)](https://github.com/TomSchimansky/CustomTkinter)

---

## 📑 Inhaltsverzeichnis

1. [Projektübersicht (Abstract)](#1-projektübersicht-abstract)
2. [Problemstellung und wissenschaftliche Motivation](#2-problemstellung-und-wissenschaftliche-motivation)
3. [Theoretischer und medizinischer Hintergrund](#3-theoretischer-und-medizinischer-hintergrund)
4. [Klinische Metriken und analytische Indizes](#4-klinische-metriken-und-analytische-indizes)
5. [Systemarchitektur und algorithmische Methodik](#5-systemarchitektur-und-algorithmische-methodik)
6. [Die Machine-Learning-Pipeline](#6-die-machine-learning-pipeline)
7. [Applikationsspezifische Funktionalitäten (UI/UX)](#7-applikationsspezifische-funktionalitäten-uiux)
8. [Installations- und Konfigurationsrichtlinien](#8-installations--und-konfigurationsrichtlinien)
9. [Operationelle Bedienungsanleitung](#9-operationelle-bedienungsanleitung)
10. [Strukturelle Organisation des Projekts](#10-strukturelle-organisation-des-projekts)
11. [Perspektivische Entwicklungen (Ausblick)](#11-perspektivische-entwicklungen-ausblick)

---

## 1. 💡 Projektübersicht (Abstract)

Bei dem vorliegenden System, designiert als **IGNITE**, handelt es sich um eine durch künstliche Intelligenz augmentierte, computergestützte diagnostische Apparatur (Computer-Aided Diagnosis, CAD). Die primäre teleologische Ausrichtung dieser Software liegt in der automatisierten Detektion, exakten Quantifizierung sowie der formalen Dokumentation inflammatorischer Prozesse an den distalen Extremitäten (ossa tarsi et metatarsi sowie ossa carpi et metacarpi) auf Basis von Infrarot-Thermografie.

Die softwareseitige Architektur implementiert modernste Verfahren der Computer Vision (unter Rückgriff auf die OpenCV-Bibliothek) zur Segmentierung thermaler Abbildungen. Ergänzt wird dies durch ein dediziert trainiertes Machine-Learning-Modell auf Basis einer Random-Forest-Regression, welches der präzisen Prädiktion anatomischer Charakteristika dient. Anstatt einer fehleranfälligen Evaluation absoluter Temperaturwerte wird durch das System IGNITE die **bilaterale Symmetrie** des Organismus analysiert. Entzündungsherde werden mittels selbst entwickelter, standardisierter Metriken, insbesondere dem *Thermal Divergence Index (TDI)*, quantifiziert. Die resultierenden diagnostischen Erhebungen werden in einem grafischen Interface visualisiert und können in Form eines formalen klinischen Berichtes exportiert werden.

---

## 2. 🚨 Problemstellung und wissenschaftliche Motivation

Physiologische Anomalien, wie sie durch Entzündungen, rheumatische Affektionen, traumatische Läsionen oder vaskuläre Insuffizienzen hervorgerufen werden, induzieren lokale Alterationen der metabolischen Aktivität. Diese Alterationen manifestieren sich in einer messbaren Modifikation der thermischen Emission der Dermis. Obgleich konventionelle Wärmebildkamerasysteme diese Emissionen visuell zugänglich machen, ist die rein optische, humanbasierte Evaluation jener Abbildungen durch gravierende Fehleranfälligkeiten gekennzeichnet:

* **Extrinsische Störfaktoren:** Die Raumtemperatur, Luftfeuchtigkeit sowie die thermale Kalibrierung der optischen Sensoren verfälschen die absolute Temperaturmessung in erheblichem Maße.
* **Intrinsische, physiologische Varianz:** Es bedarf der Berücksichtigung interindividueller Varianzen hinsichtlich der basalen Körpertemperatur, insbesondere an den peripheren Extremitäten.
* **Sensortechnische Limitationen:** Apparativ bedingtes Bildrauschen und optische Artefakte an den Konturrändern der Gliedmaßen erschweren die exakte Lokalisierung.

**Die abgeleitete Lösungsstrategie:** Es bedurfte der Entwicklung einer automatisierten, streng referenzbasierten Auswertungsmethodik, bei welcher die thermale Signatur der gesunden, kontralateralen Extremität als absolute Vergleichsbasis (Baseline) für die vermutete pathologische Region herangezogen wird.

---

## 3. 🧬 Theoretischer und medizinischer Hintergrund

Eine essenzielle Prämisse der vorliegenden Methodik ist die Annahme, dass der unversehrte menschliche Organismus unter kontrollierten Umgebungsbedingungen eine signifikante thermale Symmetrie aufweist. Folglich müssen korrespondierende anatomische Strukturen der linken und rechten Körperhälfte (beispielsweise der linke und rechte Hallux) nahezu identische Temperaturprofile emittieren.

Die Apparatur IGNITE operationalisiert das Paradigma der **bilateralsymmetrischen Thermografie**. Im Falle einer lokalen Inflammation resultiert eine gesteigerte Perfusion (Hyperämie) sowie eine erhöhte zelluläre Stoffwechselrate. Derartige pathologische Prozesse generieren eine quantifizierbare asymmetrische Temperaturdifferenz ($\Delta T$). Die algorithmische Evaluation identifiziert diese Asymmetrien auf mathematischem Wege und ist durch geeignete Filtermechanismen in der Lage, natürliche, irrelevante Varianzen zu eliminieren.

---

## 4. 📊 Klinische Metriken und analytische Indizes

Zur Gewährleistung einer objektiven Datengrundlage für medizinisches Fachpersonal errechnet die Systemarchitektur in Echtzeit vier fundamentale, diagnostische Metriken:

### Thermal Divergence Index (TDI)
Der TDI dient der Standardisierung der Temperaturdifferenz, vollkommen unabhängig von den apparativen Spezifikationen der genutzten Infrarotkamera oder der applizierten Farbpalette. Der Index reflektiert die prozentuale thermale Diskrepanz zwischen der linken und rechten Körperhälfte.
* **Mathematische Formulierung:** $TDI = \left( \frac{|T_{left} - T_{right}|}{T_{max\_sensor}} \right) \times 100$
* **Klinische Schwellenwerte:** Ein TDI > 8% indiziert einen pathologischen Verdachtsmoment, ein TDI > 15% qualifiziert sich als schwerwiegender Befund. Diese Schwellenwerte unterliegen der manuellen Kalibrierbarkeit durch den Operateur.

### Foot Asymmetry Index (FAI)
Während der TDI auf isolierte distale Phalanxen fokussiert ist, quantifiziert der FAI die systemische thermale Verteilung über die gesamte Extremität. Hierfür wird das arithmetische Mittel aller bilateralen Paare gebildet. Eine erhebliche FAI-Elevation (> 10%) lässt sich korrelativ eher mit systemischen Vaskulopathien (z.B. der peripheren arteriellen Verschlusskrankheit) in Verbindung bringen als mit streng lokalisierten Inflammationen.

### Hotspot Area (Pixel-Fläche)
Diese Metrik kalkuliert die exakte räumliche Ausdehnung des thermischen Herdes. Unter Zuhilfenahme einer adaptiven Binarisierung werden ausschließlich jene Pixel aggregiert, deren thermale Intensität innerhalb der obersten 10% des identifizierten lokalen Temperaturmaximums rangiert.

### Thermischer Gradient ($\nabla T$)
Der Gradient quantifiziert die Steilheit des Temperaturabfalls vom Epizentrum der Inflammation hin zum umliegenden, unversehrten Gewebe.
* **Diagnostische Implikation:** Ein abrupter, steiler Gradient lässt auf eine akute, fokal begrenzte Entzündungsprozession schließen (bspw. Unguis incarnatus). Ein flacher Gradient hingegen suggeriert eine diffus ausgedehnte, chronische Entzündung oder Durchblutungsanomalie.

---

## 5. ⚙️ Systemarchitektur und algorithmische Methodik

Die konzeptionelle Architektur der Software ist von einer strikten Modularität geprägt, um eine Höchstmaß an Performanz und künftiger Erweiterbarkeit (Maintainability) zu sichern.

1. **Preprocessing (`loader.py`):** Es erfolgt ein sicheres, fehlerresistentes Einlesen der Thermogramme über NumPy-basierte Byte-Streams. Dieses Vorgehen eliminiert systemimmanente Restriktionen (beispielsweise Inkompatibilitäten von OpenCV mit spezifischen Zeichenkodierungen). Im Anschluss wird eine Transformation in monochromatische Intensitätsmatrizen vorgenommen.
2. **Segmentierung (`geometry.py`):** * Zunächst wird ein **Gaußscher Weichzeichner (Gaussian Blur)** appliziert, um hochfrequente Störsignale zu dämpfen.
   * Anschließend erfolgt eine **Adaptive Otsu-Binarisierung**. Dieses stochastische Verfahren berechnet vollautomatisch den optimalen Schwellenwert zur Separation des biologischen Gewebes vom Hintergrund.
   * Der Segmentierungsprozess wird durch morphologische Operationen (Opening und Closing) zur Bereinigung etwaiger Vakanzen in der Binärmaske komplettiert.
3. **Deep-Sensor-Algorithmus:** Statt einer rudimentären Kantendetektion platziert das System dynamische Regionen von besonderem Interesse (Region of Interests, ROIs) um die prädizierten anatomischen Zentren. Innerhalb dieser Maskierungen wird tiefgreifend nach der absoluten maximalen thermalen Intensität gesucht.

---

## 6. 🧠 Die Machine-Learning-Pipeline

Zur Gewährleistung einer robusten Identifikation der zehn Phalangen – selbst bei Vorliegen komplexer Varianzen wie eng anliegenden Zehen oder invertierten Ausrichtungen – wurde eine umfassende Pipeline auf Basis des überwachten Lernens (Supervised Learning) konstruiert:

### 1. Rapid Annotation (`annotate_dataset.py`)
Es wurde ein dediziertes Softwarewerkzeug entwickelt, welches die zeitnahe, manuelle Annotation beträchtlicher Volumina an Wärmebildern ermöglicht. Die resultierenden geometrischen Koordinaten werden strukturiert in einer CSV-Datenbank (`labels.csv`) persistiert.

### 2. Data Augmentation (`augment_data.py`)
Zur Prävention von Overfitting und zur signifikanten Steigerung der Modelltoleranz wird der initiale Datensatz algorithmisch um ein Vielfaches potenziert:
* **Affine Transformationen:** Mathematische Rotation der Abbildungen in einem Spektrum von -15° bis +15°, einhergehend mit einer matrixbasierten Rekalkulation sämtlicher Annotationskoordinaten.
* **Intensitätsmodulationen:** Algorithmische Variation von Luminanz und Kontrast zur künstlichen Simulation abweichender Umgebungstemperaturen.

### 3. Merkmalsextraktion und Modelltraining (`train_ai.py`)
Die geometrische Konstitution der Extremitäten wird über die Berechnung von **Hu-Momenten** (sieben translations-, rotations- und skalierungsinvariante Kennzahlen) parametrisiert. Auf Basis dieser Parameter wird ein **Random-Forest-Regressor** (unter Einsatz von 100 Estimators) trainiert. Das Modell erlernt die nicht-lineare Abbildung von globalen morphologischen Merkmalen auf die exakten relativen Koordinaten der distalen Extremitäten.

### 4. KI-Inferenz und Fallback-Mechanismus
Während des operativen Betriebs prädiziert das trainierte Modell die anatomischen Koordinaten in Bruchteilen einer Sekunde. Sollte die Konfidenz der Prädiktion aufgrund gravierender Bildanomalien unterschritten werden, initiiert die Systemarchitektur automatisch einen manuellen Kalibrierungsmodus.

---

## 7. 🖥️ Applikationsspezifische Funktionalitäten (UI/UX)

Das grafische Benutzerinterface (Frontend) wurde vollumfänglich unter Verwendung der Bibliothek **CustomTkinter** realisiert, um zeitgemäßen Standards der User Experience zu entsprechen:

* **Dark-Mode-Interface:** Reduzierung der visuellen Ermüdung des Operateurs bei der Nutzung in abgedunkelten medizinischen Räumlichkeiten.
* **Dynamische Parameteranpassung:** Eine latenzfreie Modifikation der TDI-Schwellenwerte via Schieberegler ist implementiert. Die Aktualisierung der visuellen und tabellarischen Auswertung erfolgt in Echtzeit.
* **Reiterbasierte Navigation (Tab-System):** Eine rigorose strukturelle Trennung zwischen der visuellen Befunddarstellung, der Darlegung roher Messdaten und der softwareseitigen Konfiguration.
* **Automatisierte Reportgenerierung (`fpdf`):** Auf Anforderung erzeugt das System ein formalisiertes, mehrseitiges PDF-Dokument. Dieses umfasst die berechneten Metriken, den ermittelten FAI sowie das visualisierte Thermogramm, um eine nahtlose Integration in die digitale Patientenakte zu gewährleisten.

---

## 8. 🛠️ Installations- und Konfigurationsrichtlinien

### Systemvoraussetzungen
* Betriebssystem: Microsoft Windows 10/11, macOS oder eine kompatible Linux-Distribution
* Laufzeitumgebung: Python in Version 3.10 oder einer aktuelleren Iteration

### Instruktionen zur Implementierung

1. **Klonen des Repositories**
   ```bash
   git clone [https://github.com/noackjona-hash/Entzuendungserkennung-Waermebild.git](https://github.com/noackjona-hash/Entzuendungserkennung-Waermebild.git)
   cd Entzuendungserkennung-Waermebild

2. **Virtuelle Umgebung erstellen & aktivieren**
    Windows:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate```

    MacOS / Linux:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate ```
    
    Abhängigkeiten installieren

Bash
pip install -r requirements.txt
Software starten

Bash
python main.py
🖥️ Bedienungsanleitung
Bild laden: Wähle ein thermografisches Bild (Füße/Hände) aus.

Deep Scan (KI): Klicke auf diesen Button, um die Machine-Learning-Inferenz zu starten. Die Software segmentiert das Bild automatisch.

Manuelle Messung (Fallback): Sollte die Ausrichtung des Bildes stark von den Trainingsdaten abweichen, können die 10 Messpunkte per Fadenkreuz manuell gesetzt werden.

Einstellungen: Wechsle in den Tab "Einstellungen", um die Sensibilität des Algorithmus anzupassen (Live-Update des Bildes).

PDF Export: Klicke in der Seitenleiste auf "Klinischen Report (PDF)", um den Befund abzuspeichern.

🏗️ Architektur & Module
main.py: Controller & CustomTkinter GUI.

modules/loader.py: Sicheres Einlesen von Bildern via NumPy (Bypass für OpenCV-Umlaut-Bugs).

modules/geometry.py: Computer Vision, Hu-Moments und KI-Inferenz.

modules/analysis.py: Berechnung von TDI, Area, Gradient und FAI sowie Rendering der Overlays.

train_ai.py / augment_data.py: Skripte zur Pflege des ML-Modells.

🔭 Ausblick
Für zukünftige Versionen ist geplant:

Unterstützung für DICOM-Dateien (medizinischer Bildstandard).

Integration eines Convolutional Neural Networks (CNN) zur pixelgenauen semantischen Segmentierung.

Live-Video-Feed-Analyse direkt über angeschlossene FLIR-Wärmebildkameras.

Jugend Forscht 2026 - Innovation in der digitalen Medizintechnik.