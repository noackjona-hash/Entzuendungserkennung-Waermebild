import cv2
import numpy as np
import math
import logging
import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# ==============================================================================
# KONFIGURATION & SETUP
# ==============================================================================

@dataclass
class ThermalConfig:
    """
    Zentrale Konfigurationsklasse für alle Algorithmus-Parameter.
    Erlaubt das einfache Feintuning der gesamten Pipeline.
    """
    # Temperatur-Kalibrierung (Pixel zu Celsius)
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    
    # Erkennungs-Parameter
    basis_schwellenwert_pixel: int = 210  # Fallback, falls dynamische Berechnung fehlschlägt
    suchradius_pixel: int = 70            # Radius um den Zeh/Finger
    min_kontur_flaeche: int = 30          # Minimale Pixelanzahl für eine Entzündung
    
    # Scoring-Gewichte (Für die Heuristik)
    score_gewicht_absolut: float = 0.4    # Gewichtung der absoluten Temperatur
    score_gewicht_kontrast: float = 0.3   # Gewichtung des lokalen Kontrasts
    score_gewicht_asymmetrie: float = 0.2 # Gewichtung der Links/Rechts Differenz
    score_gewicht_form: float = 0.1       # Gewichtung der Zirkularität (runde Form)
    min_confidence_score: float = 65.0    # Mindest-Score (0-100), um als Entzündung zu gelten
    
    # NMS (Non-Maximum Suppression) Parameter
    nms_overlap_distanz: int = 40         # Wenn zwei Zentren näher als X Pixel sind, sind sie Duplikate

# ==============================================================================
# DATENSTRUKTUREN (Data Classes) - WISSENSCHAFTLICHES LEVEL
# ==============================================================================

@dataclass
class TemperatureStats:
    """Kapselt alle statistischen Temperaturdaten eines Bereichs."""
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "mean": round(self.mean_val, 2),
            "median": round(self.median_val, 2),
            "std_dev": round(self.std_dev, 2),
            "variance": round(self.variance, 2)
        }

@dataclass
class MorphologyFeatures:
    """Kapselt geometrische Eigenschaften einer gefundenen Entzündung."""
    flaeche: float = 0.0
    umfang: float = 0.0
    zirkularitaet: float = 0.0       # 1.0 ist ein perfekter Kreis
    aspect_ratio: float = 0.0        # Verhältnis Breite/Höhe der Bounding Box
    solidity: float = 0.0            # Verhältnis Fläche zu Convex Hull Fläche
    
    def to_dict(self) -> dict:
        return {
            "flaeche_px": round(self.flaeche, 2),
            "zirkularitaet": round(self.zirkularitaet, 3),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "solidity": round(self.solidity, 3)
        }

@dataclass
class HeuristicScore:
    """Speichert die detaillierte Zusammensetzung der Konfidenz-Bewertung."""
    absolut_score: float = 0.0
    kontrast_score: float = 0.0
    asymmetrie_score: float = 0.0
    form_score: float = 0.0
    total_confidence: float = 0.0    # 0.0 bis 100.0 Prozent
    is_valid: bool = False           # True, wenn total_confidence >= min_confidence_score

@dataclass
class ThermalProfile:
    """Speichert den Temperatur-Gradienten entlang eines markierten Segments."""
    distanzen: List[float] = field(default_factory=list)
    temperaturen_c: List[float] = field(default_factory=list)
    max_temp_c: float = 0.0
    mean_temp_c: float = 0.0

@dataclass
class Entzuendung:
    """
    Haupt-Datenstruktur. Beinhaltet alle berechneten Daten einer Anomalie.
    Abwärtskompatibel zu main.py durch die ersten Attribute.
    """
    # --- Kompatibilität mit bestehender GUI ---
    gelenk_name: str
    groesse_px: float       
    staerke: float # Entspricht der durchschnittlichen Pixelintensität (0-255)
    zentrum: Tuple[int, int] 
    kontur: np.ndarray 
    
    # --- Erweiterte wissenschaftliche Attribute ---
    stats_raw: TemperatureStats = field(default_factory=TemperatureStats)
    stats_celsius: TemperatureStats = field(default_factory=TemperatureStats)
    morphology: MorphologyFeatures = field(default_factory=MorphologyFeatures)
    score: HeuristicScore = field(default_factory=HeuristicScore)
    profil: ThermalProfile = field(default_factory=ThermalProfile)
    
    konturen_ebenen: Dict[str, np.ndarray] = field(default_factory=dict) # 'core', 'mid', 'outer'
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0) # x, y, w, h
    roi_bild: Optional[np.ndarray] = None # Ausgeschnittener Bildbereich

# ==============================================================================
# HAUPTKLASSE DER ANALYSE
# ==============================================================================

class ThermalAnalyzer:
    """
    Professionelle Pipeline zur Analyse medizinischer Wärmebilder.
    Beinhaltet: Preprocessing, Körper-Isolation, Asymmetrie-Prüfung,
    Multilevel-Segmentation, Heuristik-Scoring und Non-Maximum Suppression.
    """
    
    def __init__(self, bild_pfad: str, segmente: List[dict]):
        self.bild_pfad = bild_pfad
        self.segmente = segmente
        self.ausgabe_ordner = os.path.dirname(self.bild_pfad)
        self.basis_dateiname = os.path.splitext(os.path.basename(self.bild_pfad))[0]
        self.config = ThermalConfig()
        
        # Logging initialisieren
        self._setup_logger()
        self.logger.info("="*60)
        self.logger.info(f"START THERMAL ANALYSIS PIPELINE v2.0")
        self.logger.info(f"Bild: {bild_pfad}")
        self.logger.info(f"Anzahl markierter Segmente: {len(segmente)}")
        
        # 1. Bild sicher laden
        self.original_bild = self._lade_bild_sicher(self.bild_pfad)
        self.bild_hoehe, self.bild_breite = self.original_bild.shape[:2]
        
        # 2. Preprocessing & Isolation
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        self.vorverarbeitetes_bild = self._preprocess_image(self.graustufen_bild)
        
        # 3. Globale Körperstatistiken berechnen (Ignoriert Hintergrund)
        self._berechne_globale_koerper_statistiken()
        
        # 4. Asymmetrie-Analyse vorbereiten (Links vs. Rechts)
        self._bereite_symmetrie_analyse_vor()
        
        # Speicher für Ergebnisse
        self.gefundene_entzuendungen: List[Entzuendung] = []
        self.alle_kandidaten: List[Entzuendung] = []

    def _setup_logger(self):
        """Erstellt ein rotierendes, formatiertes Log für Debugging."""
        self.logger = logging.getLogger("ThermalAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            log_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            log_datei = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_analyse.log")
            fh = logging.FileHandler(log_datei, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(log_format)
            self.logger.addHandler(fh)
            
            # Auch auf der Konsole ausgeben
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(log_format)
            self.logger.addHandler(ch)

    def _lade_bild_sicher(self, pfad: str) -> np.ndarray:
        """Sicheres Laden via Numpy, verhindert Crashs bei Windows-Umlauten."""
        try:
            with open(pfad, 'rb') as f:
                bytes_data = f.read()
            array = np.frombuffer(bytes_data, dtype=np.uint8)
            bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bild is None:
                raise ValueError("Decode-Fehler.")
            return bild
        except Exception as e:
            self.logger.critical(f"Konnte Bild nicht laden: {e}")
            raise FileNotFoundError(f"Bildladefehler: {pfad}")

    def _preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
        """Wendet Denoising und Adaptive Histogram Equalization (CLAHE) an."""
        self.logger.debug("Führe Preprocessing durch (Denoising + CLAHE)...")
        # NLM Denoising erhält Kanten besser als Gaussian Blur
        denoised = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        # CLAHE für lokalen Kontrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced

    def _pixel_zu_celsius(self, pixel_wert: float) -> float:
        """Lineare Interpolation von Pixelintensität zu geschätzter Temperatur."""
        prozent = max(0.0, min(1.0, pixel_wert / 255.0))
        return self.config.temp_min_celsius + (prozent * (self.config.temp_max_celsius - self.config.temp_min_celsius))

    def _berechne_globale_koerper_statistiken(self):
        """
        Trennt den Körper vom kalten Hintergrund (mittels Otsu's Methode)
        und berechnet die echte Durchschnittstemperatur der Haut.
        Das verhindert, dass ein kalter Hintergrund die Werte verfälscht.
        """
        # Otsu Thresholding zur Hintergrund-Eliminierung
        _, koerper_maske = cv2.threshold(self.vorverarbeitetes_bild, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Berechne Werte nur auf dem Körper
        koerper_pixel = self.vorverarbeitetes_bild[koerper_maske == 255]
        
        if len(koerper_pixel) > 0:
            self.global_mean_temp_raw = float(np.mean(koerper_pixel))
            self.global_std_temp_raw = float(np.std(koerper_pixel))
            
            # Sortieren um die Top 5% Hitze zu finden (für dynamischen Threshold)
            sorted_pixels = np.sort(koerper_pixel)
            top_5_percent_idx = int(len(sorted_pixels) * 0.95)
            self.global_hot_threshold = float(sorted_pixels[top_5_percent_idx])
            
            self.logger.info(f"Globale Körper-Statistik -> Mean: {self.global_mean_temp_raw:.1f}, "
                             f"StdDev: {self.global_std_temp_raw:.1f}, Top 5% Schwelle: {self.global_hot_threshold:.1f}")
        else:
            self.logger.warning("Konnte Körper nicht vom Hintergrund trennen. Nutze Fallback.")
            self.global_mean_temp_raw = 128.0
            self.global_std_temp_raw = 30.0
            self.global_hot_threshold = 200.0

    def _bereite_symmetrie_analyse_vor(self):
        """
        Teilt die Segmente in Links und Rechts auf, um asymmetrische Entzündungen 
        zu finden. (Ein gesunder Mensch ist meist thermisch symmetrisch).
        """
        if not self.segmente:
            return
            
        # Finde den mittleren X-Wert aller Segmente
        alle_x = [(s['start'][0] + s['end'][0]) / 2 for s in self.segmente]
        bild_mitte_x = sum(alle_x) / len(alle_x) if alle_x else self.bild_breite / 2
        
        self.segmente_links = []
        self.segmente_rechts = []
        
        temp_sum_links, count_links = 0.0, 0
        temp_sum_rechts, count_rechts = 0.0, 0
        
        # Analysiere jedes Segment grob vor
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            
            # Lokale Max-Temp bestimmen
            maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokal = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=maske)
            _, max_val, _, _ = cv2.minMaxLoc(lokal, mask=maske)
            
            seg['pre_max_temp'] = max_val
            
            if mX < bild_mitte_x:
                self.segmente_links.append(seg)
                temp_sum_links += max_val
                count_links += 1
            else:
                self.segmente_rechts.append(seg)
                temp_sum_rechts += max_val
                count_rechts += 1
                
        self.mean_temp_links = temp_sum_links / count_links if count_links > 0 else 0
        self.mean_temp_rechts = temp_sum_rechts / count_rechts if count_rechts > 0 else 0
        
        self.logger.info(f"Symmetrie-Analyse: Links Ø {self.mean_temp_links:.1f} (Raw) | Rechts Ø {self.mean_temp_rechts:.1f} (Raw)")
        
        # Wenn eine Seite deutlich heißer ist, stufen wir das Risiko für die kalte Seite herab
        self.asymmetrie_faktor = abs(self.mean_temp_links - self.mean_temp_rechts)
        self.heissere_seite = "links" if self.mean_temp_links > self.mean_temp_rechts else "rechts"

    def _berechne_statistiken(self, maske: np.ndarray) -> Tuple[TemperatureStats, TemperatureStats]:
        """Extrahiert Min, Max, Mean, Var für Pixel innerhalb einer Maske."""
        pixel_werte = self.vorverarbeitetes_bild[maske == 255]
        if len(pixel_werte) == 0:
            return TemperatureStats(), TemperatureStats()
            
        raw = TemperatureStats(
            min_val=float(np.min(pixel_werte)),
            max_val=float(np.max(pixel_werte)),
            mean_val=float(np.mean(pixel_werte)),
            median_val=float(np.median(pixel_werte)),
            std_dev=float(np.std(pixel_werte)),
            variance=float(np.var(pixel_werte))
        )
        
        celsius_werte = np.array([self._pixel_zu_celsius(p) for p in pixel_werte])
        celsius = TemperatureStats(
            min_val=float(np.min(celsius_werte)),
            max_val=float(np.max(celsius_werte)),
            mean_val=float(np.mean(celsius_werte)),
            median_val=float(np.median(celsius_werte)),
            std_dev=float(np.std(celsius_werte)),
            variance=float(np.var(celsius_werte))
        )
        return raw, celsius

    def _berechne_morphologie(self, kontur: np.ndarray) -> MorphologyFeatures:
        """Analysiert die geometrische Form der Entzündung (z.B. Zirkularität)."""
        flaeche = cv2.contourArea(kontur)
        umfang = cv2.arcLength(kontur, True)
        
        # Zirkularität = 4 * pi * Fläche / Umfang^2. Ein perfekter Kreis hat 1.0.
        zirkularitaet = 0.0
        if umfang > 0:
            zirkularitaet = (4 * math.pi * flaeche) / (umfang * umfang)
            
        x, y, w, h = cv2.boundingRect(kontur)
        aspect_ratio = float(w) / h if h > 0 else 0.0
        
        hull = cv2.convexHull(kontur)
        hull_flaeche = cv2.contourArea(hull)
        solidity = float(flaeche) / hull_flaeche if hull_flaeche > 0 else 0.0
        
        return MorphologyFeatures(
            flaeche=flaeche, 
            umfang=umfang, 
            zirkularitaet=zirkularitaet, 
            aspect_ratio=aspect_ratio, 
            solidity=solidity
        )

    def _erstelle_thermisches_profil(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> ThermalProfile:
        """
        Extrahiert die Pixelintensitäten exakt auf der gezeichneten Linie.
        Erzeugt einen Temperatur-Graphen.
        """
        x1, y1 = p1
        x2, y2 = p2
        length = int(np.hypot(x2-x1, y2-y1))
        
        if length == 0:
            return ThermalProfile()
            
        x_indices = np.linspace(x1, x2, length)
        y_indices = np.linspace(y1, y2, length)
        
        distanzen = []
        temperaturen_c = []
        raw_vals = []
        
        for i in range(length):
            xi, yi = int(round(x_indices[i])), int(round(y_indices[i]))
            if 0 <= xi < self.bild_breite and 0 <= yi < self.bild_hoehe:
                raw_val = float(self.vorverarbeitetes_bild[yi, xi])
                temp_c = self._pixel_zu_celsius(raw_val)
                distanzen.append(float(i))
                temperaturen_c.append(temp_c)
                raw_vals.append(raw_val)
                
        if not temperaturen_c:
            return ThermalProfile()
            
        return ThermalProfile(
            distanzen=distanzen,
            temperaturen_c=temperaturen_c,
            max_temp_c=max(temperaturen_c),
            mean_temp_c=sum(temperaturen_c)/len(temperaturen_c)
        )

    def _bewerte_kandidat(self, seg_name: str, peak_raw: float, lokal_mean: float, 
                          morph: MorphologyFeatures, is_linke_seite: bool) -> HeuristicScore:
        """
        Das Herzstück der Intelligenz: Ein KI-ähnliches Scoring-System.
        Verhindert False-Positives, indem es Anomalien multidimensional bewertet.
        """
        # 1. Absoluter Score: Wie heiß ist es verglichen mit dem absoluten Maximum (255)?
        # Wenn Peak < globaler Mean, gibt es 0 Punkte.
        baseline = max(150.0, self.global_mean_temp_raw)
        if peak_raw <= baseline:
            abs_score = 0.0
        else:
            abs_score = min(100.0, ((peak_raw - baseline) / (255.0 - baseline)) * 100.0)
            
        # 2. Kontrast Score: Wie viel heißer ist die Stelle als ihre direkte Umgebung?
        # Eine echte Entzündung hebt sich stark ab (Delta > 30).
        delta = peak_raw - lokal_mean
        kontrast_score = min(100.0, max(0.0, (delta / 50.0) * 100.0))
        
        # 3. Asymmetrie Score: Ist dieser Fuß heißer als der andere?
        # Straft den kalten Fuß ab (verhindert False Positives wie am rechten Fuß im Beispiel).
        asym_score = 100.0
        if self.asymmetrie_faktor > 15.0: # Wenn deutliche Asymmetrie vorliegt
            if (is_linke_seite and self.heissere_seite == "rechts") or \
               (not is_linke_seite and self.heissere_seite == "links"):
                asym_score = 30.0 # Harte Strafe für die kühlere Seite
                
        # 4. Form Score (Zirkularität): Entzündungen sind oft rund/elliptisch (Blob).
        # Zu zerklüftete Formen (Zirkularität < 0.2) sind meistens Rauschen oder Gefäße.
        form_score = min(100.0, morph.zirkularitaet * 100.0)
        if morph.zirkularitaet < 0.2:
            form_score = 0.0
            
        # Zusammenrechnen der Heuristik (Gewichtetes Mittel)
        total = (abs_score * self.config.score_gewicht_absolut) + \
                (kontrast_score * self.config.score_gewicht_kontrast) + \
                (asym_score * self.config.score_gewicht_asymmetrie) + \
                (form_score * self.config.score_gewicht_form)
                
        is_valid = total >= self.config.min_confidence_score
        
        self.logger.debug(f"Scoring [{seg_name}]: Peak={peak_raw:.0f}, Abs={abs_score:.0f}, "
                          f"Kontrast={kontrast_score:.0f}, Asym={asym_score:.0f}, Form={form_score:.0f} "
                          f"-> TOTAL: {total:.1f}% (Valid: {is_valid})")
                          
        return HeuristicScore(
            absolut_score=abs_score,
            kontrast_score=kontrast_score,
            asymmetrie_score=asym_score,
            form_score=form_score,
            total_confidence=total,
            is_valid=is_valid
        )

    def _non_maximum_suppression(self, kandidaten: List[Entzuendung]) -> List[Entzuendung]:
        """
        NMS (Non-Maximum Suppression):
        Entfernt Duplikate! Wenn zwei Segmente dieselbe Entzündung finden (weil ihre
        Suchradien überlappen), wird nur die stärkere Meldung behalten.
        """
        if not kandidaten:
            return []
            
        # Sortiere nach Konfidenz (Stärkste zuerst)
        kandidaten.sort(key=lambda x: x.score.total_confidence, reverse=True)
        
        gefiltert: List[Entzuendung] = []
        for kand in kandidaten:
            is_duplicate = False
            for etabliert in gefiltert:
                # Berechne Euklidischen Abstand zwischen Zentren
                distanz = math.hypot(kand.zentrum[0] - etabliert.zentrum[0], 
                                     kand.zentrum[1] - etabliert.zentrum[1])
                
                # Wenn Zentren zu nah beieinander sind, ist es dieselbe Entzündung
                if distanz < self.config.nms_overlap_distanz:
                    is_duplicate = True
                    self.logger.info(f"NMS: Unterdrücke '{kand.gelenk_name}' ({kand.score.total_confidence:.1f}%), "
                                     f"da es mit '{etabliert.gelenk_name}' überlappt (Dist: {distanz:.1f}px).")
                    break
                    
            if not is_duplicate:
                gefiltert.append(kand)
                
        return gefiltert

    def analysiere(self, temperatur_schwellenwert: Optional[int] = None, max_distanz: Optional[int] = None, **kwargs) -> List[Entzuendung]:
        """
        Haupt-Pipeline: 
        Führt Thresholding, Segmentierung, Scoring und NMS aus.
        Akzeptiert die alten Argumente aus Kompatibilitätsgründen zur GUI (main.py).
        """
        # Abwärtskompatibilität zur main.py sicherstellen
        if temperatur_schwellenwert is not None:
            self.config.basis_schwellenwert_pixel = temperatur_schwellenwert
        if max_distanz is not None:
            self.config.suchradius_pixel = max_distanz
            
        self.alle_kandidaten = []
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            is_links = mX < (self.bild_breite / 2)
            
            # Profil erstellen
            profil = self._erstelle_thermisches_profil(s, e)
            
            # Maske für den lokalen Suchbereich (nur Zeh-Umgebung)
            lokale_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(lokale_maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokaler_bereich = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=lokale_maske)
            
            # Finde Hitzepunkt
            _, max_val, _, max_loc = cv2.minMaxLoc(lokaler_bereich, mask=lokale_maske)
            lokal_mean = cv2.mean(self.vorverarbeitetes_bild, mask=lokale_maske)[0]
            
            # Dynamischer Schwellenwert für dieses Segment (Peak - 30, mind. aber Haut-Durchschnitt)
            lokaler_schwelle = max(self.global_mean_temp_raw + 10, int(max_val) - 30)
            
            # Multi-Level Thresholding
            schwellen = {
                'core': max(lokaler_schwelle + 20, int(max_val) - 10),
                'mid': max(lokaler_schwelle + 10, int(max_val) - 20),
                'outer': lokaler_schwelle
            }
            
            konturen_dict = {}
            haupt_kontur = None
            kernel = np.ones((3, 3), np.uint8)
            
            for ebene, schwelle in schwellen.items():
                _, thresh = cv2.threshold(lokaler_bereich, schwelle, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                konturen, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if konturen:
                    # Finde größte Kontur in der Nähe des Peaks
                    gueltige_konturen = [cnt for cnt in konturen if cv2.contourArea(cnt) >= self.config.min_kontur_flaeche]
                    if gueltige_konturen:
                        groesste_kontur = max(gueltige_konturen, key=cv2.contourArea)
                        konturen_dict[ebene] = groesste_kontur
                        if ebene == 'outer':
                            haupt_kontur = groesste_kontur
                            
            if haupt_kontur is None and 'core' in konturen_dict:
                haupt_kontur = konturen_dict['core']
                
            if haupt_kontur is not None:
                # 1. Zentrum & Bounding Box
                M = cv2.moments(haupt_kontur)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else max_loc[0]
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else max_loc[1]
                x, y, w, h = cv2.boundingRect(haupt_kontur)
                
                # 2. Morphologie
                morph = self._berechne_morphologie(haupt_kontur)
                
                # 3. Statistiken der Entzündung
                entz_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
                cv2.drawContours(entz_maske, [haupt_kontur], -1, 255, -1)
                raw_stats, celsius_stats = self._berechne_statistiken(entz_maske)
                
                # 4. Scoring (Bewertung)
                score = self._bewerte_kandidat(seg['name'], max_val, lokal_mean, morph, is_links)
                
                # 5. ROI Extrahieren
                padding = 15
                y1 = max(0, y - padding)
                y2 = min(self.bild_hoehe, y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(self.bild_breite, x + w + padding)
                roi = self.original_bild[y1:y2, x1:x2].copy()
                
                kandidat = Entzuendung(
                    gelenk_name=seg['name'],
                    groesse_px=morph.flaeche,
                    staerke=raw_stats.mean_val,
                    zentrum=(cX, cY),
                    kontur=haupt_kontur,
                    stats_raw=raw_stats,
                    stats_celsius=celsius_stats,
                    morphology=morph,
                    score=score,
                    profil=profil,
                    konturen_ebenen=konturen_dict,
                    bounding_box=(x, y, w, h),
                    roi_bild=roi
                )
                
                # Wenn der Score hoch genug ist, füge ihn als Kandidaten hinzu
                if score.is_valid:
                    self.alle_kandidaten.append(kandidat)
                    
        # 6. Duplikate entfernen (Non-Maximum Suppression)
        self.gefundene_entzuendungen = self._non_maximum_suppression(self.alle_kandidaten)
        self.logger.info(f"Analyse abgeschlossen. {len(self.gefundene_entzuendungen)} finale Entzündung(en) nach NMS.")
        
        # 7. Daten exportieren
        self._exportiere_daten()
        self._speichere_rois()
        
        return self.gefundene_entzuendungen

    # ==========================================================================
    # DATEN EXPORT (JSON, CSV, BERICHT)
    # ==========================================================================
    
    def _exportiere_daten(self):
        """Exportiert wissenschaftliche JSON und CSV Dateien."""
        json_pfad = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_daten.json")
        csv_pfad = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_daten.csv")
        txt_pfad = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_details.txt")
        
        export_daten = []
        for e in self.gefundene_entzuendungen:
            data = {
                "gelenk": e.gelenk_name,
                "score_percent": round(e.score.total_confidence, 2),
                "geometrie": e.morphology.to_dict(),
                "zentrum": {"x": e.zentrum[0], "y": e.zentrum[1]},
                "temperatur_celsius": e.stats_celsius.to_dict()
            }
            export_daten.append(data)
            
        try:
            with open(json_pfad, 'w', encoding='utf-8') as f:
                json.dump(export_daten, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"JSON Export Fehler: {e}")

        try:
            if export_daten:
                with open(csv_pfad, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(['Gelenk', 'Confidence_%', 'Flaeche_px', 'Temp_Max_C', 'Temp_Mean_C'])
                    for e in export_daten:
                        writer.writerow([
                            e['gelenk'], e['score_percent'], e['geometrie']['flaeche_px'],
                            e['temperatur_celsius']['max'], e['temperatur_celsius']['mean']
                        ])
        except Exception as e:
            self.logger.error(f"CSV Export Fehler: {e}")
            
        # Generiere eine lesbare Text-Zusammenfassung
        self._erstelle_text_bericht(txt_pfad)

    def _erstelle_text_bericht(self, pfad: str):
        """Generiert einen detaillierten, menschenlesbaren Report (für Jugend Forscht Jury)."""
        lines = [
            "==================================================",
            "   THERMOGRAFIE ANALYSE - JUGEND FORSCHT 2026",
            "==================================================",
            f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysiertes Bild: {self.basis_dateiname}",
            f"Körper-Basis-Temperatur (Raw): {self.global_mean_temp_raw:.1f}",
            f"Linke Körperhälfte Ø: {self.mean_temp_links:.1f} | Rechte Körperhälfte Ø: {self.mean_temp_rechts:.1f}",
            f"Asymmetrie-Faktor: {self.asymmetrie_faktor:.1f} (Heißere Seite: {self.heissere_seite})",
            "--------------------------------------------------",
            f"GEFUNDENE ANOMALIEN (Nach NMS): {len(self.gefundene_entzuendungen)}",
            "--------------------------------------------------"
        ]
        
        for idx, e in enumerate(self.gefundene_entzuendungen, 1):
            lines.append(f"[{idx}] {e.gelenk_name}")
            lines.append(f"    - Konfidenz-Score : {e.score.total_confidence:.1f}%")
            lines.append(f"    - Max Temperatur  : {e.stats_celsius.max_val:.1f} °C")
            lines.append(f"    - Ø Temperatur    : {e.stats_celsius.mean_val:.1f} °C")
            lines.append(f"    - Zirkularität    : {e.morphology.zirkularitaet:.2f} (1.0 = Kreis)")
            lines.append(f"    - Entzündungsherd : {int(e.morphology.flaeche)} px² (Fläche)")
            lines.append("")
            
        lines.append("METHODIK:")
        lines.append("- Die Analyse nutzt Non-Maximum Suppression (NMS), um überlappende False-Positives zu filtern.")
        lines.append("- Ein KI-heuristisches Scoring-System vergleicht Kontrast, absolute Hitze, Asymmetrie und geometrische Form.")
        
        try:
            with open(pfad, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
        except Exception as e:
            self.logger.error(f"Text-Report Fehler: {e}")

    def _speichere_rois(self):
        """Speichert die isolierten Entzündungsbereiche in einen Ordner."""
        roi_ordner = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_ROIs")
        if self.gefundene_entzuendungen and not os.path.exists(roi_ordner):
            os.makedirs(roi_ordner)
            
        for e in self.gefundene_entzuendungen:
            if e.roi_bild is not None:
                sicherer_name = str(e.gelenk_name).replace(" ", "_").replace("/", "_")
                roi_pfad = os.path.join(roi_ordner, f"ROI_{e.score.total_confidence:.0f}perc_{sicherer_name}.png")
                erfolg, buffer = cv2.imencode('.png', e.roi_bild)
                if erfolg:
                    with open(roi_pfad, 'wb') as f:
                        f.write(buffer)

    # ==========================================================================
    # BILD AUSGABE (RENDERING)
    # ==========================================================================

    def render_output(self, output_pfad: str):
        """
        Rendert das wissenschaftliche finale Ergebnisbild.
        Zeichnet Linien, Konturen, Bounding Boxes, Zentren und Konfidenz-Scores.
        """
        ausgabe = self.original_bild.copy()
        
        # 1. Zeichne alle Basis-Segmente ganz dünn im Hintergrund
        for seg in self.segmente:
            cv2.line(ausgabe, seg['start'], seg['end'], (255, 100, 0), 1, cv2.LINE_AA)
            cv2.circle(ausgabe, seg['start'], 2, (150, 150, 150), -1)
            cv2.circle(ausgabe, seg['end'], 2, (150, 150, 150), -1)

        # 2. Zeichne validierte Entzündungen
        farben = {'outer': (0, 255, 255), 'mid': (0, 165, 255), 'core': (0, 0, 255)}
        
        for entz in self.gefundene_entzuendungen:
            # Bounding Box (Dezent)
            x, y, w, h = entz.bounding_box
            cv2.rectangle(ausgabe, (x, y), (x+w, y+h), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Konturen Zeichnen (Multi-Level)
            for ebene, kontur in entz.konturen_ebenen.items():
                farbe = farben.get(ebene, (255, 255, 255))
                dicke = -1 if ebene == 'core' else 2
                cv2.drawContours(ausgabe, [kontur], -1, farbe, dicke, cv2.LINE_AA)
                
            # Zentrum markieren
            cv2.drawMarker(ausgabe, entz.zentrum, (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.drawMarker(ausgabe, entz.zentrum, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

            # Professionelles GUI Label mit Konfidenz!
            label_name = f"{entz.gelenk_name}"
            label_temp = f"Temp: {entz.stats_celsius.max_val:.1f}C"
            label_conf = f"Conf: {entz.score.total_confidence:.1f}%"
            
            # Textgrößen berechnen für das Hintergrund-Rechteck
            t1, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t2, _ = cv2.getTextSize(label_temp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t3, _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_w = max(t1[0], t2[0], t3[0])
            
            # Positionieren (über der Box)
            bg_rect_start = (x, max(0, y - 50))
            bg_rect_end = (x + max_w + 6, max(0, y - 2))
            
            # Schwarzer Hintergrund für Lesbarkeit
            cv2.rectangle(ausgabe, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            
            # Texte Zeichnen
            text_x = x + 3
            text_y_base = y - 40
            cv2.putText(ausgabe, label_name, (text_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (text_x, text_y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Farbe für Konfidenz (Grün wenn sehr sicher, Gelb wenn unsicherer)
            conf_color = (0, 255, 0) if entz.score.total_confidence > 80 else (0, 255, 255)
            cv2.putText(ausgabe, label_conf, (text_x, text_y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1, cv2.LINE_AA)
                        
        # UMLAUT-SICHERES SPEICHERN
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            with open(output_pfad, 'wb') as f:
                f.write(buffer)
            self.logger.info(f"Finales Ergebnisbild gespeichert unter: {output_pfad}")
        else:
            self.logger.error("Fehler beim Encodieren des Ergebnisbildes.")
            raise IOError("Fehler: Das Ergebnisbild konnte nicht kodiert werden.")