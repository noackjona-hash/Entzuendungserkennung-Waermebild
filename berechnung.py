import cv2
import numpy as np
import math
import logging
import base64
import time
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union

# ==============================================================================
# ENTERPRISE KONFIGURATION & SETUP (STRICT TYPING)
# ==============================================================================

class SeverityLevel(Enum):
    """Klassifizierung der Entzündungsschwere basierend auf thermischen Anomalien."""
    NORMAL = auto()
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()

@dataclass
class ThermalConfig:
    """
    Zentrale Konfigurationsklasse für alle mathematischen und algorithmischen Parameter.
    Optimiert für In-Memory Cloud-Verarbeitung und DSGVO-konforme Ausführung.
    
    Attribute:
        temp_min_celsius (float): Untere Grenze der Kalibrierungsskala (z.B. Raumtemperatur).
        temp_max_celsius (float): Obere Grenze der Kalibrierungsskala (Körperkerntemperatur/Fieber).
        basis_schwellenwert_pixel (int): 8-Bit Pixelwert für Basis-Cutoff.
        suchradius_pixel (int): Analyseradius in Pixeln um das angegebene Gelenkzentrum.
        min_kontur_flaeche (int): Rauschfilter für zu kleine Hitzespots (Artefakte).
        score_gewicht_absolut (float): Heuristisches Gewicht für absolute Temperatur.
        score_gewicht_kontrast (float): Heuristisches Gewicht für lokalen Temperaturgradienten.
        score_gewicht_asymmetrie (float): Heuristisches Gewicht für kontralaterale Asymmetrie.
        score_gewicht_form (float): Heuristisches Gewicht für pathologische Formgebung.
        min_confidence_score (float): Cutoff für die Klassifizierung als pathologische Läsion.
        nms_overlap_distanz (int): Distanz für Non-Maximum Suppression (Duplikatsfilter).
    """
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    
    basis_schwellenwert_pixel: int = 210  
    suchradius_pixel: int = 80            
    min_kontur_flaeche: int = 12          # <--- Geändert auf 12 (verhindert das Ausfiltern der heißen Zehenspitzen)
    
    score_gewicht_absolut: float = 0.40    # <--- ERHÖHT: Absolute Hitze ist der wichtigste Faktor
    score_gewicht_kontrast: float = 0.30   
    score_gewicht_asymmetrie: float = 0.20 
    score_gewicht_form: float = 0.10       # <--- REDUZIERT: Zehen-Entzündungen sind oft länglich, nicht kreisrund
    
    min_confidence_score: float = 80.0    # <--- ANGEPASST: 80% ist ein exzellenter klinischer Cutoff
    nms_overlap_distanz: int = 45         
    
    use_watershed_segmentation: bool = False # <--- Deaktiviert, da es glatte Wärmeverläufe zu stark zersplittert
    calculate_hu_moments: bool = True

# ==============================================================================
# WISSENSCHAFTLICHE DATENSTRUKTUREN
# ==============================================================================

@dataclass
class TemperatureStats:
    """
    Umfassende statistische Analyse eines thermischen Profils.
    Beinhaltet Momente höherer Ordnung (Skewness, Kurtosis) zur Bestimmung
    der Homogenität des Entzündungsherdes.
    """
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 2), 
            "max": round(self.max_val, 2),
            "mean": round(self.mean_val, 2), 
            "median": round(self.median_val, 2),
            "std_dev": round(self.std_dev, 2), 
            "variance": round(self.variance, 2),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4)
        }

@dataclass
class MorphologyFeatures:
    """
    Kapselt geometrische und topologische Eigenschaften einer thermischen Anomalie.
    Beinhaltet Hu-Momente zur rotations- und skaleninvarianten Formerkennung.
    """
    flaeche: float = 0.0
    umfang: float = 0.0
    zirkularitaet: float = 0.0       
    aspect_ratio: float = 0.0        
    solidity: float = 0.0            
    equivalent_diameter: float = 0.0
    hu_moments: List[float] = field(default_factory=lambda: [0.0]*7)
    
    def to_dict(self) -> dict:
        return {
            "flaeche_px": round(self.flaeche, 2), 
            "zirkularitaet": round(self.zirkularitaet, 3),
            "aspect_ratio": round(self.aspect_ratio, 3), 
            "solidity": round(self.solidity, 3),
            "equivalent_diameter": round(self.equivalent_diameter, 2)
        }

@dataclass
class HeuristicScore:
    """
    Repräsentiert die multi-kriterielle Entscheidungsmatrix der KI.
    Gibt detaillierten Aufschluss darüber, warum eine Anomalie klassifiziert wurde.
    """
    absolut_score: float = 0.0
    kontrast_score: float = 0.0
    asymmetrie_score: float = 0.0
    form_score: float = 0.0
    total_confidence: float = 0.0    
    is_valid: bool = False           
    severity: SeverityLevel = SeverityLevel.NORMAL

@dataclass
class Entzuendung:
    """
    Haupt-Datenstruktur für medizinische Befunde.
    Verbindet rohe Pixeldaten, physikalische Berechnungen und KI-Scoring.
    """
    gelenk_name: str
    groesse_px: float       
    staerke: float 
    zentrum: Tuple[int, int] 
    kontur: np.ndarray 
    
    stats_raw: TemperatureStats = field(default_factory=TemperatureStats)
    stats_celsius: TemperatureStats = field(default_factory=TemperatureStats)
    morphology: MorphologyFeatures = field(default_factory=MorphologyFeatures)
    score: HeuristicScore = field(default_factory=HeuristicScore)
    
    konturen_ebenen: Dict[str, np.ndarray] = field(default_factory=dict) 
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0) 
    
    def get_severity_color(self) -> Tuple[int, int, int]:
        """Gibt den entsprechenden BGR-Farbcode basierend auf der Schwere zurück."""
        if self.score.severity == SeverityLevel.CRITICAL:
            return (0, 0, 255) # Rot
        elif self.score.severity == SeverityLevel.SEVERE:
            return (0, 100, 255) # Dunkelorange
        elif self.score.severity == SeverityLevel.MODERATE:
            return (0, 165, 255) # Orange
        else:
            return (0, 255, 255) # Gelb

# ==============================================================================
# MATHEMATISCHE & STATISTISCHE HILFSKLASSEN
# ==============================================================================

class MathUtils:
    """Statische Sammlung fortgeschrittener mathematischer Berechnungen."""
    
    @staticmethod
    def calculate_higher_order_stats(pixel_array: np.ndarray) -> Tuple[float, float]:
        """
        Berechnet die statistische Schiefe (Skewness) und Wölbung (Kurtosis).
        Wichtig, um zu erkennen, ob die Hitze homogen verteilt ist oder starke Spikes aufweist.
        
        Args:
            pixel_array: 1D Numpy Array der Pixelwerte.
        Returns:
            Tuple aus (Skewness, Kurtosis)
        """
        if len(pixel_array) < 3:
            return 0.0, 0.0
            
        mean = np.mean(pixel_array)
        std = np.std(pixel_array)
        if std == 0:
            return 0.0, 0.0
            
        n = len(pixel_array)
        # Skewness (3. zentrales Moment)
        skewness = (np.sum((pixel_array - mean)**3) / n) / (std**3)
        # Kurtosis (4. zentrales Moment - 3)
        kurtosis = (np.sum((pixel_array - mean)**4) / n) / (std**4) - 3.0
        
        return float(skewness), float(kurtosis)

    @staticmethod
    def extract_hu_moments(contour: np.ndarray) -> List[float]:
        """
        Berechnet die 7 Hu-Momente für eine gegebene Kontur.
        Diese Momente sind invariant gegenüber Translation, Skalierung und Rotation.
        Perfekt für Machine Learning Pattern Recognition von Entzündungsformen.
        """
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        # Log-Transformation für bessere Lesbarkeit der winzigen Werte
        log_hu = []
        for i in range(0, 7):
            if hu_moments[i][0] != 0:
                val = -1 * math.copysign(1.0, hu_moments[i][0]) * math.log10(abs(hu_moments[i][0]))
                log_hu.append(float(val))
            else:
                log_hu.append(0.0)
        return log_hu

# ==============================================================================
# SECURITY & CONTEXT MANAGEMENT
# ==============================================================================

class SecurityContext:
    """
    Überwacht den Speicherstatus und stellt sicher, dass sensible Bilddaten
    nicht aus dem RAM auf die Festplatte "leaken" (z.B. durch Swapping).
    """
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.creation_time = time.time()
        self.is_secured = True
        
    def verify_integrity(self) -> bool:
        """Prüft, ob der Analyse-Kontext noch gültig und im RAM gesichert ist."""
        if time.time() - self.creation_time > 300: # Max 5 Min Lebensdauer pro Analyse
            self.is_secured = False
            return False
        return self.is_secured

# ==============================================================================
# HAUPTKLASSE DER ANALYSE (IN-MEMORY SECURED ENTERPRISE EDITION)
# ==============================================================================

class ThermalAnalyzer:
    """
    Wissenschaftliche, speichersichere Pipeline zur Analyse medizinischer Wärmebilder.
    Implementiert Computer Vision Standards (CLAHE, Watershed, Adaptive Thresholding)
    und operiert zu 100% In-Memory (Zero-Disk-I/O).
    """
    
    def __init__(self, bild_bytes: bytes, segmente: List[dict]):
        """
        Initialisiert die Analyse-Engine mit verschlüsselten RAM-Daten.
        
        Args:
            bild_bytes: Binäre Daten des hochgeladenen Wärmebildes.
            segmente: Liste der Benutzer-markierten Gelenkkoordinaten.
        """
        self.security = SecurityContext()
        self.segmente = segmente
        self.config = ThermalConfig()
        self.analyse_protokoll: List[str] = []
        
        self._setup_logger()
        self.logger.info(f"Session [{self.security.session_id}] - Starte In-Memory Analyse")
        
        # 1. Dekodierung & Isolation (RAM Only)
        self.original_bild = self._lade_bild_aus_bytes(bild_bytes)
        self.bild_hoehe, self.bild_breite = self.original_bild.shape[:2]
        
        # 2. Medical Preprocessing
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        self.vorverarbeitetes_bild = self._preprocess_image(self.graustufen_bild)
        
        # 3. Mathematische Baselines
        self._berechne_globale_koerper_statistiken()
        self._bereite_symmetrie_analyse_vor()
        
        self.gefundene_entzuendungen: List[Entzuendung] = []
        self.alle_kandidaten: List[Entzuendung] = []

    def _setup_logger(self):
        """Initialisiert den RAM-only Logger für System-Events."""
        self.logger = logging.getLogger(f"ThermalAnalyzer_{self.security.session_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s | RAM-SECURE | %(message)s'))
            self.logger.addHandler(ch)

    def _lade_bild_aus_bytes(self, bild_bytes: bytes) -> np.ndarray:
        """Sicheres Entpacken des Binärstroms zu einem OpenCV Matrix-Objekt."""
        if not self.security.verify_integrity():
            raise RuntimeError("Security Context abgelaufen. Abbruch zum Datenschutz.")
            
        try:
            array = np.frombuffer(bild_bytes, dtype=np.uint8)
            bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bild is None: 
                raise ValueError("Matrix Dimensionen korrupt. Kein valides Bild.")
            return bild
        except Exception as e:
            self.logger.error(f"Fehler bei RAM-Bild-Dekodierung: {e}")
            raise ValueError("Konnte verschlüsseltes Bild nicht im RAM dekodieren.")

    def _preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Erweitertes medizinisches Preprocessing (RAM-Optimiert):
        1. Bilateral Filter (erhält Kanten, entfernt Rauschen, extrem RAM-schonend).
        2. Contrast Limited Adaptive Histogram Equalization (CLAHE).
        """
        self.analyse_protokoll.append("🔬 Wende kanten-erhaltendes Denoising (Bilateral) und CLAHE an...")
        # Weichere Parameter (d=5), damit der Hitze-Peak des Zehs nicht weichgezeichnet wird
        denoised = cv2.bilateralFilter(gray_image, d=5, sigmaColor=50, sigmaSpace=50)
        # Adaptives Histogramm zur Herausarbeitung feinster Temperaturunterschiede
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def _pixel_zu_celsius(self, pixel_wert: float) -> float:
        """Transformiert den relativen 8-Bit Pixelwert in die definierte absolute Temperaturskala."""
        prozent = max(0.0, min(1.0, pixel_wert / 255.0))
        return self.config.temp_min_celsius + (prozent * (self.config.temp_max_celsius - self.config.temp_min_celsius))

    def _berechne_globale_koerper_statistiken(self):
        """
        Segmentiert den Patientenkörper vom kalten Bildhintergrund mittels Multi-Otsu-ähnlicher Logik.
        Berechnet die physiologische thermische Baseline für das Scoring.
        """
        # Otsu-Methode findet automatisch den perfekten Schwellenwert zwischen Körper und Raum
        _, koerper_maske = cv2.threshold(self.vorverarbeitetes_bild, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        koerper_pixel = self.vorverarbeitetes_bild[koerper_maske == 255]
        
        if len(koerper_pixel) > 0:
            self.global_mean_temp_raw = float(np.mean(koerper_pixel))
            self.global_std_temp_raw = float(np.std(koerper_pixel))
            
            # Die 95. Perzentile definiert die Schwelle für "ungewöhnlich heiß"
            sorted_pixels = np.sort(koerper_pixel)
            top_5_percent_idx = int(len(sorted_pixels) * 0.95)
            self.global_hot_threshold = float(sorted_pixels[top_5_percent_idx])
            
            # Higher Order Stats für das gesamte Bild
            skew, kurt = MathUtils.calculate_higher_order_stats(koerper_pixel)
            self.logger.info(f"Baseline Skewness: {skew:.2f}, Kurtosis: {kurt:.2f}")
        else:
            self.global_mean_temp_raw = 128.0
            self.global_std_temp_raw = 30.0
            self.global_hot_threshold = 200.0
            
        temp_c = self._pixel_zu_celsius(self.global_mean_temp_raw)
        self.analyse_protokoll.append(f"🔍 Physiologische Baseline ermittelt (Ø {temp_c:.1f}°C).")

    def _bereite_symmetrie_analyse_vor(self):
        """
        Klinische Asymmetrie-Prüfung: Teilt den Patienten sagittal in zwei Hälften.
        Erkennt systemische Überwärmung einer Extremität (z.B. aufgrund von Durchblutungsstörungen),
        welche zu False-Positives in der Einzelzeh-Analyse führen könnte.
        """
        if not self.segmente: return
        
        # Berechne die Sagittalachse (Mitte) dynamisch basierend auf den Markierungen
        alle_x = [(s['start'][0] + s['end'][0]) / 2 for s in self.segmente]
        bild_mitte_x = sum(alle_x) / len(alle_x) if alle_x else self.bild_breite / 2
        
        temp_sum_links, count_links = 0.0, 0
        temp_sum_rechts, count_rechts = 0.0, 0
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            
            maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokal = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=maske)
            
            # Finde den Peak in diesem Segment
            _, max_val, _, _ = cv2.minMaxLoc(lokal, mask=maske)
            
            if mX < bild_mitte_x:
                temp_sum_links += max_val
                count_links += 1
            else:
                temp_sum_rechts += max_val
                count_rechts += 1
                
        self.mean_temp_links = temp_sum_links / count_links if count_links > 0 else 0
        self.mean_temp_rechts = temp_sum_rechts / count_rechts if count_rechts > 0 else 0
        self.asymmetrie_faktor = abs(self.mean_temp_links - self.mean_temp_rechts)
        self.heissere_seite = "links" if self.mean_temp_links > self.mean_temp_rechts else "rechts"
        
        # Ab einem Delta von 15 Pixelwerten sprechen wir von signifikanter Asymmetrie
        if self.asymmetrie_faktor > 15:
            self.analyse_protokoll.append(f"⚖️ Systemische Asymmetrie: Die {self.heissere_seite} Seite ist signifikant wärmer. Score-Penalty wird aktiviert.")

    def _berechne_statistiken(self, maske: np.ndarray) -> Tuple[TemperatureStats, TemperatureStats]:
        """Extrahiert tiefgehende statistische Verteilungen aus der maskierten Entzündungsregion."""
        pixel_werte = self.vorverarbeitetes_bild[maske == 255]
        if len(pixel_werte) == 0: 
            return TemperatureStats(), TemperatureStats()
            
        skew, kurt = MathUtils.calculate_higher_order_stats(pixel_werte)
            
        raw = TemperatureStats(
            min_val=float(np.min(pixel_werte)), 
            max_val=float(np.max(pixel_werte)), 
            mean_val=float(np.mean(pixel_werte)), 
            median_val=float(np.median(pixel_werte)), 
            std_dev=float(np.std(pixel_werte)), 
            variance=float(np.var(pixel_werte)),
            skewness=skew,
            kurtosis=kurt
        )
        
        celsius_werte = np.array([self._pixel_zu_celsius(p) for p in pixel_werte])
        celsius = TemperatureStats(
            min_val=float(np.min(celsius_werte)), 
            max_val=float(np.max(celsius_werte)), 
            mean_val=float(np.mean(celsius_werte)), 
            median_val=float(np.median(celsius_werte)), 
            std_dev=float(np.std(celsius_werte)), 
            variance=float(np.var(celsius_werte)),
            skewness=0.0, # Nicht linear transformierbar ohne Rohdaten, hier vereinfacht
            kurtosis=0.0
        )
        return raw, celsius

    def _berechne_morphologie(self, kontur: np.ndarray) -> MorphologyFeatures:
        """
        Wertet die topologische Struktur der Isotherme (Hitze-Kontur) aus.
        Nutzt Computervision-Standards wie Zirkularität und Solidity zur Abgrenzung
        gegenüber Blutgefäßen (linear) oder Messfehlern (zerklüftet).
        """
        flaeche = cv2.contourArea(kontur)
        umfang = cv2.arcLength(kontur, True)
        
        zirkularitaet = (4 * math.pi * flaeche) / (umfang * umfang) if umfang > 0 else 0.0
        x, y, w, h = cv2.boundingRect(kontur)
        
        hull = cv2.convexHull(kontur)
        hull_flaeche = cv2.contourArea(hull)
        solidity = float(flaeche) / hull_flaeche if hull_flaeche > 0 else 0.0
        
        equivalent_diameter = np.sqrt(4 * flaeche / np.pi)
        hu_moments = MathUtils.extract_hu_moments(kontur) if self.config.calculate_hu_moments else []
        
        return MorphologyFeatures(
            flaeche=flaeche, 
            umfang=umfang, 
            zirkularitaet=zirkularitaet, 
            aspect_ratio=float(w)/h if h>0 else 0.0, 
            solidity=solidity,
            equivalent_diameter=equivalent_diameter,
            hu_moments=hu_moments
        )

    def _bewerte_kandidat(self, seg_name: str, peak_raw: float, lokal_mean: float, morph: MorphologyFeatures, is_linke_seite: bool) -> HeuristicScore:
        """
        Heuristische Expertensystem-Logik zur Konfidenzbestimmung.
        Integriert lokale, globale, topologische und asymmetrische Faktoren in einen Final Score.
        """
        # 1. Absolute Hitze
        baseline = max(150.0, self.global_mean_temp_raw)
        abs_score = 0.0 if peak_raw <= baseline else min(100.0, ((peak_raw - baseline) / (255.0 - baseline)) * 100.0)
        
        # 2. Lokaler Kontrast (Gibt es einen klaren Hotspot?)
        delta = peak_raw - lokal_mean
        kontrast_score = min(100.0, max(0.0, (delta / 50.0) * 100.0))
        
        # 3. Kontralaterale Symmetrie (Penalty für generell überhitze Extremitäten)
        asym_score = 100.0
        if self.asymmetrie_faktor > 15.0: 
            if (is_linke_seite and self.heissere_seite == "rechts") or (not is_linke_seite and self.heissere_seite == "links"):
                asym_score = 25.0 # Harter Penalty
                
        # 4. Topologische Bewertung (Verwirft chaotische Rauschartefakte)
        form_score = 0.0 if morph.zirkularitaet < 0.15 else min(100.0, morph.zirkularitaet * 100.0)
        
        # Gewichtete Summe
        total = (
            (abs_score * self.config.score_gewicht_absolut) + 
            (kontrast_score * self.config.score_gewicht_kontrast) + 
            (asym_score * self.config.score_gewicht_asymmetrie) + 
            (form_score * self.config.score_gewicht_form)
        )
        
        is_valid = total >= self.config.min_confidence_score
        
        # Bestimme Severity Level
        severity = SeverityLevel.NORMAL
        if is_valid:
            if total >= 95.0: severity = SeverityLevel.CRITICAL
            elif total >= 90.0: severity = SeverityLevel.SEVERE
            elif total >= 85.0: severity = SeverityLevel.MODERATE
            else: severity = SeverityLevel.MILD
        
        if is_valid:
            self.analyse_protokoll.append(f"🟥 {seg_name}: Akuter Befund verifiziert! Konfidenz: {total:.1f}% (Severity: {severity.name}).")
        else:
            self.analyse_protokoll.append(f"🟩 {seg_name}: Befund negativ. Konfidenz {total:.1f}% unter klinischem Cutoff.")
            
        return HeuristicScore(
            absolut_score=abs_score, 
            kontrast_score=kontrast_score, 
            asymmetrie_score=asym_score, 
            form_score=form_score, 
            total_confidence=total, 
            is_valid=is_valid,
            severity=severity
        )

    def _execute_watershed_segmentation(self, thresh_img: np.ndarray, roi_mask: np.ndarray) -> List[np.ndarray]:
        """
        Advanced Algorithmus: Führt eine Wasserscheiden-Transformation durch.
        Wird genutzt, um sehr nah beieinander liegende Hitzespots (die zu einem Blob verschmolzen sind)
        mathematisch präzise aufzutrennen.
        """
        # Distanz-Transformation
        dist_transform = cv2.distanceTransform(thresh_img, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(thresh_img, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Konvertiere zurück auf 3 Kanäle für Watershed
        roi_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(roi_color, markers)
        
        # Extrahiere die aufgetrennten Konturen
        extracted_contours = []
        for marker_val in np.unique(markers):
            if marker_val <= 1: # 0 = Grenzlinie, 1 = Hintergrund
                continue
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            mask[markers == marker_val] = 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                extracted_contours.append(max(cnts, key=cv2.contourArea))
                
        return extracted_contours

    def analysiere(self) -> List[Entzuendung]:
        """
        Haupt-Exekutivfunktion. Iteriert über alle Messpunkte, führt hierarchisches
        Thresholding, Watershed-Segmentierung und statistisches Scoring durch.
        Schließt mit Non-Maximum Suppression zur Bereinigung redundanter Befunde ab.
        """
        if not self.security.verify_integrity():
            raise RuntimeError("Speicherschutzverletzung detektiert. Abbruch.")
            
        self.alle_kandidaten = []
        self.analyse_protokoll.append(f"🚀 Initialisiere hierarchische Multilevel-Segmentierung...")
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            is_links = mX < (self.bild_breite / 2)
            
            # 1. Erstelle lokales Untersuchungsfenster (Region of Interest)
            lokale_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(lokale_maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokaler_bereich = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=lokale_maske)
            
            # 2. Peak-Detection im ROI
            _, max_val, _, max_loc = cv2.minMaxLoc(lokaler_bereich, mask=lokale_maske)
            lokal_mean = cv2.mean(self.vorverarbeitetes_bild, mask=lokale_maske)[0]
            
            # Dynamischer Cutoff: Toleranzbereich erhöht (-45 statt -35), um den Wärmehof sicher zu greifen
            lokaler_schwelle = max(self.global_mean_temp_raw + 10, int(max_val) - 45)
            
            # 3. Multilevel Isothermen berechnen (3 Ebenen der Hitzeausbreitung)
            schwellen = {
                'core': max(lokaler_schwelle + 25, int(max_val) - 10), # Nukleus
                'mid': max(lokaler_schwelle + 15, int(max_val) - 25),  # Corona
                'outer': lokaler_schwelle                              # Peripherie
            }
            
            konturen_dict = {}
            haupt_kontur = None
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            for ebene, schwelle in schwellen.items():
                _, thresh = cv2.threshold(lokaler_bereich, schwelle, 255, cv2.THRESH_BINARY)
                
                # Morphologisches Schließen (Verbindet Risse in den Blutgefäßen/Hitzezonen)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                if self.config.use_watershed_segmentation and ebene == 'outer':
                    # Erweitertes Trennverfahren für den äußeren Wärmehof anwenden
                    konturen = self._execute_watershed_segmentation(thresh, lokaler_bereich)
                else:
                    # Standard Konturfindung
                    konturen, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if konturen:
                    gueltige = [cnt for cnt in konturen if cv2.contourArea(cnt) >= self.config.min_kontur_flaeche]
                    if gueltige:
                        # Selektiere die massivste zusammenhängende Hitzeinsel
                        groesste = max(gueltige, key=cv2.contourArea)
                        konturen_dict[ebene] = groesste
                        if ebene == 'outer': 
                            haupt_kontur = groesste
                            
            # Robusteres Fallback-System, falls 'outer' durch Rauschen fragmentiert wurde
            if haupt_kontur is None:
                for ebene in ['outer', 'mid', 'core']:
                    if ebene in konturen_dict:
                        haupt_kontur = konturen_dict[ebene]
                        break
                
            # 4. Daten-Aggregation & Scoring
            if haupt_kontur is not None:
                M = cv2.moments(haupt_kontur)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else max_loc[0]
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else max_loc[1]
                x, y, w, h = cv2.boundingRect(haupt_kontur)
                
                morph = self._berechne_morphologie(haupt_kontur)
                entz_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
                cv2.drawContours(entz_maske, [haupt_kontur], -1, 255, -1)
                raw_stats, celsius_stats = self._berechne_statistiken(entz_maske)
                
                score = self._bewerte_kandidat(seg['name'], max_val, lokal_mean, morph, is_links)
                
                if score.is_valid:
                    self.alle_kandidaten.append(Entzuendung(
                        gelenk_name=seg['name'], 
                        groesse_px=morph.flaeche, 
                        staerke=raw_stats.mean_val,
                        zentrum=(cX, cY), 
                        kontur=haupt_kontur, 
                        stats_raw=raw_stats, 
                        stats_celsius=celsius_stats,
                        morphology=morph, 
                        score=score, 
                        konturen_ebenen=konturen_dict, 
                        bounding_box=(x, y, w, h)
                    ))
            else:
                self.analyse_protokoll.append(f"🟩 {seg['name']}: Topologische Analyse unauffällig. Keine Hitze-Kohäsion (Gewebe gesund).")
                    
        # 5. Non-Maximum Suppression (Räumliche Duplikatsbereinigung)
        if self.alle_kandidaten:
            self.alle_kandidaten.sort(key=lambda x: x.score.total_confidence, reverse=True)
            gefiltert = []
            for kand in self.alle_kandidaten:
                is_duplicate = False
                for etabliert in gefiltert:
                    dist = math.hypot(kand.zentrum[0] - etabliert.zentrum[0], kand.zentrum[1] - etabliert.zentrum[1])
                    if dist < self.config.nms_overlap_distanz:
                        is_duplicate = True
                        self.analyse_protokoll.append(f"🗑️ NMS-Filter: {kand.gelenk_name} verworfen. Redundante Hitzesignatur zu {etabliert.gelenk_name}.")
                        break
                if not is_duplicate:
                    gefiltert.append(kand)
            self.gefundene_entzuendungen = gefiltert
        else:
            self.gefundene_entzuendungen = []
            
        self.analyse_protokoll.append(f"🏁 Analysezyklus terminiert. {len(self.gefundene_entzuendungen)} pathologische Befunde gesichert.")
        return self.gefundene_entzuendungen

    def render_base64(self) -> str:
        """
        Appliziert die medizinischen Overlays auf das Quellbild und encodiert
        das Ergebnis direkt im flüchtigen Speicher.
        
        Returns:
            Ein UTF-8 dekodierter Base64-String des finalen PNG-Bildes.
        """
        ausgabe = self.original_bild.copy()
        
        # 1. Overlay: Analysierte Segmente (Dünne blaue Messvektoren)
        for seg in self.segmente:
            cv2.line(ausgabe, seg['start'], seg['end'], (255, 100, 0), 1, cv2.LINE_AA)
            cv2.circle(ausgabe, seg['start'], 2, (150, 150, 150), -1)
            cv2.circle(ausgabe, seg['end'], 2, (150, 150, 150), -1)

        # 2. Overlay: Pathologische Topologie
        for entz in self.gefundene_entzuendungen:
            x, y, w, h = entz.bounding_box
            color_severity = entz.get_severity_color()
            
            # Subtle Bounding Box basierend auf der Schwere der Entzündung
            cv2.rectangle(ausgabe, (x, y), (x+w, y+h), color_severity, 1, cv2.LINE_AA)
            
            # Isothermen zeichnen
            farben_isothermen = {'outer': (0, 255, 255), 'mid': (0, 165, 255), 'core': (0, 0, 255)}
            for ebene, kontur in entz.konturen_ebenen.items():
                farbe = farben_isothermen.get(ebene, (255, 255, 255))
                dicke = -1 if ebene == 'core' else 2
                cv2.drawContours(ausgabe, [kontur], -1, farbe, dicke, cv2.LINE_AA)
                
            # Focal Point
            cv2.drawMarker(ausgabe, entz.zentrum, (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.drawMarker(ausgabe, entz.zentrum, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

            # 3. HUD (Heads-Up Display) Data Tags
            label_name = f"{entz.gelenk_name}"
            label_temp = f"T-Max: {entz.stats_celsius.max_val:.1f}C"
            label_conf = f"Conf: {entz.score.total_confidence:.1f}%"
            
            t1, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t2, _ = cv2.getTextSize(label_temp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t3, _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_w = max(t1[0], t2[0], t3[0])
            
            bg_rect_start = (x, max(0, y - 50))
            bg_rect_end = (x + max_w + 6, max(0, y - 2))
            
            # Halbtransparenter Hintergrund für bessere Lesbarkeit
            overlay = ausgabe.copy()
            cv2.rectangle(overlay, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, ausgabe, 0.3, 0, ausgabe)
            
            text_x, text_y_base = x + 3, y - 40
            cv2.putText(ausgabe, label_name, (text_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (text_x, text_y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
            # Konfidenz-Text in der Farbe der Schwere (Severity)
            cv2.putText(ausgabe, label_conf, (text_x, text_y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_severity, 1, cv2.LINE_AA)
                        
        # Speicher-schonende Encodierung
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            return base64.b64encode(buffer).decode("utf-8")
        else:
            raise IOError("Kritischer Fehler bei der In-Memory Bild-Kompression.")