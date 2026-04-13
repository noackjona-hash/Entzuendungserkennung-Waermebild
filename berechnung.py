import cv2
import numpy as np
import json
import os
import logging
import datetime
from enum import Enum, auto
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Tuple, Optional, Any

# =============================================================================
# ENTERPRISE LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("clinical_engine.log")]
)
logger = logging.getLogger("ThermoAI-Core")

# =============================================================================
# ENTERPRISE DEFINITIONEN & KLASSIFIZIERUNGEN (Mit Pydantic Validierung)
# =============================================================================

class SeverityLevel(Enum):
    """Klinische Klassifizierung der thermischen Anomalie."""
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4

class ThermalConfig(BaseModel):
    """Zentrale Konfiguration für die Computer Vision Engine mit Auto-Validierung."""
    temp_min_celsius: float = Field(20.0, description="Minimale kalibrierte Temperatur")
    temp_max_celsius: float = Field(42.0, description="Maximale kalibrierte Temperatur")
    suchradius_pixel: int = Field(80, ge=10, le=500)
    min_kontur_flaeche: int = Field(15, ge=5)
    score_gewicht_absolut: float = 0.35
    score_gewicht_kontrast: float = 0.25
    score_gewicht_asymmetrie: float = 0.30
    score_gewicht_form: float = 0.10
    min_confidence_score: float = Field(65.0, ge=0.0, le=100.0)

    @validator('temp_max_celsius')
    def validate_temps(cls, v, values):
        if 'temp_min_celsius' in values and v <= values['temp_min_celsius']:
            raise ValueError('T-Max muss strikt größer als T-Min sein.')
        return v

class TemperatureStats(BaseModel):
    min_val: float
    max_val: float
    mean_val: float
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 1), 
            "max": round(self.max_val, 1), 
            "mean": round(self.mean_val, 1)
        }

class Entzuendung:
    """
    Repräsentiert einen pathologischen Befund im Wärmebild.
    Kombiniert rein mathematische Bounding-Boxen mit klinischen Metriken.
    """
    def __init__(
        self, gelenk_name: str, zentrum: Tuple[int, int], konturen_ebenen: Dict[str, np.ndarray],
        bounding_box: Tuple[int, int, int, int], stats_celsius: TemperatureStats, score_total: float
    ):
        self.gelenk_name = gelenk_name
        self.zentrum = zentrum
        self.konturen_ebenen = konturen_ebenen
        self.bounding_box = bounding_box
        self.stats_celsius = stats_celsius
        self.score_total = score_total
        self.severity = SeverityLevel.NORMAL
        self.symmetrie_alarm = False
        self.delta_t_gegenseite = 0.0
        self._calculate_severity()

    def _calculate_severity(self):
        """Weist den klinischen Schweregrad basierend auf dem Konfidenz-Score zu."""
        if self.score_total >= 90: self.severity = SeverityLevel.CRITICAL
        elif self.score_total >= 80: self.severity = SeverityLevel.SEVERE
        elif self.score_total >= 70: self.severity = SeverityLevel.MODERATE
        else: self.severity = SeverityLevel.MILD

# =============================================================================
# KLINISCHES THERAPIE-MONITORING (TREND-MANAGER)
# =============================================================================

class TrendManager:
    """Verwaltet Langzeitdatenbank für das Therapie-Monitoring pro Patient."""
    DATABASE_FILE = "clinical_database.json"

    @staticmethod
    def save_scan(patient_id: str, ergebnisse: List[Entzuendung]) -> List[dict]:
        logger.info(f"Speichere Scan-Historie für Patient: {patient_id}")
        history = TrendManager.load_history()
        
        max_t = round(max([e.stats_celsius.max_val for e in ergebnisse]), 1) if ergebnisse else 0.0
        severity_peak = max([e.severity.value for e in ergebnisse]) if ergebnisse else 0
        
        scan_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "anomalien_count": len(ergebnisse),
            "max_temp": max_t,
            "severity_peak": severity_peak,
            "system_version": "ThermoAI-V16.0"
        }
        
        if patient_id not in history:
            history[patient_id] = []
            
        history[patient_id].append(scan_entry)
        
        try:
            with open(TrendManager.DATABASE_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        except IOError as e:
            logger.error(f"Fehler beim Schreiben der Datenbank: {str(e)}")
            
        return history[patient_id]

    @staticmethod
    def load_history() -> dict:
        if not os.path.exists(TrendManager.DATABASE_FILE):
            return {}
        try:
            with open(TrendManager.DATABASE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Datenbankdatei korrupt. Beginne mit leerer Historie.")
            return {}

# =============================================================================
# CORE VISION ENGINE (THERMAL ANALYZER)
# =============================================================================

class ThermalAnalyzer:
    """
    Hauptklasse zur Analyse von medizinischen Wärmebildern.
    Implementiert Adaptive Morphologie, High-End Denoising und Symmetrie-Prüfung.
    """
    def __init__(self, bild_cv: np.ndarray, messpunkte: List[dict], suchradius: int = 80):
        if bild_cv is None or bild_cv.size == 0:
            raise ValueError("Ungültige Bilddaten an ThermalAnalyzer übergeben.")
            
        self.original_bild = bild_cv
        self.messpunkte = messpunkte
        self.config = ThermalConfig(suchradius_pixel=suchradius)
        self.analyse_protokoll: List[str] = []
        
        self.h, self.w = bild_cv.shape[:2]
        self.gray = cv2.cvtColor(bild_cv, cv2.COLOR_BGR2GRAY)
        
        # High-End Rauschunterdrückung & Kontrastverstärkung
        logger.info("Führe Bilateral-Filtering und CLAHE Optimierung durch...")
        self.denoised = cv2.bilateralFilter(self.gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.processed = clahe.apply(self.denoised)
        
        self.global_mean = float(np.mean(self.processed))
        self.gefundene_befunde: List[Entzuendung] = []

    def _pixel_zu_celsius(self, val: float) -> float:
        """Konvertiert 8-Bit Pixelwerte (0-255) in kalibrierte Celsius-Werte."""
        ratio = val / 255.0
        temp_range = self.config.temp_max_celsius - self.config.temp_min_celsius
        return self.config.temp_min_celsius + (ratio * temp_range)

    def _analyze_point(self, name: str, x: int, y: int) -> Optional[Entzuendung]:
        """Prüft eine Region of Interest (ROI) auf Anomalien und extrahiert Isothermen."""
        # Sicherheits-Checks für Bildgrenzen
        if not (0 <= x < self.w and 0 <= y < self.h):
            logger.warning(f"Messpunkt {name} liegt außerhalb der Bildgrenzen. Wird ignoriert.")
            return None

        mask = np.zeros(self.processed.shape, dtype="uint8")
        cv2.circle(mask, (x, y), self.config.suchradius_pixel, 255, -1)
        roi = cv2.bitwise_and(self.processed, self.processed, mask=mask)
        
        _, max_val, _, _ = cv2.minMaxLoc(roi, mask=mask)
        
        ebenen = {}
        main_cnt = None
        # Multilevel-Segmentierung (Isothermen)
        thresholds = {'core': max_val - 10, 'mid': max_val - 25, 'outer': max_val - 40}
        
        for k, th in thresholds.items():
            if th < 50: continue # Ignoriere zu kalte Bereiche (Hintergrundrauschen)
            _, binarized = cv2.threshold(roi, th, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                best = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(best) >= self.config.min_kontur_flaeche:
                    ebenen[k] = best
                    if k == 'mid': main_cnt = best
        
        if main_cnt is None: 
            return None
        
        m_x, m_y, m_w, m_h = cv2.boundingRect(main_cnt)
        temp_stats = TemperatureStats(
            min_val=self._pixel_zu_celsius(max_val - 40), # Approximation
            max_val=self._pixel_zu_celsius(max_val),
            mean_val=self._pixel_zu_celsius(cv2.mean(self.processed, mask=binarized)[0])
        )
        
        # Heuristische Konfidenzberechnung (Abstand zum globalen Mittelwert)
        score = min(100.0, ((max_val - self.global_mean) / 60.0) * 100.0)
        
        if score < self.config.min_confidence_score:
            logger.debug(f"ROI {name} verworfen: Konfidenz {score:.1f}% zu niedrig.")
            return None

        return Entzuendung(
            gelenk_name=name, zentrum=(x, y), konturen_ebenen=ebenen,
            bounding_box=(m_x, m_y, m_w, m_h), stats_celsius=temp_stats, score_total=score
        )

    def _check_pair_symmetry(self):
        """Klinischer Seitenvergleich: Erkennt Asymmetrie als starken Entzündungsindikator."""
        paare = {}
        for b in self.gefundene_befunde:
            # Erwartet ein Format wie "Linker Fuß - Zeh 1"
            parts = b.gelenk_name.split(" - ")
            if len(parts) > 1:
                key = parts[1] # Gruppiere nach Körperteil (z.B. "Zeh 1")
                if key not in paare: paare[key] = []
                paare[key].append(b)
        
        for key, members in paare.items():
            if len(members) >= 2:
                # Vergleiche die beiden stärksten Befunde dieses Gelenktyps
                members = sorted(members, key=lambda x: x.stats_celsius.max_val, reverse=True)
                t1 = members[0].stats_celsius.max_val
                t2 = members[1].stats_celsius.max_val
                delta = abs(t1 - t2)
                
                if delta >= 0.8: # Ab 0.8°C klinisch relevant
                    msg = f"⚖️ SYMMETRIE-ALARM {key}: Δ {delta:.1f}°C"
                    self.analyse_protokoll.append(msg)
                    logger.warning(msg)
                    
                    for m in members[:2]:
                        m.symmetrie_alarm = True
                        m.delta_t_gegenseite = delta
                        # Erhöhe den Score aufgrund der Asymmetrie
                        m.score_total = min(100.0, m.score_total + (delta * 12))
                        m._calculate_severity() # Schweregrad neu berechnen

    def analysiere(self) -> List[Entzuendung]:
        """Führt den kompletten Scan-Zyklus über alle bereitgestellten Messpunkte durch."""
        baseline_temp = self._pixel_zu_celsius(self.global_mean)
        msg_start = f"🚀 Starte Deep-Scan. (Baseline: {baseline_temp:.1f}°C, {len(self.messpunkte)} Ankerpunkte)"
        self.analyse_protokoll.append(msg_start)
        logger.info(msg_start)
        
        raw_candidates = []
        for mp in self.messpunkte:
            res = self._analyze_point(mp['name'], mp['punkt'][0], mp['punkt'][1])
            if res: 
                raw_candidates.append(res)
            
        self.gefundene_befunde = raw_candidates
        
        if self.gefundene_befunde:
            self._check_pair_symmetry()
            # Nach Konfidenz sortieren
            self.gefundene_befunde.sort(key=lambda x: x.score_total, reverse=True)
            
        msg_end = f"🏁 Analyse beendet. {len(self.gefundene_befunde)} pathologische Muster detektiert."
        self.analyse_protokoll.append(msg_end)
        logger.info(msg_end)
        
        return self.gefundene_befunde