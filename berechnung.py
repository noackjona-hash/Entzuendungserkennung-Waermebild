import cv2
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class Entzuendung:
    """
    Datenstruktur zur Speicherung der Informationen einer gefundenen Entzündung.
    """
    gelenk_name: str
    groesse_px: float       
    staerke: float          
    zentrum: tuple[int, int] 
    kontur: np.ndarray      

class ThermalAnalyzer:
    """
    Klasse zur Analyse von Wärmebildern. Unterstützt Segmente (Start- und Endpunkt).
    """
    
    def __init__(self, bild_pfad: str, segmente: list[dict]):
        self.bild_pfad = bild_pfad
        self.segmente = segmente
        
        # FIX für Pfade mit Umlauten beim LADEN
        try:
            with open(bild_pfad, 'rb') as f:
                bytes_data = f.read()
            array = np.frombuffer(bytes_data, dtype=np.uint8)
            self.original_bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise FileNotFoundError(f"Fehler beim Laden des Bildes: {e}")

        if self.original_bild is None:
            raise FileNotFoundError(f"Bild konnte nicht dekodiert werden. Pfad prüfen: {bild_pfad}")
            
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        self.gefundene_entzuendungen: list[Entzuendung] = []

    def analysiere(self, temperatur_schwellenwert: int = 210, max_distanz: int = 60) -> list[Entzuendung]:
        self.gefundene_entzuendungen = []
        
        _, thresh = cv2.threshold(self.graustufen_bild, temperatur_schwellenwert, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        konturen, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for kontur in konturen:
            if cv2.contourArea(kontur) < 40:
                continue
                
            M = cv2.moments(kontur)
            if M["m00"] == 0: continue
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            for seg in self.segmente:
                s, e = seg['start'], seg['end']
                mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
                
                distanz = math.sqrt((cX - mX)**2 + (cY - mY)**2)
                
                if distanz < max_distanz:
                    maske = np.zeros(self.graustufen_bild.shape, dtype="uint8")
                    cv2.drawContours(maske, [kontur], -1, 255, -1)
                    staerke = cv2.mean(self.graustufen_bild, mask=maske)[0]
                    
                    self.gefundene_entzuendungen.append(Entzuendung(
                        gelenk_name=seg['name'],
                        groesse_px=cv2.contourArea(kontur),
                        staerke=staerke,
                        zentrum=(cX, cY),
                        kontur=kontur
                    ))
                    break
                    
        return self.gefundene_entzuendungen

    def render_output(self, output_pfad: str):
        ausgabe = self.original_bild.copy()
        
        # Zeichne Finger/Zeh-Segmente
        for seg in self.segmente:
            cv2.line(ausgabe, seg['start'], seg['end'], (255, 150, 0), 2)
            cv2.circle(ausgabe, seg['start'], 4, (255, 255, 255), -1)
            cv2.circle(ausgabe, seg['end'], 4, (255, 255, 255), -1)

        # Zeichne Entzündungen
        for entz in self.gefundene_entzuendungen:
            cv2.drawContours(ausgabe, [entz.kontur], -1, (0, 0, 255), 2)
            label = f"{entz.gelenk_name}: S:{int(entz.staerke)}"
            cv2.putText(ausgabe, label, (entz.zentrum[0], entz.zentrum[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        
        # FIX für Pfade mit Umlauten beim SPEICHERN
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            with open(output_pfad, 'wb') as f:
                f.write(buffer)
        else:
            raise IOError("Fehler: Das Ergebnisbild konnte nicht kodiert werden.")