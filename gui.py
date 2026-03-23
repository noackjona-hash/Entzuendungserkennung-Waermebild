import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import datetime
from berechnung import ThermalAnalyzer

class ThermalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jugend Forscht 2026 - Entzündungserkennung")
        self.root.geometry("1050x800")
        
        self.bild_pfad = None
        self.segmente = [] # Liste von {'name': str, 'start': (x,y), 'end': (x,y)}
        self.aktueller_start = None
        self.skalierungsfaktor = 1.0
        
        # UI Top Frame
        top_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_laden = tk.Button(top_frame, text="1. Bild laden", command=self.lade_bild, font=("Arial", 10, "bold"))
        self.btn_laden.pack(side=tk.LEFT, padx=10)
        
        self.btn_analyse = tk.Button(top_frame, text="2. Analyse starten", command=self.starte_analyse, state=tk.DISABLED, font=("Arial", 10, "bold"), bg="#4CAF50", fg="white")
        self.btn_analyse.pack(side=tk.LEFT, padx=10)

        self.btn_reset = tk.Button(top_frame, text="Zurücksetzen", command=self.reset_ui, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.RIGHT, padx=10)
        
        self.info_label = tk.Label(root, text="Willkommen Jona! Lade ein Wärmebild, um zu beginnen.", font=("Arial", 12))
        self.info_label.pack(pady=5)
        
        # Canvas für das Bild
        self.canvas = tk.Canvas(root, bg="#222222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_bind_id = self.canvas.bind("<Button-1>", self.on_click)
        
        self.tk_img = None

    def reset_ui(self):
        """Setzt die UI zurück, um ein neues Bild zu bearbeiten."""
        self.bild_pfad = None
        self.segmente = []
        self.aktueller_start = None
        self.canvas.delete("all")
        self.btn_analyse.config(state=tk.DISABLED)
        self.btn_reset.config(state=tk.DISABLED)
        self.info_label.config(text="Zurückgesetzt. Lade ein neues Bild.")
        # Klicks wieder erlauben
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.config(cursor="crosshair")

    def lade_bild(self):
        pfad = filedialog.askopenfilename(filetypes=[("Bilder", "*.png *.jpg *.jpeg *.bmp")])
        if not pfad: return
        
        self.reset_ui() # Vorherige Daten bereinigen
        self.bild_pfad = pfad
        
        img = Image.open(pfad)
        orig_w, orig_h = img.size
        max_size = (1000, 650)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        view_w, view_h = img.size
        
        self.skalierungsfaktor = orig_w / view_w
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(width=view_w, height=view_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        self.btn_analyse.config(state=tk.NORMAL)
        self.btn_reset.config(state=tk.NORMAL)
        self.info_label.config(text="Klicke auf START und dann auf ENDE eines Fingers/Zehs.")

    def on_click(self, event):
        if not self.bild_pfad: return
        
        # Umrechnen auf Originalgröße
        ox, oy = int(event.x * self.skalierungsfaktor), int(event.y * self.skalierungsfaktor)
        
        if self.aktueller_start is None:
            # Erster Klick (Start)
            self.aktueller_start = (ox, oy)
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="yellow", tags="temp")
            self.info_label.config(text="Klicke jetzt auf das ENDE des Fingers/Zehs.")
        else:
            # Zweiter Klick (Ende)
            segment_name = f"Zeh/Finger {len(self.segmente) + 1}"
            self.segmente.append({
                'name': segment_name,
                'start': self.aktueller_start,
                'end': (ox, oy)
            })
            
            # Zeichne dauerhafte Linie auf Canvas
            sx_view = self.aktueller_start[0] / self.skalierungsfaktor
            sy_view = self.aktueller_start[1] / self.skalierungsfaktor
            self.canvas.create_line(sx_view, sy_view, event.x, event.y, fill="#00FFFF", width=2)
            self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="#00FFFF")
            self.canvas.create_text(event.x + 15, event.y, text=segment_name, fill="#00FFFF", anchor=tk.W, font=("Arial", 9, "bold"))
            self.canvas.delete("temp")
            
            self.aktueller_start = None
            self.info_label.config(text=f"{segment_name} markiert. Markiere weitere oder starte die Analyse.")

    def starte_analyse(self):
        if not self.segmente:
            messagebox.showwarning("Fehler", "Bitte markiere mindestens einen Finger/Zeh (Start & Ende)!")
            return
            
        self.info_label.config(text="Berechne Analyse... Bitte warten (max. 30s).")
        self.root.update() # UI aktualisieren, damit der Text angezeigt wird
            
        try:
            analyzer = ThermalAnalyzer(self.bild_pfad, self.segmente)
            ergebnisse = analyzer.analysiere(temperatur_schwellenwert=210)
            
            # Bildausgabe generieren
            ordner = os.path.dirname(self.bild_pfad)
            basisname = os.path.splitext(os.path.basename(self.bild_pfad))[0]
            output_bild_pfad = os.path.join(ordner, f"{basisname}_ergebnis.png")
            bericht_pfad = os.path.join(ordner, f"{basisname}_bericht.html")
            
            analyzer.render_output(output_bild_pfad)
            
            # Ergebnis direkt in der GUI anzeigen
            self.zeige_ergebnis_bild(output_bild_pfad)
            
            # Bericht erstellen
            self.erstelle_html_bericht(ergebnisse, output_bild_pfad, bericht_pfad)
            
            self.info_label.config(text=f"Analyse abgeschlossen! Bericht gespeichert unter: {bericht_pfad}")
            messagebox.showinfo("Fertig", f"{len(ergebnisse)} Entzündungen gefunden.\nBericht und Bild wurden im selben Ordner wie das Originalbild gespeichert.")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Bei der Analyse trat ein Fehler auf:\n{str(e)}")
            self.info_label.config(text="Fehler bei der Analyse.")

    def zeige_ergebnis_bild(self, pfad):
        """Ersetzt das Originalbild im Canvas durch das berechnete Ergebnisbild."""
        img = Image.open(pfad)
        # Gleiche Skalierung wie beim Original anwenden
        max_size = (1000, 650)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # Klicks nach der Analyse deaktivieren, damit man das Bild nicht versehentlich weiter markiert
        self.canvas.unbind("<Button-1>", self.canvas_bind_id)
        self.canvas.config(cursor="arrow")

    def erstelle_html_bericht(self, ergebnisse, bild_pfad, bericht_pfad):
        """Erstellt einen sauberen HTML Bericht mit dem Screenshot und einer Tabelle."""
        now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        bild_dateiname = os.path.basename(bild_pfad)
        
        html = f"""<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>Jugend Forscht 2026 - Analysebericht</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; background-color: #f4f7f6; }}
        .header {{ background-color: #0056b3; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        h1 {{ margin: 0; }}
        .container {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .image-box {{ flex: 1.5; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .data-box {{ flex: 1; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #0056b3; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .highlight {{ color: #d9534f; font-weight: bold; font-size: 1.2em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Thermografie Analysebericht</h1>
        <p><strong>Projekt:</strong> Jugend Forscht 2026 - Entzündungserkennung per Wärmebildkamera</p>
        <p><strong>Autor:</strong> Jona Noack</p>
        <p><strong>Datum der Analyse:</strong> {now}</p>
    </div>
    
    <div class="container">
        <div class="image-box">
            <h2>Ergebnisbild (Screenshot)</h2>
            <img src="{bild_dateiname}" alt="Wärmebild Analyse Ergebnis">
        </div>
        
        <div class="data-box">
            <h2>Diagnosedaten</h2>
"""
        if not ergebnisse:
            html += "<p>Es wurden <strong>keine</strong> signifikanten Entzündungen an den markierten Stellen gefunden.</p>"
        else:
            html += f"<p>Es wurden <span class='highlight'>{len(ergebnisse)} Entzündung(en)</span> detektiert.</p>"
            html += """
            <table>
                <tr>
                    <th>Gelenk / Position</th>
                    <th>Größe (Fläche in px)</th>
                    <th>Durchschn. Temperatur-Intensität</th>
                </tr>
"""
            for e in ergebnisse:
                html += f"""
                <tr>
                    <td><strong>{e.gelenk_name}</strong></td>
                    <td>{int(e.groesse_px)} px²</td>
                    <td>{int(e.staerke)} / 255</td>
                </tr>
"""
            html += "</table>"
            
        html += """
        </div>
    </div>
</body>
</html>
"""
        with open(bericht_pfad, "w", encoding="utf-8") as f:
            f.write(html)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalApp(root)
    root.mainloop()