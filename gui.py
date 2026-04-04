import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import datetime
from berechnung import ThermalAnalyzer

class ThermalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JF 2026 - Entzündungserkennung (Lokale Diagnose)")
        self.root.geometry("1050x800")
        
        self.bild_pfad = None
        self.segmente = [] 
        self.skalierungsfaktor = 1.0
        
        # UI Top Frame
        top_frame = tk.Frame(root, bg="#2b2b2b", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_laden = tk.Button(top_frame, text="1. Bild laden", command=self.lade_bild, font=("Arial", 10, "bold"))
        self.btn_laden.pack(side=tk.LEFT, padx=10)
        
        self.btn_analyse = tk.Button(top_frame, text="2. Manuelle Analyse starten", command=self.starte_analyse, state=tk.DISABLED, font=("Arial", 10, "bold"), bg="#06b6d4", fg="white")
        self.btn_analyse.pack(side=tk.LEFT, padx=10)

        self.btn_reset = tk.Button(top_frame, text="Punkte löschen", command=self.reset_points, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.RIGHT, padx=10)
        
        self.info_label = tk.Label(root, text="Willkommen Jona! Lade ein Wärmebild für die lokale Offline-Auswertung.", font=("Arial", 11), fg="#333")
        self.info_label.pack(pady=5)
        
        self.canvas = tk.Canvas(root, bg="#1e1e1e", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_bind_id = self.canvas.bind("<Button-1>", self.on_click)
        
        self.tk_img = None

    def reset_points(self):
        self.segmente = []
        self.btn_analyse.config(state=tk.DISABLED)
        self.info_label.config(text="Punkte gelöscht. Klicke auf Gelenke, um sie zu markieren.")
        self.zeichne_bild_neu()

    def lade_bild(self):
        pfad = filedialog.askopenfilename(filetypes=[("Bilder", "*.png *.jpg *.jpeg *.bmp")])
        if not pfad: return
        
        self.bild_pfad = pfad
        self.segmente = []
        self.btn_reset.config(state=tk.NORMAL)
        self.info_label.config(text="Klicke direkt auf die Gelenke (Hände, Knie, etc.) im Bild.")
        
        # Bild skalieren für Canvas
        self.original_img = Image.open(pfad)
        orig_w, orig_h = self.original_img.size
        
        self.skalierungsfaktor = orig_w / 1000.0 if orig_w > 1000 else 1.0
        view_w, view_h = int(orig_w / self.skalierungsfaktor), int(orig_h / self.skalierungsfaktor)
        
        self.view_img = self.original_img.resize((view_w, view_h), Image.Resampling.LANCZOS)
        self.zeichne_bild_neu()

    def zeichne_bild_neu(self):
        if not self.bild_pfad: return
        self.tk_img = ImageTk.PhotoImage(self.view_img)
        self.canvas.delete("all")
        self.canvas.config(width=self.view_img.width, height=self.view_img.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        for i, pt in enumerate(self.segmente):
            vx, vy = pt['punkt'][0] / self.skalierungsfaktor, pt['punkt'][1] / self.skalierungsfaktor
            self.canvas.create_oval(vx-5, vy-5, vx+5, vy+5, outline="#06b6d4", width=2)
            self.canvas.create_text(vx+15, vy, text=pt['name'], fill="#06b6d4", anchor=tk.W, font=("Arial", 10, "bold"))
            
        if self.segmente: self.btn_analyse.config(state=tk.NORMAL)

    def on_click(self, event):
        if not self.bild_pfad: return
        ox, oy = int(event.x * self.skalierungsfaktor), int(event.y * self.skalierungsfaktor)
        
        self.segmente.append({
            'name': f"Messpunkt {len(self.segmente) + 1}",
            'punkt': (ox, oy)
        })
        self.zeichne_bild_neu()

    def starte_analyse(self):
        if not self.segmente: return
        self.info_label.config(text="Lokal berechnen... (Deep Scan läuft)")
        self.root.update()
            
        try:
            # Nutzt kleinen Radius für manuelle Punkte
            analyzer = ThermalAnalyzer(bild_pfad=self.bild_pfad, messpunkte=self.segmente, suchradius=40)
            ergebnisse = analyzer.analysiere()
            
            ordner = os.path.dirname(self.bild_pfad)
            basisname = os.path.splitext(os.path.basename(self.bild_pfad))[0]
            output_bild_pfad = os.path.join(ordner, f"{basisname}_ergebnis.png")
            bericht_pfad = os.path.join(ordner, f"{basisname}_bericht.html")
            
            # Nutzt die neue Methode aus berechnung.py
            analyzer.render_image_to_file(output_bild_pfad)
            
            self.zeige_ergebnis_bild(output_bild_pfad)
            self.erstelle_html_bericht(ergebnisse, output_bild_pfad, bericht_pfad)
            
            self.info_label.config(text=f"Fertig! Befunde: {len(ergebnisse)} | Bericht gespeichert unter: {bericht_pfad}")
            messagebox.showinfo("Erfolg", f"Bericht & Overlay-Bild wurden erfolgreich im Ordner des Originals gespeichert.")
            
        except Exception as e:
            messagebox.showerror("Fehler", str(e))
            self.info_label.config(text="Fehler bei der Analyse.")

    def zeige_ergebnis_bild(self, pfad):
        self.original_img = Image.open(pfad)
        orig_w, orig_h = self.original_img.size
        view_w, view_h = int(orig_w / self.skalierungsfaktor), int(orig_h / self.skalierungsfaktor)
        self.view_img = self.original_img.resize((view_w, view_h), Image.Resampling.LANCZOS)
        self.zeichne_bild_neu()
        
    def erstelle_html_bericht(self, ergebnisse, bild_pfad, bericht_pfad):
        now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        bild_dateiname = os.path.basename(bild_pfad)
        
        html = f"""<!DOCTYPE html><html lang="de"><head><meta charset="UTF-8"><title>Diagnosebericht</title>
        <style>body{{font-family:sans-serif;margin:40px;background:#f4f7f6}} .box{{background:#fff;padding:20px;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.1);margin-bottom:20px;}} img{{max-width:100%;border-radius:4px;}} table{{width:100%;border-collapse:collapse;}} th,td{{border:1px solid #ddd;padding:10px;text-align:left;}} th{{background:#0056b3;color:#fff}}</style>
        </head><body>
        <div class="box" style="background:#0056b3;color:white;"><h1>Offline Analysebericht</h1><p>Projekt: Jugend Forscht 2026 | Forscher: Jona Noack | Datum: {now}</p></div>
        <div class="box"><img src="{bild_dateiname}" alt="Ergebnis"></div>
        <div class="box"><h2>Befunde</h2>"""
        
        if not ergebnisse: html += "<p>Keine pathologischen Hitzemuster gefunden.</p>"
        else:
            html += f"<p>Es wurden <b style='color:red;'>{len(ergebnisse)} Anomalien</b> detektiert.</p><table><tr><th>Gelenk</th><th>Temp. Max</th><th>Fläche</th><th>Konfidenz</th></tr>"
            for e in ergebnisse:
                html += f"<tr><td>{e.gelenk_name}</td><td>{e.stats_celsius.max_val:.1f} °C</td><td>{int(e.morphology.flaeche)} px²</td><td>{e.score.total_confidence:.1f}%</td></tr>"
            html += "</table>"
            
        html += "</div></body></html>"
        with open(bericht_pfad, "w", encoding="utf-8") as f: f.write(html)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalApp(root)
    root.mainloop()