import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from berechnung import ThermalAnalyzer

class ThermalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jugend Forscht 2026 - Entzündungserkennung")
        
        self.bild_pfad = None
        self.segmente = [] # Liste von {'name': str, 'start': (x,y), 'end': (x,y)}
        self.aktueller_start = None
        self.skalierungsfaktor = 1.0
        
        # UI
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(top_frame, text="Bild laden", command=self.lade_bild).pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_analyse = tk.Button(top_frame, text="Analyse starten", command=self.starte_analyse, state=tk.DISABLED)
        self.btn_analyse.pack(side=tk.LEFT, padx=5)
        
        self.info_label = tk.Label(root, text="Lade ein Bild, um zu beginnen.")
        self.info_label.pack()
        
        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.tk_img = None

    def lade_bild(self):
        pfad = filedialog.askopenfilename()
        if not pfad: return
        self.bild_pfad = pfad
        self.segmente = []
        self.aktueller_start = None
        
        img = Image.open(pfad)
        orig_w, orig_h = img.size
        max_size = (1000, 700)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        view_w, view_h = img.size
        
        self.skalierungsfaktor = orig_w / view_w
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.config(width=view_w, height=view_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        self.btn_analyse.config(state=tk.NORMAL)
        self.info_label.config(text="Klicke auf START und dann auf ENDE eines Fingers.")

    def on_click(self, event):
        if not self.bild_pfad: return
        
        # Umrechnen auf Originalgröße
        ox, oy = int(event.x * self.skalierungsfaktor), int(event.y * self.skalierungsfaktor)
        
        if self.aktueller_start is None:
            # Erster Klick (Start)
            self.aktueller_start = (ox, oy)
            self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="yellow", tags="temp")
            self.info_label.config(text="Klicke jetzt auf das ENDE des Fingers.")
        else:
            # Zweiter Klick (Ende)
            segment_name = f"Finger_{len(self.segmente) + 1}"
            self.segmente.append({
                'name': segment_name,
                'start': self.aktueller_start,
                'end': (ox, oy)
            })
            
            # Zeichne dauerhafte Linie auf Canvas
            sx_view = self.aktueller_start[0] / self.skalierungsfaktor
            sy_view = self.aktueller_start[1] / self.skalierungsfaktor
            self.canvas.create_line(sx_view, sy_view, event.x, event.y, fill="cyan", width=2)
            self.canvas.create_text(event.x, event.y+10, text=segment_name, fill="cyan")
            self.canvas.delete("temp")
            
            self.aktueller_start = None
            self.info_label.config(text=f"{segment_name} markiert. Nächster Finger oder Analyse starten.")

    def starte_analyse(self):
        if not self.segmente:
            messagebox.showwarning("Fehler", "Bitte markiere mindestens einen Finger (Start & Ende)!")
            return
            
        try:
            analyzer = ThermalAnalyzer(self.bild_pfad, self.segmente)
            # Höherer Schwellenwert (210), da die Füße im Beispielbild sehr hell sind
            ergebnisse = analyzer.analysiere(temperatur_schwellenwert=210)
            
            output = os.path.join(os.path.dirname(self.bild_pfad), "ergebnis.png")
            analyzer.render_output(output)
            
            msg = f"Gefundene Entzündungen: {len(ergebnisse)}\nGespeichert in: {output}"
            messagebox.showinfo("Fertig", msg)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalApp(root)
    root.mainloop()