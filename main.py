import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from modules.loader import load_and_preprocess
from modules.geometry import find_both_feet, extract_toes_from_contour
from modules.analysis import perform_bilateral_analysis

class IgniteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignite - Thermografische Entzuendungserkennung (Jugend Forscht)")
        self.root.geometry("1300x850") # Breiter gemacht fuer die Seitenleiste
        self.root.configure(bg="#2b2b2b")

        self.current_image_path = None
        self.cv_original = None
        self.cv_gray = None
        self.final_cv_image = None
        
        # --- Manuelle Modus Variablen ---
        self.manual_mode = False
        self.manual_clicks = []
        
        # --- UI LAYOUT ---
        # Titel
        title = tk.Label(root, text="Ignite Diagnose-Tool", font=("Arial", 24, "bold"), bg="#2b2b2b", fg="white")
        title.pack(pady=15)

        # Toolbar
        btn_frame = tk.Frame(root, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(btn_frame, text="Waermebild laden", font=("Arial", 12), bg="#0059b3", fg="white", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_analyze = tk.Button(btn_frame, text="Auto-Analyse", font=("Arial", 12), bg="#009933", fg="white", state=tk.DISABLED, command=self.run_analysis)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)
        
        # NEU: Der Manuelle-Auswahl Button
        self.btn_manual = tk.Button(btn_frame, text="Manuelle Auswahl", font=("Arial", 12), bg="#b38f00", fg="white", state=tk.DISABLED, command=self.start_manual_mode)
        self.btn_manual.pack(side=tk.LEFT, padx=10)

        self.btn_pdf = tk.Button(btn_frame, text="Als PDF speichern", font=("Arial", 12), bg="#cc3300", fg="white", state=tk.DISABLED, command=self.save_pdf)
        self.btn_pdf.pack(side=tk.LEFT, padx=10)

        # Hauptbereich mit Bild und Seitenleiste
        main_frame = tk.Frame(root, bg="#2b2b2b")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Canvas fuer die Bildanzeige und Klicks
        self.canvas = tk.Canvas(main_frame, bg="#1e1e1e", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # --- SEITENLEISTE FUER MESSWERTE ---
        sidebar = tk.Frame(main_frame, bg="#333333", width=250, bd=2, relief=tk.SUNKEN)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        tk.Label(sidebar, text="MESSWERTE", font=("Arial", 16, "bold"), bg="#333333", fg="white").pack(pady=10)
        
        # Container fuer die Zehen-Liste
        self.toe_list_frame = tk.Frame(sidebar, bg="#333333")
        self.toe_list_frame.pack(expand=True, fill=tk.BOTH, padx=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.final_cv_image = None
            self.manual_mode = False
            self.clear_toe_list()
            
            # Bild laden und im Speicher behalten!
            self.cv_original, self.cv_gray = load_and_preprocess(file_path)
            
            if self.cv_original is not None:
                self.display_image(self.cv_original)
                
                # Buttons aktivieren
                self.btn_analyze.config(state=tk.NORMAL)
                self.btn_manual.config(state=tk.NORMAL)
                self.btn_pdf.config(state=tk.DISABLED)

    def run_analysis(self):
        if self.cv_original is None: return
        self.clear_toe_list()

        # Wir arbeiten auf einer KOPIE des Originals, damit wir es nicht zerstoeren
        img_copy = self.cv_original.copy()
        
        left_contour, right_contour = find_both_feet(self.cv_gray)
        if left_contour is not None and right_contour is not None:
            left_toes = extract_toes_from_contour(left_contour, self.cv_gray)
            right_toes = extract_toes_from_contour(right_contour, self.cv_gray)
            
            # WICHTIGE PRUEFUNG: Haben wir alle 10 Zehen?
            if len(left_toes) == 5 and len(right_toes) == 5:
                # 1. Analyse durchfuehren und zeichnen
                final_image, delta_ts = perform_bilateral_analysis(img_copy, left_toes, right_toes)
                self.final_cv_image = final_image
                self.display_image(final_image)
                
                # 2. Seitenleiste aktualisieren
                self.update_toe_list(left_toes, right_toes, delta_ts)
                
                self.btn_pdf.config(state=tk.NORMAL)
            else:
                messagebox.showwarning("Analyse-Warnung", f"Die Auto-Erkennung war nicht erfolgreich (Links: {len(left_toes)} Zehen, Rechts: {len(right_toes)} Zehen).\n\nBitte nutze den Button 'Manuelle Auswahl' für eine präzise Diagnose!")
        else:
            messagebox.showerror("Analyse-Fehler", "Konnte keine zwei separaten Fuesse erkennen. Bitte nutze die manuelle Auswahl.")

    # --- MANUELLER MODUS LOGIK ---
    def start_manual_mode(self):
        if self.cv_original is None: return
        self.manual_mode = True
        self.manual_clicks = []
        self.clear_toe_list()
        self.display_image(self.cv_original) # Bild zuruecksetzen
        self.btn_pdf.config(state=tk.DISABLED)
        messagebox.showinfo("Manuelle Diagnose", "Klicke nun nacheinander auf alle 10 Zehen.\n\nFange GANZ LINKS im Bild an und arbeite dich Zeh fuer Zeh nach GANZ RECHTS durch.")

    def on_canvas_click(self, event):
        if not self.manual_mode or self.cv_original is None: return

        # Sicherstellen, dass man nicht ausserhalb des Bildes klickt
        if event.x >= self.cv_original.shape[1] or event.y >= self.cv_original.shape[0]: return

        self.manual_clicks.append((event.x, event.y))
        
        # Visuelles Feedback: Gelber Kreis an der geklickten Stelle
        r = 5
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, outline="yellow", width=2)

        # Wenn 10 Zehen markiert wurden -> Analyse starten!
        if len(self.manual_clicks) == 10:
            self.manual_mode = False
            self.process_manual_clicks()

    def process_manual_clicks(self):
        left_toes = []
        right_toes = []

        for i, (x, y) in enumerate(self.manual_clicks):
            # Wir spannen ein 20x20 Suchfenster um den Klick, um das exakte Temperatur-Zentrum zu finden
            x_start, x_end = max(0, x - 10), min(self.cv_gray.shape[1], x + 10)
            y_start, y_end = max(0, y - 10), min(self.cv_gray.shape[0], y + 10)
            
            roi = self.cv_gray[y_start:y_end, x_start:x_end]
            if roi.size > 0:
                _, max_val, _, max_loc_roi = cv2.minMaxLoc(roi)
                temp = int(max_val)
                meas_pt = (x_start + max_loc_roi[0], y_start + max_loc_roi[1])
            else:
                # Fallback, falls ROI leer ist
                temp = int(self.cv_gray[y, x])
                meas_pt = (x, y)

            data = {"tip": (x, y), "temp": temp, "sensor": meas_pt}
            
            # Die ersten 5 Klicks sind der linke Fuss, die restlichen 5 der rechte
            if i < 5:
                left_toes.append(data)
            else:
                right_toes.append(data)

        # Analyse auf einer sauberen Kopie ausfuehren
        result_img, delta_ts = perform_bilateral_analysis(self.cv_original.copy(), left_toes, right_toes)
        
        self.final_cv_image = result_img
        self.display_image(result_img)
        self.update_toe_list(left_toes, right_toes, delta_ts)
        self.btn_pdf.config(state=tk.NORMAL)
        messagebox.showinfo("Fertig", "Manuelle Analyse erfolgreich abgeschlossen!")

    # --- HILFSFUNKTIONEN FUER DIE SEITENLEISTE ---
    def clear_toe_list(self):
        for widget in self.toe_list_frame.winfo_children():
            widget.destroy()

    def update_toe_list(self, left_toes, right_toes, delta_ts):
        self.clear_toe_list()
        
        tk.Label(self.toe_list_frame, text="Zeh | Temp L | Temp R | Diff", font=("Arial", 10, "bold"), bg="#333333", fg="#aaaaaa").pack(anchor=tk.W, pady=(0, 5))
        
        # Anatomische Beschriftungen (von Klein nach Gross fuer beide Fuesse)
        toe_names = ["Kl. Zeh", "Zeh 4", "Mittelzeh", "Zeh 2", "Gr. Zeh"]
        
        # Da wir fuer den Vergleich den rechten Fuss spiegeln, muessen wir die Anzeige anpassen
        right_toes_display = list(reversed(right_toes))

        for i in range(5):
            temp_l = left_toes[i]["temp"]
            temp_r = right_toes_display[i]["temp"]
            delta_t = delta_ts[i]
            
            # Entzuendungen hervorheben (Delta T > 15)
            color = "#ff4444" if delta_t > 15 else "white"
            
            text = f"{toe_names[i]}: {temp_l:3} | {temp_r:3} | {delta_t:+3}"
            tk.Label(self.toe_list_frame, text=text, font=("Consolas", 11), bg="#333333", fg=color).pack(anchor=tk.W, pady=2)

    # --- RESTLICHE FUNKTIONEN (PDF & ANZEIGE) ---
    def save_pdf(self):
        if self.final_cv_image is None: return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Dokument", "*.pdf")], title="Diagnose-Bericht speichern")
        if save_path:
            # OpenCV (BGR) zu PIL (RGB) konvertieren
            img_rgb = cv2.cvtColor(self.final_cv_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Bild direkt als PDF speichern
            img_pil.save(save_path, "PDF", resolution=100.0)
            messagebox.showinfo("Erfolg", "Der PDF-Bericht wurde erfolgreich gespeichert!")

    def display_image(self, cv_img):
        # OpenCV (BGR) zu PIL (RGB) konvertieren fuer Tkinter
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Canvas bereinigen und Bild anzeigen
        self.canvas.delete("all")
        # Bild oben links (0, 0) im Canvas verankern
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = IgniteApp(root)
    root.mainloop()