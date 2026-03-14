import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os

from modules.loader import load_and_preprocess
from modules.geometry import find_both_feet, extract_toes_from_contour
from modules.analysis import perform_bilateral_analysis

class IgniteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignite - Thermografische Entzündungserkennung")
        self.root.geometry("1200x900")
        self.root.configure(bg="#1e1e1e")

        self.current_image_path = None
        self.cv_original = None
        self.cv_gray = None
        self.final_cv_image = None
        
        self.manual_mode = False
        self.manual_clicks = []
        
        # --- UI LAYOUT ---
        # Header
        header = tk.Frame(root, bg="#2b2b2b", height=80)
        header.pack(fill=tk.X)
        title = tk.Label(header, text="IGNITE DIAGNOSE-SYSTEM", font=("Segoe UI", 24, "bold"), bg="#2b2b2b", fg="#00ccff")
        title.pack(pady=10)

        # Toolbar
        toolbar = tk.Frame(root, bg="#333333", pady=10)
        toolbar.pack(fill=tk.X)

        self.btn_load = tk.Button(toolbar, text="📁 Bild laden", font=("Arial", 11), bg="#444444", fg="white", command=self.load_image, width=15)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_analyze = tk.Button(toolbar, text="⚡ Auto-Analyse", font=("Arial", 11), bg="#007acc", fg="white", state=tk.DISABLED, command=self.run_analysis, width=15)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)
        
        self.btn_manual = tk.Button(toolbar, text="🖱️ Manuell wählen", font=("Arial", 11), bg="#d4a017", fg="black", state=tk.DISABLED, command=self.start_manual_mode, width=15)
        self.btn_manual.pack(side=tk.LEFT, padx=10)

        self.btn_pdf = tk.Button(toolbar, text="📄 PDF Bericht", font=("Arial", 11), bg="#b32400", fg="white", state=tk.DISABLED, command=self.save_pdf, width=15)
        self.btn_pdf.pack(side=tk.RIGHT, padx=10)

        # Hauptbereich (Canvas & Tabelle)
        main_container = tk.Frame(root, bg="#1e1e1e")
        main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        self.canvas = tk.Canvas(main_container, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Statusleiste / Info-Box
        self.status_var = tk.StringVar(value="System bereit. Bitte Wärmebild laden.")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#2b2b2b", fg="#aaaaaa")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Thermal Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.cv_original, self.cv_gray = load_and_preprocess(file_path)
            
            if self.cv_original is not None:
                self.display_image(self.cv_original)
                self.btn_analyze.config(state=tk.NORMAL)
                self.btn_manual.config(state=tk.NORMAL)
                self.btn_pdf.config(state=tk.DISABLED)
                self.status_var.set(f"Bild geladen: {os.path.basename(file_path)}")

    def run_analysis(self):
        if self.cv_original is None: return
        img_copy = self.cv_original.copy()
        
        left_contour, right_contour = find_both_feet(self.cv_gray)
        if left_contour is not None and right_contour is not None:
            left_toes = extract_toes_from_contour(left_contour, self.cv_gray)
            right_toes = extract_toes_from_contour(right_contour, self.cv_gray)
            
            if len(left_toes) == 5 and len(right_toes) == 5:
                final_image = perform_bilateral_analysis(img_copy, left_toes, right_toes, self.cv_gray)
                self.final_cv_image = final_image
                self.display_image(final_image)
                self.btn_pdf.config(state=tk.NORMAL)
                self.status_var.set("Auto-Analyse abgeschlossen.")
            else:
                messagebox.showwarning("Hinweis", "Auto-Erkennung unvollständig. Bitte manuelle Auswahl nutzen.")
        else:
            messagebox.showerror("Fehler", "Füße konnten nicht getrennt werden.")

    def start_manual_mode(self):
        if self.cv_original is None: return
        self.manual_mode = True
        self.manual_clicks = []
        self.display_image(self.cv_original)
        self.status_var.set("Manueller Modus: Klicke nacheinander auf alle 10 Zehen (von links nach rechts).")

    def on_canvas_click(self, event):
        if not self.manual_mode or self.cv_original is None: return
        
        self.manual_clicks.append((event.x, event.y))
        self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="#ffcc00", outline="white")
        
        if len(self.manual_clicks) == 10:
            self.manual_mode = False
            self.process_manual_clicks()

    def process_manual_clicks(self):
        left_toes, right_toes = [], []
        for i, (x, y) in enumerate(self.manual_clicks):
            x_s, x_e = max(0, x-10), min(self.cv_gray.shape[1], x+10)
            y_s, y_e = max(0, y-10), min(self.cv_gray.shape[0], y+10)
            roi = self.cv_gray[y_s:y_e, x_s:x_e]
            
            if roi.size > 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(roi)
                temp, meas_pt = int(max_val), (x_s + max_loc[0], y_s + max_loc[1])
            else:
                temp, meas_pt = int(self.cv_gray[y, x]), (x, y)

            data = {"tip": (x, y), "temp": temp, "sensor": meas_pt}
            if i < 5: left_toes.append(data)
            else: right_toes.append(data)

        result_img = perform_bilateral_analysis(self.cv_original.copy(), left_toes, right_toes, self.cv_gray)
        self.final_cv_image = result_img
        self.display_image(result_img)
        self.btn_pdf.config(state=tk.NORMAL)
        self.status_var.set("Manuelle Analyse abgeschlossen.")

    def save_pdf(self):
        if self.final_cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            img_rgb = cv2.cvtColor(self.final_cv_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(path, "PDF", resolution=100.0)
            messagebox.showinfo("Erfolg", "Bericht wurde gespeichert.")

    def display_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = IgniteApp(root)
    root.mainloop()