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
        self.root.geometry("1000x850") # Etwas hoeher gemacht fuer den neuen Button
        self.root.configure(bg="#2b2b2b")

        self.current_image_path = None
        self.final_cv_image = None # Speichert das fertige Analyse-Bild fuer den PDF-Export
        
        # --- UI ELEMENTE ---
        title = tk.Label(root, text="Ignite Diagnose-Tool", font=("Arial", 24, "bold"), bg="#2b2b2b", fg="white")
        title.pack(pady=15)

        btn_frame = tk.Frame(root, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(btn_frame, text="Waermebild laden", font=("Arial", 14), bg="#0059b3", fg="white", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.btn_analyze = tk.Button(btn_frame, text="Symmetrie-Analyse starten", font=("Arial", 14), bg="#009933", fg="white", command=self.run_analysis)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)
        
        # NEUER BUTTON: PDF Speichern (standardmaessig deaktiviert, bis Analyse fertig ist)
        self.btn_pdf = tk.Button(btn_frame, text="Als PDF-Bericht speichern", font=("Arial", 14), bg="#cc3300", fg="white", state=tk.DISABLED, command=self.save_pdf)
        self.btn_pdf.pack(side=tk.LEFT, padx=10)

        self.image_label = tk.Label(root, bg="#1e1e1e", text="Bitte ein Bild laden...", fg="gray", font=("Arial", 16))
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Bilder", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.final_cv_image = None # Reset
            self.btn_pdf.config(state=tk.DISABLED) # PDF Button wieder deaktivieren
            
            img, _ = load_and_preprocess(file_path)
            if img is not None:
                self.display_image(img)

    def run_analysis(self):
        if not self.current_image_path:
            messagebox.showwarning("Fehler", "Bitte lade zuerst ein Bild!")
            return

        original, gray = load_and_preprocess(self.current_image_path)
        
        left_contour, right_contour = find_both_feet(gray)
        if left_contour is not None and right_contour is not None:
            left_toes = extract_toes_from_contour(left_contour, gray)
            right_toes = extract_toes_from_contour(right_contour, gray)
            
            final_image = perform_bilateral_analysis(original, left_toes, right_toes)
            
            self.final_cv_image = final_image # Bild merken fuer PDF
            self.display_image(final_image)
            self.btn_pdf.config(state=tk.NORMAL) # PDF Button aktivieren!
        else:
            messagebox.showerror("Analyse-Fehler", "Konnte keine zwei separaten Fuesse erkennen.")

    def save_pdf(self):
        if self.final_cv_image is None:
            return
            
        # Dialog zum Speichern oeffnen
        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf", 
            filetypes=[("PDF Dokument", "*.pdf")], 
            title="Diagnose-Bericht speichern"
        )
        
        if save_path:
            # OpenCV (BGR) zu PIL (RGB) konvertieren
            img_rgb = cv2.cvtColor(self.final_cv_image, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Bild direkt als PDF speichern
            img_pil.save(save_path, "PDF", resolution=100.0)
            messagebox.showinfo("Erfolg", f"Der PDF-Bericht wurde gespeichert unter:\n{save_path}")

    def display_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_label.config(image=self.img_tk, text="")

if __name__ == "__main__":
    root = tk.Tk()
    app = IgniteApp(root)
    root.mainloop()