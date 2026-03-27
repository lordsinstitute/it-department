import tkinter as tk
from tkinter import simpledialog, messagebox
from encrypt_qr import encrypt_message, generate_qr
from scan_decrypt_qr import scan_and_decrypt
from PIL import Image, ImageTk

# ---------- COLORS ----------
BG_COLOR = "#0b1c2d"
CARD_COLOR = "#13293d"
GOLD = "#c9a227"
WHITE = "#ffffff"
LIGHT_TEXT = "#b8c6db"

# ---------- FUNCTIONS ----------

def generate():
    message = entry.get()

    if not message:
        messagebox.showwarning("Warning", "Please enter a confidential message.")
        return

    password = simpledialog.askstring("Password", "Enter Password:", show="*")

    if password:
        encrypted = encrypt_message(message, password)
        generate_qr(encrypted)

        img = Image.open("secure_qr.png")
        img = img.resize((220, 220))
        img_tk = ImageTk.PhotoImage(img)
        qr_label.config(image=img_tk)
        qr_label.image = img_tk

        status_label.config(text="QR Code Generated Successfully", fg=GOLD)


def scan():
    password = simpledialog.askstring("Password", "Enter Password to Decrypt:", show="*")

    if password:
        result = scan_and_decrypt(password)
        status_label.config(text="Decrypted Message: " + result, fg=WHITE)


# ---------- WINDOW ----------
window = tk.Tk()
window.title("Secure QR Code System")
window.geometry("1000x750")
window.configure(bg=BG_COLOR)

# ---------- TOP BANNER ----------
banner = tk.Frame(window, bg=BG_COLOR)
banner.pack(pady=20)

try:
    logo_img = Image.open("Website-Logo-1__1_-removebg-preview.png")
    logo_img = logo_img.resize((500, 160))
    logo_tk = ImageTk.PhotoImage(logo_img)

    logo_label = tk.Label(banner, image=logo_tk, bg=BG_COLOR)
    logo_label.pack()
except:
    print("Logo file not found!")

# ---------- TITLE ----------
title = tk.Label(window,
                 text="Secure QR Code Generator for Encrypted Confidential Messages",
                 font=("Helvetica", 20, "bold"),
                 bg=BG_COLOR,
                 fg=WHITE,
                 wraplength=800,
                 justify="center")
title.pack(pady=10)

subtitle = tk.Label(window,
                    text="Advanced Password-Protected Encryption System",
                    font=("Helvetica", 14),
                    bg=BG_COLOR,
                    fg=GOLD)
subtitle.pack(pady=5)

# ---------- CENTER CARD ----------
card = tk.Frame(window,
                bg=CARD_COLOR,
                width=650,
                height=400)
card.pack(pady=40)

card.pack_propagate(False)

tk.Label(card,
         text="Enter Confidential Message",
         font=("Helvetica", 15),
         bg=CARD_COLOR,
         fg=WHITE).pack(pady=20)

entry = tk.Entry(card,
                 width=55,
                 font=("Helvetica", 13),
                 bd=0,
                 bg="#1f3a52",
                 fg=WHITE,
                 insertbackground=WHITE)
entry.pack(pady=15, ipady=8)

generate_btn = tk.Button(card,
                         text="Generate Secure QR",
                         command=generate,
                         bg=GOLD,
                         fg="black",
                         font=("Helvetica", 12, "bold"),
                         width=28,
                         height=2,
                         bd=0)
generate_btn.pack(pady=15)

scan_btn = tk.Button(card,
                     text="Scan & Decrypt QR",
                     command=scan,
                     bg="#1f3a52",
                     fg=WHITE,
                     font=("Helvetica", 12),
                     width=28,
                     height=2,
                     bd=0)
scan_btn.pack(pady=10)

# ---------- QR DISPLAY ----------
qr_label = tk.Label(window, bg=BG_COLOR)
qr_label.pack(pady=20)

# ---------- STATUS ----------
status_label = tk.Label(window,
                        text="",
                        font=("Helvetica", 12),
                        bg=BG_COLOR,
                        fg=LIGHT_TEXT)
status_label.pack(pady=10)

# ---------- FOOTER ----------
footer = tk.Label(window,
                  text="LORDS Institute of Engineering & Technology | Major Project 2026",
                  font=("Helvetica", 10),
                  bg=BG_COLOR,
                  fg="#6c7a89")
footer.pack(side="bottom", pady=20)

window.mainloop()