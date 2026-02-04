# webapp.py
import os
import re
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, abort

# ================= CONFIG =================
SAVE_DIR = r"C:\Users\awsmd\Desktop\Main\saved_detection"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

USERNAME = "admin"
PASSWORD_HASH = hashlib.sha256("password123".encode()).hexdigest()

app = Flask(__name__, template_folder="templates")
app.secret_key = "CHANGE_THIS_SECRET"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= HELPERS =================
def is_image_file(fn):
    return os.path.splitext(fn.lower())[1] in ALLOWED_EXT

def extract_conf_from_filename(filename):
    """
    20251221_164745_c045.jpg -> 0.45
    """
    m = re.search(r'c(\d{2,3})', filename.lower())
    if m:
        return round(int(m.group(1)) / 100.0, 2)
    return 0.0

def build_images_list():
    images = []
    for fn in sorted(os.listdir(SAVE_DIR), reverse=True):
        path = os.path.join(SAVE_DIR, fn)
        if not os.path.isfile(path):
            continue
        if not is_image_file(fn):
            continue

        avg_conf = extract_conf_from_filename(fn)
        det_count = 1 if avg_conf > 0 else 0
        dt = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")

        images.append({
            "filename": fn,
            "avg_conf": avg_conf,
            "det_count": det_count,
            "datetime": dt
        })
    return images

# ================= ROUTES =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u == USERNAME and hashlib.sha256(p.encode()).hexdigest() == PASSWORD_HASH:
            session["user"] = u
            return redirect(url_for("gallery"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def gallery():
    if "user" not in session:
        return redirect(url_for("login"))
    images = build_images_list()
    return render_template("gallery.html", images=images)

@app.route("/image/<path:filename>")
def serve_image(filename):
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(SAVE_DIR, filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
