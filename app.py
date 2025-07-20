from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import pickle
import traceback
import json
from time import perf_counter

# === Setup Flask ===
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Logging ke file ===
log_path = os.path.join(os.path.dirname(__file__), 'log.txt')

# === Load Model dan Scaler ===
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# === Halaman utama ===
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    filepath = None
    try:
        with open(log_path, 'a') as log_file:
            log_file.write('\n===== MASUK ENDPOINT /predict =====\n')

        if 'file' not in request.files:
            return render_template("upload.html", error="Tidak ada file yang dikirim")

        file = request.files['file']
        if not file or not file.filename:
            return render_template("upload.html", error="File tidak valid")

        file_content = file.read()
        if not file_content or len(file_content) < 1000:
            return render_template("upload.html", error="File kosong atau terlalu kecil")

        file.stream.seek(0)
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        with open(log_path, 'a') as log_file:
            log_file.write(f'Nama file: {filename}\n')
            log_file.write(f'Path simpan: {filepath}\n')
            log_file.write(f'Size: {len(file_content)} bytes\n')

        # Mulai stopwatch
        t_start = perf_counter()

        # === PROSES AUDIO: Fallback reader ===
        try:
            import soundfile as sf
            y, sr = sf.read(filepath)
            y = np.array(y, dtype=np.float32)
        except:
            try:
                y, sr = librosa.load(filepath, sr=22050, duration=3.0)
            except Exception as e:
                with open(log_path, 'a') as log_file:
                    log_file.write(f'Error baca audio: {str(e)}\n')
                return render_template("upload.html", error="File audio tidak valid atau rusak.")

        # Ekstraksi MFCC
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
        except Exception as e:
            with open(log_path, 'a') as log_file:
                log_file.write(f'Error ekstraksi MFCC: {str(e)}\n')
            return render_template("upload.html", error="Gagal mengekstrak fitur audio.")

        with open(log_path, 'a') as log_file:
            log_file.write(f'MFCC shape: {mfcc_mean.shape}\n')

        # Prediksi
        mfcc_scaled = scaler.transform(mfcc_mean)
        prediction = model.predict(mfcc_scaled)[0]

        t_end = perf_counter()
        elapsed_ms = (t_end - t_start) * 1000

        with open(log_path, 'a') as log_file:
            log_file.write(f'Prediksi: {prediction}, waktu proses: {elapsed_ms:.2f} ms\n')

        color = "green" if prediction.lower() == "normal" else "red"

        return f"""
        <html>
        <head>
          <title>Hasil Deteksi</title>
          <style>
            body {{
              font-family: 'Segoe UI', sans-serif;
              text-align: center;
              padding-top: 100px;
              background-color: #f9f9f9;
            }}
            .hasil {{
              font-size: 36px;
              font-weight: bold;
              color: {color};
              margin-bottom: 30px;
            }}
            .waktu {{
              font-size: 20px;
              color: #333;
              margin-bottom: 20px;
            }}
            .btn {{
              display: inline-block;
              padding: 12px 24px;
              font-size: 16px;
              font-weight: bold;
              color: white;
              background-color: #007BFF;
              border: none;
              border-radius: 8px;
              text-decoration: none;
            }}
            .btn:hover {{
              background-color: #0056b3;
            }}
          </style>
        </head>
        <body>
          <div class="hasil">Hasil Deteksi: {prediction}</div>
          <div class="waktu">Waktu proses: {elapsed_ms:.2f} ms</div>
          <a href="/" class="btn">🔁 Deteksi Lagi</a>
        </body>
        </html>
        """

    except Exception as e:
        with open(log_path, 'a') as log_file:
            log_file.write('--- ERROR UMUM ---\n')
            log_file.write(traceback.format_exc())
        return "<h4 style='color:red'>Terjadi error saat prediksi. Cek file log.txt</h4>", 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            with open(log_path, 'a') as log_file:
                log_file.write(f'File {filepath} dihapus setelah diproses\n')
