import os
import re
import io
import zipfile
import json
import uuid
import time
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import fitz
from PIL import Image
import pytesseract
import requests

app = Flask(__name__)
app.secret_key = 'ganti-dengan-string-acak-untuk-keamanan'

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-coder:480b-cloud"   # ganti sesuai model yang tersedia
DATA_FILE = os.path.join('data', 'reports.json')

if not os.path.exists('data'):
    os.makedirs('data')

temp_store = {}

def cleanup_temp_store():
    now = time.time()
    expired = [k for k, v in temp_store.items() if now - v.get('timestamp', 0) > 1800]
    for k in expired:
        del temp_store[k]

def load_reports():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
        return []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_reports(reports):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)

def save_new_report(report):
    reports = load_reports()
    reports.append(report)
    save_reports(reports)

def update_report(report_id, updated_report):
    reports = load_reports()
    for i, r in enumerate(reports):
        if r['id'] == report_id:
            reports[i] = updated_report
            save_reports(reports)
            return True
    return False

def get_report_by_id(report_id):
    reports = load_reports()
    for r in reports:
        if r['id'] == report_id:
            return r
    return None

def extract_text_from_file(file_storage):
    filename = file_storage.filename.lower()
    file_bytes = file_storage.read()
    file_storage.seek(0)

    if filename.endswith('.pdf'):
        return extract_text_from_pdf(io.BytesIO(file_bytes))
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        try:
            img = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(img, lang='ind+eng')
            if not text.strip():
                text = pytesseract.image_to_string(img, lang='eng')
            return text.strip()
        except Exception as e:
            app.logger.error(f"OCR error: {e}")
            return ""
    else:
        raise ValueError(f"Format tidak didukung: {filename}")

def extract_text_from_pdf(pdf_stream):
    pdf_document = fitz.open(stream=pdf_stream.read(), filetype="pdf")
    full_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        full_text += text + "\n"

    if len(full_text.strip()) < 100:
        full_text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang='ind+eng')
            if not text.strip():
                text = pytesseract.image_to_string(img, lang='eng')
            full_text += text + "\n"
    pdf_document.close()
    return full_text.strip()

def call_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 1024
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get('response', '').strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Gagal terhubung ke Ollama.")
    except Exception as e:
        raise RuntimeError(f"Error Ollama: {str(e)}")

def analyze_report_text(raw_text):
    prompt = (
        "Anda adalah asisten analisis rapor. Di bawah ini adalah teks hasil ekstraksi rapor siswa. "
        "Teks ini MUNGKIN gabungan beberapa file (dipisahkan '===== PEMISAH FILE ====='), yang bisa mewakili semester berbeda.\n\n"
        "TUGAS:\n"
        "1. Identifikasi **NAMA SISWA** yang konsisten.\n"
        "2. Jika terdapat LEBIH DARI SATU SEMESTER, buatlah **TABEL PERBANDINGAN NILAI** untuk setiap mata pelajaran.\n"
        "   - Kolom: Mata Pelajaran, Semester 1, Semester 2, Perubahan (Naik/Turun/Stabil).\n"
        "   - JANGAN sertakan: Pendidikan Pancasila, Pendidikan Jasmani Olahraga dan Kesehatan (PJOK), Prakarya.\n"
        "   - WAJIB sertakan: Pendidikan Agama dan Budi Pekerti.\n"
        "   - Gunakan format markdown tabel.\n"
        "3. Setelah tabel, tulis ANALISIS NARATIF singkat.\n\n"
        "KELUARAN:\n"
        "NAMA: <nama>\n"
        "TABEL:\n"
        "<markdown atau ->\n"
        "ANALISIS:\n"
        "<paragraf>\n\n"
        f"TEKS RAPOR:\n{raw_text[:6000]}"
    )
    response = call_ollama(prompt)
    nama = ""
    tabel_md = ""
    analisis = ""
    nama_match = re.search(r'NAMA\s*:\s*(.+)', response, re.IGNORECASE)
    if nama_match:
        nama = nama_match.group(1).strip()
    else:
        lines = response.split('\n')
        nama = lines[0].strip() if lines else "Nama tidak terdeteksi"

    tabel_match = re.search(r'TABEL\s*:\s*\n(.*?)(?=\nANALISIS\s*:)', response, re.IGNORECASE | re.DOTALL)
    if tabel_match:
        tabel_md = tabel_match.group(1).strip()
        if tabel_md == '-':
            tabel_md = ""

    analisis_match = re.search(r'ANALISIS\s*:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if analisis_match:
        analisis = analisis_match.group(1).strip()
    else:
        analisis = response
    return nama, tabel_md, analisis

def generate_final_description(nama, analisis_nilai, deskripsi_perilaku):
    nama_title = nama.title()
    system_prompt = f"""Anda adalah asisten AI yang menulis narasi laporan perkembangan siswa. 
Tugas Anda: menggabungkan analisis nilai dan deskripsi perilaku menjadi narasi hangat, personal, menyentuh, TANPA judul atau pembukaan yang berlebihan.

ATURAN PENTING:
- JANGAN menulis kalimat "Deskripsi Perkembangan Lengkap" atau judul sejenis.
- JANGAN menggunakan garis pemisah seperti "___" atau "---".
- JANGAN mengakhiri dengan "Hormat kami, Andhivi" – itu akan ditambahkan oleh sistem.
- Mulai langsung dengan paragraf pembuka yang apresiatif. Gunakan NAMA LENGKAP siswa ({nama_title}) di paragraf pertama.
- Selanjutnya, gunakan heading dengan format "###" (misal: ### 🌟 Kemajuan Akademik) untuk membagi bagian.
- Di bawah setiap heading, tulis paragraf naratif yang mengalir.
- Gunakan bahasa Indonesia yang personal dan menyentuh.
- Selalu sebut siswa dengan nama lengkapnya ({nama_title}) di seluruh narasi, jangan menggunakan kata ganti "ia" atau "ananda" secara berlebihan.
- **Jangan gunakan tanda pisah (—) atau em dash. Gunakan kata-kata biasa untuk menghubungkan kalimat, seperti 'yaitu', 'adalah', atau 'yang'.**

Data:
Nama siswa: {nama_title}
Analisis nilai: {analisis_nilai}
Deskripsi perilaku dari guru: {deskripsi_perilaku}

Tuliskan langsung narasi laporannya."""
    response = call_ollama(system_prompt)
    # Bersihkan garis pemisah, tanda "Hormat kami", dll.
    cleaned = response.replace('___', '').replace('---', '')
    cleaned = cleaned.replace('Hormat kami, Andhivi', '')
    # Hapus judul Deskripsi Perkembangan Lengkap jika masih muncul
    cleaned = re.sub(r'^\s*Deskripsi Perkembangan Lengkap\s*', '', cleaned)
    # Hapus em dash jika masih ada (sebagai fallback)
    cleaned = cleaned.replace('—', ' ')
    return cleaned.strip()

def generate_pdf_html(report):
    nama = report['nama']
    nama_title = nama.title()
    tabel_md = report.get('tabel', '')
    deskripsi_raw = report['deskripsi_lengkap']

    tabel_html = ""
    if tabel_md and tabel_md.strip():
        rows = tabel_md.strip().split('\n')
        clean_rows = [row for row in rows if not re.match(r'^\|?\s*[-:]+\s*\|', row)]
        if clean_rows:
            html_rows = []
            for i, row in enumerate(clean_rows):
                cells = [c.strip() for c in row.split('|') if c.strip() != '']
                if i == 0:
                    html_rows.append('<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>')
                else:
                    html_rows.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            tabel_html = '<table class="nilai-tabel">' + ''.join(html_rows) + '</table>'

    lines = deskripsi_raw.split('\n')
    html_parts = []
    current_paragraph = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('###'):
            if current_paragraph:
                html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
                current_paragraph = []
            heading_text = stripped[3:].strip()
            html_parts.append(f"<h3>{heading_text}</h3>")
        elif stripped == '':
            if current_paragraph:
                html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
                current_paragraph = []
        else:
            current_paragraph.append(stripped.lstrip())
    if current_paragraph:
        html_parts.append(f"<p>{' '.join(current_paragraph)}</p>")
    deskripsi_html = '\n'.join(html_parts)

    # Template HTML dengan placeholder, tidak ada f-string kompleks
    html_template = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <title>Laporan __NAMA__</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700&display=swap');
    body {
      font-family: 'Merriweather', serif;
      color: #1a1a1a;
      margin: 2cm;
      line-height: 1.5;
    }
    h3 {
      font-weight: 700;
      font-size: 12pt;
      color: #000;
      margin-top: 1.2em;
      margin-bottom: 0.4em;
    }
    p {
      font-size: 11pt;
      text-align: justify;
      margin-bottom: 0.6em;
      text-indent: 2em;
    }
    .judul {
      text-align: center;
      font-size: 16pt;
      font-weight: 700;
      color: #000;
      margin-bottom: 0.3em;
    }
    .nama {
      text-align: center;
      font-size: 13pt;
      margin-bottom: 1.8em;
      font-weight: 400;
    }
    .ttd {
      text-align: right;
      margin-top: 2em;
      font-style: italic;
      font-size: 10pt;
    }
    .nilai-tabel {
      width: 100%;
      border-collapse: collapse;
      margin: 1em 0 1.5em 0;
      font-size: 10pt;
    }
    .nilai-tabel th {
      background-color: #2c3e50;
      color: white;
      padding: 6px 8px;
      text-align: center;
      border: 1px solid #2c3e50;
    }
    .nilai-tabel td {
      padding: 5px 8px;
      border: 1px solid #ccc;
      text-align: center;
    }
    .nilai-tabel tr:nth-child(even) td {
      background-color: #f9f9f9;
    }
  </style>
</head>
<body>
  <div class="judul">Laporan Perkembangan Siswa</div>
  <div class="nama">__NAMA__</div>
  __TABEL_HTML__
  __DESKRIPSI_HTML__
  <div class="ttd">Hormat kami,<br><br>Andhivi</div>
</body></html>"""

    html = html_template.replace('__NAMA__', nama_title)
    html = html.replace('__TABEL_HTML__', tabel_html)
    html = html.replace('__DESKRIPSI_HTML__', deskripsi_html)
    return html

def pdf_from_report(report):
    from weasyprint import HTML
    html = generate_pdf_html(report)
    pdf_bytes = io.BytesIO()
    HTML(string=html).write_pdf(pdf_bytes)
    pdf_bytes.seek(0)
    return pdf_bytes

# ---------- Routes ----------
@app.route('/')
def index():
    reports = load_reports()
    reports_sorted = sorted(reports, key=lambda x: x['tanggal'], reverse=True)
    return render_template('index.html', reports=reports_sorted)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        cleanup_temp_store()
        return render_template('upload.html')

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return render_template('upload.html', error='Pilih minimal satu file.')

    total_size = sum(len(f.read()) for f in files)
    for f in files: f.seek(0)
    if total_size > 10 * 1024 * 1024:
        return render_template('upload.html', error='Total ukuran maksimal 10 MB.')

    all_text = []
    for f in files:
        if not f.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            return render_template('upload.html', error=f'Format {f.filename} tidak didukung.')
        extracted = extract_text_from_file(f)
        if extracted:
            all_text.append(extracted)
            print(f"✅ {f.filename}: {len(extracted)} karakter")
        else:
            print(f"❌ {f.filename}: 0 karakter")

    if not all_text:
        return render_template('upload.html', error='Tidak ada teks yang bisa diekstrak.')

    combined_text = "\n\n===== PEMISAH FILE =====\n\n".join(all_text)

    if len(combined_text.strip()) < 100:
        return render_template('upload.html', error='Teks terlalu pendek.')

    try:
        nama, tabel_md, analisis = analyze_report_text(combined_text)
        if not nama:
            nama = "Nama tidak terdeteksi"
        # Ubah ke Title Case
        nama = nama.title()

        temp_id = str(uuid.uuid4())[:8]
        temp_store[temp_id] = {
            'nama': nama,
            'tabel': tabel_md,
            'analisis': analisis,
            'debug_text': combined_text,
            'file_count': len(files),
            'timestamp': time.time()
        }
        session['temp_id'] = temp_id

        return render_template('analyze.html', nama=nama, analisis=analisis,
                               debug_text=combined_text, file_count=len(files))
    except Exception as e:
        return render_template('upload.html', error=str(e))

@app.route('/analyze-step2', methods=['POST'])
def analyze_step2():
    temp_id = session.get('temp_id')
    if not temp_id or temp_id not in temp_store:
        return redirect(url_for('upload'))

    temp = temp_store[temp_id]
    nama = temp['nama']  # sudah Title Case
    analisis = temp['analisis']
    tabel_md = temp['tabel']
    deskripsi = request.form.get('deskripsi', '').strip()

    if not deskripsi:
        return render_template('analyze.html', nama=nama, analisis=analisis,
                               error="Deskripsi tidak boleh kosong.")
    if len(deskripsi) > 1000:
        return render_template('analyze.html', nama=nama, analisis=analisis,
                               error="Deskripsi maksimal 1000 karakter.")
    try:
        deskripsi_lengkap = generate_final_description(nama, analisis, deskripsi)
        report = {
            "id": str(uuid.uuid4())[:8],
            "nama": nama,
            "tabel": tabel_md,
            "analisis_nilai": analisis,
            "deskripsi_perilaku": deskripsi,
            "deskripsi_lengkap": deskripsi_lengkap,
            "tanggal": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        save_new_report(report)

        if temp_id in temp_store:
            del temp_store[temp_id]
        session.pop('temp_id', None)

        return redirect(url_for('detail', report_id=report['id']))
    except Exception as e:
        return render_template('analyze.html', nama=nama, analisis=analisis,
                               error=str(e))

@app.route('/report/<report_id>')
def detail(report_id):
    report = get_report_by_id(report_id)
    if not report:
        return "Laporan tidak ditemukan.", 404
    return render_template('detail.html', report=report)

@app.route('/edit-report/<report_id>', methods=['POST'])
def edit_report(report_id):
    report = get_report_by_id(report_id)
    if not report:
        flash("Laporan tidak ditemukan.", "error")
        return redirect(url_for('index'))

    deskripsi_baru = request.form.get('deskripsi', '').strip()
    if not deskripsi_baru:
        flash("Deskripsi tidak boleh kosong.", "error")
        return redirect(url_for('detail', report_id=report_id))
    if len(deskripsi_baru) > 1000:
        flash("Deskripsi maksimal 1000 karakter.", "error")
        return redirect(url_for('detail', report_id=report_id))

    try:
        deskripsi_lengkap_baru = generate_final_description(
            report['nama'],  # sudah Title Case
            report['analisis_nilai'],
            deskripsi_baru
        )
        report['deskripsi_perilaku'] = deskripsi_baru
        report['deskripsi_lengkap'] = deskripsi_lengkap_baru
        report['tanggal'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        update_report(report_id, report)
        flash("Laporan berhasil diperbarui.", "success")
    except Exception as e:
        flash(f"Gagal memperbarui: {str(e)}", "error")

    return redirect(url_for('detail', report_id=report_id))

@app.route('/download/<report_id>')
def download_report(report_id):
    report = get_report_by_id(report_id)
    if not report:
        return "Laporan tidak ditemukan.", 404
    pdf_bytes = pdf_from_report(report)
    return send_file(pdf_bytes, mimetype='application/pdf', as_attachment=True,
                     download_name=f"laporan_{report['nama'].replace(' ', '_')}.pdf")

@app.route('/download-all')
def download_all():
    reports = load_reports()
    if not reports:
        flash('Belum ada laporan.', 'error')
        return redirect(url_for('index'))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for report in reports:
            pdf_bytes = pdf_from_report(report)
            zip_file.writestr(f"laporan_{report['nama'].replace(' ', '_')}.pdf", pdf_bytes.getvalue())
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='semua_laporan.zip')

@app.route('/delete-report/<report_id>', methods=['POST'])
def delete_report(report_id):
    reports = load_reports()
    updated = [r for r in reports if r['id'] != report_id]
    if len(updated) == len(reports):
        flash('Laporan tidak ditemukan.', 'error')
    else:
        save_reports(updated)
        flash('Laporan berhasil dihapus.', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)