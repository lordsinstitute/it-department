import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash
import pdfplumber
import openai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_CONTENT_LENGTH = 12 * 1024 * 1024  # 12 MB
ALLOWED_EXTENSIONS = {'pdf'}

if not OPENAI_API_KEY:
    raise RuntimeError("Please set your OPENAI_API_KEY in the environment or .env file")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "replace-me")
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(path):
    """Extract plain text from a PDF file using pdfplumber."""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts).strip()


def build_openai_prompt(report_text, include_constraints=True):
    base_prompt = """
You are a professional AI medical assistant that analyzes medical reports and provides
clear, actionable insights for both clinicians and patients.

Analyze the following medical report and produce the following sections:

**1. Summary:**  
A brief 3–4 sentence overview of what the report indicates (condition type, test purpose, main findings).

**2. Health Insights:**  
List and explain any abnormalities, patterns, or notable findings. Highlight possible risk factors,
body systems affected, or early warning indicators. (Avoid diagnosis statements—focus on data-driven insights.)

**3. Clinical Recommendations:**  
Write bullet points aimed at clinicians about possible next diagnostic or management steps,
and data that might warrant closer monitoring.

**4. Lifestyle Modifications (Patient-Friendly):**  
Suggest practical habits, diet, exercise, sleep, and stress management recommendations tailored to the findings.

**5. Next Steps / Follow-Up:**  
Briefly list what additional tests or specialist consultations may be useful.
"""
    constraints = """
**Constraints:**  
- Keep tone supportive and educational, not diagnostic.  
- Highlight missing information if the report seems incomplete.  
- Output should use clear section headings.
""" if include_constraints else ""

    prompt = f"{base_prompt}\n{constraints}\n\nMedical Report Text:\n{report_text.strip()}"
    return prompt.strip()


def call_openai_chat(prompt, model=OPENAI_MODEL, max_tokens=900):
    """Call the OpenAI ChatCompletion API."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful medical analyst that reads reports and provides insights and recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'report' not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        file = request.files['report']
        if file.filename == '':
            flash("No file selected.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only PDF files are supported.", "warning")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(tmp_path)

        try:
            extracted_text = extract_text_from_pdf(tmp_path)
            if not extracted_text:
                flash("Unable to extract text. Try an OCR PDF.", "danger")
                return redirect(request.url)

            prompt = build_openai_prompt(extracted_text)
            ai_output = call_openai_chat(prompt)

            result = {
                "filename": filename,
                "ai_output": ai_output,
                "extracted_excerpt": (extracted_text[:3000] + "...") if len(extracted_text) > 3000 else extracted_text
            }

            os.remove(tmp_path)
            return render_template("result.html", result=result)

        except Exception as e:
            flash(f"Error processing report: {e}", "danger")
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return redirect(request.url)

    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(e):
    flash("File too large. Maximum size is 12 MB.", "danger")
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
