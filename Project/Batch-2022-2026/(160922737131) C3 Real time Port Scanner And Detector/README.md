# REAL TIME PORT SCANNER AND DETECTOR (Flask + SQLite)
##team members 
Mohd Atif Ahmed (160922737131)
Shaik Daniyal (160922737149)
Syed Sameer (160922737176)


## Features
- Windows 10/11 compatible, Python 3.12
- Secure login (Werkzeug password hashing)
- Analyze page: target input OR target file upload
- Real-time scan progress (polling)
- History table (stored in SQLite)
- Dashboard with 2 Chart.js charts + summary cards
- PDF export using ReportLab
- Evidence integrity (blockchain-inspired SHA256 hash chain) stored in `ledger.json`
- Clean error handling (demo-safe)

## Default login
- Username: admin
- Password: admin123
Change it in Settings.

## Run (Developer mode)
1) Create a venv:
   python -m venv .venv
   .venv\Scripts\activate

2) Install dependencies:
   pip install -r requirements.txt

3) Start:
   python app.py

Open: http://127.0.0.1:5000

## Notes
- Alerts are simulated (terminal log + UI toasts). No email/OAuth/etc.
- Scanning is TCP connect based, safe for demo. Use responsibly.
