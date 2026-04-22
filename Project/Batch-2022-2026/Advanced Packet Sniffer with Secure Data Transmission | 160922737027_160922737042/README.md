# Packet Sniffer (Student-Friendly, Enterprise-Looking)

This is a Flask web app that performs **packet-log style analysis** (text or file upload), generates **risk scoring**, keeps a **history**, renders a **dashboard** with Chart.js, exports **PDF reports**, and maintains an **evidence integrity ledger** (blockchain-inspired SHA256 hash chain).

## Why "Packet Sniffer" without drivers?
Raw packet capture on Windows usually needs admin + Npcap/WinPcap drivers.  
This project instead focuses on **analysis and reporting**, which is demo-friendly and works on any Windows 10/11 machine with Python 3.12.

## Features
- Secure Login (Werkzeug hashing)
- Analyze page: text or file upload
- Risk scoring: Low/Medium/High/Critical
- Dashboard: 2 charts + summary cards
- History page
- PDF export (ReportLab)
- Evidence integrity ledger (SHA256 hash chain in JSON)
- Robust error handling (safe demo experience)

## Run
See the step-by-step instructions in the main response.