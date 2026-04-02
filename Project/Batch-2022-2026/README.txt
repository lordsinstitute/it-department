Blockchain-Integrated AI for Secure Data Sharing and Access
==========================================================

Description:
------------
This project demonstrates a secure data analysis platform that integrates:
- Machine Learning-based risk classification
- Blockchain-inspired hash integrity ledger
- Secure authentication
- Enterprise-style dashboard and reporting

Key Features:
-------------
- Offline ML model (TF-IDF + Logistic Regression)
- Secure login (password hashing)
- Risk scoring (Low / Medium / High / Critical)
- Blockchain-style SHA256 evidence ledger
- PDF report generation
- Advanced dashboard with Chart.js
- SQLite auto-created database
- PyInstaller EXE compatible

System Requirements:
--------------------
- Windows 10 / 11
- Python 3.12
- No internet required at runtime
- No external APIs
- No Docker / GPU

How to Run:
-----------
1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. python app.py
5. Open browser: http://127.0.0.1:5000

EXE Packaging:
--------------
pyinstaller --onefile ^
 --add-data "templates;templates" ^
 --add-data "static;static" ^
 --add-data "database;database" ^
 --add-data "uploads;uploads" ^
 --add-data "ml;ml" ^
 app.py

Educational Use:
----------------
Ideal for:
- Final-year engineering projects
- Cybersecurity demonstrations
- Blockchain + AI integration studies
- Secure system architecture learning

License:
--------
Academic / Educational use only