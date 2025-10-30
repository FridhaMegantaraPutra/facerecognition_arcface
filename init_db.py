# init_db.py
from database import db, init_db
from app import app

with app.app_context():
    print(" Inisialisasi database dan tabel...")
    db.create_all()
    print("Semua tabel sudah siap!")
