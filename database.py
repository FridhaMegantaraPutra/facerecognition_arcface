from flask_sqlalchemy import SQLAlchemy
import numpy as np
import json
import base64
import pickle

db = SQLAlchemy()

class Face(db.Model):
    """Model untuk menyimpan data wajah di database"""
    __tablename__ = 'faces'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    embedding = db.Column(db.LargeBinary, nullable=False)  # Menyimpan embedding sebagai binary data
    
    def __init__(self, name, embedding):
        self.name = name
        # Konversi numpy array ke binary data dengan pickle
        self.embedding = pickle.dumps(embedding)
    
    def get_embedding(self):
        """Mengambil embedding sebagai numpy array"""
        return pickle.loads(self.embedding)

def init_db(app):
    """Inisialisasi database"""
    db.init_app(app)
    
    
