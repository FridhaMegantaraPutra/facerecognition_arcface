from flask import Flask, render_template, jsonify, request
import os
from database import init_db, Face, db
from face_utils import FaceRecognition

os.makedirs('templates', exist_ok=True)

if not os.path.exists('templates/index.html') and os.path.exists('index.html'):
    os.system('cp index.html templates/')

app = Flask(__name__)

# Konfigurasi database

import os

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL', 
    'postgresql://postgres:150802@localhost:5432/face_db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimal 16MB untuk upload

# Inisialisasi database
init_db(app)

# Inisialisasi face recognition
face_recognition = FaceRecognition()

# Rute utama
@app.route('/')
def index():
    return render_template('index.html')

# API Endpoint
@app.route('/api/face', methods=['GET'])
def get_faces():
    """Mendapatkan daftar semua wajah di database"""
    try:
        faces = Face.query.all()
        result = []
        
        for face in faces:
            # Ambil embedding dan potong untuk preview
            embedding = face.get_embedding()
            # Ambil 10 nilai pertama untuk preview
            embedding_preview = embedding[:10].tolist()
            # Format dengan 4 angka di belakang koma
            embedding_preview = ["{:.4f}".format(x) for x in embedding_preview]
            
            result.append({
                'id': face.id,
                'name': face.name,
                'embedding_preview': embedding_preview
            })
            
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/face/register', methods=['POST'])
def register_face():
    """Mendaftarkan wajah baru ke database"""
    try:
        if 'name' not in request.form:
            return jsonify({
                'status': 'error',
                'message': 'Nama tidak boleh kosong'
            }), 400
            
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Gambar tidak boleh kosong'
            }), 400
        
        name = request.form['name']
        image_file = request.files['image']
        
        # Proses gambar
        image_data = image_file.read()
        embedding, face_location, process_images = face_recognition.process_image(image_data)
        
        if embedding is None:
            return jsonify({
                'status': 'error',
                'message': face_location  # pesan error
            }), 400
        
        # Simpan ke database
        new_face = Face(name=name, embedding=embedding)
        db.session.add(new_face)
        db.session.commit()
        
        # Ambil preview embedding untuk ditampilkan
        embedding_preview = embedding[:10].tolist()
        embedding_preview = ["{:.4f}".format(x) for x in embedding_preview]
        
        return jsonify({
            'status': 'success',
            'message': 'Wajah berhasil didaftarkan',
            'data': {
                'id': new_face.id, 
                'name': new_face.name,
                'embedding_preview': embedding_preview,
                'process_images': process_images
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/face/recognize', methods=['POST'])
def recognize_face():
    """Mengenali wajah dari gambar"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'Gambar tidak boleh kosong'
            }), 400
        
        image_file = request.files['image']
        
        # Proses gambar
        image_data = image_file.read()
        embedding, face_location, process_images = face_recognition.process_image(image_data)
        
        if embedding is None:
            return jsonify({
                'status': 'error',
                'message': face_location  # pesan error
            }), 400
        
        # Ambil semua embedding dari database
        faces = Face.query.all()
        database_embeddings = [(face.id, face.name, face.get_embedding()) for face in faces]
        
        # Cocokkan dengan database
        match = face_recognition.match_face(embedding, database_embeddings)
        
        # Ambil preview embedding untuk ditampilkan
        embedding_preview = embedding[:10].tolist()
        embedding_preview = ["{:.4f}".format(x) for x in embedding_preview]
        
        if match is None:
            return jsonify({
                'status': 'success',
                'message': 'Wajah tidak dikenali',
                'data': {
                    'embedding_preview': embedding_preview,
                    'process_images': process_images
                }
            }), 200
        
        return jsonify({
            'status': 'success',
            'message': 'Wajah dikenali',
            'data': {
                'id': match['id'],
                'name': match['name'], 
                'confidence': match['confidence'],
                'embedding_preview': embedding_preview,
                'process_images': process_images
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/face/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Menghapus wajah dari database berdasarkan ID"""
    try:
        face = Face.query.get(face_id)
        
        if face is None:
            return jsonify({
                'status': 'error',
                'message': f'Wajah dengan ID {face_id} tidak ditemukan'
            }), 404
        
        db.session.delete(face)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': f'Wajah dengan ID {face_id} berhasil dihapus'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
  
    with app.app_context():
        
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
