import cv2
import numpy as np
import onnxruntime as ort
import os
import base64
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognition:
    """Kelas untuk pengenalan wajah menggunakan ArcFace model"""
    
    def __init__(self):
        # Inisialisasi model deteksi wajah
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Gunakan model ArcFace yang sudah ada
        model_path = 'models/arc.onnx'
        
        # Pastikan direktori models ada
        os.makedirs('models', exist_ok=True)
        
        # Jika file model tidak ada di direktori models, coba cari di root
        if not os.path.exists(model_path) and os.path.exists('arc.onnx'):
            # Salin file model dari root ke direktori models
            os.system('cp arc.onnx models/')
        
        # Inisialisasi model ArcFace
        self.arcface_session = ort.InferenceSession(model_path)
        
        # Threshold untuk pencocokan wajah
        self.similarity_threshold = 0.2
    
    def detect_faces(self, image):
        """Deteksi wajah dalam gambar menggunakan OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def preprocess_face(self, image, face_location):
        """Mempersiapkan wajah untuk input ArcFace"""
        x, y, w, h = face_location
        
        # Potong wajah
        face = image[y:y+h, x:x+w]
        
        # Resize ke format input model
        face = cv2.resize(face, (112, 112))
        
        # Konversi ke RGB
        if len(face.shape) == 2:  # grayscale
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        else:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
        # Normalisasi
        face = face.astype(np.float32) / 255.0
        
        # Reshape ke format input model
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def extract_features(self, preprocessed_face):
        """Ekstrak fitur dari wajah menggunakan ArcFace"""
        inputs = {self.arcface_session.get_inputs()[0].name: preprocessed_face}
        outputs = self.arcface_session.run(None, inputs)
        
        # Ambil embedding
        embedding = outputs[0].squeeze()
        
        # Normalisasi embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def match_face(self, embedding, database_embeddings):
        """Mencocokkan embedding wajah dengan database"""
        best_match = None
        best_similarity = -1
        
        for face_id, name, db_embedding in database_embeddings:
            # Hitung cosine similarity
            similarity = cosine_similarity(embedding.reshape(1, -1), db_embedding.reshape(1, -1))[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (face_id, name, similarity)
        
        # Jika similarity melebihi threshold, kembalikan match
        if best_match is not None and best_match[2] >= self.similarity_threshold:
            return {'id': best_match[0], 'name': best_match[1], 'confidence': float(best_match[2])}
        
        return None
    
    def process_image(self, image_data):
        """Proses gambar untuk deteksi dan ekstraksi fitur"""
        # Konversi image_data ke numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Simpan gambar asli untuk proses visualisasi
        original_image = image.copy()
        
        # Deteksi wajah
        faces = self.detect_faces(image)
        
        # Jika tidak ada wajah terdeteksi
        if len(faces) == 0:
            return None, "Tidak ada wajah terdeteksi", None
            
        # Ambil wajah pertama
        face_location = faces[0]
        x, y, w, h = face_location
        
        # Gambar kotak pada wajah yang terdeteksi
        detection_image = original_image.copy()
        cv2.rectangle(detection_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Potong wajah untuk visualisasi
        cropped_face = original_image[y:y+h, x:x+w]
        
        # Praproses wajah
        preprocessed_face = self.preprocess_face(image, face_location)
        
        # Resize preprocessed_face untuk visualisasi
        vis_preprocessed = cv2.resize(preprocessed_face[0], (112, 112))
        vis_preprocessed = (vis_preprocessed * 255).astype(np.uint8)
        
        # Ekstrak fitur
        embedding = self.extract_features(preprocessed_face)
        
        # Buat visualisasi proses
        process_images = {
            'original': original_image,
            'detection': detection_image,
            'cropped': cropped_face,
            'preprocessed': vis_preprocessed
        }
        
        # Encode gambar proses sebagai base64 untuk ditampilkan di web
        process_images_encoded = {}
        for name, img in process_images.items():
            # Konversi ke RGB jika perlu
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Encode ke JPEG
            _, buffer = cv2.imencode('.jpg', img)
            img_str = base64.b64encode(buffer).decode('utf-8')
            process_images_encoded[name] = img_str
        
        return embedding, face_location, process_images_encoded