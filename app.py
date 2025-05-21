from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import json
from bson.objectid import ObjectId
import logging
from PIL import Image
import hashlib

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/plantdoc_db"
mongo = PyMongo(app)

# Initialize collections
diseases_col = mongo.db.diseases
diagnoses_col = mongo.db.diagnoses

def initialize_disease_database():
    """Initialize disease database with PlantDoc dataset information"""
    if diseases_col.count_documents({}) == 0:
        # Load PlantDoc dataset information
        plantdoc_diseases = [
            {
                "disease_id": "apple_scab",
                "scientific_name": "Apple___Apple_scab",
                "common_name": "Apple Scab",
                "treatment": """1. Apply sulfur sprays every 7-10 days\n2. Remove infected leaves\n3. Adjust pH to 6.0-6.5""",
                "prevention": """1. Plant resistant varieties\n2. Disinfect tools\n3. Avoid overhead watering""",
                "severity": "Moderate",
                "image_hashes": []  # Will store hashes of known images
            },
            {
                "disease_id": "pepper_bacterial_spot",
                "scientific_name": "Pepper__bell___Bacterial_spot",
                "common_name": "Bell Pepper Bacterial Spot",
                "treatment": """1. Apply copper-based bactericides\n2. Remove infected plants\n3. Avoid overhead irrigation""",
                "prevention": """1. Use disease-free seeds\n2. Practice crop rotation\n3. Maintain proper spacing""",
                "severity": "High",
                "image_hashes": []
            },
            # Add all other PlantDoc diseases here...
            {
                "disease_id": "healthy",
                "scientific_name": "Pepper__bell___healthy",
                "common_name": "Healthy Bell Pepper",
                "treatment": "No treatment needed",
                "prevention": """1. Maintain proper nutrition\n2. Monitor for pests\n3. Regular plant inspections""",
                "severity": "None",
                "image_hashes": []
            }
        ]
        diseases_col.insert_many(plantdoc_diseases)
        logger.info("Initialized disease database with PlantDoc data")

def calculate_image_hash(filepath):
    """Calculate MD5 hash of an image file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_stream):
    try:
        img = Image.open(file_stream)
        img.verify()
        file_stream.seek(0)
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {e}")
        return False

def preprocess_image(img_path, target_size=(256, 256)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def load_plant_model():
    """Load model from SavedModel or .keras format"""
    try:
        # Try SavedModel format (directory)
        if os.path.exists('models/plantdoc_model'):
            model = tf.keras.models.load_model('models/plantdoc_model')
            logger.info("Model loaded successfully from SavedModel format")
            return model
        
        # Try .keras format
        if os.path.exists('models/plantdoc_model.keras'):
            model = tf.keras.models.load_model('models/plantdoc_model.keras')
            logger.info("Model loaded successfully from .keras format")
            return model
        
        logger.error("No model files found in models/ directory")
        return None
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None

# Initialize disease database and load model
initialize_disease_database()
model = load_plant_model()

try:
    with open('models/class_indices.json') as f:
        class_indices = json.load(f)
    CLASS_NAMES = list(class_indices.keys())
    logger.info(f"Loaded {len(CLASS_NAMES)} class names")
except Exception as e:
    logger.error(f"Error loading class indices: {e}")
    CLASS_NAMES = [disease['scientific_name'] for disease in diseases_col.find()]

def find_disease_by_hash(image_hash):
    """Check if image exists in database by hash"""
    return diseases_col.find_one({"image_hashes": image_hash})

def find_disease_by_scientific_name(scientific_name):
    """Find disease by scientific name"""
    return diseases_col.find_one({"scientific_name": scientific_name})

def predict_disease(img_path):
    """Predict disease from image, checking database first"""
    # Calculate image hash
    image_hash = calculate_image_hash(img_path)
    
    # Check if we've seen this exact image before
    existing_diagnosis = find_disease_by_hash(image_hash)
    if existing_diagnosis:
        logger.info(f"Found existing diagnosis for image hash {image_hash}")
        return {
            "disease_info": {
                "name": existing_diagnosis['common_name'],
                "treatment": existing_diagnosis['treatment'],
                "prevention": existing_diagnosis['prevention'],
                "severity": existing_diagnosis['severity']
            },
            "confidence": 1.0,
            "disease_key": existing_diagnosis['scientific_name'],
            "image_hash": image_hash,
            "from_database": True
        }
    
    # If model is not available, return error
    if model is None:
        logger.error("Model not loaded - returning error")
        return {
            "disease_info": {
                "name": "Diagnosis Error",
                "treatment": "Please try again with a clearer photo",
                "prevention": "Ensure good lighting and focus",
                "severity": "N/A"
            },
            "confidence": 0.0,
            "disease_key": "error",
            "image_hash": image_hash,
            "from_database": False
        }
    
    try:
        # Preprocess and predict
        img_array = preprocess_image(img_path)
        if img_array is None:
            logger.error("Image preprocessing failed")
            return {
                "disease_info": {
                    "name": "Diagnosis Error",
                    "treatment": "Please try again with a clearer photo",
                    "prevention": "Ensure good lighting and focus",
                    "severity": "N/A"
                },
                "confidence": 0.0,
                "disease_key": "error",
                "image_hash": image_hash,
                "from_database": False
            }
        
        pred = model.predict(img_array)
        class_idx = np.argmax(pred[0])
        confidence = float(np.max(pred[0]))
        
        if class_idx < len(CLASS_NAMES):
            disease_key = CLASS_NAMES[class_idx]
            disease_data = find_disease_by_scientific_name(disease_key)
            
            if disease_data:
                disease_info = {
                    "name": disease_data['common_name'],
                    "treatment": disease_data['treatment'],
                    "prevention": disease_data['prevention'],
                    "severity": disease_data['severity']
                }
                logger.info(f"Predicted: {disease_key} with confidence {confidence:.2f}")
            else:
                disease_info = {
                    "name": "Unknown Disease",
                    "treatment": "Consult a plant specialist for diagnosis",
                    "prevention": "Maintain good plant hygiene",
                    "severity": "Unknown"
                }
                logger.warning(f"No treatment info found for {disease_key}")
        else:
            disease_info = {
                "name": "Unknown Disease",
                "treatment": "Consult a plant specialist for diagnosis",
                "prevention": "Maintain good plant hygiene",
                "severity": "Unknown"
            }
            disease_key = "unknown"
            logger.warning(f"Class index {class_idx} out of range")
        
        return {
            "disease_info": disease_info,
            "confidence": confidence,
            "disease_key": disease_key,
            "image_hash": image_hash,
            "from_database": False
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "disease_info": {
                "name": "Diagnosis Error",
                "treatment": "Please try again with a clearer photo",
                "prevention": "Ensure good lighting and focus",
                "severity": "N/A"
            },
            "confidence": 0.0,
            "disease_key": "error",
            "image_hash": image_hash,
            "from_database": False
        }

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', message="No file uploaded")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', message="No file selected")
            
        if not allowed_file(file.filename):
            return render_template('error.html', message="Invalid file type")
            
        if not validate_image(file.stream):
            return render_template('error.html', message="Invalid image file")
            
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict disease or find in database
            prediction = predict_disease(filepath)
            
            # Store diagnosis in database
            diagnosis = {
                "image_path": filepath,
                "image_hash": prediction['image_hash'],
                "disease_name": prediction['disease_info']['name'],
                "scientific_name": prediction['disease_key'],
                "confidence": prediction['confidence'],
                "treatment": prediction['disease_info']['treatment'],
                "prevention": prediction['disease_info']['prevention'],
                "severity": prediction['disease_info']['severity'],
                "timestamp": datetime.now(),
                "from_database": prediction['from_database']
            }
            diagnoses_col.insert_one(diagnosis)
            
            # If this was a new prediction (not from database), add hash to disease record
            if not prediction['from_database'] and prediction['disease_key'] not in ['unknown', 'error']:
                diseases_col.update_one(
                    {"scientific_name": prediction['disease_key']},
                    {"$push": {"image_hashes": prediction['image_hash']}}
                )
            
            return render_template('result.html',
                                disease_name=prediction['disease_info']['name'],
                                scientific_name=prediction['disease_key'],
                                confidence=round(prediction['confidence']*100, 2),
                                treatment=prediction['disease_info']['treatment'],
                                prevention=prediction['disease_info']['prevention'],
                                severity=prediction['disease_info']['severity'],
                                image_path=filepath,
                                from_database=prediction['from_database'])
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return render_template('error.html', message="Processing error")
    
    return render_template('index.html')

@app.route('/history')
def history():
    try:
        diagnoses = list(diagnoses_col.find().sort("timestamp", -1).limit(20))
        # Convert ObjectId to string for template rendering
        for diagnosis in diagnoses:
            diagnosis['_id'] = str(diagnosis['_id'])
            diagnosis['formatted_timestamp'] = diagnosis['timestamp'].strftime('%Y-%m-%d %H:%M')
        return render_template('history.html', diagnoses=diagnoses)
    except Exception as e:
        logger.error(f"History error: {e}")
        return render_template('error.html', message="Could not retrieve history")

@app.route('/diagnosis/<id>')
def view_diagnosis(id):
    try:
        diagnosis = diagnoses_col.find_one({"_id": ObjectId(id)})
        if not diagnosis:
            return render_template('error.html', message="Diagnosis record not found")
        
        # Format data for display
        diagnosis['formatted_timestamp'] = diagnosis['timestamp'].strftime('%Y-%m-%d %H:%M')
        diagnosis['html_treatment'] = diagnosis['treatment'].replace('\n', '<br>')
        diagnosis['html_prevention'] = diagnosis['prevention'].replace('\n', '<br>')
        
        return render_template('view_diagnosis.html', diagnosis=diagnosis)
    except Exception as e:
        logger.error(f"View diagnosis error: {e}")
        return render_template('error.html', message="Invalid diagnosis ID")

@app.route('/delete/<id>', methods=['POST'])
def delete_diagnosis(id):
    try:
        # Remove the file from uploads
        diagnosis = diagnoses_col.find_one({"_id": ObjectId(id)})
        if diagnosis and os.path.exists(diagnosis['image_path']):
            os.remove(diagnosis['image_path'])
        
        # Remove from database
        diagnoses_col.delete_one({"_id": ObjectId(id)})
        return redirect(url_for('history'))
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        return render_template('error.html', message=f"Deletion failed: {str(e)}")

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('error.html', message="File too large (max 8MB)"), 413

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', message="Page not found"), 404

if __name__ == '__main__':
    if model is None:
        logger.error("Could not load model - running with limited functionality")
    app.run(debug=True)