from flask import Flask, request, jsonify, send_file, session
from model import Model
from flask_cors import CORS
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Necessary for session management, use a fixed key in production
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["https://samplify-app.com", "https://www.samplify-app.com", "https://api.samplify-app.com", "http://localhost:3000"]}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SYNTHETIC_FOLDER = 'synthetic'
os.makedirs(SYNTHETIC_FOLDER, exist_ok=True)

# Store user-specific data, such as model references
user_models = {}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return jsonify({'file_path': file_path}), 200

@app.route('/initialize', methods=['POST'])
def initialize():
    sample_limit = 5000  # Ensure the sample size doesn't exceed your limit
    data = request.get_json()
    csv_path = data.get('csv_path')
    num_samples = data.get('num_samples', 100)

    if num_samples > sample_limit:
        num_samples = sample_limit

    if not csv_path:
        return jsonify({'error': 'csv_path not provided'}), 400

    session_id = session.get('id')
    if not session_id:
        session_id = str(uuid.uuid4())  # Generate a new session ID
        session['id'] = session_id

    model_instance = Model(csv_path=csv_path, num_samples=num_samples)
    user_models[session_id] = model_instance

    original_filename = os.path.basename(csv_path).rsplit('.', 1)[0]
    session['original_filename'] = original_filename

    return "Model initialized!", 200

@app.route('/download-synthetic-csv')
def download_synthetic_csv():
    session_id = session.get('id')
    if not session_id or session_id not in user_models:
        return "Model not initialized!", 400

    model_instance = user_models[session_id]
    original_filename = session.get('original_filename')

    # Generate synthetic data and save it
    synthetic_data = model_instance.get_synthetic_data()
    csv_file_path = os.path.join(SYNTHETIC_FOLDER, f"synthetic_{original_filename}.csv")
    synthetic_data.to_csv(csv_file_path, index=False)

    return send_file(csv_file_path, as_attachment=True, download_name=f"synthetic_{original_filename}.csv")
