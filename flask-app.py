from flask import Flask, request, jsonify, send_file
from model import Model
from flask_cors import CORS  # Import CORS
import os

app = Flask(__name__)
# Enable CORS for all routes but only allow requests from a specific origin
CORS(app, resources={r"/*": {"origins": "http://your-frontend-domain.com"}}) # This Domain needs to get changed to the frontend domain we end up using.

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# Global variable to store the model instance
model_instance = None

@app.route('/initialize', methods=['POST'])
def initialize():
    global model_instance
    data = request.get_json()
    csv_path = data.get('csv_path')
    num_samples = data.get('num_samples', 100)  # Default to 100 if not provided

    if not csv_path:
        return jsonify({'error': 'csv_path not provided'}), 400

    model_instance = Model(csv_path=csv_path, num_samples=num_samples)
    return "Model initialized!"

@app.route('/download-synthetic-csv')
def download_synthetic_csv():
    global model_instance
    if model_instance is None:
        return "Model not initialized!"
    
    # Get the CSV file path
    csv_file_path = model_instance.get_synthetic_csv()
    
    # Send the file to the client
    return send_file(csv_file_path, as_attachment=True, download_name=os.path.basename(csv_file_path))

if __name__ == '__main__':
    app.run(debug=True)
