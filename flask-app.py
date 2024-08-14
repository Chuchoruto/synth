from flask import Flask, request, jsonify, send_file
from model import Model
from flask_cors import CORS  # Import CORS
import os

app = Flask(__name__)
# Enable CORS for all routes but only allow requests from specific origins
CORS(app, resources={r"/*": {"origins": ["http://samplify-app.com", "http://localhost:3000"]}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
SYNTHETIC_FOLDER = 'synthetic'
os.makedirs(SYNTHETIC_FOLDER, exist_ok=True)

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
original_filename = None

@app.route('/initialize', methods=['POST'])
def initialize():
    sample_limit = 5000 # This is to make sure I don't reach 1GB maximum of sample size
    global model_instance, original_filename
    data = request.get_json()
    csv_path = data.get('csv_path')
    num_samples = data.get('num_samples', 100)  # Default to 100 if not provided

    if num_samples > sample_limit:
        num_samples = sample_limit

    if not csv_path:
        return jsonify({'error': 'csv_path not provided'}), 400

    model_instance = Model(csv_path=csv_path, num_samples=num_samples)
    original_filename = os.path.basename(csv_path).rsplit('.', 1)[0]  # Extract the base filename without extension
    return "Model initialized!"

@app.route('/download-synthetic-csv')
def download_synthetic_csv():
    global model_instance, original_filename
    if model_instance is None:
        return "Model not initialized!", 400
    
    # Get synthetic data as a DataFrame
    synthetic_data = model_instance.get_synthetic_data()
    
    # Define the file path
    csv_file_path = os.path.join(SYNTHETIC_FOLDER, f"synthetic_{original_filename}.csv")
    
    # Save the DataFrame to a CSV file
    synthetic_data.to_csv(csv_file_path, index=False)
    
    # Send the file to the client
    return send_file(csv_file_path, as_attachment=True, download_name=f"synthetic_{original_filename}.csv")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
