"""
Flask Web Interface for Lesion Pre-Screening

A minimalist, professional web interface for uploading lesion images
and receiving AI-powered risk assessments.

NOT FOR MEDICAL DIAGNOSIS - Educational purposes only.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import torch

from src.inference.predict import LesionPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Model configuration
MODEL_PATH = "data/models/best_model.pth"
CLASS_NAMES = ["Low Risk", "Intermediate Risk", "High Risk"]

# Load model on startup
print("Loading model...")
try:
    predictor = LesionPredictor(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        device='cpu'
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Run prediction
            predictions, visual_features = predictor.predict_image(filepath)
            risk_score = predictor.assess_risk(predictions, visual_features)

            # Prepare response
            response = {
                'risk_score': round(risk_score.overall_score, 1),
                'risk_level': risk_score.risk_level.value,
                'confidence': round(risk_score.confidence, 3),
                'predictions': {
                    CLASS_NAMES[i]: round(float(pred), 3)
                    for i, pred in enumerate(predictions)
                },
                'visual_features': {
                    k: round(v, 3) for k, v in visual_features.items()
                },
                'recommendations': risk_score.recommendations,
                'image_path': filepath
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/generate_report', methods=['POST'])
def generate_report():
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json
    image_path = data.get('image_path')

    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 400

    try:
        # Re-run prediction
        predictions, visual_features = predictor.predict_image(image_path)
        risk_score = predictor.assess_risk(predictions, visual_features)

        # Generate PDF
        output_pdf = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
        predictor.generate_report(
            image_path=image_path,
            risk_score=risk_score,
            output_path=output_pdf
        )

        return send_file(output_pdf, as_attachment=True, download_name='lesion_report.pdf')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
