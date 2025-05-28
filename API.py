"""
Fire Risk Prediction API
Flask-based REST API for wildfire risk prediction using trained models.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import io
import json
import base64
import logging
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FireRiskClassifier(nn.Module):
    """ResNet-based classifier for fire risk prediction"""
    
    def __init__(self, num_classes=4):
        super(FireRiskClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class FireRiskModelWrapper:
    """Wrapper class for the trained fire risk model"""
    
    def __init__(self, model_path, device=None):
        self.device = self._setup_device(device)
        self.class_names = ["High", "Low", "Moderate", "Non-burnable"]
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        logger.info(f"Model loaded successfully on {self.device}")
        
    def _setup_device(self, device):
        """Setup inference device"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        return device
    
    def _load_model(self, model_path):
        """Load the trained model"""
        model = FireRiskClassifier(num_classes=len(self.class_names))
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
        return model
    
    def _get_transform(self):
        """Get image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Make prediction for a single image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get all class probabilities
                all_probs = probabilities.squeeze().cpu().numpy()
                
                # Get top prediction
                max_prob, predicted = torch.max(probabilities, 1)
                
                result = {
                    "predicted_class_id": predicted.item(),
                    "predicted_class_name": self.class_names[predicted.item()],
                    "confidence": max_prob.item(),
                    "all_probabilities": {
                        self.class_names[i]: float(prob) 
                        for i, prob in enumerate(all_probs)
                    }
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, images):
        """
        Make predictions for a batch of images
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of prediction dictionaries
        """
        try:
            # Preprocess all images
            image_tensors = []
            for image in images:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = self.transform(image)
                image_tensors.append(image_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(image_tensors).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                results = []
                for i in range(len(images)):
                    probs = probabilities[i].cpu().numpy()
                    predicted_idx = np.argmax(probs)
                    
                    result = {
                        "predicted_class_id": int(predicted_idx),
                        "predicted_class_name": self.class_names[predicted_idx],
                        "confidence": float(probs[predicted_idx]),
                        "all_probabilities": {
                            self.class_names[j]: float(prob) 
                            for j, prob in enumerate(probs)
                        }
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model wrapper
model_wrapper = None

def create_model_wrapper():
    """Initialize the model wrapper"""
    global model_wrapper
    
    model_path = os.environ.get('MODEL_PATH', 'models/fire_risk_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_wrapper = FireRiskModelWrapper(model_path)
    logger.info("Model wrapper initialized successfully")

def parse_image_from_request():
    """Parse image from request (file upload or base64)"""
    image = None
    
    # Check for file upload
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Image loaded from file upload: {file.filename}")
            except Exception as e:
                raise ValueError(f"Invalid image file: {str(e)}")
    
    # Check for base64 data
    elif 'image' in request.form:
        try:
            encoded_image = request.form['image']
            # Remove data URL prefix if present
            if ',' in encoded_image:
                encoded_image = encoded_image.split(',')[1]
            
            image_bytes = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(image_bytes))
            logger.info("Image loaded from base64 data")
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {str(e)}")
    
    # Check for JSON data
    elif request.is_json and 'image' in request.json:
        try:
            encoded_image = request.json['image']
            if ',' in encoded_image:
                encoded_image = encoded_image.split(',')[1]
            
            image_bytes = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(image_bytes))
            logger.info("Image loaded from JSON data")
        except Exception as e:
            raise ValueError(f"Invalid JSON image data: {str(e)}")
    
    if image is None:
        raise ValueError("No image provided in request")
    
    return image

# API Routes
@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fire Risk Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
            .example { background: #e9ecef; padding: 10px; margin: 5px 0; font-family: monospace; }
        </style>
    </head>
    <body>
        <h1>🔥 Fire Risk Prediction API</h1>
        <p>This API provides wildfire risk classification for satellite imagery.</p>
        
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status</p>
            <div class="example">curl http://localhost:5000/health</div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Predict fire risk for a single image</p>
            <p><strong>Input:</strong> Image file or base64 encoded image</p>
            <p><strong>Output:</strong> Risk classification with confidence scores</p>
            <div class="example">curl -X POST -F "image=@satellite_image.jpg" http://localhost:5000/predict</div>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict/batch</h3>
            <p>Predict fire risk for multiple images</p>
            <p><strong>Input:</strong> Multiple image files</p>
            <p><strong>Output:</strong> Array of risk classifications</p>
        </div>
        
        <h2>Risk Classes</h2>
        <ul>
            <li><strong>High:</strong> Very high fire risk</li>
            <li><strong>Low:</strong> Low fire risk</li>
            <li><strong>Moderate:</strong> Moderate fire risk</li>
            <li><strong>Non-burnable:</strong> Areas unlikely to burn</li>
        </ul>
        
        <h2>Model Info</h2>
        <p>Model: ResNet-18 based classifier</p>
        <p>Input: 224x224 RGB satellite images</p>
        <p>Classes: 4 fire risk categories</p>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "model_loaded": model_wrapper is not None,
            "device": str(model_wrapper.device) if model_wrapper else "unknown",
            "classes": model_wrapper.class_names if model_wrapper else []
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict fire risk for a single image"""
    try:
        if model_wrapper is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Parse image from request
        image = parse_image_from_request()
        
        # Make prediction
        result = model_wrapper.predict(image)
        
        # Add metadata
        result["model_info"] = {
            "model_type": "ResNet-18",
            "version": "1.0",
            "classes": model_wrapper.class_names
        }
        
        logger.info(f"Prediction made: {result['predicted_class_name']} ({result['confidence']:.3f})")
        
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict fire risk for multiple images"""
    try:
        if model_wrapper is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Parse multiple images
        images = []
        
        # Handle multiple file uploads
        if 'images' in request.files:
            files = request.files.getlist('images')
            for file in files:
                if file.filename != '':
                    image_bytes = file.read()
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
        else:
            return jsonify({"error": "No images provided"}), 400
        
        if not images:
            return jsonify({"error": "No valid images found"}), 400
        
        # Make predictions
        results = model_wrapper.predict_batch(images)
        
        response = {
            "predictions": results,
            "batch_size": len(results),
            "model_info": {
                "model_type": "ResNet-18",
                "version": "1.0",
                "classes": model_wrapper.class_names
            }
        }
        
        logger.info(f"Batch prediction completed: {len(results)} images")
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model_wrapper is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        info = {
            "model_type": "ResNet-18",
            "version": "1.0",
            "classes": model_wrapper.class_names,
            "num_classes": len(model_wrapper.class_names),
            "input_size": [224, 224, 3],
            "device": str(model_wrapper.device)
        }
        
        return jsonify(info), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Main function to run the API server"""
    try:
        # Initialize model
        create_model_wrapper()
        
        # Get configuration from environment
        host = os.environ.get('API_HOST', '0.0.0.0')
        port = int(os.environ.get('API_PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting Fire Risk API server...")
        logger.info(f"Host: {host}, Port: {port}, Debug: {debug}")
        
        # Run the app
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == '__main__':
    main()
