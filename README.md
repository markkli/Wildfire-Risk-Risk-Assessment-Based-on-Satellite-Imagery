# 🔥 Wildfire Risk Prediction System

A comprehensive machine learning and deep learning approach to predict wildfire risk using both tabular environmental data and satellite imagery analysis.

## 🎯 Project Overview

This project combines traditional machine learning models for environmental tabular data with deep learning image classification to provide accurate, real-time wildfire risk predictions. The hybrid approach addresses critical gaps in current catastrophe modeling by integrating multiple data sources and providing dynamic risk assessments.

## 🚀 Features

- **Hybrid Modeling**: Combines tabular ML models with satellite image classification
- **Real-time Predictions**: API endpoints for immediate risk assessment
- **Multi-modal Data Integration**: Weather, geographic, and satellite imagery data
- **Class Balancing**: Advanced techniques to handle imbalanced fire risk data
- **Production Ready**: Containerized deployment with REST API

## 📊 Model Performance

### Tabular Data Models
- **Best Model**: XGBoost with F2 score of 56%
- **Evaluation Metric**: F2 score (prioritizes recall over precision)
- **Models Tested**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM

### Image Classification
- **Architecture**: ResNet-18 based classifier
- **Classes**: High, Low, Moderate, Non-burnable (4 classes)
- **Input**: 224x224 satellite imagery
- **Data Source**: FireRisk dataset from Hugging Face

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/wildfire-risk-prediction.git
cd wildfire-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### 1. Data Preprocessing

```bash
# Preprocess tabular data
python src/data/data_preprocessing.py --input data/raw/ --output data/processed/

# Preprocess satellite images
python src/data/image_preprocessing.py --dataset blanchon/FireRisk --output data/processed/
```

### 2. Model Training

```bash
# Train tabular models
python src/training/train_tabular.py --data data/processed/tabular_data.csv --output models/

# Train image classifier
python src/training/train_image.py --data data/processed/image_data/ --output models/
```

### 3. Model Evaluation

```bash
# Evaluate all models
python src/training/model_evaluation.py --models models/ --test-data data/processed/test/
```

### 4. API Deployment

```bash
# Run locally
python src/api/app.py

# Or using Docker
docker build -t wildfire-api .
docker run -p 5000:5000 wildfire-api
```

## 🌐 API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Predict Fire Risk (Image)
```bash
curl -X POST -F "image=@path/to/satellite/image.jpg" http://localhost:5000/predict/image
```

### Predict Fire Risk (Tabular Data)
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"temperature": 35, "humidity": 20, "wind_speed": 15, "precipitation": 0}' \
  http://localhost:5000/predict/tabular
```

## 📈 Data Sources

- **Tabular Data**: Weather stations, geographic features, historical fire records
- **Satellite Imagery**: FireRisk dataset from Hugging Face (blanchon/FireRisk)
- **Fire Records**: Historical wildfire occurrence data

## 🔧 Configuration

Key parameters can be adjusted in `src/utils/config.py`:

```python
# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Data parameters
IMAGE_SIZE = (224, 224)
TARGET_SAMPLES_PER_CLASS = 1300

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 5000
```

## 📚 References

- [FireRisk Dataset](https://huggingface.co/datasets/blanchon/FireRisk)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
