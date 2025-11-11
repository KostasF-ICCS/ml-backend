import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS for production
allowed_origins = os.getenv('CORS_ORIGINS', '*').split(',')
CORS(app, origins=allowed_origins, supports_credentials=True)

# Model loading with fallback
MODEL_PATH = os.getenv('MODEL_PATH', 'final_rf_model.pkl')
model = None

def load_model():
    """Load the trained model with error handling"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Number of estimators: {model.n_estimators}")
            return True
        else:
            logger.warning(f"⚠️  Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

# Load model on startup
load_model()


def validate_input(data):
    """Validate input data"""
    required_fields = ['car', 'taxi', 'month', 'car_lag1', 'taxi_lag1']
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f'Missing required field: {field}'
    
    # Validate ranges
    try:
        car = float(data['car'])
        taxi = float(data['taxi'])
        month = int(data['month'])
        car_lag1 = float(data['car_lag1'])
        taxi_lag1 = float(data['taxi_lag1'])
        
        if not (0 <= car <= 100):
            return False, 'Car usage must be between 0% and 100%'
        if not (0 <= taxi <= 100):
            return False, 'Taxi usage must be between 0% and 100%'
        if not (1 <= month <= 12):
            return False, 'Month must be between 1 and 12'
        if not (0 <= car_lag1 <= 100):
            return False, 'Previous car usage must be between 0% and 100%'
        if not (0 <= taxi_lag1 <= 100):
            return False, 'Previous taxi usage must be between 0% and 100%'
            
        return True, None
        
    except (ValueError, TypeError) as e:
        return False, f'Invalid data type: {str(e)}'


@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    return jsonify({
        'name': 'Metro Ridership Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'predict': '/api/predict (POST)',
            'batch_predict': '/api/batch-predict (POST)',
            'model_info': '/api/model-info (GET)',
            'health': '/health (GET)'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'uptime': 'ok'
    }), 200


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Request body (JSON):
    {
        "car": 25.5,
        "taxi": 12.0,
        "month": 11,
        "car_lag1": 24.0,
        "taxi_lag1": 11.5
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Prepare features in exact training order
        features = pd.DataFrame({
            'Car': [float(data['car'])],
            'Taxi': [float(data['taxi'])],
            'Month': [int(data['month'])],
            'Car_lag1': [float(data['car_lag1'])],
            'Taxi_lag1': [float(data['taxi_lag1'])]
        })
        
        # Make prediction
        if model is not None:
            prediction = float(model.predict(features)[0])
            
            # Calculate confidence interval from tree predictions
            tree_predictions = np.array([
                tree.predict(features)[0] 
                for tree in model.estimators_
            ])
            std_dev = np.std(tree_predictions)
            confidence_interval = [
                float(max(0, prediction - 1.96 * std_dev)),
                float(min(100, prediction + 1.96 * std_dev))
            ]
            
            logger.info(f"Prediction made: {prediction:.2f}% for month {data['month']}")
            
        else:
            # Fallback prediction if model not loaded
            logger.warning("Using fallback prediction - model not loaded")
            prediction = 25.0
            confidence_interval = [20.0, 30.0]
        
        # Return response
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'confidence_interval': [round(ci, 2) for ci in confidence_interval],
            'timestamp': datetime.now().isoformat(),
            'input_features': {
                'car': float(data['car']),
                'taxi': float(data['taxi']),
                'month': int(data['month']),
                'car_lag1': float(data['car_lag1']),
                'taxi_lag1': float(data['taxi_lag1'])
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Request body (JSON):
    {
        "predictions": [
            {"car": 25.5, "taxi": 12.0, "month": 1, "car_lag1": 24.0, "taxi_lag1": 11.5},
            {"car": 26.0, "taxi": 13.0, "month": 2, "car_lag1": 25.5, "taxi_lag1": 12.0}
        ]
    }
    """
    try:
        data = request.get_json()
        predictions_input = data.get('predictions', [])
        
        if not predictions_input:
            return jsonify({
                'success': False,
                'error': 'No predictions requested'
            }), 400
        
        if len(predictions_input) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum 100 predictions per batch request'
            }), 400
        
        results = []
        errors = []
        
        for idx, item in enumerate(predictions_input):
            # Validate each input
            is_valid, error_msg = validate_input(item)
            if not is_valid:
                errors.append({
                    'index': idx,
                    'error': error_msg,
                    'input': item
                })
                continue
            
            # Prepare features
            features = pd.DataFrame({
                'Car': [float(item['car'])],
                'Taxi': [float(item['taxi'])],
                'Month': [int(item['month'])],
                'Car_lag1': [float(item['car_lag1'])],
                'Taxi_lag1': [float(item['taxi_lag1'])]
            })
            
            # Make prediction
            if model is not None:
                prediction = float(model.predict(features)[0])
            else:
                prediction = 25.0
            
            results.append({
                'index': idx,
                'input': item,
                'prediction': round(prediction, 2)
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'errors': errors,
            'total_requested': len(predictions_input),
            'successful': len(results),
            'failed': len(errors)
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        feature_names = ['Car', 'Taxi', 'Month', 'Car_lag1', 'Taxi_lag1']
        feature_importance_dict = {
            name: float(importance) 
            for name, importance in zip(feature_names, model.feature_importances_)
        }
        
        return jsonify({
            'success': True,
            'model_type': type(model).__name__,
            'n_estimators': int(model.n_estimators),
            'max_depth': int(model.max_depth) if model.max_depth else None,
            'min_samples_split': int(model.min_samples_split),
            'min_samples_leaf': int(model.min_samples_leaf),
            'features': feature_names,
            'n_features': len(feature_names),
            'feature_importance': feature_importance_dict,
            'sklearn_version': '1.3.0'
        }), 200
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/reload-model', methods=['POST'])
def reload_model():
    """Reload the model (useful for updates)"""
    try:
        success = load_model()
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to reload model'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"🚀 Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Model loaded: {model is not None}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )