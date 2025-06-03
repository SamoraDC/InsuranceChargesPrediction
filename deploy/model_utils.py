#!/usr/bin/env python3
"""
Model utilities for insurance premium prediction - Production Deploy
Independent implementation without src/ dependencies
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = Path("../models/production_model_optimized.pkl")
FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]

# Validation ranges and values
NUMERICAL_RANGES = {
    "age": {"min": 18, "max": 64},
    "bmi": {"min": 15.0, "max": 55.0},
    "children": {"min": 0, "max": 5}
}

CATEGORICAL_VALUES = {
    "sex": ["male", "female"],
    "smoker": ["yes", "no"], 
    "region": ["northeast", "northwest", "southeast", "southwest"]
}

def load_model():
    """
    Load the trained insurance prediction model.
    
    Returns:
        Model data dictionary or None if error
    """
    try:
        model_path = Path(__file__).parent / MODEL_PATH
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
            
        # Load the complete model data (model + scaler + encoders)
        model_data = joblib.load(model_path)
        logger.info("✅ Model loaded successfully!")
        return model_data
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

def validate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate user input data.
    
    Args:
        data: Dictionary with input features
        
    Returns:
        Dictionary with validation results
    """
    errors = []
    warnings = []
    
    # Check required columns
    missing_cols = set(FEATURE_COLUMNS) - set(data.keys())
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Validate numerical ranges
    for col in ["age", "bmi", "children"]:
        if col in data:
            value = data[col]
            range_info = NUMERICAL_RANGES[col]
            
            if value < range_info["min"] or value > range_info["max"]:
                warnings.append(
                    f"{col} value {value} is outside typical range "
                    f"({range_info['min']}-{range_info['max']})"
                )
    
    # Validate categorical values
    for col in ["sex", "smoker", "region"]:
        if col in data:
            if data[col] not in CATEGORICAL_VALUES[col]:
                errors.append(
                    f"Invalid value for {col}: {data[col]}. "
                    f"Valid values: {CATEGORICAL_VALUES[col]}"
                )
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }

def preprocess_data(data: Dict[str, Any], model_data: dict) -> np.ndarray:
    """
    Preprocess input data for model prediction using trained encoders and scaler.
    
    Args:
        data: Dictionary with input features
        model_data: Complete model data with encoders and scaler
        
    Returns:
        Preprocessed numpy array ready for model
    """
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Apply trained encoders
    encoders = model_data['encoders']
    df['sex'] = encoders['sex'].transform(df['sex'])
    df['smoker'] = encoders['smoker'].transform(df['smoker'])
    df['region'] = encoders['region'].transform(df['region'])
    
    # Add interaction features (same as training)
    df['bmi_smoker'] = df['bmi'] * df['smoker']
    df['age_smoker'] = df['age'] * df['smoker']
    
    # Select features in same order as training
    feature_names = model_data['feature_names']
    df_features = df[feature_names]
    
    # Apply trained scaler
    scaler = model_data['scaler']
    X_scaled = scaler.transform(df_features)
    
    return X_scaled

def predict_premium(data: Dict[str, Any], model_data=None) -> Dict[str, Any]:
    """
    Predict insurance premium for given input.
    
    Args:
        data: Dictionary with input features
        model_data: Pre-loaded model data (optional)
        
    Returns:
        Dictionary with prediction results
    """
    start_time = pd.Timestamp.now()
    
    try:
        # Validate input
        validation = validate_input(data)
        if not validation["is_valid"]:
            return {
                "success": False,
                "error": f"Validation failed: {validation['errors']}",
                "warnings": validation["warnings"]
            }
        
        # Load model if not provided
        if model_data is None:
            model_data = load_model()
            if model_data is None:
                return {
                    "success": False,
                    "error": "Could not load model"
                }
        
        # Preprocess data
        X = preprocess_data(data, model_data)
        
        # Make prediction using the trained model
        model = model_data['model']
        prediction = model.predict(X)[0]
        
        # Calculate processing time
        processing_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
        
        # Get model performance metrics
        metrics = model_data.get('metrics', {})
        
        # Format result
        result = {
            "success": True,
            "predicted_premium": round(float(prediction), 2),
            "monthly_premium": round(float(prediction) / 12, 2),
            "input_data": data,
            "processing_time_ms": round(processing_time, 2),
            "model_info": {
                "algorithm": "Gradient Boosting",
                "version": "1.0.0",
                "performance": f"R² = {metrics.get('r2', 0.88):.3f}",
                "mae": f"${metrics.get('mae', 2650):.0f}"
            },
            "warnings": validation["warnings"]
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }

def get_risk_analysis(data: Dict[str, Any], premium: float) -> Dict[str, Any]:
    """
    Analyze risk factors affecting the premium.
    
    Args:
        data: Input data dictionary
        premium: Predicted premium
        
    Returns:
        Risk analysis results
    """
    risk_factors = []
    risk_level = "Low"
    
    # Analyze risk factors
    if data['smoker'] == 'yes':
        risk_factors.append({
            "factor": "Smoker",
            "impact": "Very High",
            "description": "Smoking significantly increases health risks"
        })
        risk_level = "High"
    
    if data['age'] > 50:
        risk_factors.append({
            "factor": "Age > 50",
            "impact": "Medium",
            "description": "Higher age correlates with increased health risks"
        })
        if risk_level == "Low":
            risk_level = "Medium"
    
    if data['bmi'] > 30:
        risk_factors.append({
            "factor": "BMI > 30 (Obesity)",
            "impact": "Medium",
            "description": "Obesity increases risk of various health conditions"
        })
        if risk_level == "Low":
            risk_level = "Medium"
    
    if data['bmi'] < 18.5:
        risk_factors.append({
            "factor": "BMI < 18.5 (Underweight)",
            "impact": "Low",
            "description": "Being underweight may indicate health issues"
        })
    
    # Calculate premium percentile (approximation)
    if premium < 5000:
        percentile = "Bottom 25%"
    elif premium < 15000:
        percentile = "25-75%"
    elif premium < 30000:
        percentile = "Top 25%"
    else:
        percentile = "Top 10%"
    
    return {
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "premium_percentile": percentile,
        "recommendations": get_recommendations(data)
    }

def get_recommendations(data: Dict[str, Any]) -> list:
    """Get health and financial recommendations based on profile."""
    recommendations = []
    
    if data['smoker'] == 'yes':
        recommendations.append("Consider smoking cessation programs to reduce premium and health risks")
    
    if data['bmi'] > 30:
        recommendations.append("Weight management programs may help reduce premium costs")
    
    if data['bmi'] < 18.5:
        recommendations.append("Consider nutritional counseling to achieve healthy weight")
    
    if data['age'] > 45:
        recommendations.append("Regular health check-ups become increasingly important")
    
    if not recommendations:
        recommendations.append("Maintain healthy lifestyle to keep premium costs low")
    
    return recommendations

if __name__ == "__main__":
    # Test the utilities
    test_data = {
        'age': 35,
        'sex': 'male',
        'bmi': 27.5,
        'children': 2,
        'smoker': 'no',
        'region': 'northeast'
    }
    
    print("🧪 Testing model utilities...")
    result = predict_premium(test_data)
    
    if result["success"]:
        print(f"✅ Prediction successful: ${result['predicted_premium']:,.2f}")
        print(f"⚡ Processing time: {result['processing_time_ms']:.2f}ms")
    else:
        print(f"❌ Prediction failed: {result['error']}") 