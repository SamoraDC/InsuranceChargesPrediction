#!/usr/bin/env python3
"""
Test script for deployment verification
Teste de verificação do deployment
"""

import sys
import os
from pathlib import Path

def test_model_loading():
    """Test model loading."""
    print("🧪 Testing model loading...")
    
    try:
        from model_utils import load_model
        model = load_model()
        
        if model is not None:
            print("✅ Model loaded successfully!")
            return True
        else:
            print("❌ Model loading failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_prediction():
    """Test prediction functionality."""
    print("\n🧪 Testing prediction...")
    
    try:
        from model_utils import predict_premium
        
        test_data = {
            'age': 35,
            'sex': 'male',
            'bmi': 25.0,
            'children': 1,
            'smoker': 'no',
            'region': 'northeast'
        }
        
        result = predict_premium(test_data)
        
        if result['success']:
            print(f"✅ Prediction successful: ${result['predicted_premium']:,.2f}")
            print(f"⚡ Processing time: {result['processing_time_ms']:.2f}ms")
            return True
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def test_streamlit_imports():
    """Test Streamlit app imports."""
    print("\n🧪 Testing Streamlit imports...")
    
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ All Streamlit dependencies imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_file_structure():
    """Test file structure."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "streamlit_app.py",
        "model_utils.py", 
        "requirements_deploy.txt",
        ".streamlit/config.toml",
        "../models/production_model_optimized.pkl"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_translation_system():
    """Test translation system."""
    print("\n🧪 Testing translation system...")
    
    try:
        # Import the translation function from streamlit_app
        import streamlit_app
        
        # Test translations
        test_pt = streamlit_app.t("main_header", "pt")
        test_en = streamlit_app.t("main_header", "en")
        
        if test_pt and test_en and test_pt != test_en:
            print("✅ Translation system working!")
            print(f"  PT: {test_pt}")
            print(f"  EN: {test_en}")
            return True
        else:
            print("❌ Translation system not working properly")
            return False
            
    except Exception as e:
        print(f"❌ Error testing translations: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 DEPLOY SYSTEM TEST / TESTE DO SISTEMA DE DEPLOY")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("Streamlit Imports", test_streamlit_imports),
        ("Translation System", test_translation_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY / RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Deploy system is ready! 🚀")
        print("🎉 Todos os testes passaram! Sistema de deploy pronto! 🚀")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before deploying.")
        print(f"⚠️  {total - passed} teste(s) falharam. Corrija os problemas antes do deploy.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 