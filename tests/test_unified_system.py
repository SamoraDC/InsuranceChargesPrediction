#!/usr/bin/env python3
"""
Test unified system in deploy folder
"""

import sys
from pathlib import Path

# Add path for local imports
sys.path.insert(0, str(Path('.') / 'src'))

try:
    from insurance_prediction.models.predictor import load_production_model, predict_insurance_premium
    print('✅ Import local system successful')
    
    # Test prediction
    result = predict_insurance_premium(
        age=25,
        sex='male',
        bmi=22.6,
        children=0,
        smoker='no',
        region='southeast'
    )
    print(f'✅ Local prediction: ${result["predicted_premium"]:,.2f}')
    print(f'Model type: {result["model_type"]}')
    print(f'Processing time: {result["processing_time_ms"]:.2f}ms')
    
except Exception as e:
    print(f'❌ Error with local system: {e}')
    print('Will fallback to deployment system')
    
    # Test fallback
    try:
        from deploy.model_utils import predict_premium
        
        test_data = {
            'age': 25,
            'sex': 'male', 
            'bmi': 22.6,
            'children': 0,
            'smoker': 'no',
            'region': 'southeast'
        }
        
        result = predict_premium(test_data)
        if result['success']:
            print(f'✅ Fallback prediction: ${result["predicted_premium"]:,.2f}')
        else:
            print(f'❌ Fallback error: {result["error"]}')
    except Exception as e2:
        print(f'❌ Fallback also failed: {e2}')

print('\n=== Testing both systems ===')

# Test local
try:
    result1 = predict_insurance_premium(age=25, sex='male', bmi=22.6, children=0, smoker='no', region='southeast')
    print(f'Local model: ${result1["predicted_premium"]:,.2f}')
except:
    print('Local model: FAILED')

# Test deployment  
try:
    from deploy.model_utils import predict_premium
    test_data = {'age': 25, 'sex': 'male', 'bmi': 22.6, 'children': 0, 'smoker': 'no', 'region': 'southeast'}
    result2 = predict_premium(test_data)
    if result2['success']:
        print(f'Deploy model: ${result2["predicted_premium"]:,.2f}')
    else:
        print(f'Deploy model: {result2["error"]}')
except Exception as e:
    print(f'Deploy model: FAILED - {e}') 