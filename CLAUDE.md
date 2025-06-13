# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a microservices-based real estate valuation application for Tokyo's 23 wards, with Django frontend and FastAPI backend deployed on AWS Lambda.

### Service Communication Flow
```
[User] → [Django Lambda] → HTTP POST → [FastAPI Lambda] → [Valuation Model]
                ↓                            ↓
        [Form Validation]            [LightweightModel]
                ↓                            ↓
        [Result Display] ← JSON ← [Price Calculation]
```

### Production URLs
- **Django**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **FastAPI**: https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/

## Common Development Commands

### Local Development
```bash
# Start all services with Docker Compose
cd deployment
docker-compose up --build

# Run Django development server only
cd valuation-app
python manage.py runserver 0.0.0.0:8080

# Run FastAPI development server only  
cd valuation-api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Model Management
```bash
# Comprehensive model management (recommended)
cd model-creation
./manage_model.sh train --model-type rf --data-source sample
./manage_model.sh evaluate
./manage_model.sh info
./manage_model.sh backup current_model

# Python model management
python model_manager.py --action create --quick
python model_manager.py --action compare --models model1,model2
python model_manager.py --action status

# Test model accuracy with real estate scenarios
python test_model_accuracy.py --endpoint http://localhost:8000 --test-scenarios all
```

### Deployment Commands
```bash
# Unified deployment (recommended - handles service dependencies)
cd deployment
./deploy_all.sh                       # Interactive deployment with environment selection
./deploy_all.sh prod django           # Deploy Django to production
./deploy_all.sh prod api              # Deploy API to production
./deploy_all.sh prod both             # Deploy both services with proper sequencing

# ECR container deployment (for ML dependencies)
./ecr-deploy.sh api prod              # Deploy API via ECR container

# Manual SAM deployment
sam build -t lambda-api-light.yml
sam deploy --stack-name satei-api-light --resolve-s3

# Lambda layer management (for shared dependencies)
./create-lambda-layer.sh

# Check deployment status
aws cloudformation describe-stacks --stack-name satei-api-light
aws lambda list-functions --query "Functions[?contains(FunctionName, 'satei')].FunctionName"
```

### Testing
```bash
# Model accuracy testing with real estate scenarios
cd model-creation
python test_model_accuracy.py --endpoint http://localhost:8000 --test-scenarios all
python test_model_accuracy.py --endpoint https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod --test-scenarios premium

# Django application tests
cd valuation-app
python manage.py test valuation

# API endpoint tests
curl -X POST https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","district":"恵比寿","land_area":100,"building_area":80,"building_age":10}'

# Test local development environment
curl http://localhost:8000/api/valuation -X POST -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","district":"六本木","land_area":150,"building_area":120,"building_age":5}'
```

## High-Level Architecture

### Service Dependencies and Deployment Strategy
The Django frontend depends on the FastAPI backend being available at the URL specified in `VALUATION_API_URL`. The `deploy_all.sh` script handles this complexity automatically:

**Deployment Order (Automatic):**
1. Deploy FastAPI first to get its URL
2. Deploy Django with the FastAPI URL injected as environment variable
3. Update Django configuration if FastAPI URL changes

**Manual Environment Variable Management:**
```bash
# Check current Lambda environment variables
aws lambda get-function-configuration --function-name satei-django-prod --query 'Environment.Variables'

# Update Django Lambda to point to correct API
aws lambda update-function-configuration --function-name satei-django-prod \
  --environment Variables='{VALUATION_API_URL=https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod,...}'
```

### Lambda Adaptation Pattern
Both services use Mangum for ASGI-to-Lambda adaptation:
```python
# Django (lambda_handler.py)
django_app = get_asgi_application()
handler = Mangum(django_app, lifespan="off")

# FastAPI (lambda_main.py)  
handler = Mangum(app, lifespan="off")
```

### Model Fallback Architecture
The FastAPI service uses a simplified approach for Lambda deployment:
```python
# Currently simplified for Lambda (valuation-api/main.py)
from models.lightweight_model import LightweightValuationModel
valuation_model = LightweightValuationModel()
```

**Current State:** Uses only `LightweightValuationModel` (rule-based calculation)
**Full Model Integration:** Available via model-creation scripts for local development

**Model Management Workflow:**
1. **Local Development:** Full ML models with scikit-learn, pandas
2. **Lambda Deployment:** Lightweight model (4.5MB vs 113MB)
3. **Model Training:** Managed via `model-creation/` directory scripts
4. **Testing:** Real estate scenario validation with `test_model_accuracy.py`

### Lambda Configuration Adjustments
Django automatically detects Lambda environment and adjusts:
- Disables CSRF middleware (commented out in settings.py)
- Sets `FORCE_SCRIPT_NAME = '/Prod'` for API Gateway routing
- Configures CORS trusted origins

### Price Calculation Logic
The valuation uses a rule-based approach with Tokyo 23 ward base prices:
- Each ward has a base price per square meter (万円/㎡)
- Building depreciation: 3% per year, minimum 30% value retained
- Final price includes ±10% random variation for realism
- Results formatted as "X,XXX万円" using Django template filter

### Deployment Strategies
1. **AWS Lambda ZIP**: Limited to 250MB, requires lightweight dependencies
2. **ECR Container**: No size limit, can include full ML stack
3. **Local Docker**: Full development environment with hot reload

The `deploy_unified.sh` script handles all deployment complexity and provides the correct deployment order.

## Critical Configuration Notes

### Django-FastAPI Field Mapping
Django forms use different field names than FastAPI expects. The mapping is handled in `valuation/views.py`:
```python
valuation_data = {
    'prefecture': form.cleaned_data['prefecture'],
    'city': form.cleaned_data['city'],
    'district': form.cleaned_data['district'],  # district → district (no change)
    'land_area': form.cleaned_data['land_area'],
    'building_area': form.cleaned_data['building_area'],
    'building_age': form.cleaned_data['building_age']  # building_age → building_age (no change)
}
```

### Lambda Package Size Management
- Use `requirements-lambda.txt` for Lambda deployments (excludes ML libraries)
- Full `requirements.txt` for local development
- ECR deployment can use full requirements

### Environment-Specific Settings
- Lambda detection: `if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ`
- API URL injection: Via CloudFormation parameters
- CSRF handling: Must be disabled for Lambda/API Gateway