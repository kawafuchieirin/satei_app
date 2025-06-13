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
# Create and train a new model
cd model-creation
python create_model.py --data-source sample --model-type rf

# Evaluate model performance
python scripts/model_evaluator.py

# Quick model creation (uses defaults)
python scripts/quick_model.py
```

### Deployment Commands
```bash
# Unified deployment script (recommended)
cd deployment
./deploy_unified.sh [target] [service] [env]
# Examples:
./deploy_unified.sh aws both prod      # Deploy both services to AWS
./deploy_unified.sh ecr api prod       # Deploy API via ECR (for large dependencies)
./deploy_unified.sh local both         # Local Docker deployment

# Manual SAM deployment
sam build -t lambda-api-light.yml
sam deploy --stack-name satei-api-light --resolve-s3

# Check deployment status
aws cloudformation describe-stacks --stack-name satei-api-light
```

### Testing
```bash
# Run Django tests
cd valuation-app
python manage.py test valuation

# Run API endpoint tests
curl -X POST https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","district":"恵比寿","land_area":100,"building_area":80,"building_age":10}'

# Test local API
curl http://localhost:8000/api/valuation -X POST -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","district":"六本木","land_area":150,"building_area":120,"building_age":5}'
```

## High-Level Architecture

### Service Dependencies
The Django frontend depends on the FastAPI backend being available at the URL specified in `VALUATION_API_URL`. The deployment order matters:
1. Deploy FastAPI first to get its URL
2. Deploy Django with the FastAPI URL as a parameter

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
The FastAPI service intelligently handles ML dependencies:
```python
try:
    from models.valuation_model import ValuationModel  # Full ML model
except ImportError:
    from models.lightweight_model import LightweightValuationModel  # Rule-based fallback
```

This allows the same codebase to work in both local (with ML libraries) and Lambda (without ML libraries) environments.

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