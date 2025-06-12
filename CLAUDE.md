# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Application Architecture

This is a **dual-service real estate valuation application** with Django frontend + FastAPI backend, designed for AWS Lambda deployment with local Docker development support.

### Service Architecture
```
[User Browser] → [Django Frontend] → [FastAPI Backend] → [ML Model + Data]
```

**Key Components:**
- **Django Frontend**: User interface, form handling, API integration (`/`)
- **FastAPI Backend**: ML inference, data processing, API endpoints (`/api/`)
- **ML Model**: Random Forest Regressor for property price prediction
- **Data Source**: MLIT (National Land Infrastructure) API + mock data fallback

### Service Communication
- Django makes HTTP requests to FastAPI backend
- **Field mapping required**: `district` ↔ `area`, `building_age` ↔ `age`
- Current production URLs:
  - Django: https://a2evu7tm1a.execute-api.ap-northeast-1.amazonaws.com/dev/
  - FastAPI: https://q118xkklh2.execute-api.ap-northeast-1.amazonaws.com/Prod/

## Common Development Commands

### Local Development
```bash
# Docker Compose (Recommended)
docker-compose up --build
# Frontend: http://localhost:8080, API: http://localhost:8000

# Individual services
python manage.py runserver 0.0.0.0:8080  # Django
cd api && uvicorn main:app --host 0.0.0.0 --port 8000 --reload  # FastAPI
```

### Model Development
```bash
# Model management
./manage_model.sh train --model-type rf --data-source sample
./manage_model.sh evaluate
./manage_model.sh info

# Manual model creation
python scripts/create_model.py --data-source sample
```

### AWS Deployment
```bash
# One-command deployment (ECR Container - Recommended)
./deploy.sh

# Unified deployment script options
cd deploy
./deploy_unified.sh ecr api dev     # ECR Container (solves dependency issues)
./deploy_unified.sh aws both dev    # Traditional ZIP deployment
./deploy_unified.sh local both      # Local Docker

# Manual deployment
sam build -t deploy/lambda-api.yml
sam deploy --template-file deploy/lambda-api.yml --stack-name satei-api-dev
```

## Key Configuration Details

### Django Lambda Configuration (`satei_project/settings.py`)
```python
# Lambda-specific settings
if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
    CSRF_TRUSTED_ORIGINS = ['https://*.execute-api.ap-northeast-1.amazonaws.com']
    FORCE_SCRIPT_NAME = '/dev'  # API Gateway stage path handling
    USE_X_FORWARDED_HOST = True
    USE_X_FORWARDED_PORT = True

# CSRF middleware is completely disabled for Lambda environment
# CSRFViewMiddleware is commented out in MIDDLEWARE list
```

### FastAPI Lambda Integration (`api/lambda_main.py`)
- Uses Mangum ASGI adapter for Lambda compatibility
- 30-second timeout for ML inference
- CORS enabled for cross-origin requests

### Requirements Management
- **Lambda Django**: Uses root `requirements.txt` (minimal, no ML libraries)
- **Lambda API**: `api/requirements.txt` (ML libraries excluded for ZIP deployment)
- **ECR Container**: `api/requirements.txt` (can include full ML stack)
- **Local Development**: Full ML stack available in root `requirements.txt`

## Deployment Strategies

### 1. ECR Container Deployment (Recommended)
- **Purpose**: Solves Lambda 250MB package size limitations
- **Files**: `deploy/lambda-container.yml`, `api/Dockerfile.lambda`, `deploy/ecr-deploy.sh`
- **Command**: `./deploy.sh` or `./deploy_unified.sh ecr api dev`

### 2. Traditional ZIP Deployment
- **Files**: `deploy/lambda-api.yml`, `deploy/lambda-django.yml`
- **Limitation**: ML dependencies must be excluded
- **Command**: `./deploy_unified.sh aws both dev`

### 3. Local Docker Development
- **Files**: `docker-compose.yml`, `deploy/docker-compose.yml`
- **Command**: `docker-compose up --build`

## Critical Development Considerations

### Field Name Mapping (`valuation/views.py`)
Django and FastAPI use different field names - mapping is required:
```python
valuation_data = {
    'area': form.cleaned_data['district'],     # district → area
    'age': form.cleaned_data['building_age']   # building_age → age
}
```

### Lambda Size Limitations
- Lambda deployment package must be < 250MB
- ML libraries (scikit-learn, pandas) excluded from Lambda requirements
- Use ECR container deployment to include full ML stack

### API Gateway Path Handling
- Django uses `FORCE_SCRIPT_NAME = '/dev'` for API Gateway stage routing
- API endpoints expect `/Prod/` or `/dev/` prefixes depending on deployment

### Model Management
- Models saved as joblib files for Lambda deployment
- Comprehensive model lifecycle via `manage_model.sh`
- Cross-validation and performance monitoring built-in
- Mock data fallback when MLIT API unavailable

### Data Source Strategy
- Primary: MLIT API for real estate transaction data
- Fallback: Enhanced mock data with realistic pricing algorithms
- Supports CSV import for custom datasets

## Troubleshooting Common Issues

### **Critical: Django 403 Forbidden - Primary Issue**
The most common deployment issue is CSRF middleware conflicts in Lambda environment:
1. **Immediate fix**: Comment out CSRFViewMiddleware in `MIDDLEWARE` list
2. **Verify**: `FORCE_SCRIPT_NAME` matches deployment stage (`/dev` or `/Prod`)
3. **Test**: Deploy and verify Django app loads without 403 errors

### "mangum module not found" in Lambda
- **Cause**: SAM not properly packaging dependencies
- **Solution**: Use ECR container deployment instead of ZIP

### Django 403 Forbidden Error
- **Cause**: CSRF middleware conflict in Lambda environment
- **Fix**: Ensure CSRFViewMiddleware is completely disabled in `settings.py`
- **Secondary cause**: API Gateway stage path mismatch
- **Fix**: Check `FORCE_SCRIPT_NAME` in `settings.py` matches deployed stage (`/dev` or `/Prod`)

### API Call Failures
- **Cause**: Field name mismatch between Django and FastAPI
- **Fix**: Verify field mapping in `valuation/views.py`

### Form Submission Not Working
- **Check**: CSRF middleware is completely disabled for Lambda
- **Verify**: API URL configuration in Django settings
- **Common fix**: Comment out CSRFViewMiddleware in MIDDLEWARE list

## File Structure Key Points

- **Entry Points**: `lambda_handler.py` (Django), `api/lambda_main.py` (FastAPI)
- **Model Code**: `api/models/` contains ML pipeline, data fetcher, evaluator
- **Templates**: `valuation/templates/` with Bootstrap styling
- **Deployment**: `deploy/` directory contains all AWS deployment configurations
- **Scripts**: `scripts/` for model management and development tools