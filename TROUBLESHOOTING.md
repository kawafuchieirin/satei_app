# トラブルシューティングガイド

このドキュメントは、不動産査定アプリケーションの開発・デプロイ時によく発生する問題と解決方法をまとめています。

## 目次
- [API関連のエラー](#api関連のエラー)
- [デプロイメントエラー](#デプロイメントエラー)
- [査定結果の表示エラー](#査定結果の表示エラー)
- [Docker/ECR関連のエラー](#dockerecr関連のエラー)
- [予防策とベストプラクティス](#予防策とベストプラクティス)

---

## API関連のエラー

### 問題: 403 Forbidden エラー
**症状**: 
- Webフォームから査定実行時に「API呼び出しに失敗しました。ステータス: 403 - {'message': 'Forbidden'}」

**原因**:
- 外部FastAPI Lambdaが存在しない
- API Gatewayの権限設定
- CSRF保護の競合

**解決方法**:
1. **Django内部処理に切り替え**
```python
# views.py - 外部API呼び出しを内部関数に変更
def valuation_form(request):
    # 変更前: response = requests.post(api_url, ...)
    # 変更後:
    result = calculate_valuation(valuation_data)
```

2. **環境変数で切り替え可能にする**
```python
# settings.py
USE_EXTERNAL_API = os.environ.get('USE_EXTERNAL_API', 'false').lower() == 'true'

# views.py
if settings.USE_EXTERNAL_API:
    response = requests.post(api_url, ...)
else:
    result = calculate_valuation(data)
```

3. **CSRF設定の確認**
```python
# APIエンドポイントに@csrf_exemptを追加
@csrf_exempt
@require_http_methods(["POST"])
def api_valuation(request):
    ...
```

---

## デプロイメントエラー

### 問題: Lambda関数が見つからない
**症状**:
- `aws lambda list-functions`でFastAPI Lambdaが表示されない
- CloudFormationスタックが存在しない

**解決方法**:
1. **デプロイステータスの確認**
```bash
# Lambda関数一覧
aws lambda list-functions --query "Functions[?contains(FunctionName, 'satei')]"

# CloudFormationスタック確認
aws cloudformation list-stacks --stack-status-filter CREATE_COMPLETE UPDATE_COMPLETE
```

2. **正しいデプロイコマンドの実行**
```bash
cd deployment/aws

# 両方デプロイ
./deploy_all.sh both prod

# APIのみ
./deploy_all.sh api prod

# Djangoのみ
./deploy_all.sh django prod
```

### 問題: ECRイメージのプッシュタイムアウト
**症状**:
- `docker push`が394MBでタイムアウト
- "Source image does not exist"エラー

**解決方法**:
1. **ECRログイン再実行**
```bash
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin 412420079063.dkr.ecr.ap-northeast-1.amazonaws.com
```

2. **軽量版の使用を検討**
```bash
# 軽量版APIをデプロイ
sam build -t lambda-api.yml
sam deploy --stack-name satei-api-light --resolve-s3
```

3. **Lambda Layerの活用**
- 大きな依存関係をLayerに分離
- メインのLambda関数を軽量化

---

## 査定結果の表示エラー

### 問題: 査定結果が0万円と表示される
**症状**:
- APIは正常に11,840万円を返すが、画面では0万円

**原因**:
- 価格の二重変換（万円 → 万円）
- テンプレートフィルターの誤動作

**解決方法**:
1. **価格フィルターの修正**
```python
# price_filters.py
@register.filter
def format_price_yen(value):
    """既に万円単位の値をフォーマット"""
    price = float(value)
    # 既に万円単位なので、カンマ区切りのみ追加
    man_yen = round(price)
    return f"{man_yen:,}万円"
```

2. **デバッグ情報の追加**
```python
# views.py
print(f"DEBUG - result: {result}")
print(f"DEBUG - estimated_price: {result['estimated_price']}")
```

---

## Docker/ECR関連のエラー

### 問題: Dockerイメージサイズが大きすぎる
**症状**:
- 394MB以上のイメージ
- Lambda制限（10GB）に近い

**解決策**:
1. **マルチステージビルドの使用**
```dockerfile
# Dockerfile.lambda.ml
FROM python:3.11-slim as builder
COPY requirements-ml.txt .
RUN pip install --target /app -r requirements-ml.txt

FROM public.ecr.aws/lambda/python:3.11
COPY --from=builder /app /opt/python
```

2. **不要なファイルの削除**
```dockerfile
# .dockerignore
*.pyc
__pycache__
.git
tests/
docs/
```

---

## 予防策とベストプラクティス

### 1. 段階的デプロイ戦略
```yaml
# Phase 1: Django内部計算のみ
VALUATION_MODE: internal

# Phase 2: 軽量FastAPI追加
VALUATION_MODE: external_light

# Phase 3: ML版FastAPI
VALUATION_MODE: external_ml
```

### 2. ローカルテストの強化
```bash
# Dockerでローカルテスト
cd deployment/local
docker-compose up --build

# APIテスト
curl http://localhost:8001/api/valuation -X POST \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"新宿区","land_area":100,"building_area":80,"building_age":5}'
```

### 3. エラーハンドリングの実装
```python
def safe_valuation(data):
    """エラーに強い査定処理"""
    try:
        # 外部API試行
        if settings.USE_EXTERNAL_API:
            return call_external_api(data)
    except Exception as e:
        logger.warning(f"External API failed: {e}")
    
    try:
        # 内部計算にフォールバック
        return calculate_valuation(data)
    except Exception as e:
        logger.error(f"Internal calculation failed: {e}")
        # デフォルト値を返す
        return {
            'estimated_price': 0,
            'confidence': 0,
            'error': '査定処理中にエラーが発生しました'
        }
```

### 4. ヘルスチェックの実装
```python
# views.py
def health_check(request):
    """システム状態の確認"""
    return JsonResponse({
        'status': 'healthy',
        'mode': settings.get('VALUATION_MODE', 'internal'),
        'ml_available': check_ml_available(),
        'timestamp': datetime.now().isoformat()
    })
```

### 5. CLAUDE.mdの定期更新
- 新しいエラーパターンを発見したら追記
- 解決方法を具体的に記載
- コマンド例を含める

---

## よくある質問（FAQ）

**Q: FastAPIで計算したいが、デプロイが失敗する**
A: まずDjango内部計算で動作確認し、その後段階的にFastAPIを追加することを推奨

**Q: MLモデルを使いたいが、Lambdaサイズ制限に引っかかる**
A: 
1. Lambda Layerを使用
2. EFSマウントでモデルファイルを外部化
3. SageMakerエンドポイントの利用を検討

**Q: 開発環境と本番環境で挙動が異なる**
A: 
1. 環境変数の確認（`VALUATION_MODE`, `VALUATION_API_URL`）
2. Docker Composeで本番同等環境を構築
3. `sam local start-api`でLambda環境をエミュレート

---

## 更新履歴
- 2024-06-15: 初版作成
- 403 Forbiddenエラーの解決方法追加
- 0万円表示問題の解決方法追加
- ECRデプロイタイムアウトの対処法追加