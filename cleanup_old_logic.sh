#!/bin/bash

# 旧ルールベースロジックのクリーンアップスクリプト

echo "🧹 Starting cleanup of old rule-based logic..."

# バックアップディレクトリの作成
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "📁 Created backup directory: $BACKUP_DIR"

# 1. valuation-api-ml内の旧ファイルをバックアップ
echo "📦 Backing up old files from valuation-api-ml..."

# 旧valuation_model.pyをバックアップ（フォールバックロジックを含む）
if [ -f "valuation-api-ml/models/valuation_model.py" ]; then
    cp valuation-api-ml/models/valuation_model.py $BACKUP_DIR/
    echo "  ✓ Backed up valuation_model.py"
fi

# mock_valuation_model.pyをバックアップ（必要ない場合）
if [ -f "valuation-api-ml/models/mock_valuation_model.py" ]; then
    cp valuation-api-ml/models/mock_valuation_model.py $BACKUP_DIR/
    echo "  ✓ Backed up mock_valuation_model.py"
fi

# 2. model-creation内の古いスクリプトをバックアップ
echo "📦 Backing up old scripts from model-creation..."

# 旧スクリプトのバックアップ
OLD_SCRIPTS=(
    "model-creation/create_model.py"
    "model-creation/quick_model.py"
    "model-creation/simple_model_trainer.py"
)

for script in "${OLD_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        cp "$script" $BACKUP_DIR/
        echo "  ✓ Backed up $(basename $script)"
    fi
done

# 3. 新しいML専用ファイルへの置き換え
echo "🔄 Replacing old files with ML-only versions..."

# valuation_model_ml_only.pyを正式なvaluation_model.pyとして使用
if [ -f "valuation-api-ml/models/valuation_model_ml_only.py" ]; then
    mv valuation-api-ml/models/valuation_model.py $BACKUP_DIR/valuation_model_old.py 2>/dev/null
    cp valuation-api-ml/models/valuation_model_ml_only.py valuation-api-ml/models/valuation_model.py
    echo "  ✓ Replaced valuation_model.py with ML-only version"
fi

# main_ml_new.pyを正式なmain_ml.pyとして使用
if [ -f "valuation-api-ml/main_ml_new.py" ]; then
    mv valuation-api-ml/main_ml.py $BACKUP_DIR/main_ml_old.py 2>/dev/null
    cp valuation-api-ml/main_ml_new.py valuation-api-ml/main_ml.py
    echo "  ✓ Replaced main_ml.py with new ML version"
fi

# 4. 不要なファイルの削除（オプション）
echo "🗑️  Cleaning up unnecessary files..."

# モックモデルの削除（本番環境では不要）
if [ -f "valuation-api-ml/models/mock_valuation_model.py" ]; then
    rm valuation-api-ml/models/mock_valuation_model.py
    echo "  ✓ Removed mock_valuation_model.py"
fi

# 5. デプロイメント設定の更新
echo "📝 Updating deployment configurations..."

# 実行権限の付与
chmod +x deployment/ecr-deploy-ml.sh
echo "  ✓ Made ecr-deploy-ml.sh executable"

# 6. サマリー
echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📋 Summary:"
echo "  - Backup directory: $BACKUP_DIR"
echo "  - Updated valuation_model.py to ML-only version"
echo "  - Updated main_ml.py with new ML API"
echo "  - Created new deployment scripts for ML version"
echo ""
echo "📌 Next steps:"
echo "  1. Train ML models: cd model-creation && python train_xgboost_model.py"
echo "  2. Fetch MLIT data: cd model-creation && python tokyo23_data_fetcher.py"
echo "  3. Deploy ML API: cd deployment && ./ecr-deploy-ml.sh api prod"
echo ""
echo "⚠️  Note: Old files are backed up in $BACKUP_DIR"