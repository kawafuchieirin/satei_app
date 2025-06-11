#!/bin/bash

# 不動産査定モデル管理スクリプト
# 使用方法: ./manage_model.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$SCRIPT_DIR/api"
API_URL="http://localhost:3001"

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE} 不動産査定モデル管理ツール${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_usage() {
    echo "使用方法: $0 [command] [options]"
    echo ""
    echo "コマンド:"
    echo "  train       - 新しいモデルを学習"
    echo "  info        - 現在のモデル情報を表示"
    echo "  evaluate    - モデルの精度評価"
    echo "  test        - モデルのテスト実行"
    echo "  backup      - モデルファイルのバックアップ"
    echo "  restore     - モデルファイルの復元"
    echo "  clean       - 古いモデルファイルを削除"
    echo ""
    echo "学習オプション:"
    echo "  --model-type [rf|gb|linear]  - モデルタイプ (デフォルト: rf)"
    echo "  --data-source [mlit|sample|csv] - データソース (デフォルト: sample)"
    echo ""
    echo "例:"
    echo "  $0 train --model-type rf --data-source sample"
    echo "  $0 evaluate"
    echo "  $0 info"
}

check_api_status() {
    if curl -s "$API_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API サーバーが起動しています${NC}"
        return 0
    else
        echo -e "${RED}❌ API サーバーに接続できません${NC}"
        echo "docker-compose up でサーバーを起動してください"
        return 1
    fi
}

train_model() {
    local model_type="rf"
    local data_source="sample"
    
    # パラメータの解析
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-type)
                model_type="$2"
                shift 2
                ;;
            --data-source)
                data_source="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}未知のオプション: $1${NC}"
                return 1
                ;;
        esac
    done
    
    echo -e "${YELLOW}🏗️  モデル学習を開始します...${NC}"
    echo "  モデルタイプ: $model_type"
    echo "  データソース: $data_source"
    echo ""
    
    cd "$API_DIR"
    
    # 学習実行
    if python train_model.py --model-type "$model_type" --data-source "$data_source" --evaluate; then
        echo -e "${GREEN}✅ モデル学習が完了しました${NC}"
        
        # APIサーバーが起動している場合は再起動を促す
        if check_api_status; then
            echo -e "${YELLOW}⚠️  新しいモデルを使用するには API サーバーの再起動が必要です${NC}"
            echo "docker-compose restart api"
        fi
    else
        echo -e "${RED}❌ モデル学習に失敗しました${NC}"
        return 1
    fi
}

show_model_info() {
    echo -e "${YELLOW}📊 モデル情報を取得中...${NC}"
    
    if check_api_status; then
        # API経由で情報取得
        echo ""
        echo -e "${BLUE}=== API経由の情報 ===${NC}"
        curl -s "$API_URL/api/model/info" | python3 -m json.tool
    fi
    
    # ローカルファイルの確認
    echo ""
    echo -e "${BLUE}=== ローカルファイル ===${NC}"
    cd "$API_DIR"
    
    if [ -f "valuation_model.joblib" ]; then
        echo -e "${GREEN}✅ valuation_model.joblib${NC} ($(stat -f%z valuation_model.joblib 2>/dev/null || stat -c%s valuation_model.joblib 2>/dev/null) bytes)"
    else
        echo -e "${RED}❌ valuation_model.joblib${NC}"
    fi
    
    if [ -f "label_encoders.joblib" ]; then
        echo -e "${GREEN}✅ label_encoders.joblib${NC} ($(stat -f%z label_encoders.joblib 2>/dev/null || stat -c%s label_encoders.joblib 2>/dev/null) bytes)"
    else
        echo -e "${RED}❌ label_encoders.joblib${NC}"
    fi
    
    # 学習履歴ファイル
    echo ""
    echo -e "${BLUE}=== 学習履歴 ===${NC}"
    for file in training_info_*.json; do
        if [ -f "$file" ]; then
            echo "📄 $file ($(date -r "$file" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -d "@$(stat -c%Y "$file" 2>/dev/null)" '+%Y-%m-%d %H:%M:%S' 2>/dev/null))"
        fi
    done
}

evaluate_model() {
    echo -e "${YELLOW}🔬 モデル評価を実行中...${NC}"
    
    if check_api_status; then
        echo ""
        echo -e "${BLUE}=== API経由の評価 ===${NC}"
        curl -s "$API_URL/api/model/evaluate" | python3 -m json.tool
        
        echo ""
        echo -e "${BLUE}=== 交差検証 ===${NC}"
        curl -s "$API_URL/api/model/cross-validate" | python3 -m json.tool
    fi
    
    # ローカルでの詳細評価
    echo ""
    echo -e "${BLUE}=== ローカル評価 ===${NC}"
    cd "$SCRIPT_DIR"
    python test_model_accuracy.py
}

test_model() {
    echo -e "${YELLOW}🧪 モデルテストを実行中...${NC}"
    
    cd "$SCRIPT_DIR"
    python test_model_accuracy.py
}

backup_model() {
    local backup_dir="$SCRIPT_DIR/model_backups"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="$backup_dir/backup_$timestamp"
    
    echo -e "${YELLOW}💾 モデルファイルをバックアップ中...${NC}"
    
    mkdir -p "$backup_path"
    cd "$API_DIR"
    
    # 現在のモデルファイルをバックアップ
    for file in valuation_model.joblib label_encoders.joblib scaler.joblib training_info_*.json; do
        if [ -f "$file" ]; then
            cp "$file" "$backup_path/"
            echo "✅ $file をバックアップしました"
        fi
    done
    
    echo -e "${GREEN}✅ バックアップ完了: $backup_path${NC}"
}

restore_model() {
    local backup_dir="$SCRIPT_DIR/model_backups"
    
    echo -e "${YELLOW}📁 利用可能なバックアップ:${NC}"
    if [ -d "$backup_dir" ]; then
        ls -la "$backup_dir" | grep "backup_"
    else
        echo "バックアップが見つかりません"
        return 1
    fi
    
    echo ""
    read -p "復元するバックアップのタイムスタンプを入力してください (例: 20241212_143000): " timestamp
    
    local restore_path="$backup_dir/backup_$timestamp"
    if [ ! -d "$restore_path" ]; then
        echo -e "${RED}❌ バックアップが見つかりません: $restore_path${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}🔄 モデルファイルを復元中...${NC}"
    cd "$API_DIR"
    
    # バックアップからファイルを復元
    for file in "$restore_path"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            cp "$file" "./$filename"
            echo "✅ $filename を復元しました"
        fi
    done
    
    echo -e "${GREEN}✅ 復元完了${NC}"
    echo -e "${YELLOW}⚠️  新しいモデルを使用するには API サーバーの再起動が必要です${NC}"
}

clean_old_models() {
    echo -e "${YELLOW}🧹 古いモデルファイルを削除中...${NC}"
    
    cd "$API_DIR"
    
    # タイムスタンプ付きファイルを削除（現在のファイルは保持）
    for pattern in "valuation_model_*_*.joblib" "label_encoders_*_*.joblib" "scaler_*_*.joblib" "training_info_*_*.json"; do
        for file in $pattern; do
            if [ -f "$file" ]; then
                rm "$file"
                echo "🗑️  $file を削除しました"
            fi
        done
    done
    
    echo -e "${GREEN}✅ クリーンアップ完了${NC}"
}

# メイン処理
main() {
    print_header
    
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    local command="$1"
    shift
    
    case "$command" in
        train)
            train_model "$@"
            ;;
        info)
            show_model_info
            ;;
        evaluate)
            evaluate_model
            ;;
        test)
            test_model
            ;;
        backup)
            backup_model
            ;;
        restore)
            restore_model
            ;;
        clean)
            clean_old_models
            ;;
        *)
            echo -e "${RED}未知のコマンド: $command${NC}"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"