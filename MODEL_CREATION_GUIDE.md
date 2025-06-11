# 不動産査定APIモデル作成ガイド

## 📋 概要

本ガイドでは、不動産査定APIで使用する機械学習モデルの作成方法について説明します。

## 🎯 モデル作成の流れ

```
データ取得 → 前処理 → モデル学習 → 評価 → 保存 → デプロイ
```

## 🛠️ 方法1: 学習スクリプトを使用（推奨）

### 基本的な使用方法

```bash
# 1. API ディレクトリに移動
cd api

# 2. サンプルデータでRandom Forestモデルを学習
python train_model.py --model-type rf --data-source sample

# 3. 国土交通省APIデータで学習（APIキーが必要）
python train_model.py --model-type rf --data-source mlit

# 4. 評価も同時に実行
python train_model.py --model-type rf --data-source sample --evaluate
```

### 利用可能なオプション

#### モデルタイプ (`--model-type`)
- `rf`: Random Forest（推奨）
- `gb`: Gradient Boosting
- `linear`: 線形回帰

#### データソース (`--data-source`)
- `sample`: サンプルデータ（1000件）
- `mlit`: 国土交通省API
- `csv`: CSVファイル

#### その他のオプション
- `--output-dir`: 出力ディレクトリ
- `--evaluate`: 学習後に評価を実行

### 実行例

```bash
# Random Forest + サンプルデータ
python train_model.py --model-type rf --data-source sample --evaluate

# Gradient Boosting + MLIT データ
python train_model.py --model-type gb --data-source mlit

# 線形回帰 + カスタムディレクトリ
python train_model.py --model-type linear --output-dir ./models
```

## 🔧 方法2: 管理スクリプトを使用

### 使用方法

```bash
# 管理スクリプトを実行可能にする
chmod +x manage_model.sh

# モデル学習
./manage_model.sh train --model-type rf --data-source sample

# モデル情報表示
./manage_model.sh info

# モデル評価
./manage_model.sh evaluate

# 簡易テスト
./manage_model.sh test
```

### 利用可能なコマンド

| コマンド | 説明 |
|----------|------|
| `train` | 新しいモデルを学習 |
| `info` | 現在のモデル情報を表示 |
| `evaluate` | モデルの精度評価 |
| `test` | モデルのテスト実行 |
| `backup` | モデルファイルのバックアップ |
| `restore` | モデルファイルの復元 |
| `clean` | 古いモデルファイルを削除 |

## 🎯 方法3: APIエンドポイント経由

### 学習開始

```bash
# Random Forest モデルを学習
curl -X POST "http://localhost:3001/api/model/retrain?model_type=rf&data_source=sample"
```

### モデル情報取得

```bash
# 現在のモデル情報を取得
curl "http://localhost:3001/api/model/info" | jq
```

## 📊 データについて

### 使用する特徴量

1. **都道府県** - カテゴリカル変数
2. **市区町村** - カテゴリカル変数  
3. **地区** - カテゴリカル変数
4. **土地面積** - 数値変数（㎡）
5. **建物面積** - 数値変数（㎡）
6. **築年数** - 数値変数（年）

### ターゲット変数

- **取引価格** - 回帰予測対象（円）

### データソースの詳細

#### 1. サンプルデータ (`sample`)
- 件数: 1,000件
- 内容: 人工的に生成されたデータ
- 用途: 開発・テスト用

#### 2. 国土交通省API (`mlit`)
- 件数: 可変（API制限あり）
- 内容: 実際の不動産取引データ
- 用途: 本番用
- 注意: APIキーまたは認証が必要

#### 3. CSVファイル (`csv`)
- ファイル名: `training_data.csv`
- 場所: `api/` ディレクトリ
- 用途: カスタムデータ

## 🔍 モデル評価

### 評価指標

1. **MAE (Mean Absolute Error)**: 平均絶対誤差
2. **R² Score**: 決定係数
3. **交差検証スコア**: 5分割交差検証

### 評価実行

```bash
# 詳細評価
python api/evaluate_model.py

# 簡易テスト
python test_model_accuracy.py

# API経由での評価
curl "http://localhost:3001/api/model/evaluate"
```

## 📁 生成されるファイル

### 本番用ファイル
- `valuation_model.joblib` - メインモデル
- `label_encoders.joblib` - カテゴリカル変数エンコーダー
- `scaler.joblib` - 数値変数スケーラー（線形回帰の場合）

### 履歴ファイル
- `valuation_model_rf_20241212_143000.joblib` - タイムスタンプ付きモデル
- `training_info_rf_20241212_143000.json` - 学習情報

### 学習情報の例

```json
{
  "model_type": "rf",
  "training_date": "2024-12-12T14:30:00",
  "data_size": 1000,
  "train_size": 800,
  "test_size": 200,
  "metrics": {
    "test_mae": 16919357.30,
    "test_r2": 0.515,
    "cv_r2_mean": 0.487,
    "cv_r2_std": 0.052
  }
}
```

## 🚀 デプロイ手順

### 1. 開発環境での学習

```bash
# モデル学習
./manage_model.sh train --model-type rf --data-source sample

# 評価確認
./manage_model.sh evaluate
```

### 2. ファイルの確認

```bash
# 生成されたファイルを確認
ls -la api/*.joblib api/*.json
```

### 3. APIサーバーの再起動

```bash
# Dockerコンテナの場合
docker-compose restart api

# ローカル実行の場合
# サーバーを停止して再起動
```

### 4. 動作確認

```bash
# ヘルスチェック
curl http://localhost:3001/health

# テスト予測
curl -X POST http://localhost:3001/api/valuation \
  -H "Content-Type: application/json" \
  -d '{
    "prefecture": "東京都",
    "city": "渋谷区", 
    "district": "恵比寿",
    "land_area": 100.0,
    "building_area": 80.0,
    "building_age": 10
  }'
```

## ⚠️ 注意事項

### パフォーマンス
- Random Forest: 高精度、やや重い
- Gradient Boosting: 非常に高精度、重い
- 線形回帰: 軽量、精度は劣る

### データ品質
- 異常値は自動で除去されます
- 欠損値がある場合は該当行が除外されます

### バックアップ
- 重要なモデルは必ずバックアップを作成
- `./manage_model.sh backup` でバックアップ可能

### セキュリティ
- 本番環境では適切な認証を設定
- モデルファイルへのアクセス制限を設定

## 🔧 トラブルシューティング

### よくある問題

#### 1. MLIT APIに接続できない
```
解決策:
- インターネット接続を確認
- APIキーの設定を確認  
- サンプルデータで代替
```

#### 2. メモリ不足エラー
```
解決策:
- データサイズを縮小
- より軽量なモデル（線形回帰）を使用
- マシンのメモリを増設
```

#### 3. モデルの精度が低い
```
解決策:
- より多くのデータを使用
- 特徴量エンジニアリングを実施
- ハイパーパラメータ調整
```

#### 4. APIサーバーでモデルが読み込まれない
```
解決策:
- ファイルパスを確認
- ファイル権限を確認
- サーバーを再起動
```

## 📚 参考情報

### ドキュメント
- [scikit-learn公式ドキュメント](https://scikit-learn.org/)
- [国土交通省 不動産取引価格情報API](https://www.reinfolib.mlit.go.jp/help/apiManual/)

### 関連ファイル
- `api/models/valuation_model.py` - モデル実装
- `api/models/data_fetcher.py` - データ取得
- `api/models/model_evaluator.py` - モデル評価
- `test_model_accuracy.py` - 簡易テスト

---

## 🆘 サポート

問題が発生した場合は、以下を確認してください：

1. ログファイルの確認
2. 依存関係のインストール状況
3. ファイル権限の確認
4. ディスク容量の確認

詳細なエラー情報とともにサポートまでお問い合わせください。