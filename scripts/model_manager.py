#!/usr/bin/env python3
"""
統合モデル管理スクリプト

全てのモデル作成・管理機能を統合したメインスクリプトです。

使用方法:
python model_manager.py [command] [options]

コマンド:
- quick: クイックモデル作成
- create: 詳細モデル作成
- batch: バッチ学習
- evaluate: モデル評価
- compare: モデル比較
- deploy: モデルデプロイ
"""

import argparse
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelManager:
    """
    統合モデル管理クラス
    """
    
    def __init__(self, scripts_dir=None, api_dir=None):
        self.scripts_dir = Path(scripts_dir or Path(__file__).parent)
        self.api_dir = Path(api_dir or Path(__file__).parent.parent / 'api')
        
        # スクリプトパス
        self.scripts = {
            'quick': self.scripts_dir / 'quick_model.py',
            'create': self.scripts_dir / 'create_model.py',
            'batch': self.scripts_dir / 'batch_model_training.py'
        }
    
    def print_header(self, title):
        """
        ヘッダー表示
        """
        print("\n" + "=" * 70)
        print(f"🏠 {title}")
        print("=" * 70)
    
    def quick_create(self, args):
        """
        クイックモデル作成
        """
        self.print_header("クイックモデル作成")
        
        preset = getattr(args, 'preset', 'balanced')
        cmd = [sys.executable, str(self.scripts['quick']), '--preset', preset]
        
        if hasattr(args, 'output_dir') and args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"クイックモデル作成に失敗しました: {e}")
            return False
    
    def detailed_create(self, args):
        """
        詳細モデル作成
        """
        self.print_header("詳細モデル作成")
        
        cmd = [sys.executable, str(self.scripts['create'])]
        
        # オプション追加
        if hasattr(args, 'data_source'):
            cmd.extend(['--data-source', args.data_source])
        if hasattr(args, 'output_dir') and args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        if hasattr(args, 'no_grid_search') and args.no_grid_search:
            cmd.append('--no-grid-search')
        if hasattr(args, 'cv_folds'):
            cmd.extend(['--cv-folds', str(args.cv_folds)])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"詳細モデル作成に失敗しました: {e}")
            return False
    
    def batch_training(self, args):
        """
        バッチ学習
        """
        self.print_header("バッチ学習")
        
        cmd = [sys.executable, str(self.scripts['batch'])]
        
        # オプション追加
        if hasattr(args, 'data_source'):
            cmd.extend(['--data-source', args.data_source])
        if hasattr(args, 'output_dir') and args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        if hasattr(args, 'max_workers'):
            cmd.extend(['--max-workers', str(args.max_workers)])
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"バッチ学習に失敗しました: {e}")
            return False
    
    def evaluate_models(self, args):
        """
        モデル評価
        """
        self.print_header("モデル評価")
        
        # 管理スクリプトを使用
        manage_script = Path(__file__).parent.parent / 'manage_model.sh'
        
        if manage_script.exists():
            try:
                subprocess.run([str(manage_script), 'evaluate'], check=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"評価に失敗しました: {e}")
        
        # フォールバック: 直接テストスクリプトを実行
        test_script = Path(__file__).parent.parent / 'test_model_accuracy.py'
        if test_script.exists():
            try:
                subprocess.run([sys.executable, str(test_script)], check=True)
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"テストに失敗しました: {e}")
        
        return False
    
    def compare_models(self, args):
        """
        モデル比較
        """
        self.print_header("モデル比較")
        
        # 学習履歴ファイルを検索
        info_files = list(self.api_dir.glob('*training_info*.json'))
        batch_files = list(self.api_dir.glob('*batch_training_results*.json'))
        quick_files = list(self.api_dir.glob('*quick_model_info*.json'))
        
        all_files = info_files + batch_files + quick_files
        
        if not all_files:
            print("❌ 比較可能なモデル情報が見つかりません。")
            print("まずモデルを作成してください。")
            return False
        
        print(f"📊 {len(all_files)} 個のモデル情報を発見しました。")
        print()
        
        models_data = []
        
        for file_path in sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # ファイルタイプによって情報抽出方法を変更
                if 'batch_training' in file_path.name:
                    best_model = data.get('best_model', {})
                    models_data.append({
                        'file': file_path.name,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'type': 'Batch',
                        'model_name': best_model.get('name', 'Unknown'),
                        'test_r2': best_model.get('metrics', {}).get('test_r2', 0),
                        'test_mae': best_model.get('metrics', {}).get('test_mae', 0),
                        'data_size': data.get('data_info', {}).get('total_samples', 0)
                    })
                elif 'quick_model' in file_path.name:
                    models_data.append({
                        'file': file_path.name,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'type': 'Quick',
                        'model_name': data.get('preset', 'Unknown'),
                        'test_r2': data.get('metrics', {}).get('test_r2', 0),
                        'test_mae': data.get('metrics', {}).get('test_mae', 0),
                        'data_size': data.get('data_info', {}).get('total_samples', 0)
                    })
                else:
                    # 通常の学習情報
                    models_data.append({
                        'file': file_path.name,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'type': 'Detailed',
                        'model_name': data.get('best_model', 'Unknown'),
                        'test_r2': data.get('metrics', {}).get('test_r2', 0),
                        'test_mae': data.get('metrics', {}).get('test_mae', 0),
                        'data_size': data.get('data_size', 0)
                    })
                
            except Exception as e:
                logger.warning(f"ファイル読み込みエラー {file_path}: {e}")
                continue
        
        if not models_data:
            print("❌ 有効なモデル情報が見つかりません。")
            return False
        
        # 結果表示
        print("📈 モデル比較結果:")
        print("-" * 100)
        print(f"{'タイムスタンプ':<15} {'タイプ':<8} {'モデル名':<20} {'Test R²':<10} {'Test MAE':<15} {'データ数':<10}")
        print("-" * 100)
        
        for model in sorted(models_data, key=lambda x: x['test_r2'], reverse=True):
            timestamp = model['timestamp'][:10] if len(model['timestamp']) > 10 else model['timestamp']
            print(f"{timestamp:<15} {model['type']:<8} {model['model_name']:<20} "
                  f"{model['test_r2']:<10.3f} ¥{model['test_mae']:<13,.0f} {model['data_size']:<10,}")
        
        # 最優秀モデル
        best_model = max(models_data, key=lambda x: x['test_r2'])
        print()
        print(f"🏆 最優秀モデル: {best_model['model_name']} (R² = {best_model['test_r2']:.3f})")
        
        return True
    
    def deploy_model(self, args):
        """
        モデルデプロイ
        """
        self.print_header("モデルデプロイ")
        
        # 現在のモデルファイルを確認
        model_file = self.api_dir / 'valuation_model.joblib'
        encoders_file = self.api_dir / 'label_encoders.joblib'
        
        if not model_file.exists():
            print("❌ デプロイ対象のモデルファイルが見つかりません。")
            print("まずモデルを作成してください。")
            return False
        
        print("📦 デプロイ可能なファイル:")
        print(f"  ✅ {model_file.name} ({model_file.stat().st_size:,} bytes)")
        
        if encoders_file.exists():
            print(f"  ✅ {encoders_file.name} ({encoders_file.stat().st_size:,} bytes)")
        else:
            print(f"  ⚠️ {encoders_file.name} (見つかりません)")
        
        print()
        print("🚀 デプロイ手順:")
        print("1. docker-compose restart api  # APIサーバー再起動")
        print("2. モデルテストの実行:")
        print("   python test_model_accuracy.py")
        print("3. 本番環境への展開:")
        print("   ./deploy/ecr-deploy.sh  # ECRデプロイ")
        
        # APIサーバー再起動の確認
        restart = input("\nAPIサーバーを再起動しますか？ (y/n): ").lower().strip()
        if restart == 'y':
            try:
                subprocess.run(['docker-compose', 'restart', 'api'], 
                             cwd=Path(__file__).parent.parent, check=True)
                print("✅ APIサーバーが再起動されました。")
                
                # 簡易テスト
                import time
                print("🔄 モデル読み込みを待機中...")
                time.sleep(5)
                
                test_script = Path(__file__).parent.parent / 'test_model_accuracy.py'
                if test_script.exists():
                    subprocess.run([sys.executable, str(test_script)], check=False)
                
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"APIサーバー再起動に失敗しました: {e}")
                return False
        
        return True
    
    def show_status(self):
        """
        現在の状態表示
        """
        self.print_header("モデル管理状態")
        
        # モデルファイルの確認
        model_file = self.api_dir / 'valuation_model.joblib'
        encoders_file = self.api_dir / 'label_encoders.joblib'
        
        print("📁 モデルファイル状態:")
        if model_file.exists():
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            print(f"  ✅ valuation_model.joblib ({model_file.stat().st_size:,} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print(f"  ❌ valuation_model.joblib (見つかりません)")
        
        if encoders_file.exists():
            mtime = datetime.fromtimestamp(encoders_file.stat().st_mtime)
            print(f"  ✅ label_encoders.joblib ({encoders_file.stat().st_size:,} bytes, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print(f"  ❌ label_encoders.joblib (見つかりません)")
        
        # 学習履歴
        info_files = list(self.api_dir.glob('*training_info*.json'))
        info_files += list(self.api_dir.glob('*batch_training_results*.json'))
        info_files += list(self.api_dir.glob('*quick_model_info*.json'))
        
        print(f"\n📊 学習履歴: {len(info_files)} 件")
        for file_path in sorted(info_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"  📄 {file_path.name} ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        
        # APIサーバー状態
        print(f"\n🌐 APIサーバー状態:")
        try:
            import requests
            response = requests.get('http://localhost:3001/health', timeout=3)
            if response.status_code == 200:
                print("  ✅ APIサーバー稼働中")
                
                # モデル情報取得試行
                try:
                    model_info = requests.get('http://localhost:3001/api/model/info', timeout=3)
                    if model_info.status_code == 200:
                        info_data = model_info.json()
                        print(f"  📊 モデル読み込み済み: {info_data.get('model_loaded', False)}")
                    else:
                        print("  ⚠️ モデル情報取得エラー")
                except:
                    print("  ⚠️ モデル情報取得失敗")
            else:
                print("  ❌ APIサーバー応答エラー")
        except:
            print("  ❌ APIサーバー接続失敗")
        
        # 推奨アクション
        print(f"\n💡 推奨アクション:")
        if not model_file.exists():
            print("  1. モデルを作成してください: python model_manager.py quick")
        else:
            print("  1. モデル評価: python model_manager.py evaluate")
            print("  2. モデル比較: python model_manager.py compare")
            print("  3. 新しいモデル作成: python model_manager.py create")


def main():
    parser = argparse.ArgumentParser(description='統合モデル管理スクリプト')
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # クイック作成
    quick_parser = subparsers.add_parser('quick', help='クイックモデル作成')
    quick_parser.add_argument('--preset', choices=['fast', 'balanced', 'best'], 
                             default='balanced', help='学習プリセット')
    quick_parser.add_argument('--output-dir', help='出力ディレクトリ')
    
    # 詳細作成
    create_parser = subparsers.add_parser('create', help='詳細モデル作成')
    create_parser.add_argument('--data-source', choices=['sample', 'mlit', 'csv'], 
                              default='sample', help='データソース')
    create_parser.add_argument('--output-dir', help='出力ディレクトリ')
    create_parser.add_argument('--no-grid-search', action='store_true',
                              help='ハイパーパラメータ調整をスキップ')
    create_parser.add_argument('--cv-folds', type=int, default=5,
                              help='交差検証の分割数')
    
    # バッチ学習
    batch_parser = subparsers.add_parser('batch', help='バッチ学習')
    batch_parser.add_argument('--data-source', choices=['sample', 'mlit', 'csv'], 
                             default='sample', help='データソース')
    batch_parser.add_argument('--output-dir', help='出力ディレクトリ')
    batch_parser.add_argument('--max-workers', type=int, help='最大ワーカー数')
    
    # 評価
    subparsers.add_parser('evaluate', help='モデル評価')
    
    # 比較
    subparsers.add_parser('compare', help='モデル比較')
    
    # デプロイ
    subparsers.add_parser('deploy', help='モデルデプロイ')
    
    # 状態表示
    subparsers.add_parser('status', help='現在の状態表示')
    
    args = parser.parse_args()
    
    if not args.command:
        # コマンドが指定されていない場合は状態表示
        args.command = 'status'
    
    # モデル管理器の初期化
    manager = ModelManager()
    
    # コマンド実行
    if args.command == 'quick':
        success = manager.quick_create(args)
    elif args.command == 'create':
        success = manager.detailed_create(args)
    elif args.command == 'batch':
        success = manager.batch_training(args)
    elif args.command == 'evaluate':
        success = manager.evaluate_models(args)
    elif args.command == 'compare':
        success = manager.compare_models(args)
    elif args.command == 'deploy':
        success = manager.deploy_model(args)
    elif args.command == 'status':
        manager.show_status()
        success = True
    else:
        print(f"未知のコマンド: {args.command}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()