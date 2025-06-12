#!/usr/bin/env python3
"""
不動産査定モデルの精度検証スクリプト
"""

import json
import sys
import logging
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

from models.model_evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    メイン実行関数
    """
    print("=" * 60)
    print("不動産査定モデル 精度検証")
    print("=" * 60)
    
    try:
        # モデル評価器の初期化
        evaluator = ModelEvaluator()
        
        print("\n1. モデルとデータの読み込み中...")
        evaluator.load_model_and_data()
        print(f"   データ件数: {len(evaluator.data):,}件")
        
        print("\n2. モデル性能評価中...")
        performance = evaluator.evaluate_model_performance()
        
        print("\n3. 交差検証実行中...")
        cv_results = evaluator.cross_validate_model(cv_folds=5)
        
        print("\n4. 予測サンプル生成中...")
        samples = evaluator.generate_prediction_samples(n_samples=5)
        
        # 結果表示
        print_evaluation_results(performance, cv_results, samples)
        
        # 結果をJSONファイルに保存
        save_results_to_file(performance, cv_results, samples)
        
    except Exception as e:
        logger.error(f"評価中にエラーが発生しました: {e}")
        sys.exit(1)


def print_evaluation_results(performance: dict, cv_results: dict, samples: list):
    """
    評価結果の表示
    """
    print("\n" + "=" * 60)
    print("評価結果")
    print("=" * 60)
    
    # 全体的な性能指標
    metrics = performance['overall_metrics']
    print(f"\n📊 全体的な性能指標:")
    print(f"   R² Score (決定係数):     {metrics['r2_score']:.3f}")
    print(f"   MAE (平均絶対誤差):      ¥{metrics['mae']:,.0f}")
    print(f"   RMSE (二乗平均平方根誤差): ¥{metrics['rmse']:,.0f}")
    print(f"   MAPE (平均絶対パーセント誤差): {metrics['mape']:.1f}%")
    
    # 交差検証結果
    print(f"\n🔄 交差検証結果 ({cv_results['cv_folds']}分割):")
    print(f"   R² Score: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    print(f"   MAE:      ¥{cv_results['mae_mean']:,.0f} ± ¥{cv_results['mae_std']:,.0f}")
    
    # 価格帯別精度
    if performance['price_range_accuracy']:
        print(f"\n💰 価格帯別精度:")
        for price_range, accuracy in performance['price_range_accuracy'].items():
            print(f"   {price_range}:")
            print(f"     件数: {accuracy['count']:,}件")
            print(f"     R² Score: {accuracy['r2_score']:.3f}")
            print(f"     MAE: ¥{accuracy['mae']:,.0f}")
            print(f"     MAPE: {accuracy['mape']:.1f}%")
    
    # 特徴量重要度
    if performance['feature_importance']['features']:
        print(f"\n🔍 特徴量重要度 (上位5位):")
        features = performance['feature_importance']['features'][:5]
        importance = performance['feature_importance']['importance'][:5]
        for i, (feature, imp) in enumerate(zip(features, importance), 1):
            print(f"   {i}. {feature}: {imp:.3f}")
    
    # 評価サマリー
    print(f"\n📝 評価サマリー:")
    for summary in performance['evaluation_summary']:
        print(f"   • {summary}")
    
    # 予測サンプル
    if samples:
        print(f"\n🎯 予測サンプル (実際の取引データとの比較):")
        for i, sample in enumerate(samples[:3], 1):
            input_data = sample['input']
            print(f"\n   サンプル {i}:")
            print(f"     所在地: {input_data['prefecture']} {input_data['city']} {input_data['district']}")
            print(f"     土地面積: {input_data['land_area']:.1f}㎡")
            print(f"     建物面積: {input_data['building_area']:.1f}㎡")
            print(f"     築年数: {input_data['building_age']}年")
            print(f"     実際の価格: ¥{sample['actual_price']:,.0f}")
            print(f"     予測価格:   ¥{sample['predicted_price']:,.0f}")
            print(f"     誤差率:     {sample['error_rate']:.1f}%")


def save_results_to_file(performance: dict, cv_results: dict, samples: list):
    """
    結果をJSONファイルに保存
    """
    results = {
        'model_performance': performance,
        'cross_validation': cv_results,
        'prediction_samples': samples
    }
    
    output_file = Path(__file__).parent / 'model_evaluation_results.json'
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 評価結果をファイルに保存しました: {output_file}")
        
    except Exception as e:
        logger.warning(f"結果の保存に失敗しました: {e}")


if __name__ == "__main__":
    main()