"""
スコア算出・フィードバック生成エンジン
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Scorer:
    def __init__(self):
        """スコア算出エンジン初期化"""
        # 各項目の重要度設定（講師の重視するポイントを反映）
        self.weights = {
            'text': 0.4,      # 文字の重要度40%（最重要）
            'composition': 0.3, # 構図30%
            'color': 0.3      # 色彩30%
        }
        
        # スコア判定基準
        self.thresholds = {
            'excellent': 85,  # 優秀
            'good': 70,      # 良好
            'fair': 50,      # 普通
            'poor': 30       # 要改善
        }
        
        logger.info("Scorer initialized")
    
    def calculate_total_score(self, scores: Dict[str, int]) -> int:
        """
        総合スコア算出
        
        Args:
            scores: 各項目のスコア {'text': 45, 'composition': 78, 'color': 72}
            
        Returns:
            総合スコア (0-100)
        """
        try:
            total = sum(scores[key] * self.weights[key] for key in scores if key in self.weights)
            total_score = max(0, min(100, int(total)))
            
            logger.info(f"Total score calculated: {total_score} from {scores}")
            return total_score
            
        except Exception as e:
            logger.error(f"Score calculation error: {str(e)}")
            return 50  # デフォルトスコア
    
    def generate_feedback(self, scores: Dict[str, int]) -> Dict[str, str]:
        """
        フィードバック生成
        
        Args:
            scores: 各項目のスコア
            
        Returns:
            各項目のフィードバックメッセージ
        """
        try:
            feedback = {}
            
            # 文字フィードバック
            feedback['text'] = self._generate_text_feedback(scores.get('text', 50))
            
            # 構図フィードバック
            feedback['composition'] = self._generate_composition_feedback(scores.get('composition', 50))
            
            # 色彩フィードバック
            feedback['color'] = self._generate_color_feedback(scores.get('color', 50))
            
            logger.info("Feedback generated successfully")
            return feedback
            
        except Exception as e:
            logger.error(f"Feedback generation error: {str(e)}")
            return self._get_default_feedback()
    
    def _generate_text_feedback(self, text_score: int) -> str:
        """文字フィードバック生成"""
        if text_score >= self.thresholds['excellent']:
            return "文字サイズ・配置ともに優秀です。タイトルが非常に読みやすく、視認性も抜群です。プロレベルの文字配置になっています。"
        
        elif text_score >= self.thresholds['good']:
            return "文字サイズは適切です。タイトルがしっかり読める大きさで配置されています。さらに目立たせたい場合は、縁取りや影の追加を検討してください。"
        
        elif text_score >= self.thresholds['fair']:
            return "文字サイズは概ね適切ですが、もう少し大きくするとより読みやすくなります。10-15%程度の拡大をお勧めします。"
        
        else:
            return "タイトルが小さすぎます。20%以上拡大し、縁取りの追加を強く推奨します。現在の状態では読みにくく、注目度が低下しています。背景とのコントラストも確認してください。"
    
    def _generate_composition_feedback(self, composition_score: int) -> str:
        """構図フィードバック生成"""
        if composition_score >= self.thresholds['excellent']:
            return "構図が非常に優れています。プロレベルの配置バランスで、視覚的インパクトが強い作品です。三分割法や黄金比が効果的に活用され、見る人の視線を自然に引きつけます。"
        
        elif composition_score >= self.thresholds['good']:
            return "構図バランスは良好です。キャラクターの配置が自然で、視線誘導も適切に行われています。さらに向上させるには、空間の使い方を意識してみてください。"
        
        elif composition_score >= self.thresholds['fair']:
            return "構図バランスは悪くありませんが、改善の余地があります。三分割法を意識して、主要素の配置を調整してみてください。"
        
        else:
            return "構図バランスに大きな改善が必要です。キャラクターや要素の配置を見直し、三分割法やバランスの取り方を学習することをお勧めします。中心に寄りすぎている可能性があります。"
    
    def _generate_color_feedback(self, color_score: int) -> str:
        """色彩フィードバック生成"""
        if color_score >= self.thresholds['excellent']:
            return "色彩設計が素晴らしいです。統一感がありながらも、メリハリの効いた配色になっています。彩度とコントラストのバランスが絶妙で、非常に魅力的な仕上がりです。"
        
        elif color_score >= self.thresholds['good']:
            return "色彩バランスは良好です。魅力的な色使いで注目を集める効果があります。より印象的にしたい場合は、アクセントカラーの活用を検討してみてください。"
        
        elif color_score >= self.thresholds['fair']:
            return "色彩バランスは普通です。もう少しコントラストを強くするか、彩度を調整すると、より目を引く表紙になるでしょう。"
        
        else:
            return "色彩バランスの改善が必要です。コントラストが不足しているか、色の統一感に欠ける可能性があります。主色・補色の関係を見直し、より魅力的な配色を心がけてください。"
    
    def get_score_level(self, score: int) -> str:
        """スコアレベル判定"""
        if score >= self.thresholds['excellent']:
            return 'excellent'
        elif score >= self.thresholds['good']:
            return 'good'
        elif score >= self.thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def get_improvement_priority(self, scores: Dict[str, int]) -> list:
        """
        改善優先度の算出
        
        Args:
            scores: 各項目のスコア
            
        Returns:
            改善優先度順のリスト [('text', 45), ('color', 60), ...]
        """
        try:
            # 重要度×低スコアで優先度計算
            priorities = []
            for key, score in scores.items():
                if key in self.weights:
                    # スコアが低く、重要度が高いほど優先度が高い
                    priority = self.weights[key] * (100 - score)
                    priorities.append((key, score, priority))
            
            # 優先度順にソート
            priorities.sort(key=lambda x: x[2], reverse=True)
            
            return [(item[0], item[1]) for item in priorities]
            
        except Exception as e:
            logger.error(f"Priority calculation error: {str(e)}")
            return [('text', 50), ('composition', 50), ('color', 50)]
    
    def _get_default_feedback(self) -> Dict[str, str]:
        """デフォルトフィードバック"""
        return {
            'text': '文字に関する分析を実行できませんでした。画像を確認して再度お試しください。',
            'composition': '構図に関する分析を実行できませんでした。画像を確認して再度お試しください。',
            'color': '色彩に関する分析を実行できませんでした。画像を確認して再度お試しください。'
        }
    
    def generate_summary_feedback(self, total_score: int, scores: Dict[str, int]) -> str:
        """
        総合評価コメント生成
        
        Args:
            total_score: 総合スコア
            scores: 各項目スコア
            
        Returns:
            総合評価コメント
        """
        try:
            level = self.get_score_level(total_score)
            priorities = self.get_improvement_priority(scores)
            
            if level == 'excellent':
                return f"素晴らしい作品です！総合{total_score}点の高品質な表紙に仕上がっています。商業レベルの完成度です。"
            
            elif level == 'good':
                return f"良好な仕上がりです。総合{total_score}点で、基本的な要素は適切に配置されています。"
            
            elif level == 'fair':
                lowest_item = priorities[0][0] if priorities else 'text'
                item_names = {'text': '文字', 'composition': '構図', 'color': '色彩'}
                return f"総合{total_score}点です。特に{item_names.get(lowest_item, '文字')}の改善を優先することをお勧めします。"
            
            else:
                return f"総合{total_score}点です。基本的な要素から改善していきましょう。段階的に品質を向上させることで、魅力的な表紙になります。"
                
        except Exception as e:
            logger.error(f"Summary feedback generation error: {str(e)}")
            return f"総合{total_score}点の評価結果です。"