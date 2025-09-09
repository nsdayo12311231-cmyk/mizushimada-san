"""
画像解析エンジン
構図・色彩・文字配置を分析する
"""

import cv2
import numpy as np
from PIL import Image, ImageStat
import io
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        """画像解析エンジン初期化"""
        logger.info("ImageAnalyzer initialized")
    
    def process_image(self, image_bytes: bytes) -> np.ndarray:
        """
        バイト列から画像を読み込み、OpenCV形式に変換
        
        Args:
            image_bytes: 画像のバイト列
            
        Returns:
            OpenCV形式の画像配列 (BGR)
        """
        try:
            # バイト列をnumpy配列に変換
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # OpenCVで画像をデコード
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            logger.info(f"Image processed: shape={image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            raise
    
    def analyze_composition(self, image: np.ndarray) -> int:
        """
        構図解析
        
        Args:
            image: OpenCV画像配列
            
        Returns:
            構図スコア (0-100)
        """
        try:
            h, w = image.shape[:2]
            
            # 三分割法チェック
            rule_of_thirds_score = self._check_rule_of_thirds(image)
            
            # 中心バランスチェック
            balance_score = self._check_balance(image)
            
            # エッジ密度チェック（構図の複雑さ）
            edge_score = self._check_edge_density(image)
            
            # 総合構図スコア
            composition_score = (
                rule_of_thirds_score * 0.4 +
                balance_score * 0.4 +
                edge_score * 0.2
            )
            
            score = max(0, min(100, int(composition_score)))
            logger.info(f"Composition analysis: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Composition analysis error: {str(e)}")
            return 50  # デフォルトスコア
    
    def analyze_color(self, image: np.ndarray) -> int:
        """
        色彩解析
        
        Args:
            image: OpenCV画像配列
            
        Returns:
            色彩スコア (0-100)
        """
        try:
            # BGR → RGB変換（PIL用）
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # 彩度チェック
            saturation_score = self._check_saturation(image)
            
            # コントラストチェック
            contrast_score = self._check_contrast(pil_image)
            
            # 色相バランスチェック
            hue_balance_score = self._check_hue_balance(image)
            
            # 総合色彩スコア
            color_score = (
                saturation_score * 0.35 +
                contrast_score * 0.35 +
                hue_balance_score * 0.3
            )
            
            score = max(0, min(100, int(color_score)))
            logger.info(f"Color analysis: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Color analysis error: {str(e)}")
            return 50  # デフォルトスコア
    
    def analyze_text(self, image: np.ndarray) -> int:
        """
        文字解析
        
        Args:
            image: OpenCV画像配列
            
        Returns:
            文字スコア (0-100)
        """
        try:
            h, w = image.shape[:2]
            
            # グレースケール変換
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # テキスト領域の検出
            text_regions = self._detect_text_regions(gray)
            
            # 文字領域のサイズ評価
            size_score = self._evaluate_text_size(text_regions, (h, w))
            
            # 視認性評価
            visibility_score = self._evaluate_text_visibility(image, text_regions)
            
            # 配置評価
            position_score = self._evaluate_text_position(text_regions, (h, w))
            
            # 総合文字スコア
            text_score = (
                size_score * 0.4 +
                visibility_score * 0.4 +
                position_score * 0.2
            )
            
            score = max(0, min(100, int(text_score)))
            logger.info(f"Text analysis: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Text analysis error: {str(e)}")
            return 50  # デフォルトスコア
    
    def _check_rule_of_thirds(self, image: np.ndarray) -> float:
        """三分割法チェック"""
        h, w = image.shape[:2]
        
        # 三分割線の位置
        h_lines = [h // 3, 2 * h // 3]
        v_lines = [w // 3, 2 * w // 3]
        
        # エッジ検出
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 三分割線上のエッジ密度を計算
        score = 0
        total_checks = 0
        
        for y in h_lines:
            line_edges = np.sum(edges[max(0, y-2):min(h, y+3), :])
            score += min(100, line_edges / w * 10)
            total_checks += 1
            
        for x in v_lines:
            line_edges = np.sum(edges[:, max(0, x-2):min(w, x+3)])
            score += min(100, line_edges / h * 10)
            total_checks += 1
        
        return score / total_checks if total_checks > 0 else 50
    
    def _check_balance(self, image: np.ndarray) -> float:
        """画像のバランスチェック"""
        h, w = image.shape[:2]
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 重心計算
        M = cv2.moments(gray)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2
        
        # 中心からのずれを評価
        center_x, center_y = w // 2, h // 2
        x_offset = abs(cx - center_x) / (w // 2)
        y_offset = abs(cy - center_y) / (h // 2)
        
        # バランススコア（中心に近いほど高い）
        balance = 100 - (x_offset + y_offset) * 50
        return max(0, balance)
    
    def _check_edge_density(self, image: np.ndarray) -> float:
        """エッジ密度チェック"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / total_pixels
        
        # 適度なエッジ密度が良い（5-15%程度）
        optimal_density = 0.1
        density_diff = abs(edge_density - optimal_density)
        score = max(0, 100 - density_diff * 1000)
        
        return score
    
    def _check_saturation(self, image: np.ndarray) -> float:
        """彩度チェック"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 彩度の統計情報
        sat_mean = np.mean(hsv[:, :, 1])
        sat_std = np.std(hsv[:, :, 1])
        
        # 適度な彩度が好ましい（100-200程度）
        optimal_sat = 150
        sat_diff = abs(sat_mean - optimal_sat)
        sat_score = max(0, 100 - sat_diff / 2)
        
        # 彩度のバリエーション（標準偏差）も考慮
        variety_score = min(100, sat_std)
        
        return (sat_score + variety_score) / 2
    
    def _check_contrast(self, pil_image: Image.Image) -> float:
        """コントラストチェック"""
        # PIL統計を使用
        stat = ImageStat.Stat(pil_image)
        
        # RGBチャンネルの標準偏差の平均（コントラストの指標）
        contrast = sum(stat.stddev) / len(stat.stddev)
        
        # 適度なコントラストが好ましい（30-80程度）
        optimal_contrast = 55
        contrast_diff = abs(contrast - optimal_contrast)
        score = max(0, 100 - contrast_diff * 2)
        
        return score
    
    def _check_hue_balance(self, image: np.ndarray) -> float:
        """色相バランスチェック"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        
        # 色相ヒストグラム
        hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
        hist = hist.flatten()
        
        # 色相の分散を計算（多様性の指標）
        hue_variance = np.var(hist)
        
        # 適度な分散が好ましい
        score = min(100, hue_variance / 1000)
        
        return score
    
    def _detect_text_regions(self, gray_image: np.ndarray) -> list:
        """テキスト領域検出"""
        # MSERを使用してテキスト候補領域を検出
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)
        
        # テキスト領域の候補をフィルタリング
        text_regions = []
        for region in regions:
            if len(region) > 50 and len(region) < 10000:  # サイズフィルタ
                x, y, w, h = cv2.boundingRect(region)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # アスペクト比フィルタ
                    text_regions.append((x, y, w, h))
        
        return text_regions
    
    def _evaluate_text_size(self, text_regions: list, image_size: Tuple[int, int]) -> float:
        """文字サイズ評価"""
        if not text_regions:
            return 30  # テキスト領域が見つからない場合
        
        h, w = image_size
        image_area = h * w
        
        # 最大のテキスト領域を評価
        max_area = max([region[2] * region[3] for region in text_regions])
        area_ratio = max_area / image_area
        
        # 適切なサイズ比率（2-10%程度）
        optimal_ratio = 0.05
        ratio_diff = abs(area_ratio - optimal_ratio)
        score = max(0, 100 - ratio_diff * 2000)
        
        return score
    
    def _evaluate_text_visibility(self, image: np.ndarray, text_regions: list) -> float:
        """文字視認性評価"""
        if not text_regions:
            return 30
        
        scores = []
        for x, y, w, h in text_regions:
            # テキスト領域の色情報
            text_area = image[y:y+h, x:x+w]
            if text_area.size == 0:
                continue
                
            # 周辺領域との色差を計算
            bg_area = self._get_surrounding_area(image, x, y, w, h)
            if bg_area.size == 0:
                continue
                
            text_mean = np.mean(text_area, axis=(0, 1))
            bg_mean = np.mean(bg_area, axis=(0, 1))
            
            # 色差計算（簡易版）
            color_diff = np.linalg.norm(text_mean - bg_mean)
            visibility_score = min(100, color_diff * 2)
            scores.append(visibility_score)
        
        return np.mean(scores) if scores else 30
    
    def _evaluate_text_position(self, text_regions: list, image_size: Tuple[int, int]) -> float:
        """文字配置評価"""
        if not text_regions:
            return 50
        
        h, w = image_size
        
        # 最大のテキスト領域の位置を評価
        largest_region = max(text_regions, key=lambda r: r[2] * r[3])
        x, y, tw, th = largest_region
        
        # 中心位置
        center_x = x + tw // 2
        center_y = y + th // 2
        
        # 上部配置が好ましい（タイトル用）
        vertical_score = 100 - (center_y / h) * 50
        
        # 水平方向は中央が好ましい
        horizontal_score = 100 - abs(center_x - w // 2) / (w // 2) * 100
        
        return (vertical_score + horizontal_score) / 2
    
    def _get_surrounding_area(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """テキスト領域周辺のエリアを取得"""
        img_h, img_w = image.shape[:2]
        
        # 周辺領域のマージン
        margin = max(w, h) // 4
        
        # 周辺領域の座標
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)
        
        # 周辺領域を取得（テキスト領域を除く）
        surrounding = image[y1:y2, x1:x2]
        
        # テキスト領域をマスクで除外
        mask = np.ones(surrounding.shape[:2], dtype=bool)
        text_x1 = max(0, x - x1)
        text_y1 = max(0, y - y1)
        text_x2 = min(x2 - x1, text_x1 + w)
        text_y2 = min(y2 - y1, text_y1 + h)
        
        if text_x2 > text_x1 and text_y2 > text_y1:
            mask[text_y1:text_y2, text_x1:text_x2] = False
        
        return surrounding[mask]