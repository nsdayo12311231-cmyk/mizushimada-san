  """
  みずしまださん ~ AI表紙添削エンジン
  FastAPI メインアプリケーション
  """

  from fastapi import FastAPI, File, UploadFile, HTTPException
  from fastapi.middleware.cors import CORSMiddleware
  from fastapi.responses import JSONResponse
  import uvicorn
  import os
  import time
  import logging
  from datetime import datetime
  from typing import Dict, Any

  # ローカルインポート
  from app.core.image_analyzer import ImageAnalyzer
  from app.core.scorer import Scorer

  # ログ設定
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)

  # FastAPI アプリケーション初期化
  app = FastAPI(
      title="みずしまださん",
      description="AI表紙添削エンジン ~ 添削くんとの連携用API",
      version="1.0.0",
      docs_url="/docs",  # Swagger UI
      redoc_url="/redoc"  # ReDoc
  )

  # CORS設定（添削くんとの連携用）
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["https://sassaku-kun.vercel.app"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # 依存関係の初期化
  analyzer = ImageAnalyzer()
  scorer = Scorer()

  @app.get("/")
  async def root() -> Dict[str, Any]:
      """
      ルートエンドポイント - ヘルスチェック
      """
      return {
          "message": "みずしまださん API",
          "version": "1.0.0",
          "status": "running",
          "timestamp": datetime.now().isoformat()
      }

  @app.get("/api/health")
  async def health_check() -> Dict[str, str]:
      """
      ヘルスチェックエンドポイント
      """
      return {
          "status": "healthy",
          "timestamp": datetime.now().isoformat()
      }

  @app.post("/api/analyze")
  async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
      """
      画像分析エンドポイント
      表紙画像をアップロードして分析結果を取得
      """
      try:
          # ファイル形式チェック
          if not file.content_type.startswith('image/'):
              raise HTTPException(
                  status_code=400,
                  detail="アップロードされたファイルは画像である必要があります"
              )

          # ファイルサイズチェック (10MB制限)
          contents = await file.read()
          if len(contents) > 10 * 1024 * 1024:  # 10MB
              raise HTTPException(
                  status_code=413,
                  detail="ファイルサイズが大きすぎます（10MB以下にしてください）"
              )

          logger.info(f"画像分析開始: {file.filename} ({len(contents)} bytes)")

          # 画像分析実行
          analysis_result = await analyzer.analyze_image(contents, file.filename)

          # スコア計算
          score_result = scorer.calculate_score(analysis_result)

          # 結果統合
          result = {
              "success": True,
              "filename": file.filename,
              "analysis": analysis_result,
              "score": score_result,
              "timestamp": datetime.now().isoformat()
          }

          logger.info(f"画像分析完了: {file.filename}")
          return result

      except HTTPException:
          raise
      except Exception as e:
          logger.error(f"画像分析エラー: {str(e)}")
          raise HTTPException(
              status_code=500,
              detail=f"画像分析中にエラーが発生しました: {str(e)}"
          )

  @app.exception_handler(404)
  async def not_found_handler(request, exc):
      """
      404エラーハンドラー
      """
      return JSONResponse(
          status_code=404,
          content={
              "error": "エンドポイントが見つかりません",
              "message": "指定されたURLは存在しません",
              "timestamp": datetime.now().isoformat()
          }
      )

  @app.exception_handler(500)
  async def internal_error_handler(request, exc):
      """
      500エラーハンドラー
      """
      return JSONResponse(
          status_code=500,
          content={
              "error": "内部サーバーエラー",
              "message": "サーバー内部でエラーが発生しました",
              "timestamp": datetime.now().isoformat()
          }
      )

  if __name__ == "__main__":
      port = int(os.environ.get("PORT", 8000))
      uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
