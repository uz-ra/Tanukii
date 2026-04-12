# Tanukii

![Logo](Tanukii.png)

ローカル実行向けの、音声文字起こし + 要約 Web アプリです。

## スクリーンショット
![demo](demo.gif)

## 構成

- バックエンド: FastAPI
- 文字起こし: faster-whisper
- 要約: gemma-2-2b-jpn-it-gguf(local) / Gemini / OpenAI（設定で切替、API設定）
- フロントエンド: HTML/CSS/JavaScript

## セットアップ

詳細手順は [SETUP.md](SETUP.md) を参照してください。

最短起動コマンド:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

起動後、ブラウザで以下にアクセス:

- http://127.0.0.1:8000

## 主な機能

- 音声ファイルの文字起こし
- **話題キーワード入力による文字起こし精度向上**
- レジュメファイル（PDF/Word）の読み込み
- 文字起こし結果とレジュメを組み合わせて要約
- 要約スタイル切替（箇条書き / 議事録 / アクション抽出）
- 設定の読込・保存
- デバッグモード時のログ表示

## API エンドポイント

- `GET /api/health`
- `POST /api/transcribe` (multipart: file, language, model)
- `POST /api/transcribe/start` (multipart: file, language, model, **initial_prompt**) - initial_promptパラメータ追加
- `GET /api/transcribe/jobs/{job_id}`
- `POST /api/summarize` (multipart: text, style, provider, model, api_key, system_prompt, user_prompt_template)
- `POST /api/extract-resume` (multipart: file)
- `GET /api/config`
- `POST /api/config`
- `GET /api/debug/logs`
- `POST /api/debug/logs/clear`

`/api/transcribe/jobs/{job_id}` は、進捗率（0-100）を `progress` として返します。値は「確定済みセグメントの終了時刻 / 音声長」を使った擬似進捗です。

## 設定ファイル

- サンプル環境変数: [.env.example](.env.example)
- 実行時設定保存先: [config/settings.json](config/settings.json) (Git 管理対象外)

## 導入時の注意

- `models/` は容量が大きいため、初回実行時のダウンロードを前提に同梱していません。

## セキュリティ

- `--host 0.0.0.0` は LAN 公開です。
- 外部公開は避け、必要に応じて `CORS_ORIGINS` を制限してください。
