# SETUP

このドキュメントは、Tanukii をローカルで動かすためのセットアップ手順です。

## 前提条件

- Python 3.11 以上
- Homebrew (Windowsの場合はffmpegを手動でインストールしてください)

## 1. リポジトリ準備

```bash
git clone <your-repo-url>
cd Tanukii
```

## 2. 仮想環境作成

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. 依存インストール

```bash
pip install -r requirements.txt
```

## 4. FFmpeg インストール

```bash
brew install ffmpeg
```

`/api/transcribe` で音声を処理するために FFmpeg が必要です。

## 5. 環境変数設定 (任意)

```bash
cp .env.example .env
```

必要に応じて `.env` を編集します。

- `WHISPER_MODEL`: `tiny` / `base` / `small` / `medium` / `large`
- `SUMMARY_PROVIDER`: `auto` / `gemini` / `openai` / `local`
- `OPENAI_API_KEY`, `OPENAI_SUMMARY_MODEL`
- `GEMINI_API_KEY`, `GEMINI_SUMMARY_MODEL`
- `CORS_ORIGINS`

## 6. 起動

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

アクセス先:

- 同一マシン: http://127.0.0.1:8000
- LAN 内別端末: http://<このPCのIP>:8000

## 7. 動作確認

```bash
curl -s http://127.0.0.1:8000/api/health
```

`{"status":"ok", ...}` が返れば起動できています。

## 8. 設定保存について

画面の「設定を保存」で設定値は `config/settings.json` に保存されます。
このファイルはローカル情報を含むため Git 管理対象外です。

## 9. Tips
音声文字起こしのモデルを最初でダウンロードしておきたいときは


```.venv/bin/python -c "from faster_whisper import WhisperModel; [WhisperModel(m, download_root='models', compute_type='int8') for m in ['tiny','base','small','medium','large']]"```