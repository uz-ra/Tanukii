# SETUP

## 前提条件

- Python 3.11 以上
- Homebrew (Windowsの場合はffmpegを手動でインストールしてください)

## 1. リポジトリ準備

```bash
git clone https://github.com/uz-ra/Tanukii
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
- `LOCAL_SUMMARY_MODEL`: 既定 `alfredplpl/gemma-2-2b-jpn-it-gguf`
- `LOCAL_SUMMARY_GGUF_FILE`: 既定 `gemma-2-2b-jpn-it-Q4_K_M.gguf`
- `LOCAL_SUMMARY_MAX_NEW_TOKENS`: ローカル要約の最大出力トークン数（既定 384）
- `LOCAL_SUMMARY_TEMPERATURE`: ローカル要約の温度（既定 0.2）
- `LOCAL_SUMMARY_CONTEXT_LENGTH`: 推論コンテキスト長（既定 4096）
- `LOCAL_SUMMARY_THREADS`: 推論スレッド数（既定: CPUコアの半分）
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
このファイルはローカル情報(APIキーなど)を含むため Git 管理対象外です。

## 9. Tips
音声文字起こしのモデルを最初でダウンロードしておきたいときは


```.venv/bin/python -c "from faster_whisper import WhisperModel; [WhisperModel(m, download_root='models', compute_type='int8') for m in ['tiny','base','small','medium','large']]"```

http://127.0.0.1:8000 で起動中のプロセスをkillしたいときは、

```bash
lsof -ti:8000 | xargs kill -9
```

または:

```bash
kill -9 $(lsof -t -i :8000)
```

Geminiのモデルは```gemini-3-flash-preview```か```gemini-3.1-flash-lite-preview```がおすすめ