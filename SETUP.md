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

## 9. 新機能：話題キーワードによる文字起こし精度向上

文字起こし画面の「話題（キーワード）」欄に認識対象となる話題やキーワードを入力することで、Whisper の文字起こし精度を向上させることができます。

### 使い方

1. 文字起こし画面に移動
2. 「話題（キーワード）」欄に、音声内容に関連するキーワードを入力
   - 複数の単語を入力する場合は、カンマで区切ってください
   - 例：`技術カンファレンス, 人工知能, プログラミング`
3. 音声ファイルを選択して「文字起こし開始」をクリック

### 効果

- 技術用語や専門用語の誤認識を削減
- 業界特有の用語や社内用語の認識精度向上
- 複数の類似音の区別が改善される傾向

### 参考

この機能は、Whisper の `initial_prompt` パラメータを使い、以下の記事で紹介されている手法を実装しています：
https://note.com/kirillovlov/n/n4af603aabb26

## 10. 新機能：レジュメ読み込み機能

要約ページ上で PDF または Word ファイル (.docx) をアップロードしてレジュメ内容を読み込むことができます。

### 対応ファイル形式

- **PDF** (.pdf)
- **Word** (.docx)

### 使い方

1. 要約ページに移動
2. 「レジュメファイル（PDF/Word）」でファイルを選択
3. 「レジュメを読み込む」ボタンをクリック
4. ファイルのテキストが「要約対象テキスト」に反映されます
5. 既に文字起こし結果がある場合は、レジュメ与える【レジュメ内容】セクション付きで結合されます
6. 「要約作成」で実行

### バックエンド API

- **エンドポイント**: `POST /api/extract-resume`
- **パラメータ**: `file` (form data, multipart)
- **レスポンス**: 
  ```json
  {
    "text": "抽出されたテキスト...",
    "filename": "resume.pdf"
  }
  ```

## 11. Tips
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