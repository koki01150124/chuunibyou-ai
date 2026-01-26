# 厨二病AI

## 概要
このプロジェクトは、厨二病特有の言い回しやセリフを生成するAIモデルです。 Geminiによって生成されたデータセットを、LSTMを用いて学習しています。

## セットアップ
1. リポジトリをクローン
   ```
   git clone https://github.com/koki01150124/chuunibyou-ai.git
   ```
2. chuunibyou-aiに移動
   ```
   cd chuunibyou-ai
   ```
3. 仮想環境を作成
   ```
   python -m venv .venv
   ```
4. 仮想環境を有効化
   ```
   .\.venv\Scripts\activate
   ```
5. 必要なライブラリをインストール
   ```
   pip install -r requirements.txt
   ```

## 実行
```
python inference.py
```

## 出力例
「我が瞳には、世界の終焉が映る。」

## 技術スタック
- Language: Python 3.11.5
- Framework: PyTorch
- Architecture: LSTM (Embedding dim: 512, Hidden dim: 512)
