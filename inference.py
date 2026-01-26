import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import json

# 設定
N = 10  # 生成回数
MAX_SENTENCE_LENGTH = 128  # 生成する文章の最大の長さ
MODEL_PATH = "models/model.pth"  # modelのパス
VOCAB_PATH = "models/vocab.json"  # vocabのパス


class SentenceGeneratorModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 512, padding_idx=0)
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = nn.LeakyReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(model, index_to_word, sentence_length, argmax=True, topk=10, temp=1.0):
    gen_text = torch.ones(size=(1, 1)).to(torch.int32).to(device)

    for i in range(0, sentence_length):
        tmp_texts = (
            F.pad(gen_text, (0, sentence_length - i), value=0)
            .to(torch.int32)
            .to(device)
        )
        output = model(tmp_texts)

        logits = output[:, i, :]

        if argmax:
            gathered_indices = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=topk, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            chosen_indices = torch.multinomial(top_k_probs, 1).squeeze(-1)
            gathered_indices = top_k_indices.gather(-1, chosen_indices.unsqueeze(-1))

        gen_text = torch.cat([gen_text, gathered_indices], dim=1)

    for I in gen_text[0][1:]:
        if I in [0, 1, 2]:
            break
        print(index_to_word[int(I)], end=" ")
    print()


with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    # JSONのキーをintに変換
    loaded_vocab = json.load(f)
    index_to_word = {int(k): v for k, v in loaded_vocab.items()}

vocab_size = len(index_to_word)

# モデルの初期化と重みのロード
model = SentenceGeneratorModel(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

# 推論モードに切り替え
model = model.to(device)
model.eval()

# 生成
for _ in range(N):
    generate(
        model, index_to_word, sentence_length=MAX_SENTENCE_LENGTH + 1, argmax=False
    )
