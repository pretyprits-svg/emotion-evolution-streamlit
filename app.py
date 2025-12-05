
import streamlit as st
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np

# Matplotlib for plotting (Streamlit will render it)
import matplotlib.pyplot as plt

# Transformers for tokenization / optional pretrained emotion model
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Emotion Evolution Visualizer", page_icon="🧠", layout="wide")

st.title("🧠 Emotion Evolution in Conversations — Streamlit Demo")
st.markdown("""
Upload a conversation and visualize the **emotion flow** across turns.
You can either:
1) **Load your trained EEM (BERT + BiLSTM) checkpoint** from the training script, or  
2) Use a **pretrained emotion classifier** (quick demo, no training needed).
""")

# ---------- Label mapping ----------
EMOTION_ID2LABEL = {
    0: "No_Emotion",
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise",
}
EMOTION_LABEL2ID = {v:k for k,v in EMOTION_ID2LABEL.items()}
NUM_LABELS = len(EMOTION_ID2LABEL)

# ---------- EEM model (matches training script) ----------
class EmotionEvolutionModel(nn.Module):
    def __init__(self, bert_name: str = "bert-base-uncased", lstm_hidden: int = 256, num_labels: int = NUM_LABELS, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.hidden_size = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(self.hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * lstm_hidden, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, turn_counts: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [N_utts, H]
        B = turn_counts.size(0)
        T_max = int(turn_counts.max().item())
        H = cls.size(-1)
        device = cls.device
        padded = torch.zeros(B, T_max, H, device=device)
        idx = 0
        for b in range(B):
            t = int(turn_counts[b].item())
            padded[b, :t, :] = cls[idx: idx + t, :]
            idx += t
        lstm_out, _ = self.bilstm(padded)
        logits = self.classifier(self.dropout(lstm_out))
        return logits

# ---------- Helpers ----------
@st.cache_resource
def load_eem(bert_name: str, ckpt_bytes: bytes) -> Dict[str, Any]:
    # Save checkpoint to a temp file
    import tempfile, torch
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp.write(ckpt_bytes)
    tmp.flush()
    ckpt = torch.load(tmp.name, map_location="cpu")

    # Use the bert_name argument directly instead of ckpt["bert_name"]
    model = EmotionEvolutionModel(
        bert_name=bert_name,
        hidden_size=128,
        num_classes=7
    )

    # Restore trained layers
    model.lstm.load_state_dict(ckpt["lstm_state_dict"])
    model.fc.load_state_dict(ckpt["fc_state_dict"])
    model.eval()

    from transformers import BertTokenizerFast
    tok = BertTokenizerFast.from_pretrained(bert_name)

    return {"model": model, "tokenizer": tok}


@st.cache_resource
def load_pretrained_classifier(model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name)
    clf.eval()
    return tok, clf

def tokenize_utterances(tokenizer, utterances: List[str], max_length: int = 64):
    enc = tokenizer(
        utterances,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc

def run_eem(model, tokenizer, utterances: List[str]):
    enc = tokenize_utterances(tokenizer, utterances)
    turn_counts = torch.tensor([len(utterances)], dtype=torch.long)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"], turn_counts)  # [1, T, C]
        preds = logits.argmax(dim=-1).squeeze(0).tolist()
    labels = [EMOTION_ID2LABEL[i] for i in preds]
    return preds, labels

def run_pretrained(tok, clf, utterances: List[str]):
    enc = tok(
        utterances,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = clf(**enc)
        preds = out.logits.argmax(dim=-1).tolist()
    # Map model-specific labels to our canonical space if possible
    # For j-hartmann model labels:
    # ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    #  'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    #  'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    id2label = clf.config.id2label
    mapped_ids = []
    mapped_labels = []
    for p in preds:
        lab = id2label[p].lower()
        # Simple heuristic mapping to our 7-class space
        if lab in ["anger", "annoyance", "disapproval"]:
            mapped_ids.append(1); mapped_labels.append("Anger")
        elif lab in ["disgust"]:
            mapped_ids.append(2); mapped_labels.append("Disgust")
        elif lab in ["fear", "nervousness"]:
            mapped_ids.append(3); mapped_labels.append("Fear")
        elif lab in ["joy", "amusement", "excitement", "optimism", "relief", "pride", "gratitude", "love", "approval", "admiration", "caring"]:
            mapped_ids.append(4); mapped_labels.append("Happiness")
        elif lab in ["sadness", "grief", "remorse", "disappointment"]:
            mapped_ids.append(5); mapped_labels.append("Sadness")
        elif lab in ["surprise", "realization", "curiosity"]:
            mapped_ids.append(6); mapped_labels.append("Surprise")
        else:
            mapped_ids.append(0); mapped_labels.append("No_Emotion")
    return mapped_ids, mapped_labels

def parse_uploaded_file(file) -> List[str]:
    name = file.name.lower()
    if name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
        utterances = [line.strip() for line in text.splitlines() if line.strip()]
        return utterances
    elif name.endswith(".json"):
        import json
        data = json.loads(file.read().decode("utf-8", errors="ignore"))
        # Accept list[str] or list[dict{text:..., speaker:...}]
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return [str(x.get("text", "")) for x in data if str(x.get("text", "")).strip()]
        else:
            raise ValueError("JSON must be a list of strings or list of objects with a 'text' field.")
    elif name.endswith(".csv"):
        df = pd.read_csv(file)
        if "text" in df.columns:
            return df["text"].astype(str).tolist()
        else:
            raise ValueError("CSV must contain a 'text' column.")
    else:
        raise ValueError("Unsupported file type. Upload .txt, .json, or .csv")

# Sidebar controls
st.sidebar.header("Inference Options")
mode = st.sidebar.radio("Choose inference mode:", ["Use my trained EEM checkpoint", "Use pretrained emotion model"], index=1)
bert_name = st.sidebar.text_input("BERT backbone (for EEM)", value="bert-base-uncased")
max_len = st.sidebar.slider("Max tokens per utterance", 16, 256, 64, 8)

st.sidebar.markdown("---")
st.sidebar.header("Upload Conversation")
uploaded = st.sidebar.file_uploader("Upload .txt / .json / .csv", type=["txt", "json", "csv"])

if mode == "Use my trained EEM checkpoint":
    ckpt_file = st.sidebar.file_uploader("Upload EEM checkpoint (.pt)", type=["pt"])
else:
    st.sidebar.caption("Pretrained model: j-hartmann/emotion-english-distilroberta-base (mapped to 7 classes)")

run_btn = st.sidebar.button("Run Inference")

# Main area
sample = [
    "Hi! How are you doing today?",
    "Not great... I think I messed up my presentation.",
    "Oh no, I'm sorry to hear that. What happened?",
    "I forgot some slides and panicked.",
    "That sounds stressful. Want me to help you prepare next time?",
    "Yes, please. That would be great!"
]

with st.expander("Sample conversation (click to use)", expanded=False):
    st.write(pd.DataFrame({"turn": list(range(1, len(sample)+1)), "text": sample}))
    if st.button("Use sample conversation"):
        uploaded = None  # force sample
        st.session_state["use_sample"] = True
    else:
        st.session_state["use_sample"] = st.session_state.get("use_sample", False)

def get_dialogue() -> List[str]:
    if st.session_state.get("use_sample", False) and uploaded is None:
        return sample
    if uploaded is None:
        return []
    try:
        return parse_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return []

dialogue = get_dialogue()

if run_btn:
    if not dialogue:
        st.warning("Please upload a conversation file or use the sample.")
    else:
        with st.spinner("Running inference..."):
            if mode == "Use my trained EEM checkpoint":
                if ckpt_file is None:
                    st.error("Please upload your trained EEM checkpoint (.pt).")
                else:
                    loaded = load_eem(bert_name, ckpt_file.read())
                    model = loaded["model"]
                    tokenizer = loaded["tokenizer"]
                    preds, labels = run_eem(model, tokenizer, dialogue)
            else:
                tok, clf = load_pretrained_classifier()
                preds, labels = run_pretrained(tok, clf, dialogue)

        if dialogue and (mode == "Use pretrained emotion model" or ckpt_file is not None):
            # Table
            df = pd.DataFrame({
                "Turn": list(range(1, len(dialogue)+1)),
                "Utterance": dialogue,
                "Predicted Emotion": labels
            })
            st.subheader("Predictions")
            st.dataframe(df, use_container_width=True)

            # Plot emotion evolution (IDs)
            st.subheader("Emotion Evolution Plot")
            fig = plt.figure()
            plt.plot(range(1, len(preds)+1), preds, marker="o")
            plt.xlabel("Turn")
            plt.ylabel("Emotion ID (mapped)")
            plt.title("Emotion Evolution")
            st.pyplot(fig)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Tip: For best results with EEM, upload the checkpoint produced by training script (eem_best.pt).")
