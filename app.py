import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")

# ---- Custom styling ----
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle {
        color: gray;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📰 News Topic Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fine-tuned BERT model that classifies news headlines into World, Sports, Business, or Sci/Tech.</p>', unsafe_allow_html=True)

# ---- Load model (cached so it only loads once) ----
@st.cache_resource
def load_model():
    model_path = "C:/bert_project_output"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

with st.spinner("Loading model..."):
    tokenizer, model, device = load_model()

labels = ["World", "Sports", "Business", "Sci/Tech"]
label_emojis = {"World": "🌍", "Sports": "⚽", "Business": "💼", "Sci/Tech": "🔬"}

# ---- Sidebar info ----
with st.sidebar:
    st.header("About")
    st.write("This app uses a BERT model fine-tuned on the AG News dataset to classify news headlines into 4 categories.")
    st.write("**Categories:**")
    for label in labels:
        st.write(f"{label_emojis[label]} {label}")
    st.write("---")
    st.write(f"**Running on:** {'GPU 🚀' if device.type == 'cuda' else 'CPU'}")

# ---- Main input ----
headline = st.text_area(
    "Enter a news headline:",
    placeholder="e.g. Apple unveils new AI-powered iPhone",
    height=100
)

col1, col2 = st.columns([1, 4])
with col1:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    if headline.strip() == "":
        st.warning("Please enter a headline first.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### {label_emojis[labels[prediction]]} Predicted Category: **{labels[prediction]}**")
        st.write(f"Confidence: **{confidence*100:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.write("**Probability breakdown:**")
        prob_df = pd.DataFrame({
            "Category": labels,
            "Probability": [probs[0][i].item() for i in range(len(labels))]
        }).sort_values("Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Category"))

# ---- Example headlines ----
st.write("---")
st.write("**Try an example:**")
examples = [
    "NASA discovers new exoplanet in habitable zone",
    "Manchester United wins dramatic match in final minute",
    "Stock markets surge after interest rate cut",
    "Floods displace thousands in southern region"
]
example_cols = st.columns(2)
for i, example in enumerate(examples):
    with example_cols[i % 2]:
        st.caption(f"• {example}")
