import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
import json
import librosa
import librosa.display
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# ---- CONFIG ----
HF_REPO = "dhayal1/bird-call-spectrogram-classifier"
SR = 22050
DURATION = 5
N_MELS = 128

st.set_page_config(page_title="Bird Call Spectrogram Classifier", page_icon="🎵", layout="centered")

# ---- LOAD MODEL + METADATA (cached) ----
@st.cache_resource
def load_model(num_classes):
    weights_path = hf_hub_download(repo_id=HF_REPO, filename="bird_call_resnet18.pth")
    model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_species_list():
    path = hf_hub_download(repo_id=HF_REPO, filename="species_list.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_results():
    path = hf_hub_download(repo_id=HF_REPO, filename="audio_robustness_results.json")
    with open(path) as f:
        return json.load(f)

species_list = load_species_list()
model = load_model(len(species_list))
robustness_results = load_results()

# ---- AUDIO -> SPECTROGRAM ----
def audio_to_melspec(y, sr=SR, duration=DURATION, n_mels=N_MELS):
    target_len = sr * duration
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32), mel_db

def plot_spectrogram(mel_db, sr=SR):
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel-Spectrogram')
    fig.tight_layout()
    return fig

# ---- UI ----
st.title("🎵 Bird Call Spectrogram Classifier")
st.markdown(
    f"A ResNet18 classifier trained on mel-spectrograms of {len(species_list)} common North American "
    "bird species, sourced from Xeno-canto — benchmarked on clean recordings vs. **simulated background "
    "noise**, since field recordings are rarely as clean as curated audio libraries."
)

tab1, tab2 = st.tabs(["🔍 Try It", "📊 Robustness Benchmark"])

with tab1:
    uploaded_file = st.file_uploader("Upload a bird call audio clip (wav, mp3)", type=["wav", "mp3", "ogg"])

    if uploaded_file:
        with st.spinner("Analyzing..."):
            y, _ = librosa.load(uploaded_file, sr=SR, duration=DURATION)
            spec_norm, spec_db = audio_to_melspec(y)

            spec_3ch = np.stack([spec_norm, spec_norm, spec_norm], axis=0)
            spec_tensor = torch.tensor(spec_3ch, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(spec_tensor)
                probs = F.softmax(logits, dim=1)[0]
                top5_probs, top5_idx = torch.topk(probs, min(5, len(species_list)))

        st.audio(uploaded_file)
        st.pyplot(plot_spectrogram(spec_db))

        st.subheader("Top predictions")
        for prob, idx in zip(top5_probs, top5_idx):
            name = species_list[idx.item()]
            st.write(f"**{name}** — {prob.item()*100:.1f}%")

        st.caption(
            f"⚠️ Trained on only {len(species_list)} common species with a small dataset "
            "(~20 clips/species from Xeno-canto). Species outside this set, or noisy/faint "
            "recordings, will still get a prediction — just not necessarily a correct one. "
            "See the Robustness Benchmark tab for measured accuracy and honest limitations."
        )
    else:
        st.info("Upload a bird call clip to see the spectrogram and predictions.")
        st.caption(f"Trained species: {', '.join(species_list)}")

with tab2:
    st.subheader("Accuracy under simulated background noise")
    st.markdown(
        "Field recordings rarely sound as clean as curated audio libraries — wind, traffic, "
        "and other birds all add background noise. This model was evaluated on held-out test "
        "clips under two conditions: as recorded, and with **simulated white noise added** "
        "before spectrogram conversion."
    )

    results_df = pd.DataFrame(robustness_results)
    results_df_display = results_df.copy()
    for col in ['accuracy', 'precision', 'recall', 'f1_macro']:
        results_df_display[col] = (results_df_display[col] * 100).round(1).astype(str) + '%'
    results_df_display.columns = ['Condition', 'Accuracy', 'Precision', 'Recall', 'F1 (macro)']
    st.dataframe(results_df_display, use_container_width=True, hide_index=True)

    st.bar_chart(results_df.set_index('condition')['accuracy'] * 100)

    clean = results_df[results_df['condition'] == 'clean'].iloc[0]
    noisy = results_df[results_df['condition'] == 'background_noise'].iloc[0]
    drop = (clean['accuracy'] - noisy['accuracy']) * 100

    st.warning(
        f"**Key finding:** accuracy drops **{drop:.1f} percentage points** "
        f"({clean['accuracy']*100:.1f}% → {noisy['accuracy']*100:.1f}%) under simulated "
        f"background noise — a substantial degradation, consistent with how sensitive "
        f"spectrogram-based classification is to recording conditions."
    )

    n_species = len(species_list)
    chance = 100 / n_species
    st.info(
        f"**Honest context:** this model was trained on a small dataset (~19 clips/species "
        f"from Xeno-canto, {n_species} species) — well above the {chance:.1f}% random-chance "
        f"baseline, but with real room to improve given more training data per species. "
        f"The clean-vs-noise *relative* degradation is the more robust signal here than the "
        f"absolute accuracy number."
    )

    st.caption(
        "Note: background noise is simulated via added Gaussian noise before spectrogram "
        "conversion, not sourced from naturally noisy field recordings. This is a standard "
        "technique for controlled robustness testing, but real-world noise (wind, overlapping "
        "calls, traffic) has different characteristics than synthetic white noise."
    )

st.markdown("---")
st.caption("Built by Dhayal R · [GitHub](https://github.com/Dhayalramesh/bird-call-spectrogram-classifier)")
