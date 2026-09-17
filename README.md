# 🎵 Bird Call Spectrogram Classifier

A ResNet18 classifier trained on mel-spectrograms of 12 common North American bird species, sourced from Xeno-canto — benchmarked on clean recordings vs. **simulated background noise**, since field recordings are rarely as clean as curated audio libraries.

**Live demo:** [bird-call-spectrogram-classifier-5m3ecm2ynjkcryxjapper4x.streamlit.app](https://bird-call-spectrogram-classifier-5m3ecm2ynjkcryxjapper4x.streamlit.app)

---

## Why this project

Identifying birds from audio is a genuinely different problem from identifying them visually — it requires converting sound into a form a CNN can learn from (spectrograms), and it comes with its own real-world failure mode: background noise. Wind, traffic, overlapping calls, and general field conditions rarely produce the clean audio found in curated recording libraries. This project measures how much classification accuracy degrades under noisy conditions, using the same "measure before claiming" approach applied across this portfolio's other projects — rather than reporting a single clean-test number and stopping there.

## What it does

- Fetches labeled bird call recordings for 12 species directly from the **Xeno-canto API** (v3, free with registration)
- Converts each audio clip to a **mel-spectrogram** (librosa) — a visual representation of frequency content over time, which a CNN can process like an image
- Fine-tunes a **ResNet18** (transfer learning) to classify species from spectrograms
- Evaluates on two conditions: clean audio, and audio with **simulated white noise** added before spectrogram conversion
- Deploys as an interactive Streamlit app: upload a bird call clip, see its spectrogram rendered live, and get species predictions

## Species covered

Northern Cardinal, Blue Jay, American Robin, House Sparrow, American Crow, Black-capped Chickadee, Mourning Dove, Song Sparrow, Red-winged Blackbird, Common Starling, House Finch, Common Grackle

## Results

Evaluated on 60 held-out test clips (5 per species, ~19 training clips/species):

| Condition | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| Clean | 61.7% | 58.4% | 61.7% | 58.8% |
| Background noise | 36.7% | 38.9% | 36.7% | 33.1% |

**Key finding:** accuracy drops **25.0 percentage points** (61.7% → 36.7%) under simulated background noise — a substantial degradation, consistent with how sensitive spectrogram-based classification is to recording conditions. This mirrors the robustness patterns found across this portfolio's other CV projects: real-world degradation (occlusion, clutter, noise) consistently costs significant accuracy versus clean-benchmark conditions.

### Out-of-distribution behavior

Testing the deployed model on a Caspian Gull recording — a species entirely outside the 12 trained classes — produced a low-confidence, spread-out prediction (top guess at 51.6%, with the remainder distributed across four other species at 5–10% each) rather than a single confidently wrong answer. This is the expected and reassuring failure mode for a fixed-class classifier facing genuinely novel input: the model has no "unknown" category, so it must pick something, but its uncertainty is visible in how thin the confidence spread is.

## Honest limitations

This project's dataset is meaningfully smaller than this portfolio's vision projects (228 training clips across 12 species vs. thousands of images in the CV projects), and the results reflect that honestly:

- **61.7% clean-test accuracy** is well above the 8.3% random-chance baseline for 12 classes, confirming the model learned real acoustic patterns — but it's a modest number in absolute terms, with real room to improve given more training data per species.
- **Train accuracy reached 98.3%** while test accuracy was 61.7% — a clear overfitting gap, expected and explainable given only ~19 training samples per class.
- The **noise robustness benchmark's relative degradation** (the 25-point drop) is a more reliable signal than either absolute accuracy number alone, since it's measured consistently on the same model and same test clips under two conditions.
- Background noise is **simulated** (Gaussian white noise added pre-spectrogram), not sourced from naturally noisy field recordings. This is a standard, controlled technique for robustness testing, but real-world noise (wind, overlapping calls, traffic) has different spectral characteristics than synthetic white noise.

## Methodology

- **Dataset:** [Xeno-canto](https://xeno-canto.org) API v3, querying `en:"=<species>" grp:birds q:">C" type:song` per species — quality-filtered, song-type recordings only, exact English-name matching
- **Audio processing:** clips resampled to 22050 Hz, truncated/padded to 5 seconds, converted to 128-bin mel-spectrograms (librosa), normalized to [0, 1], replicated to 3 channels for pretrained CNN compatibility
- **Model:** ResNet18, ImageNet-pretrained, fine-tuned end-to-end for 15 epochs (AdamW)
- **Noise condition:** Gaussian white noise (σ=0.02) added to the raw waveform before spectrogram conversion, applied only to held-out test clips
- **Evaluation:** scikit-learn precision/recall/F1 (macro-averaged across all 12 classes)

## Tech stack

Python, PyTorch, timm, librosa, matplotlib, Streamlit, Hugging Face Hub (model hosting), Xeno-canto API

## Project structure
bird-call-spectrogram-classifier/
├── bird_call_spectrogram_classifier.ipynb # full pipeline: data fetch, spectrogram conversion, training, evaluation (Colab)
├── app.py # Streamlit app (live inference + benchmark dashboard)
├── requirements.txt # Python dependencies
├── packages.txt # system-level dependencies (ffmpeg, libsndfile1 for audio decoding)
└── README.md

Trained model weights, benchmark results, and the species list are hosted on [Hugging Face Hub](https://huggingface.co/dhayal1/bird-call-spectrogram-classifier) and pulled by the app at runtime.

## Run it yourself

**Training:** open `bird_call_spectrogram_classifier.ipynb` in Google Colab (free T4 GPU), add a free [Xeno-canto API key](https://xeno-canto.org/account) in the config cell, run all cells top to bottom.

**App (locally):**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Note: requires system packages `ffmpeg` and `libsndfile1` for audio decoding (see `packages.txt`; on Debian/Ubuntu run `sudo apt-get install ffmpeg libsndfile1`).

## Author

Dhayal R — [GitHub](https://github.com/Dhayalramesh) · [LinkedIn](https://linkedin.com/in/dhayalsr)
