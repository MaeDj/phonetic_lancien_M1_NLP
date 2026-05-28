# French Vowel Acoustic Analysis — M1 NLP/TAL Phonetics Project

Acoustic analysis of French oral and nasal vowels across four speakers (2 male, 2 female), covering the full pipeline from speech recording to vowel space visualization.

**Authors**: Alexandre CHOPLIN and Maé DUGOUA-JACQUES
**Program**: Master 1 NLP/TAL — IDMC, Université de Lorraine (S8)  
**Supervisor**: Mélanie Lancien

---

## Project Overview

This project investigates inter-speaker variation in French vowel acoustics. Four speakers each read 10 French sentences designed to elicit French oral and nasal vowels. The resulting recordings are processed through a three-stage pipeline:

1. **Praat-based acoustic feature extraction** — F0, F1–F4, HNR, intensity, jitter, shimmer, CoG sampled every 5 ms
2. **Formant quality verification** — measurements validated against published French phonetic references (Gendrot & Adda-Decker 2005; Fougeron & Smith 1993)
3. **Vowel space analysis** — statistical summaries, vowel space plots, cross-speaker comparisons, and Lobanov normalisation

---

## Corpus

### Speakers

| ID | Gender | age|
|----|--------|
| S1_Female | Female | 20's|
| S1_Male | Male | 20's |
| S2_Female | Female | 50's |
| S2_Male | Male | 50's |

### Recording Sentences (10 per speaker)

1. *Lucile a mis sa belle jupe sur la chaise.*
2. *La poule picore la pâte rouge.*
3. *Le loup gris passe par la salle vide.*
4. *Arthur a vu une puce sur sa tête.*
5. *C'est une bête avec de grandes pattes.*
6. *La loupe de cette dame est très utile.*
7. *Il fume sa pipe toute la journée.*
8. *La chute de la tarte sur la table.*
9. *Lucas boit une soupe chaude.*
10. *La fête se passe dans la rue de la Harpe.*

Sentences were chosen to contain phonetic contrasts covering French rounded/unrounded front and back vowels (i, y, e, ɛ, a, ɑ, œ, ø, ɔ, o, u) and nasal vowels (ɑ̃, ɛ̃, ɔ̃).

### Recording Parameters

- Sample rate: **22 050 Hz**, mono
- Format: **WAV**
- Software: `algo/recording_software.py` (press Enter to start/stop each sentence)

---

## Repository Structure

```
.
├── algo/
│   ├── recording_software.py        # Interactive speech recorder
│   └── fixed_formants_extraction.praat  # Praat script for acoustic extraction
│
├── {Speaker}/                       # One folder per speaker
│   ├── wav/                         # Raw .wav recordings
│   ├── textGrid/                    # Praat TextGrid phone-level annotations
│   ├── result/                      # Raw acoustic CSVs (one per sentence)
│   └── txt/                         # Sentence transcriptions (S2 speakers)
│
├── quality_verified/
│   ├── {Speaker}/                   # Validated formant CSVs
│   ├── formant_quality_verification.py  # Quality-verification script
│   └── vowel_analysis.py            # Analysis & plotting script (alt. copy)
│
├── plots/                           # All generated figures (PNG)
├── tables/                          # Statistical summary CSVs
│
├── formant_quality_verification.py  # Quality-verification script (root copy)
├── vowel_analysis.py                # Main analysis & plotting script
│
└── report/
    └── Phonetic_M1_NLP_CHOPLIN_DUGOUA.pdf  # Final written report
```

---

## Pipeline

### Step 1 — Speech Recording

```bash
python algo/recording_software.py
# Prompt: Enter speaker name (e.g., S1_Male, S2_Female)
```

Recordings are saved to `audio/{speaker_name}_sentence_XX.wav`.

### Step 2 — Acoustic Feature Extraction (Praat)

Open `algo/fixed_formants_extraction.praat` in Praat and configure:

| Parameter | Description |
|-----------|-------------|
| `textgrids_folder` | Path to TextGrid files for the speaker |
| `wavefiles_folder` | Path to WAV files for the speaker |
| `results_file` | Output directory for CSV results |
| `speakers_gender` | `F` for female (ceiling 5500 Hz), `M` for male (5000 Hz) |

The script extracts **every 5 ms**:
- **F0** (pitch, Hz)
- **F1–F4** (formants, Hz) via Burg's method
- **HNR** (harmonics-to-noise ratio, dB)
- **Intensity** (dB)
- **Jitter**, **Shimmer**
- **CoG** (centre of gravity), Kurtosis, Skewness

Results are written as tab-separated CSVs to `{Speaker}/result/`.

### Step 3 — Quality Verification

```bash
python formant_quality_verification.py
```

For each 5 ms measurement window, the script:
1. Requires a defined F0 (50–500 Hz range)
2. Requires all formants (F1–F4) to be defined
3. Locates the TextGrid phone that **strictly contains** the window
4. Checks F1, F2, F3, F4 against per-vowel thresholds from published references

**Formant references used:**

| Vowel type | Reference |
|------------|-----------|
| French oral vowels (i, y, e, ɛ, a, œ, ø, ɔ, o, u) | Gendrot & Adda-Decker (2005), Table 4 |
| French nasal vowels (ɑ̃, ɛ̃, ɔ̃) | Fougeron & Smith (1993) ± 300 Hz tolerance |
 
Validated rows are written to `quality_verified/{Speaker}/{sentence}_quality.csv`.

### Step 4 — Vowel Space Analysis

```bash
python vowel_analysis.py
```

Reads quality-verified CSVs and generates all plots and tables below.

---

## Outputs

### Plots (`plots/`)

| File | Description |
|------|-------------|
| `{Speaker}_vowel_space.png` | F1×F2 scatter with category means and convex hull |
| `{Speaker}_F1_F2_ellipses.png` | 1-SD confidence ellipses per vowel category |
| `all_speakers_overlay.png` | All 4 speakers superposed (convex hulls) |
| `female_comparison.png` | S1_Female vs S2_Female |
| `male_comparison.png` | S1_Male vs S2_Male |
| `S1_comparison.png` | S1_Female vs S1_Male |
| `S2_comparison.png` | S2_Female vs S2_Male |
| `formant_boxplots.png` | F1 & F2 distributions per vowel and speaker |
| `facet_per_vowel.png` | Per-vowel F1×F2 scatter grid, all speakers |
| `f0_distribution.png` | F0 (pitch) KDE per speaker |
| `lobanov_normalised.png` | Lobanov z-score normalised vowel space |

Axes follow the phonetic convention: F2 inverted (front vowels left), F1 inverted (high vowels top).

### Tables (`tables/`)

| File | Description |
|------|-------------|
| `{Speaker}_stats.csv` | Mean, SD, N per vowel for F1, F2, F3 |
| `all_speakers_stats.csv` | Same, all speakers stacked |
| `vowel_space_areas.csv` | Convex hull area (Hz²) per speaker |
| `euclidean_distances.csv` | Pairwise vowel Euclidean distances (Hz) per speaker |

---

## Installation

```bash
pip install sounddevice scipy matplotlib numpy pandas
```

[Praat](https://www.fon.hum.uva.nl/praat/) is required for Step 2 (acoustic extraction). It is not a Python package — download and install it separately.

---
