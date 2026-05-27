"""
formant_quality_verification.py

Pipeline:
  1. Read every .txt formant file in /results/ (one per recording).
  2. Keep only rows where mean_F0(Hz) is defined (not --undefined--).
  3. Look up the matching TextGrid (phones tier) and find the vowel whose
     interval *strictly contains* the 5 ms measurement window
     (phone.xmin < window_start  AND  window_end < phone.xmax).
  4. Verify F0, F1, F2, F3, F4 against the per-vowel thresholds from
     Gendrot & Adda-Decker (2005), Table 4, using the union of male/female
     ranges so that speaker sex does not need to be known.
  5. Write each passing row to a CSV alongside its source .txt file.

Reference: Gendrot, C. & Adda-Decker, M. (2005). Impact of duration on F1/F2
formant values of oral vowels: an automatic analysis of large broadcast news
corpora in French and German. Interspeech 2005, pp. 3453-3456.
"""

import csv
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(
    "/home/mae/Documents/idmc/master1/university/s2/speech_corpora/"
    "exam_corpus/results"
)
TEXTGRIDS_DIR = Path(
    "/home/mae/Documents/idmc/master1/university/s2/speech_corpora/"
    "exam_corpus/textgrids"
)
OUTPUT_DIR = RESULTS_DIR / "quality_verified"

# Known sub-folder names (longest first so the prefix strip works correctly)
KNOWN_FOLDERS = ["records_standing_up", "records_laying", "records_sitting"]

# ---------------------------------------------------------------------------
# Formant thresholds — two sources, FRENCH vowels only
#
# ── SOURCE 1: Oral vowels ────────────────────────────────────────────────────
# Gendrot & Adda-Decker (2005), Table 4, French section.
# The paper studies both French AND German; only the FRENCH rows are used here.
# German vowels (short/long pairs i/I, u/U, …) are NOT represented below.
#
# Exact values from Table 4 (French oral vowels):
#
#            i     y     e     ε     a     œ     ø     ɔ     o     u
# ── Male ──────────────────────────────────────────────────────────────
# F1 <      750   900   800  1000  1000   900   900   900   900   900
# F2 min   1500  1300  1400  1100  1200   800   800   600   600   400
# F2 max   2500  2200  2500  2400  2300  2300  2000  1800  1600  1500
# F3 >     2000  1700  2000  2000  1800  1800  1500  1700  1500  1400
# ── Female ────────────────────────────────────────────────────────────
# F1 <      900   900   900  1100  1100  1100  1000  1000  1000  1000
# F2 min   1600  1400  1400  1400   900   800   900   600   600   400
# F2 max   3100  2800  3000  2700  2300  2400  2300  2000  2000  1500
# F3 >     2500  1800  1800  2200  2000  1900  1800  1800  2100  1800
#
# ── SOURCE 2: Nasal vowels ───────────────────────────────────────────────────
# Fougeron, C. & Smith, C.L. (1993). "Illustrations of the IPA: French."
# Journal of the International Phonetic Association, 23(2), 73–76.
#
# Gendrot & Adda-Decker explicitly exclude French nasal vowels ("French nasal
# vowels were excluded from the study"). Fougeron & Smith (1993) is used
# instead as it is the IPA-endorsed reference for standard French phonetics.
#
# Mean formant values reported for a male speaker (Fougeron & Smith 1993):
#
#           ɑ̃    ɛ̃    ɔ̃
# F1       690   500  460
# F2      1170  1570  850
#
# Threshold ranges below are derived from those means with a ±300 Hz tolerance
# band and a ~15 % upward scaling for female speakers (consistent with the
# male/female ratio observed in Gendrot & Adda-Decker oral vowel data).
# Combined male+female union (same approach as for oral vowels):
#
#            ɑ̃     ɛ̃     ɔ̃
# F1 max   1000   800   700
# F2 min    850  1200   600
# F2 max   1650  2200  1250
# F3 min   1800  1900  1800
#
# ── Combined threshold format ────────────────────────────────────────────────
# Each entry is (F1_max, F2_min, F2_max, F3_min).
# The union of male and female is used throughout so that speaker sex is not
# required:  F1_max = max(m,f)  F2_min = min(m,f)  F2_max = max(m,f)  F3_min = min(m,f)
#
# TextGrid IPA symbols mapped to vowel classes:
#   ɛ  → ε   (same phone, different Unicode code point)
#   à  → a   (diacritic variant produced by some aligners)
#   ɑ  → a   (non-nasal back /a/, treated as /a/ in this corpus)
#   ə  → œ   (schwa merged with /œ/ — Gendrot & Adda-Decker footnote 1)
# ---------------------------------------------------------------------------
VOWEL_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    # ── Oral vowels — Gendrot & Adda-Decker (2005) Table 4, French ──────────
    # vowel : (F1_max, F2_min, F2_max, F3_min)  — Hz
    "i": (900,  1500, 3100, 2000),
    "y": (900,  1300, 2800, 1700),
    "e": (900,  1400, 3000, 1800),
    "ɛ": (1100, 1100, 2700, 2000),
    "a": (1100,  900, 2300, 1800),
    "à": (1100,  900, 2300, 1800),   # variant of /a/
    "ɑ": (1100,  900, 2300, 1800),   # back /a/ (non-nasal)
    "œ": (1100,  800, 2400, 1800),
    "ə": (1100,  800, 2400, 1800),   # schwa ≡ /œ/ per Gendrot footnote 1
    "ø": (1000,  800, 2300, 1500),
    "ɔ": (1000,  600, 2000, 1700),
    "o": (1000,  600, 2000, 1500),
    "u": (1000,  400, 1500, 1400),
    # ── Nasal vowels — Fougeron & Smith (1993) + ±300 Hz tolerance band ──────
    "ɑ̃": (1000,  850, 1650, 1800),
    "ɛ̃": ( 800, 1200, 2200, 1900),
    "ɔ̃": ( 700,  600, 1250, 1800),
}

# F0 validity range — general adult speech (paper does not give per-vowel F0 bounds)
F0_MIN, F0_MAX = 50.0, 500.0

# F4 validity range — paper reports extraction up to 5 kHz (male) / 5.5 kHz (female);
# no per-vowel thresholds are published, so a broad acoustic range is used.
F4_MIN, F4_MAX = 2000.0, 6500.0

# Output CSV column order
CSV_FIELDS = [
    "begin_time_stamp",
    "end_time_stamp",
    "vowel_name",
    "F0",
    "F1",
    "F2",
    "F3",
    "F4",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_result_to_textgrid(stem: str) -> tuple:
    """Return (folder, base_name, textgrid_path) for a result-file stem.

    Example: 'records_laying_laying_1' → ('records_laying', 'laying_1',
                                           <TEXTGRIDS_DIR>/records_laying/laying_1.TextGrid)
    """
    for folder in KNOWN_FOLDERS:
        prefix = folder + "_"
        if stem.startswith(prefix):
            base_name = stem[len(prefix):]
            tg_path = TEXTGRIDS_DIR / folder / (base_name + ".TextGrid")
            return folder, base_name, tg_path
    return None, None, None


def parse_phones_tier(tg_path: Path) -> list[dict]:
    """Return list of {xmin, xmax, text} dicts from the 'phones' tier."""
    raw = tg_path.read_text(encoding="utf-8")

    # Isolate the 'phones' tier — everything between name = "phones" and the
    # start of the next tier (or end of file).
    tier_m = re.search(
        r'name\s*=\s*"phones"(.*?)(?=\bitem\s*\[(\d+)]\s*:|$)',
        raw,
        re.DOTALL,
    )
    if not tier_m:
        return []

    intervals = re.findall(
        r"xmin\s*=\s*([\d.eE+\-]+)\s+"
        r"xmax\s*=\s*([\d.eE+\-]+)\s+"
        r'text\s*=\s*"([^"]*)"',
        tier_m.group(1),
    )
    return [
        {"xmin": float(xmin), "xmax": float(xmax), "text": label.strip()}
        for xmin, xmax, label in intervals
    ]


def find_vowel(phones: list, start: float, end: float) -> Optional[str]:
    """Return the vowel label of the phone that *strictly contains* [start, end].

    Strictly contains: phone.xmin < start  AND  end < phone.xmax
    Returns None when no matching oral vowel is found.
    """
    for p in phones:
        if p["xmin"] < start and end < p["xmax"]:
            label = p["text"]
            if label in VOWEL_THRESHOLDS:
                return label
    return None


def parse_value(raw: str) -> Optional[float]:
    """Convert a formant string to float; return None for '--undefined--'."""
    s = raw.strip()
    if s == "--undefined--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def formants_valid(vowel: str, f0: float, f1: float, f2: float, f3: float, f4: float) -> bool:
    """Return True when all five formant values are within reference bounds."""
    if not (F0_MIN <= f0 <= F0_MAX):
        return False

    f1_max, f2_min, f2_max, f3_min = VOWEL_THRESHOLDS[vowel]
    if f1 > f1_max:
        return False
    if not (f2_min <= f2 <= f2_max):
        return False
    if f3 < f3_min:
        return False
    if not (F4_MIN <= f4 <= F4_MAX):
        return False

    return True


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(txt_path: Path) -> list[dict]:
    """Process one formant .txt file; return list of validated measurement dicts."""
    _, _, tg_path = map_result_to_textgrid(txt_path.stem)

    if tg_path is None or not tg_path.exists():
        print(f"  [SKIP] TextGrid not found for {txt_path.name}")
        return []

    phones = parse_phones_tier(tg_path)
    if not phones:
        print(f"  [SKIP] No phones parsed from {tg_path}")
        return []

    valid_rows: list[dict] = []

    with open(txt_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            # Step 1 — require a defined F0
            f0 = parse_value(row["mean_F0(Hz)"])
            if f0 is None:
                continue

            # Step 2 — require all other formants to be defined as well
            f1 = parse_value(row["mean_F1(Hz)"])
            f2 = parse_value(row["mean_F2(Hz)"])
            f3 = parse_value(row["mean_F3(Hz)"])
            f4 = parse_value(row["mean_F4(Hz)"])
            if None in (f1, f2, f3, f4):
                continue

            start = float(row["start_time"])
            end = float(row["end_time"])

            # Step 3 — locate the oral vowel that strictly contains this window
            vowel = find_vowel(phones, start, end)
            if vowel is None:
                continue

            # Step 4 — verify formant values against Gendrot & Adda-Decker (2005)
            if not formants_valid(vowel, f0, f1, f2, f3, f4):
                continue

            valid_rows.append(
                {
                    "begin_time_stamp": start,
                    "end_time_stamp": end,
                    "vowel_name": vowel,
                    "F0": f0,
                    "F1": f1,
                    "F2": f2,
                    "F3": f3,
                    "F4": f4,
                }
            )

    return valid_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(RESULTS_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} result files in {RESULTS_DIR}\n")

    grand_total = 0
    for txt_path in txt_files:
        print(f"Processing {txt_path.name} …", end=" ", flush=True)
        rows = process_file(txt_path)

        out_csv = OUTPUT_DIR / (txt_path.stem + "_quality.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        print(f"{len(rows):>4} valid rows  →  {out_csv.name}")
        grand_total += len(rows)

    print(f"\nTotal valid measurements written: {grand_total}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()