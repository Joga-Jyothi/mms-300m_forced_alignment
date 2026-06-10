import json
import csv
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

HUMAN_DIR = Path("human_annotations")
MMS300M_DIR = Path("mms300m_annotations")

THRESHOLD_MS = 80
THRESHOLD_SEC = THRESHOLD_MS / 1000.0

# =====================================================
# STATS
# =====================================================

total_files = 0
red_flag_files = 0

total_words = 0
flagged_words = 0

total_matching_words = 0
total_start_error_sec = 0.0
total_end_error_sec = 0.0
total_combined_error_sec = 0.0

# =====================================================
# REPORTS
# =====================================================

red_flag_report = []
word_error_report = []

# =====================================================
# PARSE HUMAN TXT
# =====================================================

def load_human_annotation(txt_file):
    words = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 3:
                continue

            start = float(parts[0])
            end = float(parts[1])
            word = " ".join(parts[2:]).upper()

            words.append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
                }
            )

    return words


# =====================================================
# PARSE MMS300M TXT
# =====================================================

def load_mms300m_annotation(txt_file):
    words = []

    with open(txt_file, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            if not row or not row.get("word"):
                continue

            words.append(
                {
                    "word": row["word"].strip().upper(),
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                }
            )

    return words


# =====================================================
# MAIN COMPARISON
# =====================================================

for human_file in sorted(HUMAN_DIR.glob("*_Annotated.txt")):

    stem = human_file.stem.replace("_Annotated", "")

    mms_file = MMS300M_DIR / f"{stem}.txt"

    if not mms_file.exists():
        print(f"Missing MMS300M file: {stem}.txt")
        continue

    total_files += 1

    human_words = load_human_annotation(human_file)
    mms_words = load_mms300m_annotation(mms_file)

    file_has_flag = False
    file_flagged_words = 0

    # -------------------------------------------------
    # WORD COUNT MISMATCH
    # -------------------------------------------------

    if len(human_words) != len(mms_words):

        red_flag_files += 1

        red_flag_report.append(
            {
                "file": stem,
                "reason": "WORD_COUNT_MISMATCH",
                "human_words": len(human_words),
                "mms300m_words": len(mms_words),
            }
        )

        print(
            f"[FLAG] {stem} word count mismatch "
            f"({len(human_words)} vs {len(mms_words)})"
        )

        continue

    # -------------------------------------------------
    # WORD BY WORD COMPARISON
    # -------------------------------------------------

    for idx in range(len(human_words)):

        h = human_words[idx]
        m = mms_words[idx]

        total_words += 1

        # ---------------------------------------------
        # WORD MISMATCH
        # ---------------------------------------------

        if h["word"] != m["word"]:

            file_has_flag = True
            file_flagged_words += 1
            flagged_words += 1

            word_error_report.append(
                {
                    "file": stem,
                    "word_index": idx,
                    "human_word": h["word"],
                    "mms300m_word": m["word"],
                    "start_diff_ms": "",
                    "end_diff_ms": "",
                    "reason": "WORD_MISMATCH",
                }
            )

            continue

        # ---------------------------------------------
        # TIMESTAMP DIFFERENCE
        # ---------------------------------------------

        start_diff = abs(h["start"] - m["start"])
        end_diff = abs(h["end"] - m["end"])
        combined_diff = start_diff + end_diff

        total_matching_words += 1
        total_start_error_sec += start_diff
        total_end_error_sec += end_diff
        total_combined_error_sec += combined_diff

        if (
            start_diff > THRESHOLD_SEC
            or
            end_diff > THRESHOLD_SEC
        ):

            file_has_flag = True
            file_flagged_words += 1
            flagged_words += 1

            word_error_report.append(
                {
                    "file": stem,
                    "word_index": idx,
                    "human_word": h["word"],
                    "mms300m_word": m["word"],
                    "start_diff_ms": round(start_diff * 1000, 2),
                    "end_diff_ms": round(end_diff * 1000, 2),
                    "reason": "TIMESTAMP_DIFF",
                }
            )

    if file_has_flag:

        red_flag_files += 1

        red_flag_report.append(
            {
                "file": stem,
                "flagged_words": file_flagged_words,
                "total_words": len(human_words),
            }
        )

# =====================================================
# SUMMARY
# =====================================================

percentage = 0.0

if total_words > 0:
    percentage = (flagged_words / total_words) * 100

mean_start_error_ms = 0.0
mean_end_error_ms = 0.0
mean_combined_error_ms = 0.0

if total_matching_words > 0:

    mean_start_error_ms = (
        total_start_error_sec / total_matching_words
    ) * 1000

    mean_end_error_ms = (
        total_end_error_sec / total_matching_words
    ) * 1000

    mean_combined_error_ms = (
        total_combined_error_sec / total_matching_words
    ) * 1000

summary = {
    "files_compared": total_files,
    "red_flag_files": red_flag_files,
    "total_words": total_words,
    "flagged_words": flagged_words,
    "percentage_flagged_words": round(percentage, 4),
    "mean_start_error_ms": round(mean_start_error_ms, 2),
    "mean_end_error_ms": round(mean_end_error_ms, 2),
    "mean_combined_error_ms": round(mean_combined_error_ms, 2),
    "matching_words_evaluated": total_matching_words,
}

# =====================================================
# SAVE SUMMARY
# =====================================================

with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

# =====================================================
# SAVE RED FLAG FILES
# =====================================================

with open("red_flag_files.csv", "w", newline="", encoding="utf-8") as f:

    fieldnames = [
        "file",
        "reason",
        "human_words",
        "mms300m_words",
        "flagged_words",
        "total_words",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )

    writer.writeheader()

    for row in red_flag_report:
        writer.writerow(row)

# =====================================================
# SAVE WORD ERRORS
# =====================================================

with open("word_level_errors.csv", "w", newline="", encoding="utf-8") as f:

    fieldnames = [
        "file",
        "word_index",
        "human_word",
        "mms300m_word",
        "start_diff_ms",
        "end_diff_ms",
        "reason",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(word_error_report)

# =====================================================
# PRINT SUMMARY
# =====================================================

print("\n" + "=" * 50)
print(f"Files compared       : {total_files}")
print(f"Red flag files       : {red_flag_files}")
print(f"Total words          : {total_words}")
print(f"Words > 80 ms        : {flagged_words}")
print(f"Percentage flagged   : {percentage:.2f}%")
print(f"Mean Start Error     : {mean_start_error_ms:.2f} ms")
print(f"Mean End Error       : {mean_end_error_ms:.2f} ms")
print(f"Mean Combined Error  : {mean_combined_error_ms:.2f} ms")
print("=" * 50)

print("\nGenerated:")
print("summary.json")
print("red_flag_files.csv")
print("word_level_errors.csv")