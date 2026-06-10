import os
import torch
from pathlib import Path

from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# ==========================
# CONFIG
# ==========================

AUDIO_DIR = "data/audio"
TRANSCRIPT_DIR = "data/transcripts"
OUTPUT_DIR = "mms300m_annotations"

LANGUAGE = "eng"
ROMANIZE = False

# ==========================
# VALIDATION
# ==========================

if not Path(AUDIO_DIR).exists():
    print(f"ERROR: Audio directory not found: {Path(AUDIO_DIR).absolute()}")
    exit(1)

if not Path(TRANSCRIPT_DIR).exists():
    print(f"ERROR: Transcript directory not found: {Path(TRANSCRIPT_DIR).absolute()}")
    exit(1)

# ==========================
# SETUP
# ==========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

alignment_model, alignment_tokenizer = load_alignment_model(
    device=device,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
)

# ==========================
# PROCESS FILES
# ==========================

wav_files = sorted(Path(AUDIO_DIR).glob("*.wav"))

print(f"Found {len(wav_files)} audio files")

if len(wav_files) == 0:
    print("ERROR: No audio files found!")
    exit(1)

files_processed = 0
files_failed = 0
files_skipped = 0

for wav_path in wav_files:

    stem = wav_path.stem
    transcript_path = Path(TRANSCRIPT_DIR) / f"{stem}.txt"

    if not transcript_path.exists():
        print(f"Missing transcript: {transcript_path}")
        files_skipped += 1
        continue

    print(f"\nProcessing: {stem}")

    try:
        # --------------------------
        # Load audio
        # --------------------------

        waveform = load_audio(
            str(wav_path),
            alignment_model.dtype,
            alignment_model.device,
        )

        # --------------------------
        # Read transcript
        # --------------------------

        with open(transcript_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print("Empty transcript")
            files_skipped += 1
            continue

        # --------------------------
        # Generate emissions
        # --------------------------

        emissions, stride = generate_emissions(
            alignment_model,
            waveform,
            batch_size=16,
        )

        # --------------------------
        # Preprocess transcript
        # --------------------------

        tokens_starred, text_starred = preprocess_text(
            text,
            romanize=ROMANIZE,
            language=LANGUAGE,
        )

        # --------------------------
        # Align
        # --------------------------

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(
            tokens_starred,
            segments,
            blank_token,
        )

        word_timestamps = postprocess_results(
            text_starred,
            spans,
            stride,
            scores,
        )

        # DEBUG: Print structure to see what we're working with
        if word_timestamps:
            print(f"DEBUG: First item type: {type(word_timestamps[0])}")
            print(f"DEBUG: First item: {word_timestamps[0]}")
            if isinstance(word_timestamps[0], dict):
                print(f"DEBUG: Dictionary keys: {word_timestamps[0].keys()}")

        # --------------------------
        # Save output
        # --------------------------

        output_file = Path(OUTPUT_DIR) / f"{stem}.txt"

        with open(output_file, "w", encoding="utf-8") as out:

            out.write("word,start,end\n")

            for word in word_timestamps:
                out.write(
                    f"{word['text']},{word['start']:.3f},{word['end']:.3f}\n"
                )

        print(f"Saved: {output_file}")
        files_processed += 1

    except Exception as e:
        print(f"FAILED: {stem}")
        print(e)
        files_failed += 1

print(f"\n{'='*50}")
print(f"Files processed: {files_processed}")
print(f"Files failed:    {files_failed}")
print(f"Files skipped:   {files_skipped}")
print(f"Total:           {len(wav_files)}")
print("Done.")