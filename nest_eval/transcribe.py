"""Generate Whisper transcripts for NEST audio segments.

Walks `data/segments/<dataset>/audio/*.mp4`, runs Whisper, and writes
one JSON per dataset to `data/transcripts/<dataset>.json`. Each file maps
the segment's *silent* path (the one in question CSVs under `--use-audio`)
to its transcript text. `nest_eval.run` merges all per-dataset files
before handing them to the evaluator.

Usage (from repo root):
    python -m nest_eval.transcribe                  # all datasets in data/segments
    python -m nest_eval.transcribe --resume         # skip segments already transcribed
    python -m nest_eval.transcribe --model-size large-v3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import whisper
from tqdm import tqdm

logger = logging.getLogger(__name__)


def silent_key_for(audio_path: Path, dataset: str) -> str:
    """`data/segments/<dataset>/silent/<seg>.mp4` — what the evaluator stores
    as Question.media_path under --benchmark-path . in --use-audio mode."""
    return os.path.join("data", "segments", dataset, "silent", audio_path.name)


def transcribe_dataset(model, dataset_dir: Path, dataset: str, output: Path,
                       resume: bool) -> tuple[int, int]:
    """Returns (n_transcribed, n_failed)."""
    audio_dir = dataset_dir / "audio"
    audio_files = sorted(audio_dir.glob("*.mp4"))
    if not audio_files:
        logger.warning(f"[{dataset}] no .mp4s under {audio_dir}")
        return 0, 0

    transcripts: dict[str, str] = {}
    if resume and output.exists():
        transcripts = json.loads(output.read_text())
        logger.info(f"[{dataset}] resuming with {len(transcripts)} existing transcripts")

    output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    n_done = 0
    for i, audio_path in enumerate(tqdm(audio_files, file=sys.stderr,
                                        unit="seg", desc=dataset)):
        key = silent_key_for(audio_path, dataset)
        if resume and key in transcripts:
            continue
        try:
            transcripts[key] = model.transcribe(str(audio_path))["text"].strip()
            n_done += 1
        except Exception as e:
            logger.error(f"[{dataset}] FAILED {audio_path.name}: {e}")
            transcripts[key] = ""
            failed += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(audio_files):
            output.write_text(json.dumps(transcripts, indent=2, ensure_ascii=False))

    logger.info(f"[{dataset}] wrote {len(transcripts)} transcripts to {output}")
    return n_done, failed


def main():
    parser = argparse.ArgumentParser(description="Whisper transcripts for NEST audio segments")
    parser.add_argument("--segments-dir", type=Path, default=Path("data/segments"),
                        help="Root containing <dataset>/audio/*.mp4 (default: data/segments)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/transcripts"),
                        help="Directory for per-dataset transcript JSONs (default: data/transcripts)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of dataset subdirs to transcribe (default: all under --segments-dir)")
    parser.add_argument("--model-size", default="large-v3",
                        help="Whisper model size (default: large-v3)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip segments already present in the per-dataset JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.datasets:
        dataset_dirs = [args.segments_dir / d for d in args.datasets]
    else:
        dataset_dirs = sorted(p for p in args.segments_dir.iterdir()
                              if p.is_dir() and (p / "audio").is_dir())
    if not dataset_dirs:
        logger.error(f"No dataset directories with audio/ under {args.segments_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Whisper {args.model_size} on {device}")
    model = whisper.load_model(args.model_size, device=device)

    total_done = total_failed = 0
    for dataset_dir in dataset_dirs:
        dataset = dataset_dir.name
        output = args.output_dir / f"{dataset}.json"
        n, f = transcribe_dataset(model, dataset_dir, dataset, output, args.resume)
        total_done += n
        total_failed += f

    logger.info(f"Done. {total_done} new transcripts; {total_failed} failed.")


if __name__ == "__main__":
    main()
