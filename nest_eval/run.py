"""Run NEST evaluation end-to-end.

Pipeline:
1. Authenticate with HuggingFace (HF_TOKEN env, .env, or `huggingface-cli login`).
2. Load the NEST dataset (mahuynh/NEST).
3. Compose prompts via nest_eval.build_prompts and write a questions CSV to
   data/experiment_files/<exp_name>.csv.
4. Hand off to evaluation.run_evaluation with --benchmark-path .

Outputs land at results/<exp_name>/<model_name>/.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("nest_eval")


HF_DATASET_REPO = "mahuynh/NEST"


def _load_dotenv_if_present(repo_root: Path) -> None:
    """Load HF_TOKEN from a repo-root .env if present. No external dep."""
    dotenv = repo_root / ".env"
    if not dotenv.exists():
        return
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def authenticate_huggingface(repo_root: Path) -> None:
    """Try HF_TOKEN env -> .env -> existing hf cache. Login only if a token is found."""
    _load_dotenv_if_present(repo_root)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("Authenticated with HuggingFace via HF_TOKEN")
        return
    # Fall through: rely on ~/.cache/huggingface/token (huggingface-cli login).
    logger.info("No HF_TOKEN set; relying on existing huggingface-cli login cache.")


def load_nest(split: str, dataset_repo: str = HF_DATASET_REPO):
    from datasets import load_dataset
    ds = load_dataset(dataset_repo, "NEST", split=split)
    logger.info(f"Loaded {dataset_repo} (split={split}, n={len(ds)})")
    return ds


def main():
    p = argparse.ArgumentParser(description="Run NEST evaluation")

    # Mode flags
    p.add_argument("--use-audio", action="store_true",
                   help="Use the audio segment .mp4s instead of silent video. "
                        "(Visual+audio mode; segments must exist under data/segments/<dataset>/audio/.)")
    p.add_argument("--blind", action="store_true",
                   help="Strip media — text-only evaluation.")

    # Build-time options
    p.add_argument("--name", default=None,
                   help="Optional suffix appended to the experiment name.")
    p.add_argument("--split", default="test", help="Dataset split (default: test)")
    p.add_argument("--dataset-repo", default=HF_DATASET_REPO,
                   help=f"HF dataset repo to load (default: {HF_DATASET_REPO}). "
                        "Override to test against a different/gated release, "
                        "e.g. a private repo you have HF_TOKEN access to.")
    p.add_argument("--rebuild", action="store_true",
                   help="Regenerate the questions CSV even if it already exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of questions (for smoke tests / dev).")
    p.add_argument("--require-media", action="store_true",
                   help="Filter out rows whose media file is missing on disk.")

    # Evaluator pass-through
    p.add_argument("--model", default="vllm/Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--judge-model", default="vllm/Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--judge-gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--judge-max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--judge-port", type=int, default=9001)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--log-level", default="INFO")

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    repo_root = Path(__file__).resolve().parent.parent

    # Mode = combination of (visual|blind) × (no-audio|with-transcript).
    if args.blind and args.use_audio:
        mode = "audio_only"      # text + transcript, no video
    elif args.blind:
        mode = "blind"           # text only
    elif args.use_audio:
        mode = "audio_visual"    # silent video + transcript
    else:
        mode = "visual"          # silent video only
    exp_name = f"{mode}_{args.name}" if args.name else mode

    # Audio mode requires precomputed per-dataset transcripts under
    # data/transcripts/. We merge them into a single file the evaluator reads.
    transcripts_dir = repo_root / "data" / "transcripts"
    transcript_file = None
    if args.use_audio:
        if not transcripts_dir.is_dir() or not any(transcripts_dir.glob("*.json")):
            p.error(f"--use-audio requires per-dataset JSONs in {transcripts_dir} "
                    f"(run `python -m nest_eval.transcribe` first).")
        merged: dict = {}
        for f in sorted(transcripts_dir.glob("*.json")):
            if f.name == "_merged.json":
                continue
            merged.update(json.loads(f.read_text()))
        transcript_file = transcripts_dir / "_merged.json"
        transcript_file.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        logger.info(f"Merged {len(merged)} transcripts into {transcript_file}")

    exp_dir = repo_root / "data" / "experiment_files"
    exp_dir.mkdir(parents=True, exist_ok=True)
    questions_csv = exp_dir / f"{exp_name}.csv"

    # Build prompts CSV
    if args.rebuild or not questions_csv.exists():
        authenticate_huggingface(repo_root)
        ds = load_nest(split=args.split, dataset_repo=args.dataset_repo)
        df = ds.to_pandas()

        from nest_eval.build_prompts import build_dataframe
        out_df = build_dataframe(df, blind=args.blind)

        if args.require_media and not args.blind:
            mask = out_df["media_path"].apply(
                lambda rel: (repo_root / rel).exists() if rel else False)
            n_missing = int((~mask).sum())
            if n_missing:
                logger.warning(f"Filtering out {n_missing} rows with missing media")
            out_df = out_df[mask].reset_index(drop=True)

        if args.limit is not None:
            out_df = out_df.head(args.limit).reset_index(drop=True)

        out_df.to_csv(questions_csv, index=False)
        logger.info(f"Wrote {len(out_df)} questions to {questions_csv}")
    else:
        logger.info(f"Reusing existing questions file: {questions_csv} "
                    f"(use --rebuild to regenerate)")

    # Run evaluator
    model_name = args.model.split("/")[-1]
    output_dir = repo_root / "results" / exp_name / model_name
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "evaluation.run_evaluation",
        "--model", args.model,
        "--judge-model", args.judge_model,
        "--questions-file", str(questions_csv),
        # The evaluator computes its media root as <benchmark_path>/data,
        # so passing the repo root makes the media root resolve to data/.
        "--benchmark-path", ".",
        "--output-dir", str(output_dir),
        "--cache-dir", str(cache_dir),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--judge-gpu-memory-utilization", str(args.judge_gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--judge-max-model-len", str(args.judge_max_model_len),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--batch-size", str(args.batch_size),
        "--port", str(args.port),
        "--judge-port", str(args.judge_port),
        "--log-level", args.log_level,
        "--log-file", str(output_dir / "evaluation.log"),
    ]
    if args.no_cache:
        cmd.append("--no-cache")
    if args.blind:
        cmd.append("--no-media")
    if args.use_audio:
        cmd.extend(["--transcript-file", str(transcript_file)])

    logger.info("Launching evaluator: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
