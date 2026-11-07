# Running the NEST evaluation

End-to-end pipeline: load NEST from HuggingFace, compose prompts, run a
vision-language model, score the responses.

## Expected dataset structure

`nest_eval` reads videos from `<repo>/data/segments/`:

```
data/
  segments/
    childplay/
      silent/<segment_id>.mp4   # default — visual mode
      audio/<segment_id>.mp4    # for --use-audio
    childes/
      silent/<segment_id>.mp4
      audio/<segment_id>.mp4
```

`<segment_id>` is the NEST-anonymized identifier (e.g.
`childplay0023_seg02`). The downloader pipeline that produces this layout
lives in [`DATA.md`](DATA.md).

## HuggingFace authentication

NEST is gated. First request access on
<https://huggingface.co/datasets/mahuynh/NEST>, then provide a token from
<https://huggingface.co/settings/tokens> via any of:

1. A `.env` file at the repo root with `HF_TOKEN=hf_...` (recommended;
   already gitignored).
2. The `HF_TOKEN` environment variable.
3. The standard cache: `huggingface-cli login` once and paste a token.

## Run

```bash
bash nest_eval/run.sh

# Pick a model:
MODEL=vllm/Qwen/Qwen3-VL-4B-Instruct bash nest_eval/run.sh

# Use the larger judge (for final reported numbers):
JUDGE_MODEL="$JUDGE_MODEL_LARGE" bash nest_eval/run.sh

# Blind (text-only):
EXTRA_FLAGS="--blind" bash nest_eval/run.sh

# Audio-visual (silent video + Whisper transcript):
python -m nest_eval.transcribe        # one-time; writes data/transcripts/
EXTRA_FLAGS="--use-audio" bash nest_eval/run.sh

# Audio-only (text + transcript, no video):
EXTRA_FLAGS="--blind --use-audio" bash nest_eval/run.sh
```

Outputs land at `results/<exp>/<model>/evaluation_results.json`, with
the generated questions CSV at `data/experiment_files/<exp>.csv`.
`<exp>` is `visual` by default (`visual_<name>` if you pass `--name`).

## Main flags

Set via `MODEL=...`/`JUDGE_MODEL=...` (env), or pass through with
`EXTRA_FLAGS="..."`. Full list: `python -m nest_eval.run --help`.

| Flag / env | Default | Purpose |
|---|---|---|
| `MODEL` | Qwen3-VL-8B | Model under test (any vLLM-supported VLM, or a litellm API id). |
| `JUDGE_MODEL` | `$JUDGE_MODEL_TEST` (Qwen3-VL-4B) | Judge for open-ended questions. Use `$JUDGE_MODEL_LARGE` (Qwen3-VL-32B) for paper numbers. |
| `--limit N` | unset | Cap to first N questions (smoke tests). |
| `--require-media` | off | Drop rows whose segment .mp4 is missing on disk. |
| `--rebuild` | off | Regenerate the prompts CSV. |
| `--name SUFFIX` | none | Suffix appended to the experiment name. |
| `--no-cache` | off | Disable the response cache. |

## Reproducibility constants

`run.sh` sets these to mirror the configuration used for our reported
numbers: `MAX_TOKENS=256`, `MAX_MODEL_LEN=32768`, `GPU_MEM=0.7`,
`TEMPERATURE=0.1`, `BATCH_SIZE=1`. Override by editing `run.sh` or
prepending `EXTRA_FLAGS="--max-tokens ..."`.
