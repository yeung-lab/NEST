"""Build evaluation prompts from NEST rows.

NEST ships each row with a `prefix` field that already contains the
description block and (for I-prefix questions) the interpersonal critical
rule. This module just concatenates `prefix` + `question_text`, shuffles
multiple-choice option order with a per-question seed, and remaps the
ground-truth letter. Open-ended rows keep their text ground_truth and
have no options.
"""

import json
import random
import re
from typing import Optional, Tuple

import pandas as pd

OUTPUT_COLUMNS = [
    "question_id", "question_group", "question_type", "media_type",
    "question_text", "media_path", "ground_truth", "options",
    "human_responses", "annotators", "data_source", "difficulty",
    "video_id", "frame_number", "start_frame", "end_frame",
    "start_time", "end_time",
]


def compose_prompt(prefix: str, question_text: str) -> str:
    prefix = (prefix or "").strip()
    if not prefix or prefix.lower() == "nan":
        return question_text
    return f"{prefix}\nQuestion: {question_text}"


def shuffle_options(options_json: str, gt_letter: str, seed: str
                    ) -> Tuple[str, str]:
    """Shuffle option order with a per-question seed; remap the GT letter.

    Strips the leading letter prefix (e.g. "A. ", "A) ") from each option,
    shuffles the texts, and re-letters from A. Returns (json_options, new_letter).
    """
    items = json.loads(options_json)
    texts = [re.sub(r"^[A-Z][.):]\s*", "", str(o)).strip() for o in items]

    gt_idx = ord(gt_letter.strip().upper()) - ord("A")
    if not 0 <= gt_idx < len(texts):
        raise ValueError(f"ground_truth letter {gt_letter!r} out of range for "
                         f"{len(texts)} options")
    gt_text = texts[gt_idx]

    rng = random.Random(seed)
    rng.shuffle(texts)

    letters = [chr(ord("A") + i) for i in range(len(texts))]
    new_options = [f"{l}. {t}" for l, t in zip(letters, texts)]
    new_letter = letters[texts.index(gt_text)]
    return json.dumps(new_options), new_letter


def media_path_for(row: pd.Series, blind: bool) -> str:
    """Path relative to repo root (the evaluator's --benchmark-path).

    Always points at the silent .mp4. Audio is delivered via Whisper
    transcripts injected into the prompt at evaluation time, not by feeding
    the audio container to the VLM. Blind mode emits an empty path which
    the evaluator strips when --no-media is passed.
    """
    if blind:
        return ""
    return f"data/segments/{row['dataset']}/silent/{row['segment_id']}.mp4"


def build_row(row: pd.Series, blind: bool) -> Optional[dict]:
    qid = str(row["question_id"])
    qtype = str(row["question_type"])
    prompt = compose_prompt(row.get("prefix", ""), row["question_text"])

    if qtype == "multiple_choice":
        options_json, gt = shuffle_options(row["options"], str(row["ground_truth"]), seed=qid)
    elif qtype == "open_ended":
        options_json = None
        gt = str(row["ground_truth"])
    else:
        raise ValueError(f"unknown question_type {qtype!r} for {qid}")

    return {
        "question_id": qid,
        "question_group": row.get("question_group"),
        "question_type": qtype,
        "media_type": "video",
        "question_text": prompt,
        "media_path": media_path_for(row, blind),
        "ground_truth": gt,
        "options": options_json,
        "human_responses": None,
        "annotators": None,
        "data_source": row.get("dataset"),
        "difficulty": None,
        "video_id": row.get("video_id"),
        "frame_number": row.get("keyframe"),
        "start_frame": row.get("start_frame"),
        "end_frame": row.get("end_frame"),
        "start_time": None,
        "end_time": None,
    }


def build_dataframe(nest_df: pd.DataFrame, blind: bool) -> pd.DataFrame:
    rows = [build_row(r, blind) for _, r in nest_df.iterrows()]
    rows = [r for r in rows if r is not None]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
