# evaluation/metrics/scoring.py
"""Evaluation scoring system for multiple-choice and open questions"""

import re
import json
import hashlib
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import logging
from pydantic import BaseModel

from evaluation.models.litellm_wrapper import LiteLLMWrapper

logger = logging.getLogger(__name__)

# Bump to invalidate old cache entries when scoring methodology changes
_JUDGE_CACHE_VERSION = 2

# Pydantic model for structured LLM judge output
class JudgeEvaluation(BaseModel):
    """Structured output model for LLM judge evaluation (MMBench-Video scale)"""
    model_config = {
        "extra": "ignore",  # ignore extra fields from local models
        # OpenAI strict json_schema mode requires every property in
        # "required", even ones with a Python default (see
        # response_formats.py's MultipleChoiceResponse/OpenQuestionResponse
        # for the same pattern already used for the primary model).
        "json_schema_extra": {"required": ["score", "reasoning"]},
    }

    score: int  # 0, 1, 2, or 3 (MMBench-Video scale)
    reasoning: Optional[str] = None

class ClaimMatch(BaseModel):
    """A single atomic claim with its match status"""
    model_config = {"extra": "ignore"}
    claim: str
    matched: bool

class ClaimRecallEvaluation(BaseModel):
    """Claim-level recall evaluation for action-response descriptions"""
    model_config = {"extra": "ignore"}
    gt_claims: List[ClaimMatch]  # GT claims; matched=True if correctly covered

@dataclass
class ScoreResult:
    """Result of scoring a single question"""
    question_id: str
    question_type: str
    is_correct: bool
    score: float  # MC: 0.0 or 1.0 | open-ended: 0-3 (MMBench-Video)
    predicted_answer: str
    ground_truth: str
    predicted_reasoning: Optional[str] = None
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class MultipleChoiceScorer:
    """Scorer for multiple-choice questions"""
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
    
    def score(self, 
              predicted_answer: str, 
              ground_truth: str, 
              question_id: str,
              options: Optional[List[str]] = None) -> ScoreResult:
        """Score a multiple-choice answer"""
        
        # Clean and normalize answers
        pred_clean = self._extract_choice(predicted_answer)
        gt_clean = self._extract_choice(ground_truth)
        
        if not self.case_sensitive:
            pred_clean = pred_clean.upper()
            gt_clean = gt_clean.upper()
        
        # Check if correct
        is_correct = pred_clean == gt_clean
        score = 1.0 if is_correct else 0.0
        
        return ScoreResult(
            question_id=question_id,
            question_type="multiple_choice",
            is_correct=is_correct,
            score=score,
            predicted_answer=pred_clean,
            ground_truth=gt_clean,
            predicted_reasoning=None,
            confidence=None,  # Keep it simple - no confidence scoring
            details={
                "raw_predicted": predicted_answer,
                "raw_ground_truth": ground_truth,
                "options": options
            }
        )
    
    def _extract_choice(self, answer: str) -> str:
        """Extract choice letter from answer text"""
        # Look for patterns like "A)", "A.", "A:", "A", etc.
        patterns = [
            r'^([A-Z])\)',  # A)
            r'^([A-Z])\.',  # A.
            r'^([A-Z]):',   # A:
            r'^([A-Z])\s',  # A 
            r'^([A-Z])$',   # A
        ]
        
        for pattern in patterns:
            match = re.match(pattern, answer.strip())
            if match:
                return match.group(1)
        
        # If no pattern matches, return first character
        return answer.strip()[0] if answer.strip() else ""
    

class OpenQuestionScorer:
    """Scorer for open-ended questions using various metrics"""

    def __init__(self,
                 use_llm_judge: bool = True,
                 llm_judge_model: Optional[str] = None,
                 judge_port: int = 8003,
                 judge_gpu_memory_utilization: float = 0.7,
                 judge_max_model_len: int = 4096,
                 judge_temperature: float = 0.1,
                 judge_recall_max_tokens: int = 1024,
                 cache_dir: Optional[str] = None,
                 match_criterion: Optional[str] = None):
        self.use_llm_judge = use_llm_judge
        self.llm_judge_model = llm_judge_model
        self.llm_judge = None

        if use_llm_judge:
            self.llm_judge = LLMJudgeScorer(llm_judge_model, port=judge_port,
                                            gpu_memory_utilization=judge_gpu_memory_utilization,
                                            max_model_len=judge_max_model_len,
                                            temperature=judge_temperature,
                                            recall_max_tokens=judge_recall_max_tokens,
                                            cache_dir=cache_dir,
                                            match_criterion=match_criterion)
    
    def score(self,
              predicted_answer: str,
              ground_truth: str,
              question_id: str,
              question_text: str = "",
              human_responses: Optional[List[str]] = None) -> ScoreResult:
        """Score an open-ended answer"""

        if self.use_llm_judge and self.llm_judge:
            return self.llm_judge.score(
                predicted_answer, ground_truth, question_id,
                question_text=question_text,
                human_responses=human_responses,
            )
        else:
            # Fallback to simple text similarity
            return self._simple_similarity_score(
                predicted_answer, ground_truth, question_id
            )
    
    def _simple_similarity_score(self, 
                                predicted: str, 
                                ground_truth: str, 
                                question_id: str) -> ScoreResult:
        """Simple text similarity scoring as fallback"""
        # Basic word overlap scoring
        pred_words = set(predicted.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if not gt_words:
            return ScoreResult(
                question_id=question_id,
                question_type="open_ended",
                is_correct=False,
                score=0.0,
                predicted_answer=predicted,
                ground_truth=ground_truth,
                predicted_reasoning=None,
                confidence=None,
                details={"method": "simple_similarity", "error": "empty_ground_truth"}
            )
        
        intersection = pred_words.intersection(gt_words)
        union = pred_words.union(gt_words)
        
        jaccard_score = len(intersection) / len(union) if union else 0.0
        is_correct = jaccard_score > 0.5  # Threshold for correctness

        return ScoreResult(
            question_id=question_id,
            question_type="open_ended",
            is_correct=is_correct,
            score=jaccard_score,
            predicted_answer=predicted,
            ground_truth=ground_truth,
            predicted_reasoning=None,
            confidence=None,  # Keep it simple - no confidence scoring
            details={
                "method": "simple_similarity",
                "jaccard_score": jaccard_score,
                "intersection_size": len(intersection),
                "union_size": len(union)
            }
        )

# The "definition of matched" portion of the recall prompt. This is the
# ONLY part of the recall prompt that's overridable — the rest of the
# prompt (header, claim decomposition, example, fields) is fixed because
# the meta-judge audit + downstream parsing assume that structure.
#
# Override via `LLMJudgeScorer(match_criterion=...)` or extract to a file
# for the prompt-update flow:
#   python -c "from evaluation.metrics.scoring import RECALL_MATCH_CRITERION_DEFAULT;
#              print(RECALL_MATCH_CRITERION_DEFAULT, end='')" > match_criterion.txt
RECALL_MATCH_CRITERION_DEFAULT = (
    "   - matched: true — the candidate mentions this claim and the "
    "details are factually correct (verified against the ground truth "
    "and any reference answers).\n"
    "   - matched: false — the claim is absent from the candidate, OR "
    "the candidate mentions it but with wrong or hallucinated details."
)


class LLMJudgeScorer:
    """LLM-as-a-judge scorer for open-ended questions using LiteLLMWrapper"""

    def __init__(self, model_id: Optional[str] = "gpt-4o-mini", port: int = 8003,
                 gpu_memory_utilization: float = 0.7,
                 max_model_len: int = 4096,
                 temperature: float = 0.1,
                 recall_max_tokens: int = 1024,
                 cache_dir: Optional[str] = None,
                 match_criterion: Optional[str] = None):
        self.model_id = model_id
        self.judge_port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        # Default 1024 matches what every original (Qwen-judged) production
        # run used successfully. Reasoning judges like gpt-5 need more (their
        # hidden reasoning tokens eat into this budget before the visible
        # answer), but raising it for a *local* vLLM judge instead inflates
        # --judge-max-model-len's KV-cache memory need, risking the exact
        # OOM this judge server config was tuned to avoid — so this is
        # judge-specific, not a single shared constant.
        self.recall_max_tokens = recall_max_tokens
        self.match_criterion = (
            match_criterion if match_criterion is not None
            else RECALL_MATCH_CRITERION_DEFAULT
        )
        self.model_available = False
        self.model = None
        self._setup_attempted = False
        # Lazy init: don't start server here — called on first score() so the
        # main model server has been stopped before the judge claims the GPU.

        # Judge score cache: persists across runs so partial judge progress
        # accumulates instead of being lost when the judge crashes.
        self._score_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_file = None
        if cache_dir:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_file = os.path.join(cache_dir, "judge_scores.json")
            if os.path.exists(self._cache_file):
                try:
                    with open(self._cache_file, "r") as f:
                        self._score_cache = json.load(f)
                    logger.info(f"Loaded {len(self._score_cache)} cached judge scores")
                except Exception:
                    self._score_cache = {}

    def _setup_judge_model(self):
        """Setup the LLM judge model using LiteLLMWrapper"""
        try:
            self.model = LiteLLMWrapper.get_or_create_model(
                model_id=self.model_id,
                max_tokens=256,
                temperature=self.temperature,
                auto_start_server=True,
                port=self.judge_port,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
            )
            self.model_available = True
            logger.info(f"LLM Judge initialized with LiteLLMWrapper: {self.model_id} on port {self.judge_port}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM judge model: {e}")
            self.model_available = False
            self.model = None
    
    def _save_score_cache(self):
        """Persist judge score cache to disk."""
        if self._cache_file:
            try:
                with open(self._cache_file, "w") as f:
                    json.dump(self._score_cache, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save judge score cache: {e}")

    def score(self,
              predicted_answer: str,
              ground_truth: str,
              question_id: str,
              question_text: str = "",
              human_responses: Optional[List[str]] = None) -> ScoreResult:
        """Score using LLM-as-a-judge with structured outputs via LiteLLMWrapper"""

        # Technical failures (generation exception, or max-retries exceeded —
        # see evaluator.py's _generate_responses) are recorded with "ERROR"
        # as/in the response, not a real candidate answer. Score these
        # directly as 0 without spending a judge call on them: asking an LLM
        # to judge "ERROR" against the ground truth is wasted cost, and doing
        # so also produces a normal-looking cached verdict that's
        # indistinguishable from a real semantic judgment, which goes stale
        # once a later retry replaces "ERROR" with a real answer. Nothing is
        # cached here, so there's nothing to invalidate once a retry succeeds.
        if "ERROR" in predicted_answer:
            return ScoreResult(
                question_id=question_id,
                question_type="open_ended",
                is_correct=False,
                score=0,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                predicted_reasoning=None,
                confidence=None,
                details={"method": "technical_failure",
                         "reason": "no response generated (ERROR sentinel)"},
            )

        # Check judge score cache first.
        # Accept old entries that used llm_judge_structured (valid scores)
        # but re-evaluate any open-ended entry that is missing recall data
        # (recall is now computed for every open-ended question, not just C5).
        #
        # Also re-evaluate if the cached entry recorded a hash of the answer
        # it actually judged and that no longer matches predicted_answer —
        # this happens when a question's generation failed once (judged and
        # cached as "ERROR"), then a later retry produced a real answer:
        # since the cache is keyed only by question_id, the stale "ERROR"
        # verdict would otherwise be served forever against the new answer.
        # Entries cached before this check existed have no stored hash and
        # are trusted as-is (recomputing the whole historical cache to
        # backfill the hash would be wasteful).
        if question_id in self._score_cache:
            cached = self._score_cache[question_id]
            details = cached.get("details", {})
            is_judge_entry = (
                cached.get("cache_version") == _JUDGE_CACHE_VERSION
                or details.get("method") == "llm_judge_structured"
                or details.get("scoring") == "mmbench_video"
            )
            needs_recall = "recall" not in details
            current_hash = hashlib.md5(predicted_answer.encode()).hexdigest()
            stale_answer = ("predicted_answer_hash" in cached
                             and cached["predicted_answer_hash"] != current_hash)
            if is_judge_entry and not needs_recall and not stale_answer:
                # Migrate old entries: stamp current version & scoring tag
                if cached.get("cache_version") != _JUDGE_CACHE_VERSION:
                    cached["cache_version"] = _JUDGE_CACHE_VERSION
                    details.setdefault("scoring", "mmbench_video")
                    self._save_score_cache()
                return ScoreResult(
                    question_id=question_id,
                    question_type="open_ended",
                    is_correct=cached["is_correct"],
                    score=cached["score"],
                    predicted_answer=predicted_answer,
                    ground_truth=ground_truth,
                    predicted_reasoning=None,
                    confidence=None,
                    details=details
                )

        # Lazy init: start judge server on first call (main model server already stopped)
        if not self._setup_attempted:
            self._setup_attempted = True
            self._setup_judge_model()

        if not self.model_available or not self.model:
            # Fallback to simple similarity
            scorer = OpenQuestionScorer(use_llm_judge=False)
            return scorer._simple_similarity_score(predicted_answer, ground_truth, question_id)

        # Retry loop: restart the judge server on InternalServerError (same
        # vLLM EngineCore crash bug that affects the main model server).
        max_retries = 2
        for attempt in range(1 + max_retries):
            try:
                return self._score_with_judge(
                    predicted_answer, ground_truth, question_id,
                    question_text, human_responses,
                )
            except Exception as e:
                is_server_error = ("InternalServerError" in type(e).__name__
                                   or "InternalServerError" in str(e))
                if is_server_error and attempt < max_retries and self.model:
                    logger.warning(
                        f"[JUDGE SERVER ERROR] {question_id} - restarting judge "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    try:
                        LiteLLMWrapper._stop_server(self.model.port)
                        self.model._ensure_server_running()
                        continue
                    except Exception as restart_e:
                        logger.error(f"[JUDGE RESTART FAILED] {restart_e}")
                # Non-server error or exhausted retries → fallback
                logger.error(f"LLM Judge error for question {question_id}: {e}")
                scorer = OpenQuestionScorer(use_llm_judge=False)
                return scorer._simple_similarity_score(
                    predicted_answer, ground_truth, question_id
                )

    def _score_with_judge(self, predicted_answer, ground_truth, question_id,
                          question_text, human_responses):
        """Run the actual judge scoring (called by score() with retry wrapper)."""
        prompt = self._create_evaluation_prompt(
            predicted_answer, ground_truth, question_text, human_responses
        )

        # max_tokens=1024: reasoning judge models (e.g. gpt-5) spend part of
        # this budget on hidden reasoning tokens before the visible JSON
        # answer, so the default 256 set in _setup_judge_model() (tuned for
        # non-reasoning local judges) truncates them to an empty response.
        judgment = self.model.generate_structured(
            prompt=prompt,
            response_model=JudgeEvaluation,
            max_tokens=1024,
        )

        # MMBench-Video: score is 0-3 integer, correct if ≥ 2
        score = judgment.score
        is_correct = score >= 2

        details = {
            "method": "llm_judge_structured",
            "scoring": "mmbench_video",
            "model": self.model_id,
            "reasoning": judgment.reasoning,
        }

        # Claim-level recall evaluation for every open-ended question.
        try:
            recall_prompt = self._create_recall_prompt(
                predicted_answer, ground_truth, question_text,
                human_responses,
            )
            recall_eval = self.model.generate_structured(
                prompt=recall_prompt,
                response_model=ClaimRecallEvaluation,
                max_tokens=self.recall_max_tokens,
            )
            n_gt = len(recall_eval.gt_claims)
            recall = (sum(1 for c in recall_eval.gt_claims if c.matched)
                      / n_gt) if n_gt else 0.0
            details["recall"] = recall
            details["gt_claims"] = [
                {"claim": c.claim, "matched": c.matched}
                for c in recall_eval.gt_claims
            ]
        except Exception as e:
            logger.warning(f"Recall evaluation failed for {question_id}: {e}")

        # Cache the judge result with version tag + a hash of the answer it
        # judged, so a future call with a different answer for the same
        # question_id (e.g. after a retry updates responses.json) is
        # detected as stale instead of silently served from cache.
        self._score_cache[question_id] = {
            "is_correct": is_correct,
            "score": score,
            "details": details,
            "cache_version": _JUDGE_CACHE_VERSION,
            "predicted_answer_hash": hashlib.md5(predicted_answer.encode()).hexdigest(),
        }
        self._save_score_cache()

        return ScoreResult(
            question_id=question_id,
            question_type="open_ended",
            is_correct=is_correct,
            score=score,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            predicted_reasoning=None,
            confidence=None,
            details=details
        )
    
    def _create_evaluation_prompt(self,
                                 predicted: str,
                                 ground_truth: str,
                                 question_text: str = "",
                                 human_responses: Optional[List[str]] = None) -> str:
        """Create evaluation prompt using MMBench-Video semantic similarity rubric"""

        prompt = (
            "You are an expert evaluator. Score the semantic similarity between "
            "the candidate answer and the ground-truth answer on a scale of 0-3.\n\n"
            "0 — No similarity: The candidate answer is completely irrelevant or incorrect.\n"
            "1 — Low similarity: The candidate answer is largely incorrect but "
            "shares minor elements with the ground truth.\n"
            "2 — High similarity: The candidate answer is largely correct but "
            "may differ in some details.\n"
            "3 — Complete similarity: The candidate answer is entirely correct "
            "and matches the ground truth in meaning.\n\n"
            "Focus on semantic meaning, not grammar or phrasing.\n\n"
            f"Question: {question_text}\n"
            f"Groundtruth answer: {ground_truth}\n"
            f"Candidate answer: {predicted}\n"
        )
        if human_responses:
            prompt += "\nHuman reference responses:\n"
            for i, resp in enumerate(human_responses, 1):
                prompt += f"{i}. {resp}\n"
        prompt += "\nYour response:"
        return prompt
    
    def _create_recall_prompt(self,
                              predicted: str,
                              ground_truth: str,
                              question_text: str = "",
                              human_responses: Optional[List[str]] = None) -> str:
        """Create prompt for claim-level recall evaluation"""

        prompt = (
            "You are an expert evaluator assessing claim-level recall.\n\n"
            "Instructions:\n"
            "1. Decompose the ground-truth answer into atomic action-response "
            "claims (one discrete action or observation per claim).\n"
            "2. For each ground-truth claim, check whether the candidate answer "
            "covers it with correct details.\n"
            f"{self.match_criterion}\n\n"
            "Example JSON output:\n"
            '{"gt_claims": [\n'
            '  {"claim": "child picks up red ball", "matched": true},\n'
            '  {"claim": "adult claps in response", "matched": false}\n'
            ']}\n\n'
            f"Question: {question_text}\n"
            f"Ground-truth answer: {ground_truth}\n"
            f"Candidate answer: {predicted}\n"
        )
        if human_responses:
            prompt += "\nReference answers (use to verify correctness):\n"
            for i, resp in enumerate(human_responses, 1):
                prompt += f"{i}. {resp}\n"
        prompt += "\nYour response:"
        return prompt

class EvaluationMetrics:
    """Main evaluation metrics calculator"""

    def __init__(self,
                 mc_scorer: Optional[MultipleChoiceScorer] = None,
                 open_scorer: Optional[OpenQuestionScorer] = None,
                 judge_model: Optional[str] = None,
                 judge_port: int = 8003,
                 judge_gpu_memory_utilization: float = 0.7,
                 judge_max_model_len: int = 4096,
                 judge_temperature: float = 0.1,
                 judge_recall_max_tokens: int = 1024,
                 cache_dir: Optional[str] = None,
                 match_criterion: Optional[str] = None):
        self.mc_scorer = mc_scorer or MultipleChoiceScorer()
        use_llm_judge = bool(judge_model)  # None or "" → no judge
        self.open_scorer = open_scorer or OpenQuestionScorer(
            use_llm_judge=use_llm_judge, llm_judge_model=judge_model,
            judge_port=judge_port,
            judge_gpu_memory_utilization=judge_gpu_memory_utilization,
            judge_max_model_len=judge_max_model_len,
            judge_temperature=judge_temperature,
            judge_recall_max_tokens=judge_recall_max_tokens,
            cache_dir=cache_dir,
            match_criterion=match_criterion)
    
    def evaluate_question(self, 
                         question: 'Question',
                         predicted_answer: str) -> ScoreResult:
        """Evaluate a single question"""
        
        if question.question_type == "multiple_choice":
            return self.mc_scorer.score(
                predicted_answer=predicted_answer,
                ground_truth=question.ground_truth,
                question_id=question.question_id,
                options=question.options
            )
        elif question.question_type == "open_ended":
            return self.open_scorer.score(
                predicted_answer=predicted_answer,
                ground_truth=question.ground_truth,
                question_id=question.question_id,
                question_text=question.question,
                human_responses=question.human_responses,
            )
        else:
            raise ValueError(f"Unknown question type: {question.question_type}")
    
    def evaluate_batch(self, 
                      questions: List['Question'],
                      predicted_answers: List[str]) -> List[ScoreResult]:
        """Evaluate a batch of questions"""
        
        if len(questions) != len(predicted_answers):
            raise ValueError("Number of questions and answers must match")
        
        results = []
        for question, answer in zip(questions, predicted_answers):
            try:
                result = self.evaluate_question(question, answer)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating question {question.question_id}: {e}")
                # Create error result
                results.append(ScoreResult(
                    question_id=question.question_id,
                    question_type=question.question_type,
                    is_correct=False,
                    score=0.0,
                    predicted_answer=answer,
                    ground_truth=question.ground_truth,
                    predicted_reasoning=None,
                    confidence=None,
                    details={"error": str(e)}
                ))
        
        return results
    
    def calculate_group_metrics(self, 
                               results: List[ScoreResult],
                               question_groups: Dict[str, List['Question']]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each question group"""
        
        group_metrics = {}
        
        for group_name, group_questions in question_groups.items():
            group_question_ids = {q.question_id for q in group_questions}
            group_results = [r for r in results if r.question_id in group_question_ids]
            
            if not group_results:
                group_metrics[group_name] = {
                    "accuracy": 0.0,
                    "average_score": 0.0,
                    "total_questions": 0,
                    "correct_answers": 0
                }
                continue
            
            # Calculate metrics
            total_questions = len(group_results)
            correct_answers = sum(1 for r in group_results if r.is_correct)
            accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
            average_score = sum(r.score for r in group_results) / total_questions if total_questions > 0 else 0.0
            
            # Separate by question type
            mc_results = [r for r in group_results if r.question_type == "multiple_choice"]
            open_results = [r for r in group_results if r.question_type == "open_ended"]
            mc_correct = sum(1 for r in mc_results if r.is_correct)

            open_recalls = [r.details["recall"] for r in open_results
                            if r.details and "recall" in r.details]
            group_metrics[group_name] = {
                "accuracy": accuracy,
                "average_score": average_score,
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "mc_accuracy": mc_correct / len(mc_results) if mc_results else 0.0,
                "open_mean_score": (sum(r.score for r in open_results) / len(open_results)
                                    if open_results else 0.0),
                "open_mean_recall": (sum(open_recalls) / len(open_recalls)
                                     if open_recalls else None),
                "open_recall_count": len(open_recalls),
                "mc_count": len(mc_results),
                "open_count": len(open_results)
            }

        return group_metrics
    
    def calculate_overall_metrics(self, results: List[ScoreResult]) -> Dict[str, float]:
        """Calculate overall evaluation metrics"""
        
        if not results:
            return {
                "accuracy": 0.0,
                "average_score": 0.0,
                "total_questions": 0,
                "correct_answers": 0
            }
        
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        accuracy = correct_answers / total_questions
        average_score = sum(r.score for r in results) / total_questions
        
        # Separate by question type
        mc_results = [r for r in results if r.question_type == "multiple_choice"]
        open_results = [r for r in results if r.question_type == "open_ended"]

        # Calculate correct answers per type
        mc_correct = sum(1 for r in mc_results if r.is_correct)

        open_recalls = [r.details["recall"] for r in open_results
                        if r.details and "recall" in r.details]
        return {
            "accuracy": accuracy,
            "average_score": average_score,
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "mc_accuracy": mc_correct / len(mc_results) if mc_results else 0.0,
            "open_mean_score": (sum(r.score for r in open_results) / len(open_results)
                                if open_results else 0.0),
            "open_mean_recall": (sum(open_recalls) / len(open_recalls)
                                 if open_recalls else None),
            "open_recall_count": len(open_recalls),
            "mc_count": len(mc_results),
            "open_count": len(open_results)
        }
