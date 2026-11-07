# evaluation/evaluator.py
"""Main evaluation runner that orchestrates the entire evaluation process"""

import json
import logging
import os
import subprocess
from typing import List, Dict, Optional, Union, Any
import time

from .benchmark_loader import BenchmarkLoader, Question
from .models.base_model import BaseModelWrapper
from .models.litellm_wrapper import LiteLLMWrapper
from .models.response_formats import MultipleChoiceResponse, OpenQuestionResponse
from .metrics.scoring import EvaluationMetrics, ScoreResult
from .cache.result_cache import ResultCache

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an AI assistant responsible for answering questions about videos."


class EvaluationRunner:
    """Main evaluation runner for the visual communication benchmark"""
    
    def __init__(self,
                 model: BaseModelWrapper,
                 cache_dir: Optional[Union[str, os.PathLike]] = None,
                 use_cache: bool = True,
                 consistency_checks: bool = True,
                 consistency_runs: int = 3,
                 judge_model: Optional[str] = None,
                 judge_port: int = 8003,
                 judge_gpu_memory_utilization: float = 0.7,
                 judge_max_model_len: int = 4096,
                 judge_temperature: float = 0.1,
                 judge_recall_max_tokens: int = 1024,
                 max_retries: Optional[int] = None,
                 match_criterion: Optional[str] = None):
        self.model = model
        self.cache = ResultCache(cache_dir) if use_cache else None
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.consistency_checks = consistency_checks
        self.consistency_runs = consistency_runs

        # Initialize metrics with judge model
        self.metrics = EvaluationMetrics(judge_model=judge_model, judge_port=judge_port,
                                        judge_gpu_memory_utilization=judge_gpu_memory_utilization,
                                        judge_max_model_len=judge_max_model_len,
                                        judge_temperature=judge_temperature,
                                        judge_recall_max_tokens=judge_recall_max_tokens,
                                        cache_dir=cache_dir,
                                        match_criterion=match_criterion)
        
        # Results storage
        self.results: List[ScoreResult] = []
        self.responses: Dict[str, str] = {}
        self.response_reasoning: Dict[str, Optional[str]] = {}
        self.consistency_results: Dict[str, Dict[str, Any]] = {}
        self.failed_questions: Dict[str, str] = {}  # Track technical failures
    
    @staticmethod
    def _wait_for_gpu_release(timeout=60):
        """Wait until GPU memory usage drops below 1 GB (model fully unloaded)."""
        for i in range(timeout):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                used_mb = max(int(x) for x in out.stdout.strip().split("\n") if x.strip())
                if used_mb < 1024:
                    logger.info(f"GPU memory released ({used_mb} MiB used) after {i}s")
                    return
            except Exception:
                pass
            time.sleep(1)
        logger.warning(f"GPU memory still in use after {timeout}s, proceeding anyway")

    def evaluate_benchmark(self,
                          benchmark_path: Union[str, os.PathLike],
                          questions_file: str = "test_questions.json",
                          output_dir: Optional[Union[str, os.PathLike]] = None,
                          batch_size: int = 8,
                          no_media: bool = False,
                          transcript_file: Optional[str] = None) -> Dict[str, Any]:
        """Run complete evaluation on benchmark"""
        
        logger.info(f"Starting evaluation with model: {self.model.model_id}")
        start_time = time.time()
        
        # Load benchmark data (auto-detect format by extension)
        loader = BenchmarkLoader(benchmark_path)
        if questions_file.endswith(".csv"):
            loader.load_from_csv(questions_file)
        elif questions_file.endswith(".json"):
            loader.load_from_json(questions_file)
        else:
            loader.load(questions_file)  # Auto-detect
        
        # Validate questions
        issues = loader.validate_questions()
        if issues:
            logger.warning(f"Found {len(issues)} validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        # Get all questions and resolve media paths
        questions = loader.get_all_questions()
        
        # Resolve media paths relative to benchmark directory using os.path.join
        for question in questions:
            if question.media_path:
                question.media_path = os.path.join(benchmark_path, question.media_path)

        # Transcript mode: inject before no_media so media_path is still available as lookup key
        if transcript_file:
            with open(transcript_file) as f:
                transcripts = json.load(f)
            injected = 0
            for question in questions:
                key = question.media_path or ""
                transcript = (transcripts.get(key)
                              or transcripts.get(os.path.normpath(key), ""))
                if transcript:
                    question.question += f'\n\n[Audio transcript: "{transcript}"]'
                    injected += 1
            logger.info(f"TRANSCRIPT mode: injected transcripts into {injected}/{len(questions)} questions")

        # Blind mode: strip all media paths so the model sees text only
        if no_media:
            for question in questions:
                question.media_path = None
            logger.info("NO-MEDIA mode: stripped all media paths (text-only inference)")

        logger.info(f"Loaded {len(questions)} questions")

        # Preflight cache check: only start the vLLM server if there are questions
        # not yet in cache. Errors are never cached so they always need inference.
        if self.cache and isinstance(self.model, LiteLLMWrapper) and self.model.is_local_model:
            uncached = [
                q for q in questions
                if self.cache.get_response(q.question_id, self.model.model_id,
                                           q.question, q.media_path) is None
            ]
            if uncached:
                logger.info(f"{len(uncached)}/{len(questions)} questions need inference — starting server...")
                self.model._ensure_server_running()
            else:
                logger.info(f"All {len(questions)} questions cached — skipping server startup")

        # Generate responses (main model inference)
        responses = self._generate_responses(questions, batch_size)

        # If using a local LLM judge, stop the main model server first so the
        # judge server can claim the full GPU (they can't both fit in memory).
        judge = getattr(getattr(self.metrics, 'open_scorer', None), 'llm_judge', None)
        if (judge is not None and
                isinstance(self.model, LiteLLMWrapper) and
                self.model.is_local_model):
            logger.info("Stopping main model server before starting LLM judge (sequential GPU use)...")
            LiteLLMWrapper._stop_server(self.model.port)
            if self.model.port in LiteLLMWrapper._servers:
                LiteLLMWrapper._servers[self.model.port]["model"] = None
            # Wait for GPU memory to actually be released (not just a fixed sleep)
            self._wait_for_gpu_release()

        # Evaluate responses (judge runs lazily on first open-ended question)
        results = self._evaluate_responses(questions, responses)
        
        # Calculate metrics
        overall_metrics = self.metrics.calculate_overall_metrics(results)
        group_metrics = self.metrics.calculate_group_metrics(results, loader.question_groups)
        
        # Compile final results
        evaluation_results = {
            "model_info": self.model.get_model_info(),
            "evaluation_timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "evaluation_duration": time.time() - start_time,
            "benchmark_info": {
                "total_questions": len(questions),
                "question_groups": list(loader.question_groups.keys()),
                "group_summary": loader.get_group_summary()
            },
            "overall_metrics": overall_metrics,
            "group_metrics": group_metrics,
            "detailed_results": [
                {
                    "question_id": r.question_id,
                    "question_type": r.question_type,
                    "is_correct": r.is_correct,
                    "score": r.score,
                    "predicted_answer": r.predicted_answer,
                    "predicted_reasoning": r.predicted_reasoning,
                    "ground_truth": r.ground_truth,
                    "confidence": r.confidence,
                    "details": r.details
                }
                for r in results
            ],
            "failed_questions": self.failed_questions,  # Technical failures
            "consistency_results": self.consistency_results,
            "cache_stats": self.cache.get_cache_stats() if self.cache else None
        }
        
        # Save results
        if output_dir:
            self._save_results(evaluation_results, output_dir)
        
        # Store results
        self.results = results
        self.responses = responses
        
        logger.info(f"Evaluation completed in {evaluation_results['evaluation_duration']:.2f}s")
        logger.info(f"Overall accuracy: {overall_metrics['accuracy']:.3f}")
        
        return evaluation_results
    
    def _format_multiple_choice_prompt(self, question: Question) -> str:
        """Format a multiple-choice question with options"""
        prompt = question.question + "\n\n"

        if question.options:
            for option in question.options:
                prompt += f"{option}\n"
            prompt += "\nAnswer with only the letter of your choice (A, B, C, etc.)."

        return prompt

    def _generate_responses(self,
                           questions: List[Question],
                           batch_size: int) -> Dict[str, str]:
        """Generate responses for all questions"""

        responses = {}
        _restart_count = 0  # Track how many server restarts we've done (max 5)

        # Process in batches
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1}")

            # Process each question individually to handle different question types
            for question in batch_questions:
                # Initialize response_reasoning
                response_reasoning = None

                # Check cache first
                if self.cache:
                    cached_result = self.cache.get_response(
                        question.question_id,
                        self.model.model_id,
                        question.question,
                        question.media_path
                    )
                    if cached_result:
                        cached_response, cached_reasoning = cached_result
                        responses[question.question_id] = cached_response
                        self.response_reasoning[question.question_id] = cached_reasoning
                        logger.info(f"[CACHE HIT] {question.question_id} - Retrieved from cache (no cost)")
                        continue

                # Skip questions that have exceeded max retries
                if self.max_retries is not None and self.cache:
                    fail_count = self.cache.get_failure_count(
                        question.question_id, self.model.model_id,
                        question.question, question.media_path)
                    if fail_count >= self.max_retries:
                        logger.warning(f"[SKIP_RETRIES] {question.question_id} - "
                                       f"Failed {fail_count} times, max_retries={self.max_retries}")
                        self.failed_questions[question.question_id] = (
                            f"Skipped: failed {fail_count} times (max_retries={self.max_retries})")
                        responses[question.question_id] = "ERROR"
                        continue

                # Skip image questions for video-only models (e.g. LLaVA-NeXT-Video)
                if (question.media_path and
                        question.media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) and
                        hasattr(self.model, 'video_only') and self.model.video_only):
                    logger.info(f"[SKIPPED] {question.question_id} - Image input unsupported by video-only model")
                    responses[question.question_id] = "[UNSUPPORTED] Video-only model cannot process image input"
                    self.response_reasoning[question.question_id] = None
                    continue

                # Log cache miss - generating new response
                logger.info(f"[GENERATING] {question.question_id} - Running model query (incurs cost)")

                try:
                    # Generate response based on question type
                    if question.question_type == "multiple_choice":
                        # Format prompt with options
                        formatted_prompt = self._format_multiple_choice_prompt(question)

                        # Use structured generation for multiple-choice
                        structured_response = self.model.generate_structured(
                            prompt=formatted_prompt,
                            response_model=MultipleChoiceResponse,
                            media_path=question.media_path,
                            system_prompt=SYSTEM_PROMPT
                        )

                        # Extract just the letter answer
                        response = structured_response.answer
                        response_reasoning = structured_response.reasoning

                    else:
                        # Use structured generation for open questions
                        structured_response = self.model.generate_structured(
                            prompt=question.question,
                            response_model=OpenQuestionResponse,
                            media_path=question.media_path,
                            system_prompt=SYSTEM_PROMPT
                        )

                        # Extract answer and reasoning
                        response = structured_response.answer
                        response_reasoning = structured_response.reasoning

                    # Store response and reasoning
                    responses[question.question_id] = response
                    self.response_reasoning[question.question_id] = response_reasoning

                    # Cache response
                    if self.cache:
                        self.cache.store_response(
                            question.question_id,
                            self.model.model_id,
                            question.question,
                            response,
                            question.media_path,
                            response_reasoning
                        )

                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    # If the server crashed (InternalServerError), restart once and retry
                    is_server_error = "InternalServerError" in type(e).__name__ or "InternalServerError" in str(e)
                    if is_server_error and _restart_count < 10 and isinstance(self.model, LiteLLMWrapper) and self.model.is_local_model:
                        _restart_count += 1
                        logger.warning(f"[SERVER ERROR] {question.question_id} - Server EngineCore may have crashed, force restarting (attempt {_restart_count}/10)...")
                        try:
                            # Force-stop first to release GPU memory from the zombie APIServer/EngineCore
                            LiteLLMWrapper._stop_server(self.model.port)
                            self.model._ensure_server_running()
                            logger.info("[SERVER RESTARTED] Retrying question...")
                            # Retry the question with the fresh server
                            if question.question_type == "multiple_choice":
                                formatted_prompt = self._format_multiple_choice_prompt(question)
                                structured_response = self.model.generate_structured(
                                    prompt=formatted_prompt,
                                    response_model=MultipleChoiceResponse,
                                    media_path=question.media_path,
                                    system_prompt=SYSTEM_PROMPT
                                )
                            else:
                                structured_response = self.model.generate_structured(
                                    prompt=question.question,
                                    response_model=OpenQuestionResponse,
                                    media_path=question.media_path,
                                    system_prompt=SYSTEM_PROMPT
                                )
                            response = structured_response.answer
                            response_reasoning = structured_response.reasoning
                            responses[question.question_id] = response
                            self.response_reasoning[question.question_id] = response_reasoning
                            if self.cache:
                                self.cache.store_response(
                                    question.question_id, self.model.model_id,
                                    question.question, response,
                                    question.media_path, response_reasoning
                                )
                            logger.info(f"[RETRY SUCCESS] {question.question_id}")
                            continue
                        except Exception as retry_e:
                            error_msg = f"[after restart] {type(retry_e).__name__}: {str(retry_e)}"
                            logger.error(f"[RETRY FAILED] {question.question_id} - {error_msg}")
                    else:
                        logger.error(f"[FAILED] {question.question_id} - {error_msg}")
                    self.failed_questions[question.question_id] = error_msg
                    responses[question.question_id] = "ERROR"
                    # Record failure for retry limiting
                    if self.cache and self.max_retries is not None:
                        n = self.cache.record_failure(
                            question.question_id, self.model.model_id,
                            question.question, error_msg, question.media_path)
                        logger.info(f"[FAIL_COUNT] {question.question_id} - attempt {n}/{self.max_retries}")
            
            # Consistency checks for selected questions
            if self.consistency_checks and i == 0:  # Check first batch for consistency
                self._run_consistency_checks(batch_questions[:2])  # Check first 2 questions
        
        return responses
    
    def _run_consistency_checks(self, questions: List[Question]) -> None:
        """Run consistency checks on selected questions"""
        
        logger.info("Running consistency checks...")
        
        for question in questions:
            # Check if already cached
            if self.cache:
                cached_responses = self.cache.get_consistency_responses(
                    question.question_id,
                    self.model.model_id
                )
                if cached_responses:
                    logger.info(f"Using cached consistency responses for {question.question_id}")
                    responses = cached_responses
                else:
                    # Generate multiple responses
                    responses = []
                    for _ in range(self.consistency_runs):
                        try:
                            response = self.model.generate_single(
                                prompt=question.question,
                                media_path=question.media_path
                            )
                            responses.append(response)
                        except Exception as e:
                            logger.error(f"Error in consistency check: {e}")
                            responses.append(f"ERROR: {str(e)}")
                    
                    # Cache responses
                    self.cache.store_consistency_responses(
                        question.question_id,
                        self.model.model_id,
                        question.question,
                        responses,
                        question.media_path
                    )
            else:
                # Generate without caching
                responses = []
                for _ in range(self.consistency_runs):
                    try:
                        response = self.model.generate_single(
                            prompt=question.question,
                            media_path=question.media_path
                        )
                        responses.append(response)
                    except Exception as e:
                        logger.error(f"Error in consistency check: {e}")
                        responses.append(f"ERROR: {str(e)}")
            
            # Check consistency
            if self.cache:
                consistency_result = self.cache.check_consistency(
                    question.question_id,
                    self.model.model_id
                )
                self.consistency_results[question.question_id] = consistency_result
                logger.info(f"Consistency for {question.question_id}: {consistency_result['consistency_score']:.3f}")
    
    def _evaluate_responses(self, 
                           questions: List[Question], 
                           responses: Dict[str, str]) -> List[ScoreResult]:
        """Evaluate generated responses"""
        
        logger.info("Evaluating responses...")
        
        # Prepare data for batch evaluation
        question_list = []
        response_list = []
        
        for question in questions:
            question_list.append(question)
            response_list.append(responses.get(question.question_id, "NO_RESPONSE"))
        
        # Evaluate
        results = self.metrics.evaluate_batch(question_list, response_list)

        # Add reasoning to results
        for result in results:
            result.predicted_reasoning = self.response_reasoning.get(result.question_id)

        # Log summary
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)
        logger.info(f"Evaluation complete: {correct}/{total} correct ({correct/total:.3f})")

        return results
    
    def _save_results(self, 
                     results: Dict[str, Any], 
                     output_dir: Union[str, os.PathLike]) -> None:
        """Save evaluation results to files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results (fixed filename so skip-check in eval.sh works reliably)
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save detailed results CSV
        csv_file = os.path.join(output_dir, "detailed_results.csv")
        self._save_csv_results(results["detailed_results"], csv_file)

        # Save summary report
        summary_file = os.path.join(output_dir, "summary_report.txt")
        self._save_summary_report(results, summary_file)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _save_csv_results(self, 
                         detailed_results: List[Dict[str, Any]], 
                         csv_file: str) -> None:
        """Save detailed results as CSV"""
        
        import csv
        
        with open(csv_file, 'w', newline='') as f:
            if detailed_results:
                writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
                writer.writeheader()
                writer.writerows(detailed_results)
    
    def _save_summary_report(self, 
                            results: Dict[str, Any], 
                            summary_file: str) -> None:
        """Save human-readable summary report"""
        
        with open(summary_file, 'w') as f:
            f.write("Visual Communication Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            model_info = results["model_info"]
            f.write(f"Model: {model_info['model_id']}\n")
            f.write(f"Type: {model_info['type']}\n")
            f.write(f"Evaluation Time: {results['evaluation_timestamp']}\n")
            f.write(f"Duration: {results['evaluation_duration']:.2f}s\n\n")
            
            # Overall metrics
            overall = results["overall_metrics"]
            f.write("Overall Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Questions: {overall['total_questions']}\n")
            f.write(f"Correct Answers: {overall['correct_answers']}\n")
            f.write(f"Accuracy: {overall['accuracy']:.3f}\n")
            f.write(f"Average Score: {overall['average_score']:.3f}\n")
            f.write(f"MC Questions: {overall['mc_count']} (Accuracy: {overall['mc_accuracy']:.3f})\n")
            f.write(f"Open Questions: {overall['open_count']} (Mean Score: {overall['open_mean_score']:.3f} / 3.0)\n\n")

            # Group metrics
            f.write("Group Results:\n")
            f.write("-" * 15 + "\n")
            for group, metrics in results["group_metrics"].items():
                f.write(f"{group}:\n")
                f.write(f"  Questions: {metrics['total_questions']}\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
                f.write(f"  MC: {metrics['mc_count']} ({metrics['mc_accuracy']:.3f})\n")
                f.write(f"  Open: {metrics['open_count']} (Mean Score: {metrics['open_mean_score']:.3f} / 3.0)\n\n")
            
            # Consistency results
            if results["consistency_results"]:
                f.write("Consistency Results:\n")
                f.write("-" * 20 + "\n")
                for question_id, consistency in results["consistency_results"].items():
                    f.write(f"{question_id}: {consistency['consistency_score']:.3f} "
                           f"({'✓' if consistency['is_consistent'] else '✗'})\n")
                f.write("\n")
    
    def get_model_performance_by_type(self) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by question type and media type"""
        
        if not self.results:
            return {}
        
        # Group results by type
        mc_results = [r for r in self.results if r.question_type == "multiple_choice"]
        open_results = [r for r in self.results if r.question_type == "open_question"]
        
        performance = {
            "multiple_choice": {
                "accuracy": sum(1 for r in mc_results if r.is_correct) / len(mc_results) if mc_results else 0.0,
                "average_score": sum(r.score for r in mc_results) / len(mc_results) if mc_results else 0.0,
                "count": len(mc_results)
            },
            "open_question": {
                "accuracy": sum(1 for r in open_results if r.is_correct) / len(open_results) if open_results else 0.0,
                "average_score": sum(r.score for r in open_results) / len(open_results) if open_results else 0.0,
                "count": len(open_results)
            }
        }
        
        return performance
