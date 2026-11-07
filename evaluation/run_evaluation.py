#!/usr/bin/env python3
# evaluation/run_evaluation.py
"""Main script to run visual communication evaluation"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .benchmark_loader import BenchmarkLoader
from .models.litellm_wrapper import LiteLLMWrapper
from .evaluator import EvaluationRunner

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def print_statistics(results):
    """Print statistics in a formatted table"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall statistics
    overall = results["overall_metrics"]
    print(f"\nModel: {results['model_info']['model_id']}")
    print(f"Duration: {results.get('evaluation_duration', 0):.2f}s")

    print("\nOVERALL PERFORMANCE")
    print("-" * 80)
    print(f"Total Questions:         {overall['total_questions']}")
    print(f"Correct Answers:         {overall['correct_answers']}")
    print(f"Overall Accuracy:        {overall['accuracy']:.2%}")
    print(f"Average Score:           {overall['average_score']:.3f}")

    # Performance by question type
    mc_count = overall.get('mc_count', 0)
    open_count = overall.get('open_count', 0)
    if mc_count > 0 or open_count > 0:
        print("\nPERFORMANCE BY QUESTION TYPE")
        print("-" * 80)

        if mc_count > 0:
            mc_accuracy = overall.get('mc_accuracy', 0.0)
            print(f"{'Multiple Choice':<25} {mc_count:>8} questions    {mc_accuracy:>7.2%} accuracy")

        if open_count > 0:
            open_score = overall.get('open_mean_score', 0.0)
            open_recall = overall.get('open_mean_recall')
            recall_n = overall.get('open_recall_count', 0)
            recall_str = (f"{open_recall:>7.2%} recall (n={recall_n})"
                          if open_recall is not None else "  recall n/a")
            print(f"{'Open-Ended':<25} {open_count:>8} questions    "
                  f"{open_score:>7.2f} similarity (0-3)    {recall_str}")

    # Question group statistics — type-aware per group
    group_metrics = results.get("group_metrics", {})
    if group_metrics:
        print("\nPERFORMANCE BY QUESTION GROUP")
        print("-" * 80)
        for group_name, group_stats in sorted(group_metrics.items()):
            total = group_stats.get('total_questions', 0)
            g_mc = group_stats.get('mc_count', 0)
            g_open = group_stats.get('open_count', 0)
            parts = [f"{group_name:<35} {total:>4} questions"]
            if g_mc:
                parts.append(f"MC: {group_stats.get('mc_accuracy', 0.0):.2%}")
            if g_open:
                sim = group_stats.get('open_mean_score', 0.0)
                rec = group_stats.get('open_mean_recall')
                rec_str = f", recall {rec:.2%}" if rec is not None else ""
                parts.append(f"Open: {sim:.2f}/3{rec_str}")
            print("  ".join(parts))

    # Consistency results
    consistency_results = results.get('consistency_results', {})
    if consistency_results:
        print("\nCONSISTENCY RESULTS")
        print("-" * 80)
        consistent_count = sum(1 for c in consistency_results.values() if c.get('is_consistent'))
        print(f"Consistent: {consistent_count}/{len(consistency_results)}")

    # Technical failures
    failed_questions = results.get("failed_questions", {})
    if failed_questions:
        print(f"\nTechnical Failures: {len(failed_questions)} (details in JSON output)")

    print("=" * 80)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Run visual communication evaluation")

    # Model arguments
    parser.add_argument("--model", type=str, default="vllm/Qwen/Qwen2.5-VL-3B-Instruct",
                        help="Model ID to use for evaluation")
    parser.add_argument("--api-base", type=str, default=None,
                        help="API base URL for local models (e.g., http://localhost:8000/v1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for API models")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.2,
                        help="GPU memory utilization for local models (0.0-1.0)")
    parser.add_argument("--auto-start-server", action="store_true",
                        help="Automatically start vLLM server for local models")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="Maximum model context length for vLLM server (default: 32768)")
    parser.add_argument("--port", type=int, default=8002,
                        help="Port for vLLM server (default: 8002)")
    parser.add_argument("--judge-port", type=int, default=8003,
                        help="Port for judge vLLM server (default: 8003)")
    parser.add_argument("--ft-weights", type=str, default=None,
                        help="Local merged-weights directory. When set, vLLM "
                             "loads weights from here while --model is kept as "
                             "the served-model-name and the key for "
                             "model_config.yaml lookups (e.g. for fine-tuned "
                             "Qwen merged checkpoints).")

    # Data arguments
    parser.add_argument("--questions-file", type=str, default="test_questions.csv",
                        help="Path to questions file (CSV or JSON)")
    parser.add_argument("--benchmark-path", type=str, default=".",
                        help="Repo-relative root for resolving media paths in the questions CSV. "
                             "The evaluator computes the media root as <benchmark_path>/data, "
                             "so the default '.' resolves to data/ from the repo root.")

    # Evaluation arguments
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory (default: evaluation_cache_<questions-file>)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (reasoning models like GPT-5/Gemini 3.x require 1.0)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Max output tokens (default: 256 for vLLM from config, 512 for API models; reasoning models need 4096+)")
    parser.add_argument("--no-media", action="store_true",
                        help="Run in blind mode: strip all media paths so the model sees text only")
    parser.add_argument("--transcript-file", type=str, default=None,
                        help="JSON file mapping segment.mp4 absolute path → Whisper transcript text")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable response caching")
    parser.add_argument("--consistency", action="store_true",
                        help="Enable consistency checks")
    parser.add_argument("--consistency-runs", type=int, default=3,
                        help="Number of runs for consistency checking")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Model for LLM judge (default: same as --model)")
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.7,
                        help="GPU memory utilization for judge model (0.0-1.0, default: 0.7)")
    parser.add_argument("--judge-max-model-len", type=int, default=4096,
                        help="Max context length for judge vLLM server (default: 4096)")
    parser.add_argument("--judge-temperature", type=float, default=0.1,
                        help="Sampling temperature for the LLM judge (default: 0.1). "
                             "Reasoning judge models (e.g. gpt-5) reject non-default "
                             "temperature and require 1.0.")
    parser.add_argument("--judge-recall-max-tokens", type=int, default=1024,
                        help="Max output tokens for the judge's recall (claim-matching) "
                             "call (default: 1024, matches all original Qwen-judged runs). "
                             "Reasoning judges like gpt-5 spend part of this budget on "
                             "hidden reasoning tokens and may need more (e.g. 4096) — but "
                             "raising it for a LOCAL vLLM judge also raises "
                             "--judge-max-model-len's KV-cache memory need, so tune "
                             "per-judge rather than assuming one value fits all.")
    parser.add_argument("--match-criterion-file", type=Path, default=None,
                        help="Path to a text file with the recall judge's "
                             "match criterion (the matched: true / matched: "
                             "false bullets). Overrides "
                             "evaluation.metrics.scoring.RECALL_MATCH_CRITERION_DEFAULT. "
                             "Use to A/B a new criterion proposed by the "
                             "meta-judge audit without editing scoring.py.")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Max times to retry a failed question across runs. "
                             "0 = never retry (skip on first failure), 1 = retry once, etc. "
                             "Omit to disable failure tracking (unlimited retries).")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log file path (optional)")

    args = parser.parse_args()

    # Set defaults — empty string from shell ("") treated same as omitted
    if not args.judge_model:
        args.judge_model = None

    if args.cache_dir is None:
        questions_base = Path(args.questions_file).stem
        args.cache_dir = f"evaluation_cache_{questions_base}"

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    logger.info("=" * 80)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Questions file: {args.questions_file}")
    logger.info(f"Benchmark path: {args.benchmark_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Cache: {args.cache_dir} ({'disabled' if args.no_cache else 'enabled'})")
    logger.info(f"Max model len: {args.max_model_len}")
    if args.no_media:
        logger.info("NO-MEDIA (blind) mode: all media paths will be stripped")
    if args.transcript_file:
        logger.info(f"Transcript file: {args.transcript_file}")

    try:
        # Determine allowed media path from benchmark path (resolve symlinks)
        allowed_media_path = os.path.realpath(os.path.join(args.benchmark_path, "data"))
        logger.info(f"Allowed media path: {allowed_media_path}")

        # Initialize model — server is NOT started here for local models.
        # evaluate_benchmark() does a cache preflight and only starts the server
        # if there are questions that aren't already cached.
        logger.info(f"Initializing model: {args.model}")
        if args.model.startswith("vllm/"):
            logger.info(f"GPU memory utilization: {args.gpu_memory_utilization}")
            # CLI --max-tokens overrides config; fall back to config, then 256
            actual_model = args.model.replace("vllm/", "")
            _model_cfg = LiteLLMWrapper._get_model_vision_config(actual_model)
            max_tokens = args.max_tokens if args.max_tokens is not None else _model_cfg.get("max_tokens", 256)
            logger.info(f"Max tokens: {max_tokens}")
            model = LiteLLMWrapper.get_or_create_model(
                model_id=args.model,
                max_tokens=max_tokens,
                temperature=args.temperature,
                auto_start_server=False,  # evaluator starts lazily after cache check
                gpu_memory_utilization=args.gpu_memory_utilization,
                allowed_media_path=allowed_media_path,
                max_model_len=args.max_model_len,
                port=args.port,
                weights_path=args.ft_weights,
            )
        else:
            max_tokens = args.max_tokens if args.max_tokens is not None else 512
            model = LiteLLMWrapper(
                model_id=args.model,
                api_base=args.api_base,
                api_key=args.api_key,
                max_tokens=max_tokens,
                temperature=args.temperature,
                allowed_media_path=allowed_media_path
            )
        logger.info("Model initialized successfully")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        match_criterion = (
            args.match_criterion_file.read_text()
            if args.match_criterion_file else None
        )
        if match_criterion:
            logger.info("Using custom match criterion from %s",
                        args.match_criterion_file)

        # Create evaluation runner
        runner = EvaluationRunner(
            model=model,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            consistency_checks=args.consistency,
            consistency_runs=args.consistency_runs,
            judge_model=args.judge_model,
            judge_port=args.judge_port,
            judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
            judge_max_model_len=args.judge_max_model_len,
            judge_temperature=args.judge_temperature,
            judge_recall_max_tokens=args.judge_recall_max_tokens,
            max_retries=args.max_retries,
            match_criterion=match_criterion,
        )

        # Run evaluation
        logger.info("Running evaluation...")
        results = runner.evaluate_benchmark(
            benchmark_path=args.benchmark_path,
            questions_file=args.questions_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            no_media=args.no_media,
            transcript_file=args.transcript_file,
        )

        # Print statistics
        print_statistics(results)

        output_path = results.get("output_path", args.output_dir)
        print(f"\nResults saved to: {output_path}")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        LiteLLMWrapper.shutdown_all_models()
        logger.info("Cleanup completed")


if __name__ == "__main__":
    main()
