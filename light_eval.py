"""
LightEval runner for two models on math benchmarks.

By default evaluates:
- teetone/OpenR1-Distill-Qwen3-1.7B-Math
- Qwen/Qwen3-4B-Thinking-2507

On tasks:
- MATH500
- AIME24

Adjust MODEL_NAMES or TASKS below as needed.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


DEFAULT_MODEL_NAMES: List[str] = [
    "teetone/OpenR1-Distill-Qwen3-1.7B-Math",
    "Qwen/Qwen3-4B-Thinking-2507",
]

# Tasks follow LightEval naming; adjust if your install uses different aliases.
DEFAULT_TASKS: List[str] = ["MATH500", "AIME24"]


def build_pipeline(model_name: str, tasks: Iterable[str], output_dir: str, max_samples: int) -> Pipeline:
    evaluation_tracker = EvaluationTracker(output_dir=output_dir)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
    )

    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    config = TransformersModelConfig(model_name=model_name, batch_size=1)
    wrapped_model = TransformersModel.from_model(base_model, config)

    return Pipeline(
        model=wrapped_model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=list(tasks),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LightEval on math benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_NAMES, help="Model names to evaluate")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="LightEval task names")
    parser.add_argument("--output_dir", default="./results", help="Directory to store evaluation outputs")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples per task (LightEval cap)")
    args = parser.parse_args()

    for model_name in args.models:
        print(f"\n=== Evaluating {model_name} on {args.tasks} ===")
        pipeline = build_pipeline(model_name, args.tasks, args.output_dir, args.max_samples)
        results = pipeline.evaluate()
        pipeline.show_results()
        # Optionally use results programmatically
        _ = pipeline.get_results()
        print(f"Finished {model_name}")


if __name__ == "__main__":
    main()

