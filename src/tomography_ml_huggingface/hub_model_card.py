"""Shared Hugging Face model-card and artefact helpers for M11 exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"
README_NAME = "README.md"

FINAL_REPORT_URL = (
    "https://github.com/tbgitoo/gummybear-tomography/blob/master/"
    "GummyBearTomography_Final_Report.ipynb"
)
DATASET_HUB_ID = "tbhugging/gummybear-tomography"
DATASET_HUB_URL = f"https://huggingface.co/datasets/{DATASET_HUB_ID}"
REPO_URL = "https://github.com/tbgitoo/gummybear-tomography"
EVAL_RESULTS_DOCS_URL = (
    "https://huggingface.co/docs/hub/model-cards#evaluation-results"
)

METRIC_RMSE_TOTAL = "RMSE_total (Euclidean xyz)"
PIPELINE_TAG = "image-feature-extraction"


@dataclass(frozen=True)
class ModelCardEvalSpec:
    """Structured evaluation metadata for Hub YAML + prose sections."""

    task_name: str
    dataset_name: str
    dataset_config: str = "m8_1"
    source_name: str = ""
    source_url: str = FINAL_REPORT_URL
    tags: tuple[str, ...] = ()


def build_model_card_frontmatter(
    *,
    hub_id: str,
    metrics: Mapping[str, float],
    eval_spec: ModelCardEvalSpec,
) -> str:
    """Build Hub-compliant YAML metadata (pipeline_tag, metrics, model-index)."""
    try:
        from huggingface_hub import EvalResult, ModelCardData
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to write the Hub model card. "
            'Install with: pip install ".[hf]" -c requirements.txt'
        ) from exc

    eval_results: list[Any] = []
    for split_key, split_name in (
        ("validation_RMSE_total", "validation"),
        ("test_RMSE_total", "test"),
    ):
        if split_key not in metrics:
            continue
        eval_results.append(
            EvalResult(
                task_type=PIPELINE_TAG,
                task_name=eval_spec.task_name,
                dataset_type=DATASET_HUB_ID,
                dataset_name=eval_spec.dataset_name,
                dataset_config=eval_spec.dataset_config,
                dataset_split=split_name,
                metric_type="rmse",
                metric_name=METRIC_RMSE_TOTAL,
                metric_value=round(float(metrics[split_key]), 6),
                source_name=eval_spec.source_name,
                source_url=eval_spec.source_url,
            )
        )

    card_data = ModelCardData(
        license="apache-2.0",
        library_name="pytorch",
        pipeline_tag=PIPELINE_TAG,
        tags=list(eval_spec.tags),
        datasets=[DATASET_HUB_ID],
        metrics=["rmse"],
        model_name=hub_id,
        eval_results=eval_results or None,
    )
    yaml_body = card_data.to_yaml().rstrip() + "\n"
    return f"---\n{yaml_body}---\n"


def build_evaluation_results_prose(
    *,
    eval_spec: ModelCardEvalSpec,
    protocol_lines: Sequence[str],
    checkpoint_markdown: str,
) -> list[str]:
    """Markdown body for the Evaluation Results section."""
    lines = [
        "## Evaluation Results",
        "",
        "Structured scores for the Hub widget are declared in the YAML "
        "`model-index` / `metrics` metadata "
        f"([Model Cards — Evaluation Results]({EVAL_RESULTS_DOCS_URL})).",
        "",
        "### Testing Data",
        "",
        "- Dataset: "
        f"[{DATASET_HUB_ID}]({DATASET_HUB_URL}) "
        f"(config `{eval_spec.dataset_config}`)",
        "- Splits: `validation`, `test`",
        *protocol_lines,
        checkpoint_markdown,
        "",
        "### Metrics",
        "",
        "Reported error is **Euclidean RMSE** over particle `(x,y,z)`:",
        "`d_i = ||pred_i - y_i||_2`, then `RMSE_total = sqrt(mean_i d_i^2)`.",
        f"Hub metric id: `rmse` (display name `{METRIC_RMSE_TOTAL}`).",
        "",
        "### Results",
        "",
    ]
    return lines


def build_results_table(metrics: Mapping[str, float]) -> list[str]:
    """Markdown table rows for held-out RMSE scores."""
    result_rows: list[str] = []
    for key, split_name in (
        ("validation_RMSE_total", "validation"),
        ("test_RMSE_total", "test"),
    ):
        if key in metrics:
            result_rows.append(
                f"| `{split_name}` | RMSE_total (Euclidean xyz) | "
                f"`{metrics[key]:.6f}` |"
            )
    if not result_rows:
        return []
    return [
        "| Split | Metric | Value |",
        "| --- | --- | ---: |",
        *result_rows,
        "",
    ]


def write_hub_model_artifacts(
    dest: Path,
    *,
    state_dict: Mapping[str, Any],
    config: Mapping[str, Any],
    readme_markdown: str,
) -> tuple[Path, Path, Path]:
    """Write ``pytorch_model.bin``, ``config.json``, and ``README.md``."""
    import torch

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    weights_path = dest / WEIGHTS_NAME
    torch.save(dict(state_dict), weights_path)
    config_path = dest / CONFIG_NAME
    config_path.write_text(json.dumps(dict(config), indent=2) + "\n", encoding="utf-8")
    readme_path = dest / README_NAME
    readme_path.write_text(readme_markdown, encoding="utf-8")
    return weights_path, config_path, readme_path


__all__ = [
    "CONFIG_NAME",
    "DATASET_HUB_ID",
    "DATASET_HUB_URL",
    "EVAL_RESULTS_DOCS_URL",
    "FINAL_REPORT_URL",
    "METRIC_RMSE_TOTAL",
    "ModelCardEvalSpec",
    "PIPELINE_TAG",
    "README_NAME",
    "REPO_URL",
    "WEIGHTS_NAME",
    "build_evaluation_results_prose",
    "build_model_card_frontmatter",
    "build_results_table",
    "write_hub_model_artifacts",
]
