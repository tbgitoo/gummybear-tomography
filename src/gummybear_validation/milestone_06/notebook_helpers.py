"""Notebook-facing M6 helpers (tables, demo cleanup, workbook copies, previews)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from gummybear.datasets.generation_plan import ExecutionPlan, SequenceJob
from gummybear.paths import display_path, repo_relative_path
from gummybear_validation.milestone_06.inspect_cache_plan import inspect_cache_plan


def workbook_sheet_summary(workbook: Any) -> pd.DataFrame:
    """Row counts per sheet for dry-run display."""
    return pd.DataFrame(
        [
            {
                "sheet": sheet,
                "rows": len(workbook.rows(sheet)),
                "enabled_rows": len(workbook.enabled_rows(sheet)),
                "disabled_rows": len(workbook.disabled_rows(sheet)),
            }
            for sheet in workbook.sheet_names
        ]
    )


def jobs_summary_table(jobs: Sequence[SequenceJob]) -> pd.DataFrame:
    """Compact table of resolved sequence jobs for notebook display."""
    return pd.DataFrame(
        [
            {
                "sequence_id": job.sequence_id,
                "split": job.split,
                "optical_setup_id": job.optical.optical_setup_id,
                "source_intensity": job.optical.source_intensity,
                "particle_setup_id": job.particle.particle_setup_id,
                "particle_group_id": job.particle_group_id,
                "n_particles": len(job.particles),
                "diffusion_setup_id": job.diffusion.diffusion_setup_id,
                "camera_schedule_id": job.camera.camera_schedule_id,
                "views": job.camera.num_views,
                "resolution": f"{job.camera.resolution_x} x {job.camera.resolution_y}",
                "clean_cache_id": job.clean_optical_cache_id[:16],
                "particle_cache_id": job.particle_source_cache_id[:16],
            }
            for job in jobs
        ]
    )


def matrix_jobs_summary_table(jobs: Sequence[SequenceJob]) -> pd.DataFrame:
    """Job table tailored to M6.5 shared-cache / sweep review."""
    return pd.DataFrame(
        [
            {
                "sequence_id": job.sequence_id,
                "resolved_job_hash": job.resolved_job_hash,
                "clean_cache_id": job.clean_optical_cache_id,
                "particle_cache_id": job.particle_source_cache_id,
                "diffusion_setup_id": job.diffusion.diffusion_setup_id,
                "extrapolation_length": job.diffusion.extrapolation_length,
                "camera_schedule_id": job.camera.camera_schedule_id,
                "frames": len(job.camera.poses),
            }
            for job in jobs
        ]
    )


def cache_plan_rows(execution_plan: ExecutionPlan) -> pd.DataFrame:
    """Flatten ``inspect_cache_plan`` into one row per diffusion group."""
    cache_plan = inspect_cache_plan(execution_plan)
    rows: list[dict[str, Any]] = []
    for clean_group in cache_plan["clean_groups"]:
        for particle_group in clean_group["particle_groups"]:
            for diffusion_group in particle_group["diffusion_groups"]:
                rows.append(
                    {
                        "optical_setup_id": clean_group["optical_setup_id"],
                        "clean_cache_id": clean_group["clean_optical_cache_id"][:16],
                        "clean_status": clean_group["cache_status"],
                        "particle_setup_id": particle_group["particle_setup_id"],
                        "particle_group_id": particle_group.get("particle_group_id", ""),
                        "particle_count": particle_group.get("particle_count", 1),
                        "particle_cache_id": particle_group["particle_source_cache_id"][:16],
                        "particle_status": particle_group["cache_status"],
                        "diffusion_setup_id": diffusion_group["diffusion_setup_id"],
                        "job_count": diffusion_group["job_count"],
                        "frame_count": diffusion_group["frame_count"],
                    }
                )
    return pd.DataFrame(rows)


def camera_task_rows(execution_plan: ExecutionPlan) -> pd.DataFrame:
    """Flatten planned camera tasks (no rays / images)."""
    rows: list[dict[str, Any]] = []
    for clean_group in execution_plan.clean_groups:
        for particle_group in clean_group.particle_groups:
            for diffusion_group in particle_group.diffusion_groups:
                for task in diffusion_group.camera_tasks:
                    rows.append(
                        {
                            "sequence_id": task.sequence_id,
                            "frame_index": task.frame_index,
                            "angle_deg": task.angle_deg,
                            "resolution_x": task.resolution_x,
                            "resolution_y": task.resolution_y,
                        }
                    )
    return pd.DataFrame(rows)


def clear_demo_sequence_outputs(
    jobs: Sequence[SequenceJob],
    output_root: Path,
    *,
    repo_root: Path | None = None,
) -> list[Path]:
    """Remove planned sequence directories under *output_root*; keep sibling ``_cache``."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    removed: list[Path] = []
    for job in jobs:
        sequence_dir = output_root / job.sequence_id
        if sequence_dir.exists():
            shutil.rmtree(sequence_dir)
            removed.append(sequence_dir)
            label = (
                display_path(sequence_dir)
                if repo_root is None
                else repo_relative_path(sequence_dir)
            )
            print(f"Removed prior demo sequence output: {label}")
    return removed


def miss_hit_cache_tables(
    first: Any,
    second: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cache-status and stage-timing tables for an M6.4 miss→hit demo."""
    cache_status_table = pd.DataFrame(
        {
            "Forced MISS status": [
                first.clean_cache.status.upper(),
                first.particle_cache.status.upper(),
            ],
            "Forced MISS reason": [
                first.clean_cache.reason,
                first.particle_cache.reason,
            ],
            "Cache HIT status": [
                second.clean_cache.status.upper(),
                second.particle_cache.status.upper(),
            ],
            "Cache HIT reason": [
                second.clean_cache.reason,
                second.particle_cache.reason,
            ],
        },
        index=["Clean source cache", "Particle source cache"],
    )
    stage_keys = {
        "Clean source": "clean_source",
        "Particle source": "particle_source",
        "Diffusion solves": "diffusion_solves",
        "Camera capture": "camera_capture",
    }
    timing_table = pd.DataFrame(
        {
            "Forced MISS (s)": [first.stage_seconds[key] for key in stage_keys.values()],
            "Cache HIT (s)": [second.stage_seconds[key] for key in stage_keys.values()],
        },
        index=stage_keys,
    )
    timing_table["Time saved (s)"] = (
        timing_table["Forced MISS (s)"] - timing_table["Cache HIT (s)"]
    )
    timing_table["Speedup (×)"] = (
        timing_table["Forced MISS (s)"] / timing_table["Cache HIT (s)"]
    )
    timing_table.loc["Four-stage total"] = {
        "Forced MISS (s)": timing_table["Forced MISS (s)"].sum(),
        "Cache HIT (s)": timing_table["Cache HIT (s)"].sum(),
        "Time saved (s)": timing_table["Time saved (s)"].sum(),
        "Speedup (×)": (
            timing_table["Forced MISS (s)"].sum() / timing_table["Cache HIT (s)"].sum()
        ),
    }
    return cache_status_table, timing_table


def enable_sequence_in_workbook_copy(
    source_workbook: Path,
    dest_workbook: Path,
    sequence_id: str,
    *,
    enabled: bool = True,
) -> Path:
    """Write a temp workbook copy with one ``sequences`` row enabled/disabled."""
    frames = pd.read_excel(source_workbook, sheet_name=None, engine="openpyxl")
    sequences = frames["sequences"]
    mask = sequences["sequence_id"] == sequence_id
    if not mask.any():
        raise KeyError(f"sequence_id {sequence_id!r} not found in {source_workbook}")
    sequences.loc[mask, "enabled"] = enabled
    dest_workbook = Path(dest_workbook)
    dest_workbook.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(dest_workbook, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return dest_workbook


def load_sequence_manifest(sequence_dir: Path) -> dict[str, Any]:
    """Load ``manifest.json`` from a sequence directory."""
    return json.loads((Path(sequence_dir) / "manifest.json").read_text())


def manifest_audit_summary(
    manifest: Mapping[str, Any],
    *,
    validation_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact provenance + particles summary for notebook display."""
    setup_coord_keys = ("optical", "particle", "diffusion", "camera", "corruption")
    setup_coordinates: dict[str, Any] = {}
    for name in setup_coord_keys:
        setup = manifest.get("setups", {}).get(name)
        if not isinstance(setup, dict) or "workbook_name" not in setup:
            continue
        setup_coordinates[name] = {
            "workbook_name": setup["workbook_name"],
            "workbook_sheet": setup["workbook_sheet"],
            "source_excel_row": setup["source_excel_row"],
        }

    particles_block = manifest.get("setups", {}).get("particles")
    if particles_block is None:
        particle_setup = manifest["setups"]["particle"]
        particles_summary: dict[str, Any] = {
            "schema": "legacy_singleton",
            "count": 1,
            "particle_setup_id": particle_setup.get("particle_setup_id"),
            "center": [
                particle_setup.get("center_x"),
                particle_setup.get("center_y"),
                particle_setup.get("center_z"),
            ],
            "radius": particle_setup.get("radius"),
        }
    else:
        particles_summary = {
            "schema": "ordered_group",
            "particle_group_id": particles_block.get("particle_group_id"),
            "count": particles_block.get("count"),
            "order": particles_block.get("order"),
            "items": [
                {
                    "particle_setup_id": item.get("particle_setup_id"),
                    "center": [
                        item.get("center_x"),
                        item.get("center_y"),
                        item.get("center_z"),
                    ],
                    "radius": item.get("radius"),
                }
                for item in particles_block.get("items", [])
            ],
        }

    out: dict[str, Any] = {
        "workbook": manifest.get("workbook"),
        "caches": {
            "persistent_cache_used": manifest["caches"]["persistent_cache_used"],
            "clean_optical_cache_id": manifest["caches"]["clean_optical_cache_id"],
            "particle_source_cache_id": manifest["caches"]["particle_source_cache_id"],
            "diffusion_operator_cache": manifest["caches"]["diffusion_operator_cache"],
        },
        "setup_coordinates": setup_coordinates,
        "particles": particles_summary,
        "representation": manifest.get("representation"),
    }
    if validation_summary is not None:
        out["validation"] = dict(validation_summary)
    return out


def preview_sequence_roles(
    sequence_dir: Path,
    *,
    roles: Sequence[str] | None = None,
    frame_index: int = 0,
    figsize_per_role: float = 3.5,
) -> Any:
    """Show role images for one frame (or first file per role directory)."""
    import matplotlib.pyplot as plt
    from PIL import Image

    sequence_dir = Path(sequence_dir)
    manifest_path = sequence_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = load_sequence_manifest(sequence_dir)
        frame = manifest["frames"][frame_index]
        role_names = list(roles) if roles is not None else list(frame["filenames"])
        paths = [sequence_dir / frame["filenames"][role] for role in role_names]
    else:
        role_names = [
            role
            for role in (roles or ("clean", "particle", "observed", "anomaly"))
            if (sequence_dir / role).is_dir()
        ]
        paths = [sorted((sequence_dir / role).iterdir())[0] for role in role_names]

    fig, axes = plt.subplots(
        1, len(role_names), figsize=(figsize_per_role * len(role_names), 4)
    )
    if len(role_names) == 1:
        axes = [axes]
    for axis, role, path in zip(axes, role_names, paths, strict=True):
        axis.imshow(Image.open(path), cmap="gray")
        axis.set_title(role)
        axis.axis("off")
    plt.tight_layout()
    return fig
