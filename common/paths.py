"""Stable project locations independent of a module's folder."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
LEGACY_SCENE = PROJECT_ROOT / "scene.xml"
BOWL_SCENE = PROJECT_ROOT / "simulation/scenes/scene_bowl.xml"
GPU_COMPARISON = PROJECT_ROOT / "docs/benchmarks/GPU_BATCH_COMPARISON.md"
