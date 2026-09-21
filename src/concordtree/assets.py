"""Package-resource loading and integrity checks for the frozen runtime."""

from __future__ import annotations

import hashlib
import importlib.util
from functools import lru_cache
import os
import platform
from pathlib import Path
from typing import Any

import torch

from concordtree.models import QuartFormer, QuartetMLP


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_ROOT / "assets" / "models"
BACKEND_ROOT = PACKAGE_ROOT / "assets" / "backends"

ASSET_SHA256 = {
    "models/qf1.pt": "e82d87328f7cd4bd7655c9f6c9a492068c572f96f17a80ef6a0222c7f7f1fcfb",
    "models/best_mlp_model.pth": "a74e89e7b6091938f0b413a5c1446864bc18a8db7176f5c77a204018bc8e2edb",
    "models/homogeneous/qf1.pt": "303503ce5b193d9c3d7926c862c39e624553ff2fab046c097cc2a0745acdda4c",
    "models/homogeneous/best_mlp_model.pth": "8554764e33885992cc31e4d044a82a4ae8e876f119197e72b07ae3a1dcbc533b",
    "models/coeff_blocks.pt": "4786805f655e59c6d9a20153fbf8f19912043b9e4dd19e5fada0e832e8984c57",
    "models/attention_pair_masks.pt": "26456a08666c8ccabbc8e63493ee0fe066be7f6acf339052329fda93d212a43d",
    "models/quartet_matrix.pt": "3b199d6178bba0db6339075e5996cbcc8e25cef247ca3827bb718ea86fcb2112",
    "models/species_encoding.pt": "568e9945cda6f14ad6d569523a763a543caab34a42ca467e06271c3f0d0e612e",
    "backends/sequence_processor_backend.cpython-310-x86_64-linux-gnu.so": "1fdb899cce75ff834ff9d4e11ee2339a4565c6ba5cfb9bd26eded5af8fd00ff5",
    "backends/pattern_freq_cuda_backend.cpython-310-x86_64-linux-gnu.so": "90c70c0f53a618abb536024c9544bf6b43d329ffa0391929c68a1b3e7915498d",
    "backends/panel_score_backend.cpython-310-x86_64-linux-gnu.so": "ce1ec2c040507461739cff9d92b1d2a530d0c2fae24d071d102601464b494fe5",
    "backends/candidate_graph_backend.cpython-310-x86_64-linux-gnu.so": "66cd7b0e0aa50eb3bfddd418fff17a91693212d531fa4cb17efebf64da7e406b",
    "backends/learned_nni_plan_backend.cpython-310-x86_64-linux-gnu.so": "cb48c4ce4e7d2465c78cf4aa28a6f4ffa41852a7d853ca7a15ca291f425eae0b",
    "backends/split_compat_backend.cpython-310-x86_64-linux-gnu.so": "4d00cc58dcf52852e9d1f98a6f333f9f03d0a428724726d6dcb11650710cab34",
}

QUARTET_PREDICTORS = ("heterogeneous", "homogeneous")
MODEL_ASSETS = {
    "heterogeneous": {
        "mlp": "models/best_mlp_model.pth",
        "qf": "models/qf1.pt",
    },
    "homogeneous": {
        "mlp": "models/homogeneous/best_mlp_model.pth",
        "qf": "models/homogeneous/qf1.pt",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(relative: str) -> Path:
    path = PACKAGE_ROOT / "assets" / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing packaged runtime asset: {path}")
    return path


def verify_assets() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in ASSET_SHA256.items():
        digest = sha256_file(asset_path(relative))
        if digest != expected:
            raise RuntimeError(
                f"asset integrity failure for {relative}: {digest} != {expected}"
            )
        observed[relative] = digest
    return observed


def _load_extension(name: str, path: Path) -> Any:
    if platform.python_version_tuple()[:2] != ("3", "10"):
        raise RuntimeError(
            "the bundled binary backends require CPython 3.10 on Linux x86_64"
        )
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load binary extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_backends() -> tuple[Any, Any]:
    sequence = _load_extension(
        "sequence_processor_backend",
        asset_path(
            "backends/sequence_processor_backend.cpython-310-x86_64-linux-gnu.so"
        ),
    )
    experimental_pattern = os.environ.get("CONCORDTREE_EXPERIMENTAL_PATTERN_BACKEND")
    pattern_path = (
        Path(experimental_pattern).resolve(strict=True)
        if experimental_pattern
        else asset_path(
            "backends/pattern_freq_cuda_backend.cpython-310-x86_64-linux-gnu.so"
        )
    )
    pattern = _load_extension(
        "pattern_freq_cuda_backend",
        pattern_path,
    )
    return sequence, pattern


def load_panel_score_backend() -> Any:
    """Load the integer-only contextual-panel plan compiler."""

    return _load_extension(
        "panel_score_backend",
        asset_path(
            "backends/panel_score_backend.cpython-310-x86_64-linux-gnu.so"
        ),
    )


@lru_cache(maxsize=1)
def load_candidate_graph_backend() -> Any:
    """Load the exact multicore projection-pool and top-k compiler."""

    return _load_extension(
        "candidate_graph_backend",
        asset_path(
            "backends/candidate_graph_backend.cpython-310-x86_64-linux-gnu.so"
        ),
    )


@lru_cache(maxsize=1)
def load_learned_nni_plan_backend() -> Any:
    """Load the exact native tree-to-quartet plan compiler."""

    return _load_extension(
        "learned_nni_plan_backend",
        asset_path(
            "backends/learned_nni_plan_backend.cpython-310-x86_64-linux-gnu.so"
        ),
    )


@lru_cache(maxsize=1)
def load_split_compat_backend() -> Any:
    """Load the exact packed anchored-laminar selector."""

    return _load_extension(
        "split_compat_backend",
        asset_path(
            "backends/split_compat_backend.cpython-310-x86_64-linux-gnu.so"
        ),
    )


def model_asset(kind: str, quartet_predictor: str) -> Path:
    if quartet_predictor not in MODEL_ASSETS:
        raise ValueError(
            f"unknown quartet predictor {quartet_predictor!r}; "
            f"expected one of {QUARTET_PREDICTORS}"
        )
    if kind not in {"mlp", "qf"}:
        raise ValueError(f"unknown model asset kind: {kind!r}")
    return asset_path(MODEL_ASSETS[quartet_predictor][kind])


def load_mlp(
    device: torch.device, quartet_predictor: str = "heterogeneous"
) -> QuartetMLP:
    model = QuartetMLP().to(device)
    model.load_state_dict(
        torch.load(
            model_asset("mlp", quartet_predictor),
            map_location=device,
            weights_only=True,
        )
    )
    return model.eval()


def load_attention_pair_masks(device: torch.device) -> torch.Tensor:
    pair_masks = torch.load(
        asset_path("models/attention_pair_masks.pt"),
        map_location=device,
        weights_only=True,
    ).to(device)
    if pair_masks.shape != (665, 96, 16) or pair_masks.dtype != torch.int32:
        raise RuntimeError("packaged attention pair masks have an invalid schema")
    return pair_masks


def load_qf_bundle(
    device: torch.device, quartet_predictor: str = "heterogeneous"
) -> tuple[Any, Any, QuartFormer, torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence, pattern = load_backends()
    model = QuartFormer(species_num=24)
    model.load_state_dict(
        torch.load(
            model_asset("qf", quartet_predictor),
            map_location="cpu",
            weights_only=True,
        )
    )
    model = model.to(device).eval()
    coeff = torch.load(
        asset_path("models/coeff_blocks.pt"),
        map_location=device,
        weights_only=False,
    ).to(device)
    quartet_matrix = torch.load(
        asset_path("models/quartet_matrix.pt"),
        map_location=device,
        weights_only=False,
    ).to(device)
    species = torch.load(
        asset_path("models/species_encoding.pt"),
        map_location=device,
        weights_only=False,
    ).to(device)
    return sequence, pattern, model, coeff, quartet_matrix, species
