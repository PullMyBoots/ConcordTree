"""Output validation without loading statistical or neural runtime modules."""
from __future__ import annotations
from pathlib import Path
import os


def _linked_targets(data: Path):
    """Match rglob's non-following walk, using directory entry type metadata."""
    pending=[data]
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    yield Path(entry.path).resolve()
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))

def ensure_output_safe(path: Path, input_paths: list[Path]) -> Path:
    requested = path.absolute()
    output = path.resolve()
    for ancestor in Path(__file__).resolve().parents:
        if (ancestor / ".ResearchProject").exists():
            data = ancestor / "data"
            if requested.is_relative_to(data) or output.is_relative_to(data.resolve()):
                raise ValueError("output must be outside project data/")
            if data.is_dir() and any(output.is_relative_to(target) for target in _linked_targets(data)):
                raise ValueError("output must be outside linked project data/")
            break
    if any(output == source.parent or source in output.parents for source in input_paths):
        raise ValueError("output cannot be inside an input file or its directory")
    return output


def _output_path(path: Path, input_paths: list[Path]) -> Path:
    output=ensure_output_safe(path, input_paths)
    if output.exists():
        raise ValueError("output must be a new directory")
    return output
