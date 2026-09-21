# ConcordTree

ConcordTree builds an unrooted phylogenetic tree directly from an aligned DNA
or RNA sequence file on one NVIDIA GPU.

The method builds several complementary tree views from the alignment, combines
the branches supported across those views, and then checks uncertain local
branches with a neural model trained to compare four taxa at a time. You can
choose either of two model architectures:

- **Lightweight (`mlp`)** is the faster default.
- **Attention (`transformer`)** uses a larger attention-based model and more
  computation.

Both choices run the same tree-building procedure; only the neural model used
for local comparisons changes.

**[See the complete benchmark and comparison results](docs/index.html)**

## Requirements

- Linux x86-64
- CPython 3.10
- NVIDIA GPU with a driver compatible with PyTorch 2.5.1 and CUDA 12.1
- Conda for the recommended isolated installation

## Install

```bash
bash scripts/create_environment.sh
source .conda/env/bin/activate
concordtree doctor --device cuda:0
```

To install the packaged wheel into an existing compatible Python 3.10
environment:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install dist/concordtree-0.1.2-cp310-cp310-linux_x86_64.whl
concordtree doctor --device cuda:0
```

## Run

Test the installation with the included synthetic alignment:

```bash
concordtree infer \
  --msa examples/minimal24.phy \
  --work-dir runs/demo \
  --tree-type gene-tree \
  --quartet-model mlp \
  --device cuda:0
```

For your own alignment:

```bash
concordtree inspect --msa input.phy

concordtree infer \
  --msa input.phy \
  --work-dir runs/my-tree \
  --tree-type species-tree \
  --quartet-model mlp \
  --missing-data-model standard \
  --device cuda:0
```

The input must be a sequential PHYLIP nucleotide alignment with at least 24
taxa. `U` is accepted and treated as `T`; ambiguity codes, gaps, and other
non-`ACGTU` symbols are treated as missing data. The final tree is written to
`WORK_DIR/tree.nwk`.

## Choose the tree type

| Setting | Use it when |
| --- | --- |
| `--tree-type gene-tree` | The alignment represents one locus or one shared genealogy. |
| `--tree-type species-tree` | Concatenated loci may have different gene histories. This is the default. |

This choice selects the matching trained local model. ConcordTree still reads
one alignment and does not require separately estimated gene trees.

## Choose the missing-data model

Run `concordtree inspect --msa input.phy` first. It reports the coherent
sparsity index (CSI) from the alignment alone:

| CSI | Current guidance |
| --- | --- |
| `<= 0.25` | Use `standard`. |
| `0.25–0.35` | Indeterminate; compare both modes when accuracy matters. |
| `>= 0.35` | Use `coverage-aware`. |

| Setting | Use it when |
| --- | --- |
| `--missing-data-model standard` | Taxa share most aligned sites and gaps are ordinary alignment gaps. This is the default. |
| `--missing-data-model coverage-aware` | Taxon pairs share few observed sites and taxa tend to miss the same regions. |

This option changes only the distance used to construct each initial View
tree. Neural refinement and View merging are identical. 

## Choose the neural model

| Setting | Use it when |
| --- | --- |
| `--quartet-model mlp` | Use the lightweight model. This is the faster default. |
| `--quartet-model transformer` | Use the larger attention-based model. |

## Control refinement stopping

Each of the three refinement stages has an independent convergence threshold
and round budget:

```text
--view-stop-ratio             --view-max-rounds
--coordinate-stop-ratio       --coordinate-max-rounds
--saturation-stop-ratio       --saturation-max-rounds
```

A stage stops when it makes no move, reaches its enabled ratio threshold, or
uses its enabled round budget. Each value accepts a number; either member of a
pair also accepts `none`. For example, threshold-only Saturation is selected
with `--saturation-stop-ratio 0.002 --saturation-max-rounds none`. The defaults
are `0.01/24`, `0.005/4`, and `0.005/5`, respectively.

`--view-count` accepts 2–8 and defaults to 4. `--parallelism 0` selects the
CPU thread budget automatically. Run `concordtree infer --help` for all
options.

## Other commands

```bash
# Inspect missing-data structure without a GPU or reference tree
concordtree inspect --msa input.phy

# Check CUDA, native backends, and packaged model files
concordtree doctor --device cuda:0

# Compare two trees
concordtree validate \
  --msa input.phy \
  --expected expected.nwk \
  --actual runs/my-tree/tree.nwk
```

[中文说明](README.zh-CN.md) · [Results and method](docs/index.html) ·
[License](LICENSE)
