"""Layer-wise cross-lingual retrieval accuracy from extract_embeddings.py output.

Takes an embedding file produced with `--pooling llm-layers` (one
[num_layers+1, llm_dim] vector per example, pooled over the mapped-question
span at every decoder depth -- see extract_embeddings.py) and, at each
layer, measures whether a question's representation in one language
retrieves the *same question* (matched by id) in another language: for
every ordered language pair (A, B) and every id shared between them, rank
all of B's embeddings by cosine similarity to A's embedding for that id and
check whether the true match is the top-1 (or top-k) neighbor. Averaged
over all pairs and ids, this gives one retrieval-accuracy number per layer
-- a standard bitext/cross-lingual-alignment probe (the same style of
analysis LASER/MEXA use), applied here to this project's own mapped-token
representations instead of sentence encoders.

Requires numpy in the project's venv (add with `pip install numpy` if
extract_embeddings.py's torch install didn't already pull it in).

Example
-------
    python Stage3/analysis/layerwise_retrieval.py \\
        --embeddings stage1=$SCRATCH/M2-ALIGN/Stage3/analysis_out/layers_stage1_xgqa.pt \\
        --embeddings stage2=$SCRATCH/M2-ALIGN/Stage3/analysis_out/layers_stage2_xgqa.pt \\
        --embeddings stage3=$SCRATCH/M2-ALIGN/Stage3/analysis_out/layers_stage3_xgqa.pt \\
        --k 1 \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/xgqa_layerwise_retrieval.csv \\
        --plot $SCRATCH/M2-ALIGN/Stage3/analysis_out/xgqa_layerwise_retrieval.png

Produces a table like:

    stage    layer   top1_accuracy   n_pairs
    stage1   0       0.81            42
    stage1   1       0.34            42
    ...
    stage1   28      0.05            42
    stage3   0       0.81            42
    stage3   1       0.62            42
    ...
    stage3   28      0.29            42

(layer 0 is the pre-LLM mapped embedding itself, identical in kind to
extract_embeddings.py's "xm" pooling mode; layers 1..N are post-decoder-layer.)

Useful for: this is the direct mechanistic test behind the
DeepStack/sentinel-marker hypothesis -- if retrieval accuracy collapses
sharply in exactly the first few layers (where DeepStack injects visual
features at image-token positions), that's evidence the injection disrupts
the mapped span's cross-lingual alignment even though it doesn't directly
touch those positions, motivating a sentinel-token experiment that
insulates the mapped span from that disruption. Comparing curves across
stages also shows whether later-stage training (Stage3's VQA
task-specialization) *preserves* the alignment Stage1/2 built, or erodes it
by depth -- a concrete, layer-resolved version of the aggregate
"M2RB trails Baseline" gap.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys

import numpy as np
import torch


def _parse_embeddings_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--embeddings entries must be 'LABEL=PATH', got: {spec!r}")
    label, path = spec.split("=", 1)
    return label, path


def _topk_accuracy(emb_a: np.ndarray, ids_a: list, emb_b: np.ndarray, ids_b: list, k: int) -> float:
    """Fraction of A's rows whose top-k nearest B-row (cosine) shares its id."""
    a_norm = emb_a / np.clip(np.linalg.norm(emb_a, axis=1, keepdims=True), 1e-8, None)
    b_norm = emb_b / np.clip(np.linalg.norm(emb_b, axis=1, keepdims=True), 1e-8, None)
    sims = a_norm @ b_norm.T  # [n_a, n_b]
    id_to_idx_b = {id_: i for i, id_ in enumerate(ids_b)}
    k_eff = min(k, sims.shape[1])
    topk_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
    hits = 0
    for i, id_ in enumerate(ids_a):
        target_idx = id_to_idx_b[id_]
        if target_idx in topk_idx[i]:
            hits += 1
    return hits / len(ids_a) if ids_a else 0.0


def layerwise_retrieval(data: dict, k: int) -> list[tuple[int, float, int]]:
    """Returns `[(layer, mean_top-k_accuracy, n_language_pairs_averaged), ...]`."""
    embeddings = data["embeddings"]  # [N, num_layers+1, dim]
    languages = data["languages"]
    ids = data["ids"]
    if embeddings.ndim != 3:
        raise ValueError("layerwise_retrieval needs a --pooling llm-layers embedding file (3D tensor).")

    # id -> global row index, per language (rows are unique per (id, lang) by construction in extract_embeddings.py).
    by_lang: dict[str, dict] = {}
    for i, lang in enumerate(languages):
        by_lang.setdefault(lang, {})[ids[i]] = i

    unique_langs = sorted(by_lang)
    n_layers = embeddings.shape[1]
    results = []
    for layer in range(n_layers):
        layer_embeds = embeddings[:, layer, :].numpy().astype(np.float64)
        accs = []
        for lang_a, lang_b in itertools.permutations(unique_langs, 2):
            id_to_idx_a, id_to_idx_b = by_lang[lang_a], by_lang[lang_b]
            ids_shared = sorted(set(id_to_idx_a) & set(id_to_idx_b))
            if not ids_shared:
                continue
            idx_a = [id_to_idx_a[id_] for id_ in ids_shared]
            idx_b = [id_to_idx_b[id_] for id_ in ids_shared]
            acc = _topk_accuracy(layer_embeds[idx_a], ids_shared, layer_embeds[idx_b], ids_shared, k)
            accs.append(acc)
        results.append((layer, float(np.mean(accs)) if accs else 0.0, len(accs)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", action="append", required=True,
                         help="'LABEL=PATH' to an extract_embeddings.py --pooling llm-layers output .pt file. "
                              "Repeat to compare stages/checkpoints on one plot.")
    parser.add_argument("--k", type=int, default=1, help="Top-k for retrieval accuracy (default: 1).")
    parser.add_argument("--out", default=None, help="Write the table as CSV here; also printed to stdout.")
    parser.add_argument("--plot", default=None, help="If given, also save a layer-vs-accuracy line plot here (PNG).")
    args = parser.parse_args()

    specs = [_parse_embeddings_arg(s) for s in args.embeddings]
    fieldnames = ["stage", "layer", f"top{args.k}_accuracy", "n_pairs"]
    table_rows = []
    curves = {}

    for label, path in specs:
        data = torch.load(path, map_location="cpu")
        results = layerwise_retrieval(data, args.k)
        curves[label] = results
        for layer, acc, n_pairs in results:
            table_rows.append({"stage": label, "layer": layer, f"top{args.k}_accuracy": round(acc, 4), "n_pairs": n_pairs})

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)
        print(f"\nWrote {len(table_rows)} rows to {args.out}", file=sys.stderr)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for label, results in curves.items():
            layers = [r[0] for r in results]
            accs = [r[1] for r in results]
            ax.plot(layers, accs, marker="o", markersize=3, label=label)
        ax.set_xlabel("Decoder layer (0 = pre-LLM mapped embedding)")
        ax.set_ylabel(f"Top-{args.k} cross-lingual retrieval accuracy")
        ax.set_title("Layer-wise cross-lingual retrieval accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"Wrote {args.plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
