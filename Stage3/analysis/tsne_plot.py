"""t-SNE visualization of cross-lingual embeddings from extract_embeddings.py.

Loads one or more embedding files (typically one per stage/checkpoint --
Stage1, Stage2, Stage3 -- run over the same benchmark/languages/ids so the
panels are directly comparable) and plots a t-SNE projection colored by
language, one panel per input file. Works on "xm"-pooled embedding files
directly; for "llm-layers" files, pick one layer with --layer (default: the
last, i.e. the representation right before generation would start).

Requires numpy, scikit-learn and matplotlib in the project's venv --
these aren't in the base `pip install datasets requests pillow` from the
project README; add them once with `pip install scikit-learn matplotlib`.

Example
-------
    python Stage3/analysis/tsne_plot.py \\
        --embeddings stage1=$SCRATCH/M2-ALIGN/Stage3/analysis_out/xm_stage1_xgqa.pt \\
        --embeddings stage2=$SCRATCH/M2-ALIGN/Stage3/analysis_out/xm_stage2_xgqa.pt \\
        --embeddings stage3=$SCRATCH/M2-ALIGN/Stage3/analysis_out/xm_stage3_xgqa.pt \\
        --out $SCRATCH/M2-ALIGN/Stage3/analysis_out/tsne_xgqa_by_stage.png

Produces a 3-panel PNG (stage1 / stage2 / stage3), each panel a scatter of
~300*7 points colored by language (bn/de/ru/zh/pt/id/ko).

Useful for: a visual, reviewer-legible argument for *when* cross-lingual
alignment emerges. If Stage1's panel shows 7 well-separated language
clusters (each language's questions cluster with themselves, not with their
same-question counterparts in other languages) while Stage3's panel shows
mixing by *content* instead of language (points from different languages
but the same qid sit near each other), that's direct visual evidence the
mapping is doing what the method claims -- and if it *doesn't* mix, that's
equally useful negative evidence for the Discussion section, motivating
e.g. the sentinel-marker follow-up.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch


def _parse_embeddings_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--embeddings entries must be 'LABEL=PATH', got: {spec!r}")
    label, path = spec.split("=", 1)
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", action="append", required=True,
                         help="'LABEL=PATH' to an extract_embeddings.py output .pt file. Repeat for multiple panels.")
    parser.add_argument("--layer", type=int, default=-1,
                         help="For 'llm-layers'-pooled files only: which layer to plot (default: last).")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="Output PNG path.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    specs = [_parse_embeddings_arg(s) for s in args.embeddings]
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 5.5), squeeze=False)
    axes = axes[0]

    for ax, (label, path) in zip(axes, specs):
        data = torch.load(path, map_location="cpu")
        embeddings = data["embeddings"]
        if data["pooling"] == "llm-layers":
            embeddings = embeddings[:, args.layer, :]
        embeddings = embeddings.numpy().astype(np.float64)
        languages = data["languages"]

        n = embeddings.shape[0]
        perplexity = min(args.perplexity, max(5.0, (n - 1) / 3))
        proj = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, init="pca").fit_transform(embeddings)

        unique_langs = sorted(set(languages))
        cmap = plt.get_cmap("tab10" if len(unique_langs) <= 10 else "tab20")
        for i, lang in enumerate(unique_langs):
            idx = [j for j, l in enumerate(languages) if l == lang]
            ax.scatter(proj[idx, 0], proj[idx, 1], s=10, alpha=0.7, color=cmap(i), label=lang)
        ax.set_title(f"{label} (n={n}, pooling={data['pooling']})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=2, fontsize=8, loc="best")

    fig.suptitle("t-SNE of mapped question embeddings, by language")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
