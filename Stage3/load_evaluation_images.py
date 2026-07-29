"""Pre-download CVQA evaluation images into Stage3/evaluate.py's URL image cache.

Stage3/evaluate.py resolves CVQA images by URL at eval time
(``load_image_by_url``), downloading on demand into ``--image-cache-dir``.
That's fine on a workstation, but ``Stage3/job-scripts/evaluate.sh`` runs
evaluate.py via sbatch on a Narval compute node, and compute nodes there have
no internet access -- see the note in Stage2/load_image.py, which
pre-downloads WIT images for the exact same reason. Run this script from the
login node (or a workstation) before sbatch'ing any ``--benchmark cvqa``
evaluate.sh job, so every image evaluate.py needs is already sitting in the
cache and the eval job never has to reach the network.

Uses the same cache key (``sha1(image_url).hexdigest() + ".jpg"``) and the
same default cache directory as Stage3/evaluate.py's ``load_image_by_url``,
so a cache populated here is a direct hit for the eval job -- no
coordination needed beyond leaving ``--image-cache-dir`` at its default (or
passing the same override to both scripts).

Images that fail to download (dead link, non-retryable HTTP error) are just
skipped here, same as evaluate.py would skip them on the fly -- this script
only shrinks how much of that skipping happens mid-job, it doesn't change
the fallback behavior.

Usage
-----
    python Stage3/load_evaluation_images.py --languages bn,ru,zh,pt,id,ko,jv,mn,si,ga
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime

import requests
from PIL import Image

SCRATCH_ROOT = os.path.join(os.environ.get("SCRATCH", "."), "M2-ALIGN", "Stage3")

_DOWNLOAD_HEADERS = {
    "User-Agent": "M2-ALIGN/1.0 (https://github.com/MaryamTaj/M2-ALIGN)",
}
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 5
_PROGRESS_EVERY = 200


def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("stage3b_load_evaluation_images")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, f"load_evaluation_images_{ts}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def _image_cache_path(cache_dir: str, url: str) -> str:
    """Same ``sha1(url).jpg`` key Stage3/evaluate.py's load_image_by_url uses."""
    cache_key = hashlib.sha1(url.encode()).hexdigest()
    return os.path.join(cache_dir, cache_key + ".jpg")


def _download_one_image(url: str, cache_dir: str) -> str | None:
    """Fetch *url* into the cache Stage3/evaluate.py reads at eval time.

    Returns:
        None if the image ends up cached (already present, or freshly
        downloaded and decoded); otherwise a short string describing why it
        failed, for aggregating failure reasons.
    """
    path = _image_cache_path(cache_dir, url)
    if os.path.exists(path):
        return None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=15, headers=_DOWNLOAD_HEADERS)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            tmp_path = path + ".tmp"
            img.save(tmp_path, format="JPEG", quality=85)
            os.replace(tmp_path, path)
            return None
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            if status not in _RETRYABLE_STATUSES or attempt == _MAX_RETRIES - 1:
                return f"HTTP {status}"
            retry_after = exc.response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else min(30, 2 ** attempt)
            time.sleep(wait)
        except Exception as exc:
            return type(exc).__name__
    return "HTTP 429 (exhausted retries)"


def download_images(urls: list[str], cache_dir: str, max_workers: int, logger: logging.Logger) -> set[str]:
    """Download every url in *urls* into *cache_dir*, skipping already-cached ones.

    Re-running after a partial/interrupted run only fetches what's still
    missing.

    Returns:
        The subset of *urls* that ended up successfully cached.
    """
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("Downloading %d unique images -> %s", len(urls), cache_dir)

    ok: set[str] = set()
    done = 0
    failure_counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one_image, url, cache_dir): url for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            done += 1
            reason = future.result()
            if reason is None:
                ok.add(url)
            else:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if done % _PROGRESS_EVERY == 0 or done == len(urls):
                logger.info("Downloaded %d/%d images (%d failed so far)", done, len(urls), done - len(ok))

    logger.info("Image download complete: %d/%d succeeded", len(ok), len(urls))
    if failure_counts:
        top = sorted(failure_counts.items(), key=lambda kv: -kv[1])[:10]
        logger.info("Failure reasons (top %d): %s", len(top), dict(top))
    return ok


def _read_image_urls(path: str) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                urls.append(json.loads(line)["image_url"])
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download CVQA eval images for Stage3/evaluate.py.")
    parser.add_argument("--languages", type=str, required=True,
                        help="Comma-separated ISO codes with an existing Stage3/data/cvqa/<lang>.jsonl, "
                             "e.g. 'bn,ru,zh,pt,id,ko,jv,mn,si,ga'.")
    parser.add_argument("--eval-data-dir", type=str, default=os.path.join(SCRATCH_ROOT, "data", "cvqa"),
                        help="Directory holding <lang>.jsonl files written by load_evaluation_data.py.")
    parser.add_argument("--image-cache-dir", type=str,
                        default=os.path.join(SCRATCH_ROOT, "data", "cvqa", "images"),
                        help="Must match the --image-cache-dir evaluate.sh/evaluate.py will use "
                             "(defaults to the same path as evaluate.py's own default).")
    parser.add_argument("--download-workers", type=int, default=4,
                        help="Number of concurrent download threads.")
    args = parser.parse_args()

    logger = setup_logging(os.path.join(SCRATCH_ROOT, "logs"))
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]

    urls: set[str] = set()
    for lang in langs:
        path = os.path.join(args.eval_data_dir, f"{lang}.jsonl")
        lang_urls = _read_image_urls(path)
        logger.info("%s: %d rows -> %d unique image urls", lang, len(lang_urls), len(set(lang_urls)))
        urls.update(lang_urls)

    logger.info("Total unique image urls across %s: %d", langs, len(urls))
    download_images(sorted(urls), args.image_cache_dir, args.download_workers, logger)


if __name__ == "__main__":
    main()
