"""
Preprocessing pipeline: from arXiv source (.zip + .pdf) to LaTeXML HTML.

Stages:
  1. Extract .zip → src/PAPER_ID/
  2. Identify main .tex file, copy/rename to main.tex
  3. Clean .tex (remove comments, useless commands)
  4. Run LaTeXML → html/PAPER_ID/PAPER_ID.html
  5. Fix bibliography & citations using .bbl
  6. Run pdffigures2 → fig/PAPER_ID.json

Usage:
  python -m nougat.dataset.preprocess_pipeline --base-dir /path/to/data_root

Expected input layout:
  data_root/
  └── src/
      ├── 2308.13418.pdf
      ├── 2308.13418.zip   (arXiv source tarball/zip)
      └── ...

Output (created by this script):
  data_root/
  ├── src/
  │   ├── 2308.13418.pdf
  │   ├── 2308.13418.zip
  │   └── 2308.13418/        ← extracted, main.tex identified
  │       ├── main.tex
  │       ├── main.bbl
  │       └── ...
  ├── html/
  │   └── 2308.13418/
  │       └── 2308.13418.html
  ├── fig/
  │   └── 2308.13418.json
  └── ...
"""

import os
import re
import tarfile
import zipfile
import shutil
import subprocess
import argparse
import logging
from pathlib import Path

from nougat.dataset.patches.preprocess_tex import preprocess_tex
from nougat.dataset.patches.fix_bibliography import fix_bibliography
from nougat.dataset.patches.fix_citations import fix_citations

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_main_tex(tex_dir: Path) -> Path | None:
    """
    Identify the main .tex file in an extracted arXiv source directory.
    Heuristic: look for \\documentclass or \\begin{document}; fall back to common names.
    """
    tex_files = list(tex_dir.glob("*.tex"))
    if not tex_files:
        return None
    if len(tex_files) == 1:
        return tex_files[0]

    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding="utf-8", errors="ignore")
            if r"\documentclass" in content and r"\begin{document}" in content:
                return tex_file
        except Exception:
            continue

    for name in ["main.tex", "paper.tex", "root.tex", "manuscript.tex", "ms.tex"]:
        candidate = tex_dir / name
        if candidate.exists():
            return candidate

    return tex_files[0]


def find_bbl_file(tex_dir: Path) -> Path | None:
    bbl_files = list(tex_dir.glob("*.bbl"))
    if not bbl_files:
        return None
    if len(bbl_files) == 1:
        return bbl_files[0]
    for bbl in bbl_files:
        if bbl.stem == "main":
            return bbl
    return bbl_files[0]


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract .zip or .tar.gz/.tgz archive."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = archive_path.name.lower()
        if name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(dest_dir)
        elif name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
            with tarfile.open(archive_path, "r:*") as t:
                t.extractall(dest_dir, filter="data")
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
        logger.info(f"Extracted {archive_path.name} → {dest_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def run_latexml(tex_file: Path, html_dir: Path, paper_id: str) -> bool:
    out_dir = html_dir / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{paper_id}.xml"
    html_path = out_dir / f"{paper_id}.html"

    cmd_latexml = [
        "latexml",
        f"--dest={xml_path}",
        "--quiet",
        str(tex_file),
    ]
    try:
        result = subprocess.run(
            cmd_latexml, capture_output=True, text=True, timeout=300,
            cwd=str(tex_file.parent),
        )
        if result.returncode != 0:
            logger.error(f"latexml failed for {paper_id}:\n{result.stderr[:500]}")
            return False
    except FileNotFoundError:
        logger.error("latexml not found on PATH. Install LaTeXML first.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"latexml timed out for {paper_id}")
        return False

    cmd_post = [
        "latexmlpost",
        f"--dest={html_path}",
        "--format=html5",
        str(xml_path),
    ]
    try:
        result = subprocess.run(cmd_post, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"latexmlpost failed for {paper_id}:\n{result.stderr[:500]}")
            return False
    except FileNotFoundError:
        logger.error("latexmlpost not found on PATH.")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"latexmlpost timed out for {paper_id}")
        return False

    logger.info(f"LaTeXML OK: {tex_file.name} → {html_path}")
    return True


def run_pdffigures2(pdf_path: Path, fig_dir: Path, paper_id: str) -> bool:
    try:
        from nougat.dataset.pdffigures import call_pdffigures
        fig_dir.mkdir(parents=True, exist_ok=True)
        call_pdffigures(str(pdf_path), str(fig_dir))
        logger.info(f"pdffigures2 OK: {paper_id}")
        return True
    except Exception as e:
        logger.warning(f"pdffigures2 failed for {paper_id}: {e}")
        return False


def preprocess_paper(paper_id: str, base_dir: Path, recompute: bool = False) -> bool:
    src_dir = base_dir / "src"
    html_dir = base_dir / "html"
    fig_dir = base_dir / "fig"

    tex_dir = src_dir / paper_id
    pdf_path = src_dir / f"{paper_id}.pdf"

    # --- Step 1: Extract archive ---
    archive = None
    for ext in (".zip", ".tar.gz", ".tgz", ".tar"):
        candidate = src_dir / f"{paper_id}{ext}"
        if candidate.exists():
            archive = candidate
            break

    if archive and (not tex_dir.exists() or recompute):
        if not extract_archive(archive, tex_dir):
            return False
    elif not tex_dir.exists():
        logger.error(f"No archive or extracted dir for {paper_id}")
        return False

    # --- Step 2: Identify main .tex, rename/copy to main.tex ---
    main_tex = tex_dir / "main.tex"
    if not main_tex.exists():
        found_tex = find_main_tex(tex_dir)
        if found_tex is None:
            logger.error(f"No .tex file found in {tex_dir}")
            return False
        if found_tex.name != "main.tex":
            logger.info(f"Copying {found_tex.name} → main.tex")
            shutil.copy2(found_tex, main_tex)

    # --- Step 3: Preprocess .tex ---
    logger.info(f"[{paper_id}] Preprocessing main.tex")
    preprocess_tex(str(main_tex))

    # --- Step 4: Run LaTeXML ---
    html_file = html_dir / paper_id / f"{paper_id}.html"
    if not html_file.exists() or recompute:
        if not run_latexml(main_tex, html_dir, paper_id):
            return False
    else:
        logger.info(f"[{paper_id}] HTML already exists, skipping LaTeXML")

    # --- Step 5: Fix bibliography & citations ---
    bbl_file = find_bbl_file(tex_dir)
    if bbl_file and html_file.exists():
        try:
            fix_bibliography(str(html_file), str(bbl_file))
        except Exception as e:
            logger.warning(f"[{paper_id}] fix_bibliography: {e}")
        try:
            fix_citations(str(html_file), str(bbl_file))
        except Exception as e:
            logger.warning(f"[{paper_id}] fix_citations: {e}")
    else:
        logger.info(f"[{paper_id}] No .bbl file found, skipping bibliography/citation fix")

    # --- Step 6: Run pdffigures2 ---
    fig_json = fig_dir / f"{paper_id}.json"
    if pdf_path.exists() and (not fig_json.exists() or recompute):
        run_pdffigures2(pdf_path, fig_dir, paper_id)
    elif not pdf_path.exists():
        logger.warning(f"[{paper_id}] PDF not found at {pdf_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Nougat preprocessing pipeline: arXiv source → LaTeXML HTML"
    )
    parser.add_argument(
        "--base-dir", type=Path, required=True,
        help="Base data directory containing src/ with .zip and .pdf files",
    )
    parser.add_argument(
        "--papers", nargs="*",
        help="Specific paper IDs to process. Default: auto-detect from src/",
    )
    parser.add_argument("--recompute", action="store_true", help="Recompute all steps")
    args = parser.parse_args()

    src_dir = args.base_dir / "src"
    if not src_dir.exists():
        logger.error(f"src/ directory not found in {args.base_dir}")
        return

    if args.papers:
        paper_ids = args.papers
    else:
        archive_stems = set()
        for ext in ("*.zip", "*.tar.gz", "*.tgz", "*.tar"):
            for f in src_dir.glob(ext):
                stem = f.name.split(".")[0] if ".tar" in f.name else f.stem
                archive_stems.add(stem)
        pdf_stems = {f.stem for f in src_dir.glob("*.pdf")}
        paper_ids = sorted(archive_stems | pdf_stems)

    logger.info(f"Processing {len(paper_ids)} papers from {args.base_dir}")

    success, fail = 0, 0
    for paper_id in paper_ids:
        logger.info(f"{'='*60}")
        logger.info(f"Processing {paper_id}")
        if preprocess_paper(paper_id, args.base_dir, recompute=args.recompute):
            success += 1
        else:
            fail += 1

    logger.info(f"{'='*60}")
    logger.info(f"Done: {success} succeeded, {fail} failed, {len(paper_ids)} total")


if __name__ == "__main__":
    main()
