import subprocess
import os
import uuid
import re
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="TikZ to SVG API")


class TikzRequest(BaseModel):
    code: str


def cleanup_files(job_id: str):
    # Added .svg to the cleanup list
    for ext in [".tex", ".aux", ".log", ".pdf", ".svg", ".png"]:
        file_to_remove = f"{job_id}{ext}"
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)


def _normalize_tikz_block(code: str) -> str:
    """Accept full tikzpicture or bare body and normalize to a single tikzpicture block."""
    raw = (code or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="TikZ code is empty.")

    if "\\begin{tikzpicture}" in raw and "\\end{tikzpicture}" in raw:
        match = re.search(
            r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
            raw,
            flags=re.DOTALL,
        )
        if match:
            return match.group(0)

    return f"\\begin{{tikzpicture}}\n{raw}\n\\end{{tikzpicture}}"


def _build_latex_content(tikz_code: str) -> str:
    tikz_block = _normalize_tikz_block(tikz_code)
    return f"""\\documentclass[tikz, border=2mm]{{standalone}}
\\usepackage{{amsmath}}
\\begin{{document}}
{tikz_block}
\\end{{document}}"""


@app.post("/compile-svg")
async def compile_to_svg(req: TikzRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    tex_file = f"{job_id}.tex"
    pdf_file = f"{job_id}.pdf"
    svg_file = f"{job_id}.svg"

    # Schedule cleanup to run after the HTTP response is sent
    background_tasks.add_task(cleanup_files, job_id)

    # 1. Create the TeX file
    latex_content = _build_latex_content(req.code)

    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex_content)

    # 2. Compile to PDF first
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                tex_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="LaTeX compilation timed out.")

    if not os.path.exists(pdf_file):
        raise HTTPException(status_code=400, detail="Compile Error. Check TikZ syntax.")

    # 3. CONVERT PDF TO SVG
    # pdftocairo -svg <input.pdf> <output.svg>
    subprocess.run(
        ["pdftocairo", "-svg", pdf_file, svg_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not os.path.exists(svg_file):
        raise HTTPException(status_code=500, detail="SVG Conversion Error")

    # Return the SVG file with the correct media type
    return FileResponse(svg_file, media_type="image/svg+xml", filename="chart.svg")


@app.post("/compile-png")
async def compile_to_png(req: TikzRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    tex_file = f"{job_id}.tex"
    pdf_file = f"{job_id}.pdf"
    png_file = f"{job_id}.png"

    background_tasks.add_task(cleanup_files, job_id)

    latex_content = _build_latex_content(req.code)
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(latex_content)

    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-no-shell-escape",
                tex_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="LaTeX compilation timed out.")

    if not os.path.exists(pdf_file):
        raise HTTPException(status_code=400, detail="Compile Error. Check TikZ syntax.")

    # pdftocairo -png -singlefile <input.pdf> <output-prefix>
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", pdf_file, job_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if not os.path.exists(png_file):
        raise HTTPException(status_code=500, detail="PNG Conversion Error")

    return FileResponse(png_file, media_type="image/png", filename="chart.png")
