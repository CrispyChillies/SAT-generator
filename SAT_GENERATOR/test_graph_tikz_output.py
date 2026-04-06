#!/usr/bin/env python3
"""Quick tester for new TikZ-based graph question assembly.

Usage examples:
1) Build final question HTML from raw question code + TikZ code:
   python test_graph_tikz_output.py \
     --question-file temp_question.html \
     --tikz-file temp_graph.tikz \
     --long-description-file temp_long_desc.html \
     --output output/test_tikz_question.html

2) Full generation from a sample question (requires OPENAI_API_KEY):
   python test_graph_tikz_output.py \
     --run-generator \
     --questions-path questions_practice_test.json \
     --sample-index 0 \
     --output output/test_generated_question.json
"""

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


# def _read_text_direct(value: str, file_path: Optional[str]) -> str:
#     if file_path:
#         return Path(file_path).read_text(encoding="utf-8")
#     return value or ""


# def _remove_svg_and_long_desc_from_html(html: str) -> str:
#     result = re.sub(r"<svg\b.*?</svg>", "", html, flags=re.DOTALL | re.IGNORECASE)
#     result = re.sub(
#         r'<div[^>]*class="sr-only"[^>]*>.*?</div>',
#         "",
#         result,
#         flags=re.DOTALL | re.IGNORECASE,
#     )
#     return result


# def _extract_tikz_body(tikz_code: str) -> str:
#     code = (tikz_code or "").strip()
#     if not code:
#         raise ValueError("TikZ code is empty")
#     match = re.search(
#         r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}",
#         code,
#         flags=re.DOTALL,
#     )
#     if match:
#         return match.group(1).strip()
#     return code


# def _render_tikz_to_data_uri(tikz_code: str, tikz_service_url: Optional[str]) -> str:
#     service_url = tikz_service_url or os.getenv(
#         "TIKZ_COMPILER_URL", "http://127.0.0.1:8000/compile-png"
#     )
#     payload = json.dumps({"code": _extract_tikz_body(tikz_code)}).encode("utf-8")
#     req = urllib_request.Request(
#         service_url,
#         data=payload,
#         headers={"Content-Type": "application/json"},
#         method="POST",
#     )
#     try:
#         with urllib_request.urlopen(req, timeout=30) as response:
#             binary = response.read()
#             content_type = response.headers.get("Content-Type", "").lower()
#     except urllib_error.HTTPError as exc:
#         detail = exc.read().decode("utf-8", errors="ignore")
#         raise ValueError(f"TikZ compiler HTTP {exc.code}: {detail[:300]}") from exc

#     mime_type = "image/svg+xml" if "image/svg+xml" in content_type else "image/png"
#     encoded = base64.b64encode(binary).decode("ascii")
#     return f"data:{mime_type};base64,{encoded}"


# def build_question_with_tikz_figure(
#     question_text_html: str,
#     tikz_code: str,
#     long_description_html: str,
#     tikz_service_url: Optional[str],
# ) -> str:
#     clean_question_text = _remove_svg_and_long_desc_from_html(question_text_html)
#     image_uri = _render_tikz_to_data_uri(tikz_code, tikz_service_url)
#     figure_block = (
#         f'<figure style="text-align: center;">'
#         f'<img src="{image_uri}" alt="Generated graph" />'
#         f'<div class="sr-only">{long_description_html}</div>'
#         f"</figure>"
#     )

#     parts = re.split(r"(<p[^>]*>.*?</p>)", clean_question_text, flags=re.DOTALL)
#     parts = [p for p in parts if p.strip()]
#     if len(parts) >= 2:
#         return f"{''.join(parts[:-1])}\n{figure_block}\n{parts[-1]}"
#     return f"{figure_block}\n{clean_question_text}"


# def _run_merge_only(args: argparse.Namespace) -> dict:
#     question_html = _read_text_direct(args.question_code, args.question_file).strip()
#     tikz_code = _read_text_direct(args.tikz_code, args.tikz_file).strip()
#     long_desc = _read_text_direct(
#         args.long_description, args.long_description_file
#     ).strip()

#     if not question_html:
#         raise ValueError(
#             "question code is required (--question-code or --question-file)"
#         )
#     if not tikz_code:
#         raise ValueError("tikz code is required (--tikz-code or --tikz-file)")

#     merged_question = build_question_with_tikz_figure(
#         question_text_html=question_html,
#         tikz_code=tikz_code,
#         long_description_html=long_desc or "<ul><li>Generated graph.</li></ul>",
#         tikz_service_url=args.tikz_service_url,
#     )

#     return {
#         "mode": "merge_only",
#         "tikz_service_url": args.tikz_service_url
#         or os.getenv("TIKZ_COMPILER_URL", "http://127.0.0.1:8000/compile-png"),
#         "merged_question": merged_question,
#     }


def _run_full_generator(args: argparse.Namespace) -> dict:
    from generate_question_langchain import generate_new_question, load_sample_question

    sample = load_sample_question(
        questions_path=args.questions_path,
        index=args.sample_index,
        question_id=args.question_id,
    )

    result = generate_new_question(
        sample,
        use_openai_basic=True,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        model=args.model,
        creative_mode=not args.conservative_mode,
    )
    return {"mode": "run_generator", "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Test TikZ graph output pipeline")
    parser.add_argument(
        "--output", type=str, default="output/test_graph_tikz_output.json"
    )

    parser.add_argument("--question-code", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--tikz-code", type=str, default="")
    parser.add_argument("--tikz-file", type=str, default="")
    parser.add_argument("--long-description", type=str, default="")
    parser.add_argument("--long-description-file", type=str, default="")
    parser.add_argument("--tikz-service-url", type=str, default="")

    parser.add_argument("--run-generator", action="store_true")
    parser.add_argument(
        "--questions-path", type=str, default="questions_practice_test.json"
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--question-id", type=str, default=None)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--conservative-mode", action="store_true")

    args = parser.parse_args()

    payload = _run_full_generator(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if payload.get("mode") == "merge_only":
        # Save rendered HTML directly for browser preview.
        html_path = output_path.with_suffix(".html")
        html_path.write_text(payload["merged_question"], encoding="utf-8")
        payload["merged_question_html_path"] = str(html_path)

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved test output: {output_path}")
    if payload.get("merged_question_html_path"):
        print(f"Saved merged HTML: {payload['merged_question_html_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
