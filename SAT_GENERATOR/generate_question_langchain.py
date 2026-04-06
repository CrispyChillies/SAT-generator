#!/usr/bin/env python3
"""
Sinh câu hỏi mới từ câu hỏi mẫu bằng LangChain.
- Giữ nguyên category, section, type, difficulty của câu gốc.
- Sinh câu hỏi mới, explanation và đáp án (correct_answer); cùng format HTML + MathML, chỉ đổi số.
- Câu hỏi mới phải đúng format so với câu gốc (HTML + MathML).
"""

import os
import json
import re
import uuid
import base64
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError, field_validator
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền llm.")

# ---------------------------------------------------------------------------
# Import parser để lấy thông tin Graph
# ---------------------------------------------------------------------------

from mathml_parser import MathMLParser

# ---------------------------------------------------------------------------
# Cấu trúc output từ LLM: câu hỏi + explanation + đáp án
# ---------------------------------------------------------------------------


class GeneratedQuestionContent(BaseModel):
    """Câu hỏi mới, explanation và đáp án đúng; cùng format HTML+MathML, chỉ đổi số so với mẫu (dùng khi không phải multiple-choice hoặc không có 4 choices)."""

    question: str = Field(
        description="New question content in the same HTML and MathML format as the sample, with only numerical values changed"
    )
    explanation: str = Field(
        description="New explanation in the same HTML and MathML format as the sample, with only numerical values changed to match the new question"
    )
    correct_answer: str = Field(
        description="The correct answer for the new question, in the same format as the sample (e.g. HTML/MathML string of the right choice or value)"
    )


class GeneratedMultipleChoiceContent(BaseModel):
    """Câu hỏi multiple-choice: câu hỏi + explanation + đúng 4 lựa chọn (A,B,C,D) + chữ cái đáp án đúng."""

    question: str = Field(
        description="New question content, same HTML+MathML format with only numerical values changed"
    )
    explanation: str = Field(
        description="New explanation, same format with only numbers changed to match the new question"
    )
    choices: List[str] = Field(
        description="Exactly 4 answer choices in order A, B, C, D; each is HTML+MathML string with only numbers changed"
    )
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the correct answer (A, B, C, or D)"
    )

    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]


class GeneratedGraphQuestionContent(BaseModel):
    """Output cho câu hỏi có đồ thị: LLM chỉ sinh text mới + số liệu đồ thị mới (không sinh SVG)."""

    question_text: str = Field(
        description="New question text (without SVG), same format with only numbers changed"
    )
    explanation: str = Field(
        description="New explanation, same format with only numbers changed"
    )
    choices: List[str] = Field(
        description="Exactly 4 answer choices in order A, B, C, D; each with only numbers changed"
    )
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the correct answer"
    )
    tikz_code: str = Field(
        description="TikZ code used to render the new graph. Return tikzpicture body or full tikzpicture block."
    )
    new_long_description: str = Field(
        description="New long description for the graph in HTML format (<ul><li>...</li></ul>), matching the new x/y values. MUST preserve the same HTML structure as the original."
    )

    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]


class GeneratedGraphQuestionTextContent(BaseModel):
    """Text-only output cho câu hỏi multiple-choice có đồ thị."""

    question_text: str = Field(
        description="New question text (without SVG), same format with only numbers changed"
    )
    explanation: str = Field(
        description="New explanation, same format with only numbers changed"
    )
    choices: List[str] = Field(
        description="Exactly 4 answer choices in order A, B, C, D; each with only numbers changed"
    )
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(
        description="The letter of the correct answer"
    )
    new_long_description: str = Field(
        description="New long description for the graph in HTML format (<ul><li>...</li></ul>), matching the new x/y values. MUST preserve the same HTML structure as the original."
    )

    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]


class GeneratedTikzDiagramContent(BaseModel):
    """TikZ-only output produced from finalized graph text + description."""

    tikz_code: str = Field(
        description="TikZ code used to render the new graph. Must be a complete tikzpicture block."
    )


class GeneratedGraphFreeResponseContent(BaseModel):
    """Output cho câu hỏi tự luận có đồ thị: LLM sinh text mới + số liệu đồ thị mới + correct_answer (không sinh SVG)."""

    question_text: str = Field(
        description="New question text (without SVG), same format with only numbers changed"
    )
    explanation: str = Field(
        description="New explanation, same format with only numbers changed"
    )
    correct_answer: str = Field(
        description="The correct answer for the new question, in the same format as the sample (e.g. HTML/MathML string of the right value)"
    )
    tikz_code: str = Field(
        description="TikZ code used to render the new graph. Return tikzpicture body or full tikzpicture block."
    )
    new_long_description: str = Field(
        description="New long description for the graph in HTML format (<ul><li>...</li></ul>), matching the new x/y values. MUST preserve the same HTML structure as the original."
    )


class GeneratedGraphFreeResponseTextContent(BaseModel):
    """Text-only output cho câu hỏi tự luận có đồ thị."""

    question_text: str = Field(
        description="New question text (without SVG), same format with only numbers changed"
    )
    explanation: str = Field(
        description="New explanation, same format with only numbers changed"
    )
    correct_answer: str = Field(
        description="The correct answer for the new question, in the same format as the sample (e.g. HTML/MathML string of the right value)"
    )
    new_long_description: str = Field(
        description="New long description for the graph in HTML format (<ul><li>...</li></ul>), matching the new figure values. MUST preserve the same HTML structure as the original."
    )


class SourceMetadata(BaseModel):
    """Fields copied directly from the input sample (no inference)."""

    section: str = ""
    category: str = ""
    difficulty: str = ""
    type: str = ""
    skill: str = ""
    question_text: str = ""
    choices: List[str] = Field(default_factory=list)
    correct_answer: Any = None
    explanation: str = ""


class DerivedAnalysis(BaseModel):
    """Fields inferred from mathematical analysis."""

    skill: str = ""
    sub_skill: str = ""
    problem_family: str = ""
    solve_strategy: List[str] = Field(default_factory=list)
    reasoning_pattern: str = ""
    logic_invariants: List[str] = Field(default_factory=list)
    changeable_parameters: List[str] = Field(default_factory=list)
    parameter_constraints: List[str] = Field(default_factory=list)
    answer_format: str = ""
    rendering_notes: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class ProblemAnalysis(BaseModel):
    """Analysis artifact with explicit source vs derived separation."""

    source_fields: SourceMetadata
    derived_fields: DerivedAnalysis
    missing_source_fields: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class BlueprintParameter(BaseModel):
    """A parameter in the math blueprint that can be varied safely."""

    name: str
    role: str = ""
    value_type: str = "number"
    current_value: Any = None
    suggested_range: str = ""
    constraints: List[str] = Field(default_factory=list)


class ProblemBlueprint(BaseModel):
    """Blueprint combining copied source metadata with inferred analysis."""

    source_metadata: SourceMetadata
    derived_analysis: DerivedAnalysis
    known_values: Dict[str, Any] = Field(default_factory=dict)
    derived_values: Dict[str, Any] = Field(default_factory=dict)
    changeable_parameters: List[BlueprintParameter] = Field(default_factory=list)
    generation_guidance: List[str] = Field(default_factory=list)
    fallback_generation_hint: str = ""


class StructuredGeneratedInstance(BaseModel):
    """Generated instance produced from blueprint."""

    question: str
    explanation: str
    choices: Optional[List[str]] = None
    correct_answer_letter: Optional[Literal["A", "B", "C", "D"]] = None
    correct_answer: Optional[str] = None
    parameter_values: Dict[str, Any] = Field(default_factory=dict)
    derivation_summary: str = ""


class GenerationVerificationReport(BaseModel):
    """Verification output for generated instance quality and consistency."""

    is_solvable: bool
    answer_format_valid: bool
    reasoning_alignment: str = ""
    multiple_choice_unique_correct: Optional[bool] = None
    distractor_quality: Optional[str] = None
    verification_notes: List[str] = Field(default_factory=list)
    corrected_correct_answer_letter: Optional[Literal["A", "B", "C", "D"]] = None
    corrected_correct_answer: Optional[str] = None
    confidence: float = 0.0


def _build_input_snapshot(
    sample: Dict[str, Any],
    *,
    original_question_html: str,
    original_explanation: str,
    original_correct_answer: str,
    original_choices: List[str],
    correct_letter: Optional[str],
    has_graph: bool,
) -> Dict[str, Any]:
    q_block = sample.get("question") or {}
    raw_correct_answer = q_block.get("correct_answer")
    if raw_correct_answer is None:
        raw_correct_answer = sample.get("correct_answer")

    source_metadata = {
        "section": sample.get("section", ""),
        "category": sample.get("category", ""),
        "difficulty": sample.get("difficulty", ""),
        "type": sample.get("type", ""),
        "skill": sample.get("skill", ""),
        "question_text": original_question_html,
        "choices": original_choices,
        "correct_answer": raw_correct_answer,
        "explanation": original_explanation,
    }

    return {
        "id": sample.get("id"),
        "source_metadata": source_metadata,
        "has_graph": has_graph,
        "source_fields_present": [
            k for k, v in source_metadata.items() if v not in (None, "", [])
        ],
        "correct_answer_letter": correct_letter,
        "correct_answer_content": original_correct_answer,
    }


def _build_problem_analysis_prompt(
    input_snapshot: Dict[str, Any],
    step_b_trace: Optional[Dict[str, Any]] = None,
) -> str:
    trace_preview = ""
    if step_b_trace:
        trace_preview = json.dumps(
            {
                "final_result": step_b_trace.get("final_result"),
                "is_correct": step_b_trace.get("is_correct"),
                "first_steps": (step_b_trace.get("steps") or [])[:2],
            },
            ensure_ascii=False,
            indent=2,
        )

    return f"""You are an SAT Math analyst.

Analyze the original problem deeply and return a structured JSON analysis.

Requirements:
- Copy source_fields directly from input_snapshot.source_metadata. Do not rewrite or invent these fields.
- Infer ONLY derived_fields from the math logic.
- Infer missing skill/sub_skill/problem_family/solve_strategy/reasoning_pattern if needed.
- Keep logic_invariants and parameter constraints explicit.
- Keep answer_format explicit (fraction, integer, option letter, coordinate, etc.).
- Identify generation_knobs that can produce a genuinely new question while preserving the same skill family.
- Distinguish safe_variations from unsafe_variations.
- Include novelty_opportunities and validation_rules for downstream generation.


Input snapshot:
{json.dumps(input_snapshot, ensure_ascii=False, indent=2)}

Optional solver trace from Step B (may help infer strategy):
{trace_preview or "(none)"}

Return ONLY a JSON object matching the target schema.
"""


def _build_problem_blueprint_prompt(
    input_snapshot: Dict[str, Any],
    analysis: ProblemAnalysis,
) -> str:
    return f"""You are building a SAT Math problem blueprint from analysis.

Build a reusable blueprint for generation.

Blueprint requirements:
- Preserve fixed context fields from the sample: section, category, difficulty, type, and skill if present.
- Treat original question_text, choices, explanation, and correct_answer as reference examples, not immutable text.
- Build a reusable generation blueprint from the mathematical structure, not from the literal wording.
- Preserve reasoning family and core invariants, not surface phrasing.
- Capture safe generation knobs, forbidden variations, and validation rules.
- The blueprint must support producing a new instance that is mathematically valid and materially different from the source.

Input snapshot:
{json.dumps(input_snapshot, ensure_ascii=False, indent=2)}

Analysis:
{json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2)}

Rules:
- Include must_change fields for generation.
- Include allowed_variations and forbidden_variations.
- Include instance_acceptance_rules for novelty and mathematical validity.
- Prefer structural generation over number substitution.
- Add a fallback_generation_hint for robust generation.


Return ONLY a JSON object matching the target schema.
"""


def _build_structured_instance_prompt(
    input_snapshot: Dict[str, Any],
    analysis: ProblemAnalysis,
    blueprint: ProblemBlueprint,
    *,
    creative_mode: bool,
) -> str:
    generation_mode = (
        "new scenario with same solving logic"
        if creative_mode
        else "conservative parameter variation"
    )
    return f"""You are an SAT Math generator.

Generate ONE new problem instance from the provided blueprint.

Generation mode: {generation_mode}

Hard requirements:
- Generate a mathematically valid NEW instance, not a paraphrase of the source.
- Preserve the problem family and reasoning invariants, but regenerate the question text, explanation, choices, and answer from the blueprint.
- Change at least one meaningful generation knob from the blueprint.
- Re-derive the invariant for the new instance before deciding the answer.
- Determine the correct answer first, then construct choices around that answer.
- If the chosen variation changes the structure (for example, changing the number of consecutive terms), recompute all divisibility and representability conditions from scratch.
- Do not copy the original wording unless a field is explicitly marked fixed metadata.
- Reject trivial restatements of the original problem.

Novelty rules:
- The generated question must differ semantically from the original, not just lexically.
- At least one of these must change: target quantity, answer value, number of terms, task form, or choice set.
- If the final numeric answer remains unchanged, then the task form and choice design must change substantially.

Math consistency rules:
- Produce an internal witness or derivation that proves the correct answer is attainable.
- Ensure all distractors are incorrect under the same derived invariant.
- Ensure explanation matches the generated instance, not the source.


Original input snapshot:
{json.dumps(input_snapshot, ensure_ascii=False, indent=2)}

Analysis:
{json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2)}

Blueprint:
{json.dumps(blueprint.model_dump(), ensure_ascii=False, indent=2)}

Return ONLY a JSON object matching the target schema.
"""


def _build_structured_verification_prompt(
    input_snapshot: Dict[str, Any],
    analysis: ProblemAnalysis,
    blueprint: ProblemBlueprint,
    instance: StructuredGeneratedInstance,
) -> str:
    return f"""You are a SAT Math verifier.

Validate the generated instance against the original skill and blueprint.

Check:
- solvability
- answer format validity
- alignment with intended solving strategy
- if multiple-choice: unique correct answer and distractor quality
- consistency with source_metadata and derived_analysis

Important interpretation rules:
- source_metadata fields like section/category/difficulty/type/skill are fixed context and must remain consistent.
- Numerical values in the generated problem are allowed to change when they are valid changeable parameters under derived_analysis/blueprint constraints.
- Do NOT mark a problem unsolvable merely because coefficients/constants differ from the original sample.
- If choices are present in the generated instance JSON, treat the question as multiple-choice and evaluate choice consistency accordingly.

If needed, provide corrected correct answer fields.

Original input snapshot:
{json.dumps(input_snapshot, ensure_ascii=False, indent=2)}

Analysis:
{json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2)}

Blueprint:
{json.dumps(blueprint.model_dump(), ensure_ascii=False, indent=2)}

Generated instance:
{json.dumps(instance.model_dump(), ensure_ascii=False, indent=2)}

Return ONLY a JSON object matching the target schema.
"""


# ---------------------------------------------------------------------------
# Utility functions cho xử lý đồ thị
# ---------------------------------------------------------------------------
def _remove_svg_and_long_desc_from_html(html: str) -> str:
    """Loại bỏ toàn bộ SVG element và long description (sr-only div) khỏi HTML."""
    # Remove SVG
    result = re.sub(r"<svg\b.*?</svg>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove sr-only div containing long description
    result = re.sub(
        r'<div[^>]*class="sr-only"[^>]*>.*?</div>',
        "",
        result,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove visible graph long-description block from previous generations.
    result = re.sub(
        r'<div[^>]*class="graph-long-description"[^>]*>.*?</div>',
        "",
        result,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return result


import re


def _extract_tikz_body(tikz_code: str) -> str:
    """
    Ensures the returned string is a complete \\begin{tikzpicture} ... \\end{tikzpicture} block.
    Fixes cases where the LLM only returns the inner 'axis' content or includes markdown wrappers.
    """
    code = (tikz_code or "").strip()
    if not code:
        raise ValueError("TikZ code is empty")

    # Remove markdown code blocks if the LLm ignored the prompt instructions
    code = re.sub(r"```[a-z]*", "", code).replace("```", "").strip()

    # Case 1: Check if it's already a full tikzpicture block
    # We use group(0) to keep the \begin and \end tags plus any [options]
    match = re.search(
        r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})", code, flags=re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # Case 2: If the LLM only returned \begin{axis} or raw draw commands, wrap it
    # This prevents the "500 Internal Server Error" caused by missing environments
    return f"\\begin{{tikzpicture}}\n{code}\n\\end{{tikzpicture}}"


def _format_long_description_for_display(long_description_html: str) -> str:
    """Format long description HTML so bullets and text align cleanly to the left."""
    content = (long_description_html or "").strip()
    if not content:
        content = "<ul><li>Generated graph description is unavailable.</li></ul>"

    # If model wraps content in a div, keep only inner HTML for the display container.
    content = re.sub(r"^\s*<div[^>]*>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"</div>\s*$", "", content, flags=re.IGNORECASE)

    # Ensure list containers and list items render with compact, left-aligned spacing.
    content = re.sub(
        r"<ul(\s[^>]*)?>",
        '<ul style="margin:0;padding-left:1.1rem;list-style-position:outside;text-align:left;">',
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(
        r"<li(\s[^>]*)?>",
        '<li style="margin:0.2rem 0;padding-left:0.1rem;text-align:left;">',
        content,
        flags=re.IGNORECASE,
    )

    return content


def _render_tikz_to_data_uri(
    tikz_code: str, tikz_service_url: Optional[str] = None
) -> str:
    """Preprocesses TikZ code with necessary libraries and renders it via compiler service."""
    service_url = tikz_service_url or os.getenv(
        "TIKZ_COMPILER_URL", "http://localhost:8000/compile-png"
    )

    # Step 1: Extract and validate the TikZ structure
    clean_tikz_block = _extract_tikz_body(tikz_code)

    # Step 2: Construct the final LaTeX snippet for the compiler
    # If your server requires a full document, wrap this in \documentclass{standalone} \begin{document}
    final_payload_code = clean_tikz_block

    # Step 4: Prepare and Send Request
    payload = json.dumps({"code": final_payload_code}).encode("utf-8")

    # Debugging: Log the exact string sent to the compiler
    with open("debug_tikz_payload.json", "w", encoding="utf-8") as f:
        json.dump(
            {"sent_to_compiler": final_payload_code, "original_from_llm": tikz_code},
            f,
            indent=2,
        )

    req = urllib_request.Request(
        service_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=30) as response:
            binary = response.read()
            content_type = response.headers.get("Content-Type", "").lower()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ValueError(f"TikZ compiler HTTP {exc.code}: {detail[:300]}") from exc
    except Exception as exc:
        raise ValueError(f"TikZ compiler request failed: {exc}") from exc

    if not binary:
        raise ValueError("TikZ compiler returned empty image payload")

    mime_type = "image/svg+xml" if "image/svg+xml" in content_type else "image/png"
    encoded = base64.b64encode(binary).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def build_question_with_tikz_figure(
    question_text_html: str,
    tikz_code: str,
    long_description_html: str,
    tikz_service_url: Optional[str] = None,
) -> str:
    """Inject rendered TikZ figure into question HTML between intro and final question line."""
    clean_question_text = _remove_svg_and_long_desc_from_html(question_text_html)
    image_uri = _render_tikz_to_data_uri(tikz_code, tikz_service_url=tikz_service_url)
    visible_long_desc = _format_long_description_for_display(long_description_html)

    figure_block = (
        f'<figure style="text-align:center;">'
        f'<img src="{image_uri}" alt="Generated graph" style="max-width:100%;max-height:420px;height:auto;object-fit:contain;"/>'
        f'<div class="graph-long-description" style="max-width:640px;margin:12px auto 0;text-align:left;line-height:1.45;">{visible_long_desc}</div>'
        f'<div class="sr-only" style="position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0;">{long_description_html}</div>'
        f"</figure>"
    )

    parts = re.split(r"(<p[^>]*>.*?</p>)", clean_question_text, flags=re.DOTALL)
    parts = [p for p in parts if p.strip()]

    if len(parts) >= 2:
        intro_text = "".join(parts[:-1])
        question_part = parts[-1]
        return f"{intro_text}\n{figure_block}\n{question_part}"

    return f"{figure_block}\n{clean_question_text}"


def _update_explanation_for_corrected_answer(
    original_explanation: str,
    old_letter: str,
    new_letter: str,
    choices: List[str],
) -> str:
    """
    Update explanation when the correct answer letter changes.
    Replace references to the old letter with the new letter.

    Args:
        original_explanation: The explanation from LLM
        old_letter: The old (incorrect) answer letter
        new_letter: The new (correct) answer letter
        choices: List of 4 choices

    Returns:
        Updated explanation
    """
    explanation = original_explanation

    # Replace "Choice X is correct" -> "Choice Y is correct"
    explanation = re.sub(
        rf"\bChoice {old_letter} is correct\b",
        f"Choice {new_letter} is correct",
        explanation,
        flags=re.IGNORECASE,
    )

    # Replace "Choice X is the best answer" -> "Choice Y is the best answer"
    explanation = re.sub(
        rf"\bChoice {old_letter} is the best answer\b",
        f"Choice {new_letter} is the best answer",
        explanation,
        flags=re.IGNORECASE,
    )

    # Replace other letters as incorrect
    for letter in ["A", "B", "C", "D"]:
        if letter == new_letter:
            continue
        explanation = re.sub(
            rf"\bChoice {letter} is correct\b",
            f"Choice {letter} is incorrect",
            explanation,
            flags=re.IGNORECASE,
        )

    return explanation


def _build_prompt_graph_multiple_choice_freeform(
    question_text_no_svg: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Free-form prompt for reasoning LLMs - text-only graph multiple-choice questions."""
    choices_text = "\n".join(
        [f"{chr(65+i)}. {choice}" for i, choice in enumerate(original_choices)]
    )
    long_desc_html = graph_spec.get("long_description_html", "")
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)

    # Add reasoning guidance for Medium/Hard questions
    reasoning_guidance = ""
    if difficulty.lower() in ["medium", "hard"]:
        reasoning_guidance = """\n\nCRITICAL FOR MEDIUM/HARD QUESTIONS:
- REASON about what the graph must communicate before generating TikZ
- Ensure values implied by your explanation and choices are reasonable
- For trend questions: ensure the trend is clear and interpretable
- For min/max questions: make sure the extreme values are obviously distinguishable
- Check that choices are plausible but have one clear correct answer
- Avoid values that are too close together or too similar
- Make sure the graph tells a coherent story with the data"""

    return f"""You are an SAT question writer. This is a multiple-choice question with a GRAPH.

Task: Generate the finalized graph-aware text content first. Do NOT generate TikZ yet.

IMPORTANT:
- Update question text, explanation, and choices to match the graph scenario you create
- Keep HTML and MathML structure EXACTLY as in sample
- CRITICAL: Calculate which answer is correct based on YOUR generated graph
- Do NOT just copy the sample's correct letter ({correct_letter}) - it may be different for your data
- Preserve the HTML structure of the long description (<ul><li>...</li></ul>)
- Make the long description detailed enough that a separate TikZ generator can reconstruct the graph faithfully{reasoning_guidance}

Original Graph Specification:
{graph_spec_json}

Original Long Description HTML Structure (preserve this format):
{long_desc_html}

Sample question text (without SVG):
{question_text_no_svg}

Sample explanation:
{original_explanation}

Sample choices (correct: {correct_letter}):
{choices_text}

Category: {category}, Section: {section}, Difficulty: {difficulty}

Generate your response in this format:

QUESTION_TEXT:
[New question text without SVG, only numbers changed]

EXPLANATION:
[New explanation matching new graph data]

CHOICE_A:
[First answer choice]

CHOICE_B:
[Second answer choice]

CHOICE_C:
[Third answer choice]

CHOICE_D:
[Fourth answer choice]

CORRECT_ANSWER:
[Letter A, B, C, or D - CALCULATE based on your generated graph]

NEW_LONG_DESCRIPTION:
[HTML description using <ul><li>...</li></ul> format with new values]
"""


def _build_prompt_graph_multiple_choice(
    question_text_no_svg: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Prompt text-only cho câu hỏi multiple-choice có đồ thị."""
    # Do not include "Choice A:/B:/..." prefixes — UI will display A/B/C/D
    choices_text = "\n".join(original_choices)

    # Extract long_description_html for the prompt
    long_desc_html = graph_spec.get("long_description_html", "")

    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)

    return f"""You are an SAT question writer. This is a MULTIPLE-CHOICE question with a GRAPH/CHART.

Task: Generate the graph-aware textual content first. Do NOT generate TikZ yet.

IMPORTANT:
- Update the question text, explanation, choices, and long description so they all describe the SAME new graph scenario.
- Keep the same structure and wording, only change the numbers and the graph code.
- The long description must contain enough detail for a separate TikZ generator to reconstruct the intended graph faithfully.
- Make the data visually interpretable: no ambiguous ties unless the prompt clearly intends them.
- Preserve the exact HTML structure of the original long description.

CRITICAL - CORRECT ANSWER CALCULATION (READ CAREFULLY):
- DO NOT COPY the sample's correct_answer_letter ({correct_letter}). The sample letter is {correct_letter}, but your answer WILL BE DIFFERENT if your generated graph changes which choice is correct.
- YOU MUST follow these steps IN ORDER:
    1. Generate a coherent new graph scenario as text and long description.
    2. Read the question carefully to understand what it's asking (e.g., "which year has the smallest value", "which period has the greatest increase", etc.).
    3. Using your generated graph scenario, calculate which answer is correct.
    4. Set correct_answer_letter to the letter (A, B, C, or D) that corresponds to the correct answer.

Original GraphSpec:
{graph_spec_json}

Original Long Description HTML Structure (YOU MUST PRESERVE THIS EXACT HTML FORMAT for new_long_description):
---
{long_desc_html}
---

Sample question TEXT (without SVG and without long description):
---
{question_text_no_svg}
---

Sample explanation:
---
{original_explanation}
---

Sample 4 choices (correct answer in the sample is {correct_letter}, but you may need to change it based on new data):
---
{choices_text}
---

Category: {category}. Section: {section}. Difficulty: {difficulty}.

Return a JSON object with exactly these keys:
- "question_text": new question text (without long description, only numbers changed in the intro and question sentences).
- "explanation": new explanation (numbers changed to match new graph and correct answer).
- "choices": list of 4 strings (A, B, C, D order, numbers changed).
- "correct_answer_letter": The letter (A, B, C, or D) of the correct answer BASED ON YOUR GENERATED GRAPH SCENARIO. Calculate this carefully.
- "new_long_description": new graph description in HTML format, MUST use the exact same <ul><li>...</li></ul> structure as the original, only changing the numbers.

You must output ONLY valid JSON.
Do NOT include any markdown code blocks (like ```json).
Do NOT include any explanation outside the JSON.
Return ONLY the JSON object.
"""


def _build_prompt_tikz_from_graph_context(
    *,
    question_text: str,
    explanation: str,
    choices: List[str],
    correct_letter: str,
    new_long_description: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Prompt chuyên biệt để sinh TikZ từ nội dung câu hỏi đã hoàn chỉnh."""
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)
    choices_text = "\n".join(
        [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
    )

    return f"""You are a specialist TikZ figure generator for SAT math questions.

Task: Generate ONLY the TikZ diagram for the finalized math figure below.

You are NOT writing the question. The question, explanation, answer choices, and long description are already finalized.
Your job is to analyze that content, determine what kind of figure is actually needed, and produce a professional TikZ rendering that matches it.

PRIMARY GOAL:
- Build a clean, readable figure that faithfully represents the finalized scenario.
- Prioritize diagram quality, spacing, and label placement so elements do not overlap.
- Use the long description as the authoritative source of the figure structure, labels, and numeric relationships.
- Use the question/explanation/choices to understand what visual relationship must be clear.

FIRST DECIDE THE FIGURE TYPE:
Before drawing, infer which of these best matches the problem:
1. Coordinate/data graph: line graph, bar chart, scatter-style plot, or data chart where axes are necessary.
2. Geometry figure: triangle, quadrilateral, circle, angle, parallel lines with transversals, similar figures, or other Euclidean diagram.
3. Algebra/other schematic: number line, function sketch without full chart scaffolding, labeled segments, or another simple mathematical schematic.

CRITICAL AXIS RULE:
- DO NOT draw x-axis and y-axis by default.
- Draw axes ONLY if the problem is truly a coordinate/data graph and the axes are necessary to interpret the figure.
- For geometry diagrams, angle diagrams, triangles, parallel lines, transversals, labeled segments, and not-to-scale illustrations, do NOT add coordinate axes.
- If the sample/problem is a pure geometry figure like a right triangle or two parallel lines with an angle, draw only the needed geometric objects and labels.

FIGURE-SPECIFIC DRAWING RULES:
- If it is a coordinate/data graph:
  - Use manual axes with \\draw[->].
    - Normalize x coordinates to integers in 0..10 only.
    - IMPORTANT: Do NOT use raw data values directly as y coordinates.
    - Use approximate visual scaling only: map plotted y coordinates to a compact display band (recommended y in 2..8).
    - Preserve order/trend and relative differences qualitatively, not exact numeric ratio.
    - Show actual numeric values as labels near points/bars when needed for fidelity.
  - Keep labels compact and readable.
    - ANTI-ANSWER-LEAKAGE STYLING (MANDATORY):
        - Do NOT highlight any data point/bar/segment that could reveal the answer.
        - Use one uniform neutral style for all values in the same series (same color family, opacity, and line width).
        - Never use a special color for max/min values, turning points, or the value tied to the correct option.
        - Avoid attention cues on one value: no glow, bold-only emphasis, thicker stroke, unique marker, or unique fill on a single candidate.
        - If multiple series are required, use balanced styling and a legend; color must encode series identity only, not correctness.
- If it is a geometry figure:
  - Choose coordinates freely to make the figure clean and well proportioned.
  - Do not force x/y axes or graph framing.
  - Preserve key relationships visually: parallel, perpendicular, equal length, acute/obtuse angle, intersection, right angle, etc.
  - Add right-angle markers, angle arcs, tick marks, or segment labels only when they help express the intended math.
  - If the problem is marked not to scale, the figure may be visually clean without exact metric proportionality, but it must still communicate the intended relationships.
- If it is two parallel lines / angle-chasing:
  - Draw two clearly parallel lines and the relevant transversal(s).
  - Place angle labels away from intersections so they do not collide with lines.
  - Make the relevant corresponding, alternate interior, vertical, or supplementary relationships visually obvious.

TIKZ CODE REQUIREMENTS:
- Return ONLY a single \\begin{{tikzpicture}} ... \\end{{tikzpicture}} block.
- DO NOT output JSON, markdown, commentary, or any text outside the tikzpicture block.
- DO NOT use pgfplots commands or environments: \\begin{{axis}}, \\addplot, xtick, xticklabels, xmin/xmax, ymin/ymax, grid=both.
- Start with exactly: \\begin{{tikzpicture}}[line cap=round,line join=round]

LAYOUT + QUALITY RULES:
1. Coordinate bounds must satisfy abs(x) <= 12 and abs(y) <= 30.
2. Scale and position the figure to fill the canvas well without crowding the edges.
3. Use only the shapes needed for the math. Do not add redundant axes, grids, or decorations.
4. Keep labels outside strokes whenever possible.
5. If a label would overlap a segment, point, or angle marker, move it with a small offset.
6. For triangles and polygons, keep vertices separated enough that side labels and angle markers fit cleanly.
7. For angle diagrams, keep intersections uncluttered and use small arcs or labels that are easy to read.
8. For charts, ensure tick labels and axis titles do not collide with data points or each other.
9. Avoid dense loops, unnecessary shading, or complex ornamentation that could hurt compilation or readability.
10. Self-check before final output: if the chosen figure type is wrong, if axes are unnecessary, if labels overlap, or if any coordinate breaks limits, regenerate the TikZ.
11. For data graphs, keep the overall chart visually compact: target plotted height around 5-7 TikZ units (never full-range raw-value height).
12. For bar charts, bars should fit comfortably inside the frame with consistent top margin; do not let tallest bar touch the top border.
13. Final fairness check: ensure styling alone cannot be used to guess the correct answer.

FINALIZED QUESTION TEXT:
---
{question_text}
---

FINALIZED EXPLANATION:
---
{explanation}
---

FINALIZED CHOICES:
---
{choices_text}
---

CORRECT ANSWER LETTER:
{correct_letter}

FINALIZED LONG DESCRIPTION HTML:
---
{new_long_description}
---

ORIGINAL GRAPH SPEC CONTEXT:
{graph_spec_json}

Metadata:
- Category: {category}
- Section: {section}
- Difficulty: {difficulty}

Return ONLY the TikZ code block.
"""


def _build_prompt_tikz_from_free_response_context(
    *,
    question_text: str,
    explanation: str,
    correct_answer: str,
    new_long_description: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Prompt chuyên biệt để sinh TikZ từ nội dung câu hỏi tự luận đã hoàn chỉnh."""
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)

    return f"""You are a specialist TikZ figure generator for SAT math questions.

Task: Generate ONLY the TikZ diagram for the finalized math figure below.

You are NOT writing the question. The question, explanation, correct answer, and long description are already finalized.
Your job is to analyze that content, determine what kind of figure is actually needed, and produce a professional TikZ rendering that matches it.

PRIMARY GOAL:
- Build a clean, readable figure that faithfully represents the finalized scenario.
- Prioritize diagram quality, spacing, and label placement so elements do not overlap.
- Use the long description as the authoritative source of the figure structure, labels, and numeric relationships.
- Use the question/explanation/correct answer to understand what visual relationship must be clear.

FIRST DECIDE THE FIGURE TYPE:
Before drawing, infer which of these best matches the problem:
1. Coordinate/data graph: line graph, bar chart, scatter-style plot, or data chart where axes are necessary.
2. Geometry figure: triangle, quadrilateral, circle, angle, parallel lines with transversals, similar figures, or other Euclidean diagram.
3. Algebra/other schematic: number line, function sketch without full chart scaffolding, labeled segments, or another simple mathematical schematic.

CRITICAL AXIS RULE:
- DO NOT draw x-axis and y-axis by default.
- Draw axes ONLY if the problem is truly a coordinate/data graph and the axes are necessary to interpret the figure.
- For geometry diagrams, angle diagrams, triangles, parallel lines, transversals, labeled segments, and not-to-scale illustrations, do NOT add coordinate axes.
- If the sample/problem is a pure geometry figure like a right triangle or two parallel lines with an angle, draw only the needed geometric objects and labels.

FIGURE-SPECIFIC DRAWING RULES:
- If it is a coordinate/data graph:
  - Use manual axes with \\draw[->].
    - Normalize x coordinates to integers in 0..10 only.
    - IMPORTANT: Do NOT use raw data values directly as y coordinates.
    - Use approximate visual scaling only: map plotted y coordinates to a compact display band (recommended y in 2..8).
    - Preserve order/trend and relative differences qualitatively, not exact numeric ratio.
    - Show actual numeric values as labels near points/bars when needed for fidelity.
  - Keep labels compact and readable.
    - ANTI-ANSWER-LEAKAGE STYLING (MANDATORY):
        - Do NOT highlight any data point/bar/segment that could reveal the answer.
        - Use one uniform neutral style for all values in the same series (same color family, opacity, and line width).
        - Never use a special color for max/min values, turning points, or the value tied to the correct option.
        - Avoid attention cues on one value: no glow, bold-only emphasis, thicker stroke, unique marker, or unique fill on a single candidate.
        - If multiple series are required, use balanced styling and a legend; color must encode series identity only, not correctness.
- If it is a geometry figure:
  - Choose coordinates freely to make the figure clean and well proportioned.
  - Do not force x/y axes or graph framing.
  - Preserve key relationships visually: parallel, perpendicular, equal length, acute/obtuse angle, intersection, right angle, etc.
  - Add right-angle markers, angle arcs, tick marks, or segment labels only when they help express the intended math.
  - If the problem is marked not to scale, the figure may be visually clean without exact metric proportionality, but it must still communicate the intended relationships.
- If it is two parallel lines / angle-chasing:
  - Draw two clearly parallel lines and the relevant transversal(s).
  - Place angle labels away from intersections so they do not collide with lines.
  - Make the relevant corresponding, alternate interior, vertical, or supplementary relationships visually obvious.

TIKZ CODE REQUIREMENTS:
- Return ONLY a single \\begin{{tikzpicture}} ... \\end{{tikzpicture}} block.
- DO NOT output JSON, markdown, commentary, or any text outside the tikzpicture block.
- DO NOT use pgfplots commands or environments: \\begin{{axis}}, \\addplot, xtick, xticklabels, xmin/xmax, ymin/ymax, grid=both.
- Start with exactly: \\begin{{tikzpicture}}[line cap=round,line join=round]

LAYOUT + QUALITY RULES:
1. Coordinate bounds must satisfy abs(x) <= 12 and abs(y) <= 30.
2. Scale and position the figure to fill the canvas well without crowding the edges.
3. Use only the shapes needed for the math. Do not add redundant axes, grids, or decorations.
4. Keep labels outside strokes whenever possible.
5. If a label would overlap a segment, point, or angle marker, move it with a small offset.
6. For triangles and polygons, keep vertices separated enough that side labels and angle markers fit cleanly.
7. For angle diagrams, keep intersections uncluttered and use small arcs or labels that are easy to read.
8. For charts, ensure tick labels and axis titles do not collide with data points or each other.
9. Avoid dense loops, unnecessary shading, or complex ornamentation that could hurt compilation or readability.
10. Self-check before final output: if the chosen figure type is wrong, if axes are unnecessary, if labels overlap, or if any coordinate breaks limits, regenerate the TikZ.
11. For data graphs, keep the overall chart visually compact: target plotted height around 5-7 TikZ units (never full-range raw-value height).
12. For bar charts, bars should fit comfortably inside the frame with consistent top margin; do not let tallest bar touch the top border.
13. Final fairness check: ensure styling alone cannot be used to guess the correct answer.

FINALIZED QUESTION TEXT:
---
{question_text}
---

FINALIZED EXPLANATION:
---
{explanation}
---

FINALIZED CORRECT ANSWER:
---
{correct_answer}
---

FINALIZED LONG DESCRIPTION HTML:
---
{new_long_description}
---

ORIGINAL GRAPH SPEC CONTEXT:
{graph_spec_json}

Metadata:
- Category: {category}
- Section: {section}
- Difficulty: {difficulty}

Return ONLY the TikZ code block.
"""


def _build_prompt_graph_free_response_freeform(
    question_text_no_svg: str,
    original_explanation: str,
    original_correct_answer: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Free-form prompt for reasoning LLMs - text-only graph free-response questions."""
    long_desc_html = graph_spec.get("long_description_html", "")
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)

    # Add reasoning guidance for Medium/Hard questions
    reasoning_guidance = ""
    if difficulty.lower() in ["medium", "hard"]:
        reasoning_guidance = """\n\nCRITICAL FOR MEDIUM/HARD QUESTIONS:
- REASON about the graph data and what the question is asking
- Ensure x_values and y_values are REASONABLE and make practical sense
- Check that the correct answer is derivable from the graph with clear logic
- Avoid values that lead to awkward decimals or overly complex calculations
- Make sure the data tells a coherent, interpretable story
- Ensure calculations with the new data yield clean, reasonable results"""

    return f"""You are an SAT question writer. This is a free-response question with a GRAPH.

Task: Generate the finalized graph-aware text content first. Do NOT generate TikZ yet.

IMPORTANT:
- Update question text, explanation, and correct answer to match the generated graph or figure scenario
- Keep HTML and MathML structure EXACTLY as in sample
- CRITICAL: Calculate the correct answer based on YOUR generated graph
- Preserve the HTML structure of the long description (<ul><li>...</li></ul>)
- Make the long description detailed enough that a separate TikZ generator can reconstruct the intended figure faithfully{reasoning_guidance}

Original Graph Specification:
{graph_spec_json}

Original Long Description HTML Structure (preserve this format):
{long_desc_html}

Sample question text (without SVG):
{question_text_no_svg}

Sample explanation:
{original_explanation}

Sample correct answer:
{original_correct_answer}

Category: {category}, Section: {section}, Difficulty: {difficulty}

Generate your response in this format:

QUESTION_TEXT:
[New question text without SVG, only numbers changed]

EXPLANATION:
[New explanation matching new graph data]

CORRECT_ANSWER:
[The correct answer in same format as sample]

NEW_LONG_DESCRIPTION:
[HTML description using <ul><li>...</li></ul> format with new values]
"""


def _build_prompt_graph_free_response(
    question_text_no_svg: str,
    original_explanation: str,
    original_correct_answer: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Prompt text-only cho câu hỏi tự luận có đồ thị: KHÔNG truyền SVG, chỉ truyền text + GraphSpec."""

    # Extract long_description_html for the prompt
    long_desc_html = graph_spec.get("long_description_html", "")
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)
    return f"""You are an SAT question writer. This is a FREE-RESPONSE question with a GRAPH/CHART.

Task: Generate the graph-aware textual content first. Do NOT generate TikZ yet.

IMPORTANT:
- The question contains a graph or math figure (rendering will be handled separately by code).
- Update the question text, explanation, correct answer, and long description so they all match the same generated figure.
- Keep the same structure and wording, only change numbers.
- CRITICAL: You MUST calculate the correct answer based on the generated figure.
- The correct answer should match the format of the sample (e.g., if it's a number, provide a number; if it's HTML/MathML, provide HTML/MathML).
- DO NOT include the long description (<ul><li>...) in question_text. It will be added separately to the figure block.
- CRITICAL: The new_long_description MUST use the EXACT same HTML structure as the original (with <ul>, <li>, <br> tags). Only change the numbers.
- The long description must contain enough detail for a separate TikZ generator to infer the correct figure type and layout.

Original GraphSpec:
{graph_spec_json}

Original Long Description HTML Structure (YOU MUST PRESERVE THIS EXACT HTML FORMAT for new_long_description):
---
{long_desc_html}
---

Sample question TEXT (without SVG and without long description):
---
{question_text_no_svg}
---

Sample explanation:
---
{original_explanation}
---

Sample correct answer:
---
{original_correct_answer}
---

Category: {category}. Section: {section}. Difficulty: {difficulty}.

Return a JSON object with:
- question_text: new question text (without SVG, without long description, only numbers changed in the intro and question sentences)
- explanation: new explanation (numbers changed to match new graph and correct answer)
- correct_answer: the correct answer for the new question, in the same format as the sample
- new_long_description: new graph description in HTML format, MUST use the same <ul><li>...</li></ul> structure as the original, only changing the numbers

You must output ONLY valid JSON.
Do NOT include any explanation.
Do NOT include <think> tags.
Do NOT include reasoning.
Return ONLY the JSON object.
"""


def _get_question_html(sample: Dict[str, Any]) -> str:
    """Lấy nội dung câu hỏi mẫu (HTML + MathML) nguyên bản."""
    q = sample.get("question") or {}
    return (q.get("question") or "").strip()


def _get_explanation(sample: Dict[str, Any]) -> str:
    """Lấy explanation mẫu (HTML + MathML)."""
    q = sample.get("question") or {}
    return (q.get("explanation") or "").strip()


def _get_correct_answer_content(sample: Dict[str, Any]) -> str:
    """Lấy nội dung đáp án đúng (chuỗi HTML/MathML). Nếu là A/B/C/D thì map sang nội dung choice."""
    q_block = sample.get("question") or {}
    choices = q_block.get("choices") or []
    raw = q_block.get("correct_answer") or sample.get("correct_answer")
    if raw is None:
        return ""
    # Nếu raw là chuỗi dài (nội dung HTML/MathML), trả về luôn
    if isinstance(raw, str) and (len(raw) > 2 or "<" in raw or "math" in raw):
        return raw.strip()
    letter = raw[0] if isinstance(raw, (list, tuple)) and len(raw) > 0 else raw
    if not isinstance(letter, str):
        return str(raw) if raw else ""
    letter = letter.strip().upper()
    if choices and letter in ("A", "B", "C", "D"):
        idx = ["A", "B", "C", "D"].index(letter)
        if idx < len(choices):
            return (choices[idx] or "").strip()
    # raw là chuỗi 1 ký tự (A/B/C/D) nhưng không có choices → trả về raw
    if isinstance(raw, str):
        return raw.strip()
    return str(raw).strip()


def _build_prompt_freeform(
    original_question_html: str,
    original_explanation: str,
    original_correct_answer: str,
    category: str,
    section: str,
    q_type: str,
    difficulty: str,
    creative_mode: bool = True,
) -> str:
    """Free-form prompt for reasoning LLMs - free response questions."""
    if creative_mode:
        # For Medium/Hard: emphasis on reasoning and ensuring reasonable values
        reasoning_guidance = ""
        if difficulty.lower() in ["medium", "hard"]:
            reasoning_guidance = """\n\nCRITICAL FOR MEDIUM/HARD QUESTIONS:
- REASON about the mathematical logic before generating
- Ensure all numerical values are REASONABLE and make practical sense
- Check that the problem has a clear, logical solution path
- Verify the answer is achievable with the given numbers
- Make sure intermediate calculations yield clean, reasonable results
- Avoid numbers that lead to awkward decimals or irrational solutions"""

        return f"""You are an SAT question writer. Generate a NEW free-response question that tests the SAME mathematical skill as the sample, but with a DIFFERENT scenario.

Requirements:
1. SAME SKILL: Test the exact same mathematical concept/skill
2. DIFFERENT SCENARIO: Completely different context
3. DIFFERENT NUMBERS: New numerical values
4. PRESERVE FORMAT: Keep ALL HTML tags and MathML structure EXACTLY as in sample
5. Calculate the correct answer for your new question{reasoning_guidance}

Category: {category}, Section: {section}, Type: {q_type}, Difficulty: {difficulty}

Sample question:
{original_question_html}

Sample explanation:
{original_explanation}

Sample correct answer:
{original_correct_answer}

Generate your response in this format:

QUESTION:
[Your new question with proper HTML+MathML]

EXPLANATION:
[Detailed explanation for your new question]

CORRECT_ANSWER:
[The correct answer in same format as sample]
"""
    else:
        return f"""You are an SAT question writer. Change ONLY the numerical values. Keep ALL wording, HTML tags, and MathML structure identical.

Category: {category}, Section: {section}, Type: {q_type}, Difficulty: {difficulty}

Sample question:
{original_question_html}

Sample explanation:
{original_explanation}

Sample correct answer:
{original_correct_answer}

Generate your response in this format:

QUESTION:
[Question with only numbers changed]

EXPLANATION:
[Explanation with only numbers changed]

CORRECT_ANSWER:
[Correct answer with only numbers changed]
"""


def _build_prompt(
    original_question_html: str,
    original_explanation: str,
    original_correct_answer: str,
    category: str,
    section: str,
    q_type: str,
    difficulty: str,
    creative_mode: bool = True,
) -> str:
    if creative_mode:
        return f"""You are an SAT question writer. Task: Generate a NEW question that tests the SAME mathematical skill/concept as the sample, but with a DIFFERENT scenario and context.

CRITICAL REQUIREMENTS:
1. SAME SKILL: The new question must test the exact same mathematical skill, concept, or problem-solving technique as the sample
2. DIFFERENT SCENARIO: Create a completely different real-world context, story, or scenario (e.g., if sample is about distance, use temperature, money, population, etc.)
3. DIFFERENT NUMBERS: Use entirely different numerical values that make sense for your new scenario
4. SAME FORMAT: Maintain the same HTML + MathML structure and formatting style
5. SAME DIFFICULTY: Keep the same difficulty level ({difficulty})
6. CONSISTENT OUTPUT: The new question, explanation, and correct_answer must all be logically consistent

ANALYSIS INSTRUCTIONS:
- First, identify what mathematical skill/concept the sample question tests (e.g., solving equations, word problems, algebraic manipulation, etc.)
- Then create a NEW scenario that requires the SAME mathematical approach/skill to solve
- Category: {category}. Section: {section}. Type: {q_type}. Difficulty: {difficulty}

FORMAT REQUIREMENTS:
- Use proper HTML tags and MathML format exactly like the sample
- All mathematical expressions must be in <math> tags with proper MathML structure
- Provide a clear, detailed explanation showing the solution steps  
- The correct_answer must be in the same format as the sample (e.g., if sample has MathML, use MathML)

Sample question (HTML + MathML) - ANALYZE the math skill being tested:
---
{original_question_html}
---

Sample explanation (HTML + MathML) - UNDERSTAND the problem-solving approach:
---
{original_explanation}
---

Sample correct answer (content of the right answer, HTML + MathML):
---
{original_correct_answer}
---

Return a JSON object with keys: question, explanation, correct_answer.
- question: NEW question with DIFFERENT scenario testing the SAME skill, proper HTML+MathML format
- explanation: Detailed explanation for YOUR new question, showing the solution steps clearly  
- correct_answer: The correct answer for YOUR new question, in the same format as the sample (e.g., HTML/MathML string)

EXAMPLE TRANSFORMATION:
Sample: "Solve for x: 2x + 5 = 13"
Your New Question: "Solve for y: 3y - 7 = 14" 
(Both test: solving linear equations with one variable, but different equations)

You must output ONLY valid JSON.
Do NOT include any explanation.
Do NOT include <think> tags.
Do NOT include reasoning.
Return ONLY the JSON object.
"""
    else:
        # Conservative mode: only change numbers
        return f"""You are an SAT question writer. Task: take the sample question, explanation, and correct answer below and change ONLY the numerical values. Do NOT change any other content.

STRICT rules:
- Do NOT rewrite, paraphrase, or alter the wording. Keep every word, every tag, and every character exactly as in the sample except for numbers.
- Change ONLY the numerical data: replace digits in <mn> tags, in math alttext attributes, and any other numeric values in the text. Everything else must stay identical.
- The new question, explanation, and correct_answer must be consistent: the numbers you use in the new question must match the numbers in the new explanation and in the new correct answer.
- Category: {category}. Section: {section}. Type: {q_type}. Difficulty: {difficulty}.
- Output three fields: question (same HTML+MathML as sample with numbers changed), explanation (same structure with numbers changed to match the new question), correct_answer (the right answer content in same format, e.g. HTML/MathML string).

Sample question (HTML + MathML):
---
{original_question_html}
---

Sample explanation (HTML + MathML):
---
{original_explanation}
---

Sample correct answer (content of the right choice, HTML + MathML):
---
{original_correct_answer}
---

Return a JSON object with keys: question, explanation, correct_answer. Each value: same string as sample with only numbers substituted; numbers must be consistent across all three.

You must output ONLY valid JSON.
Do NOT include any explanation.
Do NOT include <think> tags.
Do NOT include reasoning.
Return ONLY the JSON object.
"""


def _get_choices(sample: Dict[str, Any]) -> List[str]:
    """Lấy danh sách 4 lựa chọn (A, B, C, D) từ câu mẫu."""
    q_block = sample.get("question") or {}
    choices = q_block.get("choices") or []
    if not isinstance(choices, list) or len(choices) < 4:
        return []
    return [(choices[i] or "").strip() for i in range(4)]


def _get_correct_answer_letter(sample: Dict[str, Any]) -> Optional[str]:
    """Lấy chữ cái đáp án đúng (A/B/C/D)."""
    q_block = sample.get("question") or {}
    raw = q_block.get("correct_answer") or sample.get("correct_answer")
    if raw is None:
        return None
    letter = raw[0] if isinstance(raw, (list, tuple)) and len(raw) > 0 else raw
    if not isinstance(letter, str):
        return None
    letter = letter.strip().upper()
    return letter if letter in ("A", "B", "C", "D") else None


def _supports_custom_temperature(model_name: str) -> bool:
    """Some models (for example GPT-5 family) only support default temperature."""
    return "gpt-5" not in (model_name or "").lower()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from model output."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("Model output does not contain valid JSON object")


def _invoke_openai_basic_structured(
    *,
    prompt_text: str,
    output_schema: type[BaseModel],
    api_key: str,
    model: str,
    temperature: float,
    debug_stage_c: bool = False,
) -> Any:
    """Direct OpenAI inference with strict system prompt and Pydantic validation."""

    required_fields = set(output_schema.model_json_schema().get("required", []))
    model_field_names = set(output_schema.model_fields.keys())

    def _properties_looks_like_payload(props: Any) -> bool:
        if not isinstance(props, dict):
            return False
        if model_field_names and not (set(props.keys()) & model_field_names):
            return False

        # If nested values look like schema metadata blocks, treat as schema.
        schema_meta_hits = 0
        for value in props.values():
            if isinstance(value, dict) and (
                {"type", "description", "title"} & set(value.keys())
            ):
                schema_meta_hits += 1
        return schema_meta_hits == 0

    def _has_required_field(obj: Any) -> bool:
        if not isinstance(obj, dict) or not required_fields:
            return False
        return any(field in obj for field in required_fields)

    def _looks_like_schema_object(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        if "$schema" in obj or "$defs" in obj:
            return True
        props = obj.get("properties")
        if _properties_looks_like_payload(props):
            return False

        # Treat explicit schema object as schema only when properties do not look like payload.
        if obj.get("type") == "object" and "properties" in obj:
            return True

        # Common schema/template shape returned by models: description/title/properties,
        # but no required payload fields like question/explanation.
        schemaish_keys = {"type", "properties", "required", "title", "description"}
        if (
            not _has_required_field(obj)
            and len(schemaish_keys.intersection(obj.keys())) >= 2
        ):
            return True

        return False

    def _candidate_payloads(obj: Any) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        if isinstance(obj, dict):
            candidates.append(obj)

            # If the response nests payload under model/class-like key.
            class_key = output_schema.__name__
            nested_by_class = obj.get(class_key)
            if isinstance(nested_by_class, dict):
                candidates.append(nested_by_class)

            # Common pattern from some models: payload nested under "properties".
            props = obj.get("properties")
            if _properties_looks_like_payload(props) and isinstance(props, dict):
                # Some responses split fields between outer object and properties payload.
                merged: Dict[str, Any] = dict(props)
                for field_name in model_field_names:
                    if field_name in obj and field_name not in merged:
                        merged[field_name] = obj[field_name]
                candidates.append(merged)
                candidates.append(dict(props))

            # Common wrapper keys returned by LLM tools/parsers.
            for key in (
                "output",
                "result",
                "data",
                "json",
                "instance",
                "payload",
                "value",
                "response",
            ):
                nested = obj.get(key)
                if isinstance(nested, dict):
                    candidates.append(nested)

            # If dict has one nested object only, it is often the actual payload.
            if len(obj) == 1:
                only_val = next(iter(obj.values()))
                if isinstance(only_val, dict):
                    candidates.append(only_val)

        # Prefer candidates that contain at least one required field.
        prioritized = [c for c in candidates if _has_required_field(c)]
        fallback = [c for c in candidates if c not in prioritized]
        return prioritized + fallback

    def _sanitize_prompt_text(text: str) -> str:
        # Remove control characters that can break downstream JSON parsing in API gateways.
        cleaned = (text or "").replace("\x00", " ")
        cleaned = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", cleaned)
        # Replace invalid Unicode sequences conservatively.
        cleaned = cleaned.encode("utf-8", errors="replace").decode(
            "utf-8", errors="replace"
        )
        return cleaned

    client = OpenAI(api_key=api_key)
    schema_json = json.dumps(
        output_schema.model_json_schema(), ensure_ascii=False, indent=2
    )
    required_list = ", ".join(sorted(required_fields)) if required_fields else "(none)"
    minimal_skeleton = (
        {k: "" for k in sorted(required_fields)} if required_fields else {}
    )
    system_prompt = f"""You are an SAT question writer.

Return ONE valid JSON object only, with no markdown and no extra text.
The JSON object must be AN INSTANCE that matches this schema:
{schema_json}

Rules:
- Preserve HTML and MathML tags exactly where present.
- Do not include reasoning or <think> tags.
- Do not include code fences.
- NEVER output a JSON Schema (no "properties", "required", "$schema", "$defs").
- Required fields that MUST appear at top level: {required_list}
- Minimal shape reminder: {json.dumps(minimal_skeleton, ensure_ascii=False)}
"""
    system_prompt = _sanitize_prompt_text(system_prompt)
    last_error = ""
    last_raw = ""
    last_parsed_keys: List[str] = []

    for attempt in range(3):
        attempt_temp = temperature if attempt == 0 else 0.0
        if attempt == 0:
            user_prompt = prompt_text
        else:
            user_prompt = (
                f"{prompt_text}\n\n"
                "Your previous output was invalid. "
                "Return ONLY a JSON INSTANCE matching the schema fields exactly. "
                "Do not return schema metadata. "
                f"Required top-level fields: {required_list}."
            )

        user_prompt = _sanitize_prompt_text(user_prompt)

        try:
            request_kwargs: Dict[str, Any] = {
                "model": model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if _supports_custom_temperature(model):
                request_kwargs["temperature"] = attempt_temp
            completion = client.chat.completions.create(**request_kwargs)
        except Exception as e:
            last_error = (
                f"OpenAI request failed on attempt {attempt + 1}: {e} "
                f"(system_len={len(system_prompt)}, user_len={len(user_prompt)})"
            )
            continue
        raw = completion.choices[0].message.content or "{}"

        if debug_stage_c:
            schema_name = output_schema.__name__
            print("\n" + "=" * 70)
            print(f"STAGE C RAW RESPONSE [{schema_name}] attempt {attempt + 1}")
            print("=" * 70)
            print(raw)
        last_raw = raw

        try:
            parsed = _extract_json_object(raw)
            if isinstance(parsed, dict):
                last_parsed_keys = list(parsed.keys())[:20]
        except Exception as e:
            last_error = f"JSON parse error: {e}"
            continue

        if _looks_like_schema_object(parsed):
            last_error = (
                "Model returned a JSON Schema object instead of a payload instance"
            )
            continue

        for candidate in _candidate_payloads(parsed):
            if _looks_like_schema_object(candidate):
                last_error = (
                    "Model returned schema-like candidate instead of payload instance"
                )
                continue

            try:
                return output_schema.model_validate(candidate)
            except ValidationError as ve:
                last_error = str(ve)

    raise ValueError(
        "OpenAI basic structured output validation failed after retries. "
        f"Last error: {last_error}. "
        f"Last parsed top-level keys: {last_parsed_keys}. "
        f"Last raw output (first 500 chars): {(last_raw or '')[:500]}"
    )


def _invoke_structured(
    *,
    prompt_text: str,
    output_schema: type[BaseModel],
    llm: Optional[ChatOpenAI],
    use_openai_basic: bool,
    api_key: Optional[str],
    model: str,
    temperature: float,
    debug_stage_c: bool = False,
) -> Any:
    """Invoke OpenAI model for strict structured output (LangChain or basic chat-completions)."""
    if use_openai_basic:
        if not api_key:
            raise ValueError("Need OPENAI_API_KEY for openai_basic structured call")
        return _invoke_openai_basic_structured(
            prompt_text=prompt_text,
            output_schema=output_schema,
            api_key=api_key,
            model=model,
            temperature=temperature,
            debug_stage_c=debug_stage_c,
        )

    if llm is None:
        raise ValueError("LLM is not initialized for structured LangChain mode")
    structured_llm = llm.with_structured_output(output_schema)
    return structured_llm.invoke([HumanMessage(content=prompt_text)])


def _validate_mc_instance(
    *,
    choices: Optional[List[str]],
    correct_letter: Optional[str],
) -> None:
    if choices is None or len(choices) != 4:
        raise ValueError(
            "Multiple-choice generated instance must have exactly 4 choices"
        )
    if not correct_letter or correct_letter not in ("A", "B", "C", "D"):
        raise ValueError(
            "Multiple-choice generated instance must have correct_answer_letter in A/B/C/D"
        )


def _try_generate_structured_openai(
    *,
    sample: Dict[str, Any],
    llm: Optional[ChatOpenAI],
    use_openai_basic: bool,
    api_key: Optional[str],
    model: str,
    creative_mode: bool,
    original_html: str,
    original_explanation: str,
    original_choices: List[str],
    original_correct_answer: str,
    correct_letter: Optional[str],
    q_type: str,
    difficulty: str,
    section: str,
    category: str,
    graph_spec: Any,
    step_b_trace: Optional[Dict[str, Any]] = None,
    debug_stage_c: bool = False,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Generate a new question through analysis-blueprint-instance-verification.

    Returns:
        (new_item_or_none, artifacts)
    """
    artifacts: Dict[str, Any] = {
        "pipeline": "structured_openai_v1",
        "status": "started",
        "mode": "openai_basic" if use_openai_basic else "openai_structured",
    }

    has_graph = graph_spec is not None
    if has_graph:
        artifacts["status"] = "fallback"
        artifacts["reason"] = "graph_questions_use_legacy_generator"
        return None, artifacts

    is_multiple_choice = (
        q_type == "multiple-choice"
        and len(original_choices) == 4
        and bool(correct_letter)
    )
    input_snapshot = _build_input_snapshot(
        sample,
        original_question_html=original_html,
        original_explanation=original_explanation,
        original_correct_answer=original_correct_answer,
        original_choices=original_choices,
        correct_letter=correct_letter,
        has_graph=has_graph,
    )
    artifacts["input_snapshot"] = input_snapshot

    temp = 0.7 if creative_mode else 0.3

    try:
        analysis_prompt = _build_problem_analysis_prompt(
            input_snapshot, step_b_trace=step_b_trace
        )
        analysis = _invoke_structured(
            prompt_text=analysis_prompt,
            output_schema=ProblemAnalysis,
            llm=llm,
            use_openai_basic=use_openai_basic,
            api_key=api_key,
            model=model,
            temperature=0.2,
            debug_stage_c=debug_stage_c,
        )

        # Force source_fields to match actual sample extraction and avoid model hallucinated source metadata.
        analysis.source_fields = SourceMetadata.model_validate(
            input_snapshot.get("source_metadata") or {}
        )
        artifacts["analysis"] = analysis.model_dump()

        blueprint_prompt = _build_problem_blueprint_prompt(input_snapshot, analysis)
        blueprint = _invoke_structured(
            prompt_text=blueprint_prompt,
            output_schema=ProblemBlueprint,
            llm=llm,
            use_openai_basic=use_openai_basic,
            api_key=api_key,
            model=model,
            temperature=0.2,
            debug_stage_c=debug_stage_c,
        )

        # Preserve strict source vs derived separation in blueprint.
        blueprint.source_metadata = analysis.source_fields
        blueprint.derived_analysis = analysis.derived_fields
        artifacts["blueprint"] = blueprint.model_dump()

        instance_prompt = _build_structured_instance_prompt(
            input_snapshot,
            analysis,
            blueprint,
            creative_mode=creative_mode,
        )
        instance = _invoke_structured(
            prompt_text=instance_prompt,
            output_schema=StructuredGeneratedInstance,
            llm=llm,
            use_openai_basic=use_openai_basic,
            api_key=api_key,
            model=model,
            temperature=temp,
            debug_stage_c=debug_stage_c,
        )
        artifacts["generated_instance"] = instance.model_dump()

        verify_prompt = _build_structured_verification_prompt(
            input_snapshot, analysis, blueprint, instance
        )
        verification = _invoke_structured(
            prompt_text=verify_prompt,
            output_schema=GenerationVerificationReport,
            llm=llm,
            use_openai_basic=use_openai_basic,
            api_key=api_key,
            model=model,
            temperature=0.0,
            debug_stage_c=debug_stage_c,
        )
        artifacts["verification"] = verification.model_dump()

        new_question_text = (instance.question or "").strip()
        new_explanation = (instance.explanation or "").strip()
        if not new_question_text or not new_explanation:
            raise ValueError("Structured instance missing question/explanation")

        # Verification models can be overly strict; only hard-fail on confident negatives.
        if verification.is_solvable is False and verification.confidence >= 0.5:
            raise ValueError(
                "Structured verification failed (unsolvable, high confidence)"
            )
        if verification.answer_format_valid is False and verification.confidence >= 0.7:
            raise ValueError(
                "Structured verification failed (answer format invalid, high confidence)"
            )

        if is_multiple_choice:
            new_choices = [str(c).strip() for c in (instance.choices or [])]
            effective_letter = instance.correct_answer_letter
            if verification.corrected_correct_answer_letter:
                effective_letter = verification.corrected_correct_answer_letter
            _validate_mc_instance(choices=new_choices, correct_letter=effective_letter)

            unique_ok = verification.multiple_choice_unique_correct
            if unique_ok is False:
                raise ValueError(
                    "Multiple-choice verification reports non-unique correct option"
                )

            new_question_content = {
                "paragraph": sample.get("question", {}).get("paragraph"),
                "question": new_question_text,
                "choices": new_choices,
                "correct_answer": [effective_letter],
                "explanation": new_explanation,
            }
        else:
            effective_answer = instance.correct_answer
            if verification.corrected_correct_answer:
                effective_answer = verification.corrected_correct_answer
            if not effective_answer:
                raise ValueError(
                    "Structured instance missing correct_answer for free response"
                )

            new_question_content = {
                "paragraph": sample.get("question", {}).get("paragraph"),
                "question": new_question_text,
                "choices": None,
                "correct_answer": str(effective_answer).strip(),
                "explanation": new_explanation,
            }

        new_item = {
            "id": str(uuid.uuid4()),
            "subject": sample.get("subject", "SAT"),
            "pool": sample.get("pool", "practice_test"),
            "section": section,
            "category": category,
            "skill": sample.get("skill", ""),
            "difficulty": difficulty,
            "type": q_type,
            "question": new_question_content,
            "image_url": sample.get("image_url"),
            "_generation_mode": "structured_openai_v1",
            "_generation_artifacts": artifacts,
        }
        artifacts["status"] = "success"
        return new_item, artifacts
    except Exception as e:
        artifacts["status"] = "fallback"
        artifacts["reason"] = str(e)
        return None, artifacts


def _build_prompt_multiple_choice_freeform(
    original_question_html: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    category: str,
    section: str,
    difficulty: str,
    creative_mode: bool = True,
) -> str:
    """Free-form prompt for reasoning LLMs - focuses on content generation, not JSON structure."""
    choices_text = "\n".join(
        [f"{chr(65+i)}. {choice}" for i, choice in enumerate(original_choices)]
    )

    if creative_mode:
        # For Medium/Hard: emphasis on reasoning and ensuring reasonable values
        reasoning_guidance = ""
        if difficulty.lower() in ["medium", "hard"]:
            reasoning_guidance = """\n\nCRITICAL FOR MEDIUM/HARD QUESTIONS:
- REASON about the mathematical logic before generating
- Ensure all numerical values are REASONABLE and make practical sense
- Check that the problem has a clear, logical solution path
- Verify the correct answer is achievable with the given numbers
- Make sure intermediate calculations yield clean, reasonable results
- Avoid numbers that lead to awkward decimals or irrational solutions
- Create plausible distractors that represent common mistakes"""

        return f"""You are an SAT question writer. Generate a NEW multiple-choice question that tests the SAME mathematical skill as the sample, but with a DIFFERENT scenario.

Requirements:
1. SAME SKILL: Test the exact same mathematical concept/skill
2. DIFFERENT SCENARIO: Completely different context (e.g., if sample is about cars, use books, temperature, etc.)
3. DIFFERENT NUMBERS: New numerical values appropriate to your scenario
4. PRESERVE FORMAT: Keep ALL HTML tags and MathML structure EXACTLY as in the sample
5. EXACTLY 4 CHOICES: Provide 4 answer options (A, B, C, D)
6. Calculate the correct answer for your new question{reasoning_guidance}

Category: {category}, Section: {section}, Difficulty: {difficulty}

Sample question:
{original_question_html}

Sample explanation:
{original_explanation}

Sample choices (correct: {correct_letter}):
{choices_text}

Generate your response in this format:

QUESTION:
[Your new question with proper HTML+MathML, testing the same skill but different scenario]

EXPLANATION:
[Detailed explanation showing how to solve YOUR new question]

CHOICE_A:
[First answer choice]

CHOICE_B:
[Second answer choice]

CHOICE_C:
[Third answer choice]

CHOICE_D:
[Fourth answer choice]

CORRECT_ANSWER:
[Letter of correct answer: A, B, C, or D]
"""
    else:
        return f"""You are an SAT question writer. Change ONLY the numerical values in this multiple-choice question. Keep ALL wording, HTML tags, and MathML structure identical.

Strict rules:
- Change ONLY numbers (digits in <mn> tags, numeric values in text)
- Keep structure, wording, and formatting IDENTICAL
- Ensure new numbers are consistent across question, explanation, and choices

Category: {category}, Section: {section}, Difficulty: {difficulty}

Sample question:
{original_question_html}

Sample explanation:
{original_explanation}

Sample choices (correct: {correct_letter}):
{choices_text}

Generate your response in this format:

QUESTION:
[Question with only numbers changed]

EXPLANATION:
[Explanation with only numbers changed]

CHOICE_A:
[First choice with only numbers changed]

CHOICE_B:
[Second choice with only numbers changed]

CHOICE_C:
[Third choice with only numbers changed]

CHOICE_D:
[Fourth choice with only numbers changed]

CORRECT_ANSWER:
[Letter of correct answer: typically {correct_letter}]
"""


def _build_prompt_multiple_choice(
    original_question_html: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    category: str,
    section: str,
    difficulty: str,
    creative_mode: bool = True,
) -> str:
    """Prompt cho multiple-choice: sinh question, explanation, 4 choices, và correct_answer_letter."""
    # Do not include "Choice A:/B:/..." prefixes — UI will display A/B/C/D
    choices_text = "\n".join(original_choices)

    if creative_mode:
        return f"""You are an SAT question writer. This is a MULTIPLE-CHOICE question. Task: Generate a NEW question that tests the SAME mathematical skill/concept as the sample, but with a DIFFERENT scenario and context.

CRITICAL REQUIREMENTS:
1. SAME SKILL: The new question must test the exact same mathematical skill, concept, or problem-solving technique as the sample
2. DIFFERENT SCENARIO: Create a completely different real-world context, story, or scenario (e.g., if sample is about cars, use books, students, temperature, etc.)
3. DIFFERENT NUMBERS: Use entirely different numerical values that make sense for your new scenario
4. SAME FORMAT: Maintain the same HTML + MathML structure and formatting style
5. SAME DIFFICULTY: Keep the same difficulty level ({difficulty})
6. EXACTLY 4 CHOICES: Provide exactly 4 answer choices (A, B, C, D) in order

ANALYSIS INSTRUCTIONS:
- First, identify what mathematical skill/concept the sample question tests (e.g., linear equations, percentages, ratios, algebraic manipulation, geometry theorems, etc.)
- Then create a NEW scenario that requires the SAME mathematical approach/skill to solve
- Ensure your new question would appear in the same category: {category}, Section: {section}

FORMAT REQUIREMENTS:
- Use proper HTML tags and MathML format exactly like the sample
- All mathematical expressions must be in <math> tags with proper MathML structure
- Keep clean, SAT-style professional wording
- Provide a clear, detailed explanation showing the solution steps
- Make sure the explanation teaches the concept, not just shows calculations

Sample question (HTML + MathML) - ANALYZE the math skill being tested:
---
{original_question_html}
---

Sample explanation (HTML + MathML) - UNDERSTAND the problem-solving approach:
---
{original_explanation}
---

Sample 4 choices (correct answer is {correct_letter}):
---
{choices_text}
---

Return a JSON object with keys: question, explanation, choices, correct_answer_letter.
- question: NEW question with DIFFERENT scenario testing the SAME skill, proper HTML+MathML format
- explanation: Detailed explanation for YOUR new question, showing the solution steps clearly
- choices: list of exactly 4 strings in order A, B, C, D (plausible distractors based on common mistakes)
- correct_answer_letter: one of "A", "B", "C", "D" (the correct answer for YOUR new question)

EXAMPLE TRANSFORMATION:
Sample: "A car travels 120 miles in 2 hours. What is its average speed?"
Your New Question: "A student reads 45 pages in 1.5 hours. What is the student's reading rate in pages per hour?"
(Both test: rate = distance/time concept, but different contexts)

You must output ONLY valid JSON.
Do NOT include any explanation.
Do NOT include <think> tags.
Do NOT include reasoning.
Return ONLY the JSON object.
"""
    else:
        # Conservative mode: only change numbers
        return f"""You are an SAT question writer. This is a MULTIPLE-CHOICE question. Task: change ONLY the numerical values in the sample below. Do NOT change wording, structure, or order. Output exactly 4 choices (A, B, C, D) and the correct answer letter.

STRICT rules:
- Do NOT rewrite or paraphrase. Keep every word and tag except numbers.
- Change ONLY numerical data (digits in <mn>, alttext, etc.). Everything else stays identical.
- The new question, explanation, and all 4 choices must use consistent numbers.
- You MUST output exactly 4 choices in order A, B, C, D. The correct answer for the new question must be the same letter as in the sample (correct answer here is {correct_letter}).
- Category: {category}. Section: {section}. Difficulty: {difficulty}.

Sample question (HTML + MathML):
---
{original_question_html}
---

Sample explanation (HTML + MathML):
---
{original_explanation}
---

Sample 4 choices (correct answer is {correct_letter}):
---
{choices_text}

Return a JSON object with keys: question, explanation, choices, correct_answer_letter.
- question: new question string (only numbers changed).
- explanation: new explanation string (only numbers changed, consistent with new question).
- choices: list of exactly 4 strings, in order A, B, C, D (only numbers changed in each).
- correct_answer_letter: one of "A", "B", "C", "D" (the correct choice for the new question; typically the same as the sample, {correct_letter}).

You must output ONLY valid JSON.
Do NOT include any explanation.
Do NOT include <think> tags.
Do NOT include reasoning.
Return ONLY the JSON object.
"""


def generate_new_question(
    sample: Dict[str, Any],
    llm: Optional[ChatOpenAI] = None,
    use_openai_basic: bool = False,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    creative_mode: Optional[bool] = None,
    step_b_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Sinh câu hỏi mới, explanation và đáp án từ câu mẫu (cùng category, đúng format, chỉ đổi số liệu).

    Args:
        sample: Một item từ questions_practice_test.json (có id, category, question, explanation, correct_answer, ...).
        llm: LangChain ChatOpenAI. Nếu None sẽ tạo mới từ API keys.
        use_openai_basic: Nếu True, dùng OpenAI chat-completions trực tiếp với system prompt JSON (không LangChain structured output).
        api_key: OpenAI API key (hoặc dùng OPENAI_API_KEY env var).
        model: Tên model OpenAI (mặc định: gpt-4o-mini).
        creative_mode: Nếu None (mặc định), tự động chọn dựa trên difficulty:
                      - Easy: False (chỉ đổi số, giữ nguyên logic)
                      - Medium/Hard: True (reasoning để đảm bảo giá trị hợp lý và logic đúng)
                      Nếu set explicit = True/False, sẽ dùng giá trị đó.

    Returns:
        Câu hỏi mới dạng dict, cùng cấu trúc với questions_practice_test.json,
        question.question, question.explanation, question.correct_answer đều được sinh; choices có thể null.
    """
    # Determine difficulty level and auto-set creative_mode if not specified
    difficulty = sample.get("difficulty", "Easy").lower()
    debug_stage_c = True

    if creative_mode is None:
        # Strategy based on difficulty:
        # - Easy: only change numbers (conservative mode)
        # - Medium/Hard: use reasoning to ensure logical consistency (creative mode)
        if difficulty == "easy":
            creative_mode = False
            print(
                f"📘 Difficulty: {difficulty.upper()} → Strategy: Conservative (only change numbers)"
            )
        else:  # medium or hard
            creative_mode = True
            print(
                f"📗 Difficulty: {difficulty.upper()} → Strategy: Reasoning (ensure logical consistency and reasonable values)"
            )
    else:
        mode_text = "creative" if creative_mode else "conservative"
        print(
            f"📙 Difficulty: {difficulty.upper()} → Strategy: {mode_text.upper()} (manually set)"
        )

    openai_key = api_key or os.getenv("OPENAI_API_KEY")
    generation_temp = 0.7 if creative_mode else 0.3

    if use_openai_basic and not openai_key:
        raise ValueError("Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền api_key.")

    if llm is None and not use_openai_basic:
        # Use OpenAI model
        if not openai_key:
            raise ValueError(
                "Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền api_key."
            )

        llm_kwargs: Dict[str, Any] = {
            "model": model,
            "api_key": openai_key,
        }
        if _supports_custom_temperature(model):
            llm_kwargs["temperature"] = generation_temp
        llm = ChatOpenAI(**llm_kwargs)  # type: ignore[arg-type]
        mode_text = "creative" if creative_mode else "conservative"
        print(f"✓ Using OpenAI model: {model} ({mode_text} mode)")
    elif use_openai_basic:
        mode_text = "creative" if creative_mode else "conservative"
        print(f"✓ Using OpenAI basic inference: {model} ({mode_text} mode)")

    category = sample.get("category", "Algebra")
    section = sample.get("section", "Math")
    q_type = sample.get("type", "multiple-choice")
    difficulty = sample.get("difficulty", "Easy")

    original_html = _get_question_html(sample)

    parser = MathMLParser()
    parsed = parser.parse(original_html)
    graph_spec = parsed.get("graph")

    if not original_html:
        raise ValueError("Câu mẫu không có nội dung question (HTML).")
    original_explanation = _get_explanation(sample)
    original_choices = _get_choices(sample)
    correct_letter = _get_correct_answer_letter(sample)
    original_correct_answer = _get_correct_answer_content(sample)
    is_multiple_choice = (
        (q_type == "multiple-choice")
        and len(original_choices) == 4
        and correct_letter
        and original_explanation
    )
    generate_full = bool(original_explanation and original_correct_answer)

    # Structured OpenAI pipeline for non-graph math questions.
    # If structured generation fails, fall back to the legacy branch below.
    structured_artifacts: Optional[Dict[str, Any]] = None
    if generate_full:
        structured_item, structured_artifacts = _try_generate_structured_openai(
            sample=sample,
            llm=llm,
            use_openai_basic=use_openai_basic,
            api_key=openai_key,
            model=model,
            creative_mode=bool(creative_mode),
            original_html=original_html,
            original_explanation=original_explanation,
            original_choices=original_choices,
            original_correct_answer=original_correct_answer,
            correct_letter=correct_letter,
            q_type=q_type,
            difficulty=difficulty,
            section=section,
            category=category,
            graph_spec=graph_spec,
            step_b_trace=step_b_trace,
            debug_stage_c=debug_stage_c,
        )
        if structured_item is not None:
            return structured_item

        reason = (structured_artifacts or {}).get("reason", "unknown")
        # Fail fast for validator high-confidence failures so UI can surface model-stuck cases.
        if isinstance(reason, str) and reason.startswith(
            "Structured verification failed"
        ):
            raise RuntimeError(reason)
        print(f"⚠ Structured pipeline fallback to legacy generator: {reason}")

    if is_multiple_choice:
        if graph_spec is not None:
            # ========== LUỒNG XỬ LÝ CÂU HỎI CÓ ĐỒ THỊ ==========
            # Loại bỏ SVG và long description khỏi HTML để giảm token
            # Long description sẽ được xử lý riêng và chèn vào figure block
            question_text_no_svg = _remove_svg_and_long_desc_from_html(original_html)

            print("Question text without SVG:", question_text_no_svg)

            # Convert GraphSpec to dict for JSON serialization
            graph_spec_dict = {
                "x_label": getattr(graph_spec, "x_label", ""),
                "y_label": getattr(graph_spec, "y_label", ""),
                "raw_long_description": getattr(graph_spec, "raw_long_description", ""),
                "long_description_html": getattr(
                    graph_spec, "long_description_html", ""
                ),
            }

            # Dùng prompt riêng cho câu hỏi có đồ thị
            if not correct_letter:
                raise ValueError("Multiple-choice question requires correct_letter")

            prompt_text = _build_prompt_graph_multiple_choice(
                question_text_no_svg,
                original_explanation,
                original_choices,
                correct_letter,
                graph_spec_dict,
                category,
                section,
                difficulty,
            )

            if use_openai_basic:
                freeform_prompt = _build_prompt_graph_multiple_choice_freeform(
                    question_text_no_svg,
                    original_explanation,
                    original_choices,
                    correct_letter,
                    graph_spec_dict,
                    category,
                    section,
                    difficulty,
                )
                result_graph_text = _invoke_openai_basic_structured(
                    prompt_text=freeform_prompt,
                    output_schema=GeneratedGraphQuestionTextContent,
                    api_key=openai_key or "",
                    model=model,
                    temperature=generation_temp,
                    debug_stage_c=debug_stage_c,
                )
            else:
                if llm is None:
                    raise ValueError(
                        "LLM chưa được khởi tạo cho structured-output mode"
                    )
                structured_llm = llm.with_structured_output(
                    GeneratedGraphQuestionTextContent
                )
                result_graph_text: GeneratedGraphQuestionTextContent = structured_llm.invoke(  # type: ignore
                    [HumanMessage(content=prompt_text)]
                )

            # Validate kết quả
            new_choices = result_graph_text.choices or []
            if len(new_choices) != 4:
                raise ValueError(
                    f"LLM phải trả về đúng 4 choices, nhận được {len(new_choices)}."
                )
            new_choices = [str(c).strip() for c in new_choices[:4]]
            new_letter = (result_graph_text.correct_answer_letter or "").strip().upper()
            if new_letter not in ("A", "B", "C", "D"):
                raise ValueError(
                    f"correct_answer_letter phải là A, B, C hoặc D, nhận được: {result_graph_text.correct_answer_letter!r}"
                )

            new_question_text_no_svg = (result_graph_text.question_text or "").strip()
            new_explanation = (result_graph_text.explanation or "").strip()
            new_long_description = (result_graph_text.new_long_description or "").strip()

            if not new_question_text_no_svg:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            if not new_long_description:
                raise ValueError("LLM không trả về new_long_description.")

            tikz_prompt = _build_prompt_tikz_from_graph_context(
                question_text=new_question_text_no_svg,
                explanation=new_explanation,
                choices=new_choices,
                correct_letter=new_letter,
                new_long_description=new_long_description,
                graph_spec=graph_spec_dict,
                category=category,
                section=section,
                difficulty=difficulty,
            )

            if use_openai_basic:
                result_tikz = _invoke_openai_basic_structured(
                    prompt_text=tikz_prompt,
                    output_schema=GeneratedTikzDiagramContent,
                    api_key=openai_key or "",
                    model=model,
                    temperature=max(0.0, min(generation_temp, 0.3)),
                    debug_stage_c=debug_stage_c,
                )
            else:
                if llm is None:
                    raise ValueError(
                        "LLM chưa được khởi tạo cho structured-output mode"
                    )
                tikz_llm = llm.with_structured_output(GeneratedTikzDiagramContent)
                result_tikz: GeneratedTikzDiagramContent = tikz_llm.invoke(  # type: ignore
                    [HumanMessage(content=tikz_prompt)]
                )

            if not (result_tikz.tikz_code or "").strip():
                raise ValueError("LLM không trả về tikz_code.")

            # Render graph through tikz_compiler service and inject to question HTML.
            new_question_text = build_question_with_tikz_figure(
                question_text_html=new_question_text_no_svg,
                tikz_code=result_tikz.tikz_code,
                long_description_html=new_long_description,
            )

            new_question_content = {
                "paragraph": sample.get("question", {}).get("paragraph"),
                "question": new_question_text,
                "choices": new_choices,
                "correct_answer": [new_letter],
                "explanation": new_explanation,
            }
        else:
            if not correct_letter:
                raise ValueError("Multiple-choice question requires correct_letter")

            prompt_text = _build_prompt_multiple_choice(
                original_html,
                original_explanation,
                original_choices,
                correct_letter,
                category,
                section,
                difficulty,
                creative_mode=creative_mode,
            )
            if use_openai_basic:
                freeform_prompt = _build_prompt_multiple_choice_freeform(
                    original_html,
                    original_explanation,
                    original_choices,
                    correct_letter,
                    category,
                    section,
                    difficulty,
                    creative_mode=creative_mode,
                )
                result_mc = _invoke_openai_basic_structured(
                    prompt_text=freeform_prompt,
                    output_schema=GeneratedMultipleChoiceContent,
                    api_key=openai_key or "",
                    model=model,
                    temperature=generation_temp,
                    debug_stage_c=debug_stage_c,
                )
            else:
                if llm is None:
                    raise ValueError(
                        "LLM chưa được khởi tạo cho structured-output mode"
                    )
                structured_llm = llm.with_structured_output(
                    GeneratedMultipleChoiceContent
                )
                result_mc: GeneratedMultipleChoiceContent = structured_llm.invoke(  # type: ignore
                    [HumanMessage(content=prompt_text)]
                )
            new_question_text = (result_mc.question or "").strip()
            new_explanation = (result_mc.explanation or "").strip()
            new_choices = result_mc.choices or []
            if len(new_choices) != 4:
                raise ValueError(
                    f"LLM phải trả về đúng 4 choices, nhận được {len(new_choices)}."
                )
            new_choices = [str(c).strip() for c in new_choices[:4]]
            new_letter = (result_mc.correct_answer_letter or "").strip().upper()
            if new_letter not in ("A", "B", "C", "D"):
                raise ValueError(
                    f"correct_answer_letter phải là A, B, C hoặc D, nhận được: {result_mc.correct_answer_letter!r}"
                )
            if not new_question_text:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            new_question_content = {
                "paragraph": sample.get("question", {}).get("paragraph"),
                "question": new_question_text,
                "choices": new_choices,
                "correct_answer": [new_letter],
                "explanation": new_explanation,
            }

    elif generate_full:
        # Không phải multiple-choice hoặc thiếu 4 choices: sinh question, explanation, correct_answer (nội dung)
        # Kiểm tra nếu câu hỏi có đồ thị → dùng luồng xử lý riêng (không truyền SVG vào prompt)
        if graph_spec is not None:
            # ========== LUỒNG XỬ LÝ CÂU HỎI TỰ LUẬN CÓ ĐỒ THỊ ==========
            # Loại bỏ SVG và long description khỏi HTML để giảm token
            question_text_no_svg = _remove_svg_and_long_desc_from_html(original_html)

            print("Free-response question text without SVG:", question_text_no_svg)

            # Convert GraphSpec to dict for JSON serialization
            graph_spec_dict = {
                "x_label": getattr(graph_spec, "x_label", ""),
                "y_label": getattr(graph_spec, "y_label", ""),
                "raw_long_description": getattr(graph_spec, "raw_long_description", ""),
                "long_description_html": getattr(
                    graph_spec, "long_description_html", ""
                ),
            }

            # Dùng prompt riêng cho câu hỏi tự luận có đồ thị
            prompt_text = _build_prompt_graph_free_response(
                question_text_no_svg,
                original_explanation,
                original_correct_answer,
                graph_spec_dict,
                category,
                section,
                difficulty,
            )

            if use_openai_basic:
                freeform_prompt = _build_prompt_graph_free_response_freeform(
                    question_text_no_svg,
                    original_explanation,
                    original_correct_answer,
                    graph_spec_dict,
                    category,
                    section,
                    difficulty,
                )
                result_free_response_text = _invoke_openai_basic_structured(
                    prompt_text=freeform_prompt,
                    output_schema=GeneratedGraphFreeResponseTextContent,
                    api_key=openai_key or "",
                    model=model,
                    temperature=generation_temp,
                    debug_stage_c=debug_stage_c,
                )
            else:
                if llm is None:
                    raise ValueError(
                        "LLM chưa được khởi tạo cho structured-output mode"
                    )
                structured_llm = llm.with_structured_output(
                    GeneratedGraphFreeResponseTextContent
                )
                result_free_response_text: GeneratedGraphFreeResponseTextContent = structured_llm.invoke(  # type: ignore
                    [HumanMessage(content=prompt_text)]
                )

            # Validate kết quả
            new_question_text_no_svg = (
                result_free_response_text.question_text or ""
            ).strip()
            new_explanation = (result_free_response_text.explanation or "").strip()
            new_correct_answer = (result_free_response_text.correct_answer or "").strip()
            new_long_description = (
                result_free_response_text.new_long_description or ""
            ).strip()

            if not new_question_text_no_svg:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            if not new_correct_answer:
                raise ValueError("LLM không trả về correct_answer.")
            if not new_long_description:
                raise ValueError("LLM không trả về new_long_description.")

            tikz_prompt = _build_prompt_tikz_from_free_response_context(
                question_text=new_question_text_no_svg,
                explanation=new_explanation,
                correct_answer=new_correct_answer,
                new_long_description=new_long_description,
                graph_spec=graph_spec_dict,
                category=category,
                section=section,
                difficulty=difficulty,
            )

            if use_openai_basic:
                result_tikz = _invoke_openai_basic_structured(
                    prompt_text=tikz_prompt,
                    output_schema=GeneratedTikzDiagramContent,
                    api_key=openai_key or "",
                    model=model,
                    temperature=max(0.0, min(generation_temp, 0.3)),
                    debug_stage_c=debug_stage_c,
                )
            else:
                if llm is None:
                    raise ValueError(
                        "LLM chưa được khởi tạo cho structured-output mode"
                    )
                tikz_llm = llm.with_structured_output(GeneratedTikzDiagramContent)
                result_tikz: GeneratedTikzDiagramContent = tikz_llm.invoke(  # type: ignore
                    [HumanMessage(content=tikz_prompt)]
                )

            if not (result_tikz.tikz_code or "").strip():
                raise ValueError("LLM không trả về tikz_code.")

            new_question_text = build_question_with_tikz_figure(
                question_text_html=new_question_text_no_svg,
                tikz_code=result_tikz.tikz_code,
                long_description_html=new_long_description,
            )

            new_question_content = {
                "paragraph": sample.get("question", {}).get("paragraph"),
                "question": new_question_text,
                "choices": None,
                "correct_answer": new_correct_answer,
                "explanation": new_explanation,
            }
        else:
            # ========== LUỒNG XỬ LÝ CÂU HỎI TỰ LUẬN KHÔNG CÓ ĐỒ THỊ ==========
            prompt_text = _build_prompt(
                original_html,
                original_explanation,
                original_correct_answer,
                category,
                section,
                q_type,
                difficulty,
                creative_mode=creative_mode,
            )
            if use_openai_basic:
                freeform_prompt = _build_prompt_freeform(
                    original_html,
                    original_explanation,
                    original_correct_answer,
                    category,
                    section,
                    q_type,
                    difficulty,
                    creative_mode=creative_mode,
                )
                result = _invoke_openai_basic_structured(
                    prompt_text=freeform_prompt,
                    output_schema=GeneratedQuestionContent,
                    api_key=openai_key or "",
                    model=model,
                    temperature=generation_temp,
                    debug_stage_c=debug_stage_c,
                )
            else:
                if llm is None:
                    raise ValueError(
                        "LLM chưa được khởi tạo cho structured-output mode"
                    )
                structured_llm = llm.with_structured_output(GeneratedQuestionContent)
                result: GeneratedQuestionContent = structured_llm.invoke(  # type: ignore
                    [HumanMessage(content=prompt_text)]
                )
            new_question_text = (result.question or "").strip()
            new_explanation = (result.explanation or "").strip()
            new_correct_answer = (result.correct_answer or "").strip()
            if not new_question_text:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            if not new_correct_answer:
                raise ValueError("LLM không trả về correct_answer.")
            new_question_content = {
                "paragraph": sample.get("question", {}).get("paragraph"),
                "question": new_question_text,
                "choices": None,
                "correct_answer": new_correct_answer,
                "explanation": new_explanation,
            }
    else:
        # Chỉ có question mẫu, không có explanation/correct_answer → chỉ sinh câu hỏi (tương thích cũ)
        class QuestionOnly(BaseModel):
            question: str = Field(
                description="New question content with proper HTML+MathML"
            )

        if use_openai_basic:
            freeform_prompt = f"""You are an SAT question writer. Change ONLY the numerical values in this question. Keep ALL HTML tags and MathML structure identical.

Sample question:
{original_html}

Generate your response in this format:

QUESTION:
[Question with only numbers changed, preserving all HTML and MathML structure]
"""
            res = _invoke_openai_basic_structured(
                prompt_text=freeform_prompt,
                output_schema=QuestionOnly,
                api_key=openai_key or "",
                model=model,
                temperature=generation_temp,
                debug_stage_c=debug_stage_c,
            )
        else:
            if llm is None:
                raise ValueError("LLM chưa được khởi tạo cho structured-output mode")
            QuestionOnlyModel = llm.with_structured_output(QuestionOnly)
            prompt_question_only = f"""You are an SAT question writer. Change ONLY the numerical values in the sample question below. Do NOT change wording or structure. Output the same HTML + MathML with only numbers substituted.

Sample:
---
{original_html}
---

Return only the new question string (same format, numbers changed)."""
            res = QuestionOnlyModel.invoke([HumanMessage(content=prompt_question_only)])

        new_question_text = (res.question or "").strip() if hasattr(res, "question") else str(res).strip()  # type: ignore
        if not new_question_text:
            raise ValueError("LLM không trả về nội dung câu hỏi.")
        new_question_content = {
            "paragraph": sample.get("question", {}).get("paragraph"),
            "question": new_question_text,
            "choices": None,
            "correct_answer": None,
            "explanation": None,
        }

    # Build full item giống schema questions_practice_test.json
    new_item = {
        "id": str(uuid.uuid4()),
        "subject": sample.get("subject", "SAT"),
        "pool": sample.get("pool", "practice_test"),
        "section": section,
        "category": category,
        "skill": sample.get("skill", ""),
        "difficulty": difficulty,
        "type": q_type,
        "question": new_question_content,
        "image_url": sample.get("image_url"),
    }
    if structured_artifacts:
        new_item["_generation_mode"] = "legacy_with_structured_fallback"
        new_item["_generation_artifacts"] = structured_artifacts
    return new_item


def load_sample_question(
    questions_path: str = "questions_practice_test.json",
    index: int = 0,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load một câu hỏi mẫu từ file JSON (theo index hoặc id)."""
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {questions_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        raise ValueError("Danh sách câu hỏi rỗng.")
    if question_id:
        for item in data:
            if item.get("id") == question_id:
                return item
        raise ValueError(f"Không tìm thấy câu hỏi id: {question_id}")
    return data[index]


def main(sample_question: str, count: int = 1, output_path: Optional[str] = None):
    """
    Sinh câu hỏi mới từ nội dung câu mẫu (HTML + MathML) truyền trực tiếp.

    Args:
        sample_question: Nội dung câu hỏi mẫu dạng str (HTML + MathML).
        count: Số câu hỏi mới sinh từ cùng một mẫu.
        output_path: File JSON để ghi kết quả; None thì in ra stdout.
    """
    sample = {
        "category": "Algebra",
        "section": "Math",
        "type": "multiple-choice",
        "difficulty": "Easy",
        "question": {"question": sample_question.strip(), "paragraph": None},
        "subject": "SAT",
        "pool": "practice_test",
        "skill": "",
        "image_url": None,
    }
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    results = []
    for _ in range(count):
        new_q = generate_new_question(sample, llm=llm)
        results.append(new_q)

    out_json = results[0] if count == 1 else results
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Đã ghi {count} câu hỏi mới vào: {output_path}")
    else:
        print(json.dumps(out_json, ensure_ascii=False, indent=2))
    return out_json


if __name__ == "__main__":
    with open("graph.json", "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    new_q = generate_new_question(graph_data, llm=llm)  # Truyền full object

    with open("new_question.json", "w", encoding="utf-8") as f:
        json.dump(new_q, f, ensure_ascii=False, indent=2)
