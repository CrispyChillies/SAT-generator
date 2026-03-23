#!/usr/bin/env python3
"""
Unified flow for generating SAT questions (Math and Reading & Writing).

Math Flow:
  A (question / explanation / correct_answer)
    ├─→ B: OpenAI solver sinh steps_function_and_meaning.json
   ├─→ C: Gen câu hỏi mới, explanation và đáp án
   └─→ D: Sinh đáp án cho câu hỏi mới (dựa vào file JSON từ B và câu hỏi từ C)

R&W Flow:
  A (paragraph / question / choices / correct_answer / explanation)
   ├─→ B: Solver analyzes original question (reasoning trace)
   ├─→ C: Gen câu hỏi mới với new scenario
   └─→ D: Validate new question (verify answer correctness)

Usage: python run_flow.py [--sample-index N] [--question-id ID] [--questions-path PATH] [--out-dir DIR]
"""

import os
import json
import argparse
import traceback
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# Import modules for Math flow
# ---------------------------------------------------------------------------
from generate_question_langchain import generate_new_question, load_sample_question
from mathml_parser import MathMLParser

try:
    from openai_basic_math_solver import OpenAIBasicMathSolver, solve_with_steps_openai_basic
    OA_BASIC_AVAILABLE = True
except ImportError:
    OA_BASIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import modules for R&W flow
# ---------------------------------------------------------------------------
from generate_rw_question import generate_new_rw_question
from rw_question_solver import solve_rw_question_simple

# ---------------------------------------------------------------------------
# Preprocess: multiple-choice A/B/C/D → giá trị đáp án (nội dung choice)
# ---------------------------------------------------------------------------

CHOICE_LETTERS = ("A", "B", "C", "D")


def _extract_mc_letter(correct_answer: Any) -> Optional[str]:
    letter = correct_answer[0] if isinstance(correct_answer, (list, tuple)) and correct_answer else correct_answer
    if not isinstance(letter, str):
        return None
    letter = letter.strip().upper()
    return letter if letter in CHOICE_LETTERS else None


def _expected_answer_for_solver(correct_answer: Any, choices: List[str]) -> Any:
    """Use choice content (not A/B/C/D letter) as expected answer for solver verification."""
    letter = _extract_mc_letter(correct_answer)
    if not letter:
        return correct_answer
    idx = CHOICE_LETTERS.index(letter)
    if 0 <= idx < len(choices):
        return (choices[idx] or "").strip() or correct_answer
    return correct_answer


def _extract_numeric_value(text: str) -> Optional[float]:
    """Extract one numeric value from text/HTML for robust answer-choice matching."""
    if not text:
        return None
    s = str(text)

    frac = re.search(r"([+-]?\d+)\s*/\s*([+-]?\d+)", s)
    if frac:
        d = int(frac.group(2))
        if d != 0:
            return float(int(frac.group(1)) / d)

    nums = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", s)
    if not nums:
        return None
    token = nums[-1].replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def _map_solver_result_to_choice_letter(
    solver_final_result: Any,
    choices: List[str],
    parser: MathMLParser,
) -> Optional[str]:
    """Map solver final result to choice letter by numeric/text equivalence."""
    if solver_final_result is None or not choices:
        return None

    solver_s = str(solver_final_result).strip()
    solver_num = _extract_numeric_value(solver_s)

    for i, choice_html in enumerate(choices[:4]):
        parsed = parser.parse(choice_html or "") if isinstance(choice_html, str) else {"text": str(choice_html)}
        choice_text = (parsed.get("text") or "").strip()

        if solver_s and choice_text and solver_s in choice_text:
            return CHOICE_LETTERS[i]

        choice_num = _extract_numeric_value(choice_text)
        if solver_num is not None and choice_num is not None and abs(solver_num - choice_num) < 1e-9:
            return CHOICE_LETTERS[i]

    return None


def _rewrite_explanation_after_answer_correction(
    explanation_html: str,
    *,
    old_letter: str,
    new_letter: str,
    old_choice_html: str,
    new_choice_html: str,
    parser: MathMLParser,
) -> str:
    """Best-effort explanation update after letter/value correction."""
    if not explanation_html:
        return explanation_html

    updated = explanation_html
    updated = re.sub(rf"\\bChoice\\s+{re.escape(old_letter)}\\s+is\\s+correct\\b", f"Choice {new_letter} is correct", updated, flags=re.IGNORECASE)
    updated = re.sub(rf"\\bChoice\\s+{re.escape(old_letter)}\\b", f"Choice {new_letter}", updated, flags=re.IGNORECASE)

    old_text = (parser.parse(old_choice_html or "").get("text") or "").strip()
    new_text = (parser.parse(new_choice_html or "").get("text") or "").strip()
    old_num = _extract_numeric_value(old_text)
    new_num = _extract_numeric_value(new_text)

    if old_num is not None and new_num is not None:
        old_int = str(int(old_num)) if float(old_num).is_integer() else str(old_num)
        new_int = str(int(new_num)) if float(new_num).is_integer() else str(new_num)

        old_comma = f"{int(old_num):,}" if float(old_num).is_integer() else old_int
        new_comma = f"{int(new_num):,}" if float(new_num).is_integer() else new_int

        updated = re.sub(rf"(?<!\\d){re.escape(old_int)}(?!\\d)", new_int, updated)
        updated = updated.replace(old_comma, new_comma)

    return updated


def preprocess_correct_answer(sample: Dict[str, Any]) -> Any:
    """
    Nếu câu hỏi là multiple-choice và correct_answer là chữ A, B, C, D
    thì chuyển thành giá trị thực của đáp án (nội dung choice tương ứng).

    Args:
        sample: Item từ questions_practice_test.json (có question.choices, question.correct_answer).

    Returns:
        correct_answer đã chuẩn hóa: hoặc chuỗi nội dung choice (HTML/MathML),
        hoặc giữ nguyên nếu không phải dạng A/B/C/D.
    """
    q_block = sample.get("question") or {}
    choices = q_block.get("choices")
    raw = q_block.get("correct_answer") or sample.get("correct_answer")

    if raw is None:
        return 

    # Lấy chữ cái đầu nếu correct_answer là list (vd: ["C"] -> "C")
    letter = raw[0] if isinstance(raw, (list, tuple)) and len(raw) > 0 else raw
    if not isinstance(letter, str):
        return raw
    letter = letter.strip().upper()

    if letter not in CHOICE_LETTERS or not choices or not isinstance(choices, list):
        return raw

    idx = CHOICE_LETTERS.index(letter)
    if idx >= len(choices):
        return raw

    return (choices[idx] or "").strip() or raw


def _save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _save_math_generation_artifacts(
    *,
    out_dir: Path,
    generation_artifacts: Optional[Dict[str, Any]],
    verbose: bool,
) -> Dict[str, str]:
    """Persist structured generation artifacts and return saved file paths."""
    if not generation_artifacts:
        return {}

    saved: Dict[str, str] = {}
    # Canonical filenames requested by the structured pipeline design.
    mapping = {
        "analysis": "original_analysis.json",
        "blueprint": "problem_blueprint.json",
        "generated_instance": "generated_instance.json",
        "verification": "verification_result.json",
    }

    for key, filename in mapping.items():
        payload = generation_artifacts.get(key)
        if payload is None:
            continue
        path = out_dir / filename
        _save_json(path, payload)
        saved[key] = str(path)

    # Backward-compatible aliases for existing tooling.
    alias_mapping = {
        "analysis": "original_problem_analysis.json",
        "verification": "generation_verification.json",
    }
    for key, alias_name in alias_mapping.items():
        payload = generation_artifacts.get(key)
        if payload is None:
            continue
        alias_path = out_dir / alias_name
        _save_json(alias_path, payload)
        saved[f"{key}_alias"] = str(alias_path)

    full_path = out_dir / "generation_artifacts.json"
    _save_json(full_path, generation_artifacts)
    saved["all"] = str(full_path)

    if verbose and saved:
        print("Saved structured generation artifacts:")
        for key, path in saved.items():
            print(f"  - {key}: {path}")

    return saved


# ---------------------------------------------------------------------------
# Question type detection
# ---------------------------------------------------------------------------

def is_reading_writing_question(sample: Dict[str, Any]) -> bool:
    """
    Determine if a question is Reading & Writing (vs Math).
    
    Args:
        sample: Question dict from questions_practice_test.json
    
    Returns:
        True if R&W question, False if Math question
    """
    section = sample.get("section", "")
    return "reading" in section.lower() and "writing" in section.lower()


# ---------------------------------------------------------------------------
# Math Flow (existing)
# ---------------------------------------------------------------------------

def run_math_flow(
    sample: Dict[str, Any],
    *,
    steps_json_path: str = "steps_function_and_meaning.json",
    out_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4.1",
    verbose: bool = True,
    open_ai_api_key: Optional[str] = None,
    creative_mode: bool = True,
) -> Dict[str, Any]:
    """
    Chạy luồng đầy đủ theo flow.md.

    - A: Lấy question, explanation, correct_answer từ sample.
    - B: OpenAI solver giải bài → sinh file steps_function_and_meaning.json.
    - C: Gen câu hỏi mới, explanation và đáp án từ cùng sample.
    - D: Dùng steps JSON + câu hỏi mới → sat_math_solver sinh đáp án cho câu mới.

    Args:
        sample: Một item từ questions_practice_test.json (có question, explanation, correct_answer).
        steps_json_path: Đường dẫn file JSON cho bước B (và dùng lại ở D).
        out_dir: Thư mục ghi file (steps JSON, câu hỏi mới, kết quả). None = dùng thư mục hiện tại.
        api_key: OpenAI API key. None = lấy từ OPENAI_API_KEY.
        model: Tên model cho LLM.
        verbose: In log chi tiết.

    Returns:
        Dict gồm:
          - steps_json_path: Đường dẫn file steps đã ghi.
          - new_question_item: Câu hỏi mới (dict), gồm question, explanation, correct_answer.
          - new_question_text: Nội dung câu hỏi mới (HTML/string).
          - answer_result: Kết quả từ sat_math_solver (final_result, steps_detail, error, ...).
            - generation_artifacts: Structured artifacts from analysis/blueprint/instance/verification (if available).
            - generation_artifact_paths: Saved artifact JSON paths.
          - error: Lỗi tổng (nếu có).
    """
    # Step B uses OpenAI basic solver.
    if not OA_BASIC_AVAILABLE:
        return {
            "error": "OpenAI basic solver not available. Check openai_basic_math_solver.py import.",
            "steps_json_path": None,
            "new_question_item": None,
            "new_question_text": None,
            "answer_result": None,
        }
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "error": "Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền api_key.",
            "steps_json_path": None,
            "new_question_item": None,
            "new_question_text": None,
            "answer_result": None,
        }

    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = Path(steps_json_path)
    if out_dir and not steps_path.is_absolute():
        steps_path = out_dir / steps_path.name

    parser = MathMLParser()

    # --- A: Trích question, explanation, correct_answer từ sample ---
    q_block = sample.get("question") or {}
    question_html = (q_block.get("question") or "").strip()
    explanation = (q_block.get("explanation") or "").strip()
    correct_answer = preprocess_correct_answer(sample)

    if not question_html:
        return {
            "error": "Sample không có nội dung question.",
            "steps_json_path": None,
            "new_question_item": None,
            "new_question_text": None,
            "answer_result": None,
        }
    if not explanation:
        return {
            "error": "Sample không có explanation.",
            "steps_json_path": None,
            "new_question_item": None,
            "new_question_text": None,
            "answer_result": None,
        }

    result_bag = {
        "steps_json_path": str(steps_path),
        "new_question_item": None,
        "new_question_text": None,
        "answer_result": None,
        "step_b_trace": None,
        "generation_artifacts": None,
        "generation_artifact_paths": {},
        "solver_verification": None,
        "error": None,
    }

    # --- B: OpenAI solver sinh steps_function_and_meaning.json ---
    if verbose:
        print("\n" + "=" * 70)
        solver_name = "OpenAI Basic Solver"
        print(f"B: {solver_name} sinh steps_function_and_meaning.json")
        print("=" * 70)
    try:
        # Use OpenAI direct inference solver (no tool-calling orchestration).
        agent = OpenAIBasicMathSolver(
            api_key=api_key,
            verification_api_key=open_ai_api_key,
            model=model,
            verbose=verbose,
        )
        trace = agent.solve(
            question=question_html,
            mathml_explanation=explanation,
            correct_answer=correct_answer,
            steps_json_path=str(steps_path),
        )
        
        if trace.error:
            result_bag["error"] = f"Agent: {trace.error}"
            if verbose:
                print("Agent error:", trace.error)
            return result_bag

        # Surface full Step-B solver trace for UI/debug.
        if hasattr(trace, "to_dict"):
            try:
                result_bag["step_b_trace"] = trace.to_dict()
            except Exception:
                result_bag["step_b_trace"] = None
        else:
            # Fallback serialization for custom trace objects.
            try:
                result_bag["step_b_trace"] = {
                    "final_result": getattr(trace, "final_result", None),
                    "is_correct": getattr(trace, "is_correct", None),
                    "error": getattr(trace, "error", None),
                    "steps": [
                        {
                            "step_number": getattr(s, "step_number", i + 1),
                            "thought": getattr(s, "thought", ""),
                            "tool_name": getattr(s, "tool_name", ""),
                            "tool_output": getattr(s, "tool_output", None),
                        }
                        for i, s in enumerate(getattr(trace, "steps", []) or [])
                    ],
                }
            except Exception:
                result_bag["step_b_trace"] = None

        if verbose and result_bag.get("step_b_trace"):
            tb = result_bag["step_b_trace"]
            steps = tb.get("steps") or []
            print("\nStep B trace summary:")
            print(f"  steps={len(steps)} final={tb.get('final_result')} is_correct={tb.get('is_correct')}")
            if steps:
                thought_preview = (steps[0].get("thought") or "").strip().replace("\n", " ")
                if thought_preview:
                    print("  thought:", thought_preview[:220] + ("..." if len(thought_preview) > 220 else ""))
    except Exception as e:
        result_bag["error"] = f"Agent: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag

    # --- C: Gen câu hỏi mới + explanation + đáp án ---
    if verbose:
        print("\n" + "=" * 70)
        generator_name = "OpenAI Basic Generator"
        print(f"C: {generator_name} sinh câu hỏi mới, explanation và đáp án")
        print("=" * 70)
    try:
        new_question_item = generate_new_question(
            sample,
            use_openai_basic=True,
            api_key=api_key,
            model=model,
            creative_mode=creative_mode,
            step_b_trace=result_bag.get("step_b_trace"),
        )
        result_bag["new_question_item"] = new_question_item
        generation_artifacts = new_question_item.get("_generation_artifacts") if isinstance(new_question_item, dict) else None
        result_bag["generation_artifacts"] = generation_artifacts
        out_dir_path = out_dir if isinstance(out_dir, Path) else Path(out_dir or ".")
        result_bag["generation_artifact_paths"] = _save_math_generation_artifacts(
            out_dir=out_dir_path,
            generation_artifacts=generation_artifacts,
            verbose=verbose,
        )
        new_q_block = new_question_item.get("question") or {}
        new_question_text = (new_q_block.get("question") or "").strip()
        new_explanation = (new_q_block.get("explanation") or "").strip()
        new_correct_answer = new_q_block.get("correct_answer")
        new_choices = new_q_block.get("choices") or []
        expected_answer_for_solver = _expected_answer_for_solver(new_correct_answer, new_choices)
        result_bag["new_question_text"] = new_question_text
        if verbose and new_question_text:
            print("Câu hỏi mới (đoạn đầu):", new_question_text[:200] + "..." if len(new_question_text) > 200 else new_question_text)
        if verbose and new_choices:
            print("4 choices (multiple-choice):", len(new_choices), "đáp án")
        if verbose and new_explanation:
            print("Explanation (đoạn đầu):", new_explanation[:200] + "..." if len(new_explanation) > 200 else new_explanation)
        if verbose and new_correct_answer is not None:
            if isinstance(new_correct_answer, (list, tuple)) and new_correct_answer:
                letter = new_correct_answer[0]
                print("Đáp án đúng (correct_answer):", letter, end="")
                if new_choices and letter in ("A", "B", "C", "D"):
                    idx = ["A", "B", "C", "D"].index(letter)
                    if idx < len(new_choices):
                        content = (new_choices[idx] or "").strip()
                        print(" →", content[:80] + "..." if len(content) > 80 else content)
                    else:
                        print()
                else:
                    print()
            else:
                s = str(new_correct_answer).strip()
                print("Đáp án (correct_answer):", s[:150] + "..." if len(s) > 150 else s)
    except Exception as e:
        result_bag["error"] = f"Gen câu hỏi: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag

    # --- D: Sinh đáp án cho câu hỏi mới (dựa vào steps JSON) ---
    if verbose:
        print("\n" + "=" * 70)
        solver_name = "OpenAI Basic Solver"
        print(f"D: {solver_name} sinh đáp án cho câu hỏi mới")
        print("=" * 70)
    try:
        # Use OpenAI direct inference solver.
        answer_result = solve_with_steps_openai_basic(
            question=new_question_text,
            steps_path=str(steps_path),
            new_correct_answer=expected_answer_for_solver,
            api_key=api_key,
            model=model,
            parser=parser,
            verbose=verbose,
        )
        
        result_bag["answer_result"] = answer_result

        # Reconcile multiple-choice answer when solver result maps to a different choice.
        current_letter = _extract_mc_letter(new_correct_answer)
        solver_mapped_letter = _map_solver_result_to_choice_letter(
            answer_result.get("final_result"),
            new_choices,
            parser,
        )
        if current_letter and solver_mapped_letter and solver_mapped_letter != current_letter:
            old_idx = CHOICE_LETTERS.index(current_letter)
            new_idx = CHOICE_LETTERS.index(solver_mapped_letter)
            old_choice_html = new_choices[old_idx] if old_idx < len(new_choices) else ""
            new_choice_html = new_choices[new_idx] if new_idx < len(new_choices) else ""

            new_q_block["correct_answer"] = [solver_mapped_letter]
            new_q_block["explanation"] = _rewrite_explanation_after_answer_correction(
                new_q_block.get("explanation") or "",
                old_letter=current_letter,
                new_letter=solver_mapped_letter,
                old_choice_html=old_choice_html,
                new_choice_html=new_choice_html,
                parser=parser,
            )
            result_bag["new_question_item"]["question"] = new_q_block

            if verbose:
                print(
                    f"Reconciled correct_answer from {current_letter} to {solver_mapped_letter} "
                    f"based on solver final_result={answer_result.get('final_result')}"
                )

        solver_verification = {
            "is_solvable": not bool(answer_result.get("error")),
            "solver_final_result": answer_result.get("final_result"),
            "solver_is_correct_vs_generated_answer": answer_result.get("is_correct"),
            "solver_error": answer_result.get("error"),
            "answer_format": "multiple_choice" if isinstance(new_correct_answer, list) else "free_response",
            "reasoning_alignment_hint": "Check generation_verification.json and Step-B trace for strategy alignment.",
            "solver_mapped_correct_letter": solver_mapped_letter,
            "generated_correct_letter_before_reconcile": current_letter,
            "reconciled": bool(current_letter and solver_mapped_letter and solver_mapped_letter != current_letter),
        }
        result_bag["solver_verification"] = solver_verification
        try:
            out_dir_path = out_dir if isinstance(out_dir, Path) else Path(out_dir or ".")
            _save_json(out_dir_path / "solver_verification.json", solver_verification)
        except Exception:
            pass

        if answer_result.get("error"):
            result_bag["error"] = result_bag["error"] or ""
            if result_bag["error"]:
                result_bag["error"] += "; "
            result_bag["error"] += f"Sinh đáp án: {answer_result['error']}"
        elif verbose:
            print("Final result (câu mới):", answer_result.get("final_result"))
    except Exception as e:
        result_bag["error"] = (result_bag["error"] or "") + f"; Sinh đáp án: {e}"
        if verbose:
            import traceback
            traceback.print_exc()

    return result_bag


# ---------------------------------------------------------------------------
# Reading & Writing Flow
# ---------------------------------------------------------------------------

def run_rw_flow(
    sample: Dict[str, Any],
    *,
    out_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run R&W question generation and validation flow.
    
    Flow:
      A: Extract original question, paragraph, choices, answer, explanation
      B: Solver analyzes original question → reasoning trace
      C: Generator creates new question (new scenario, same skill/reasoning)
      D: Solver validates new question → verifies answer is correct
    
    Args:
        sample: Sample R&W question from questions_practice_test.json
        out_dir: Output directory for saved files
        api_key: OpenAI API key (None = use OPENAI_API_KEY env)
        model: LLM model name
        verbose: Print detailed logs
    
    Returns:
        Dict with:
        - steps_json_path: None (R&W doesn't use steps, kept for consistency)
        - new_question_item: Generated question dict
        - new_question_text: Generated question text
        - answer_result: Validation result (renamed from validation_result for consistency)
        - error: Error message if any
        - _rw_original_analysis: Original question analysis (internal/debug)
    """
    out_dir = Path(out_dir) if out_dir else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "error": "OPENAI_API_KEY not set in environment",
            "original_analysis": None,
            "new_question_item": None,
            "validation_result": None,
        }    
    result_bag = {
        "steps_json_path": None,  # R&W doesn't use steps, but keep for consistency
        "new_question_item": None,
        "new_question_text": None,
        "answer_result": None,  # Renamed from validation_result for consistency
        "error": None,
        "_rw_original_analysis": None,  # Internal field for debugging
    }
    
    # --- A: Extract original question components ---
    q_block = sample.get("question", {})
    paragraph = (q_block.get("paragraph") or "").strip()
    question = (q_block.get("question") or "").strip()
    choices = q_block.get("choices") or []
    correct_answer = q_block.get("correct_answer")
    explanation = (q_block.get("explanation") or "").strip()
    
    skill = sample.get("skill", "")
    category = sample.get("category", "")
    difficulty = sample.get("difficulty", "Medium")
    
    if not paragraph or not question or not choices:
        result_bag["error"] = "Sample question missing required fields (paragraph, question, or choices)"
        return result_bag
    
    # Extract correct answer letter
    if isinstance(correct_answer, list) and len(correct_answer) > 0:
        correct_letter = correct_answer[0].strip().upper()
    elif isinstance(correct_answer, str):
        correct_letter = correct_answer.strip().upper()
    else:
        correct_letter = "A"
    
    if verbose:
        print("=" * 70)
        print("ORIGINAL R&W QUESTION")
        print("=" * 70)
        print(f"ID: {sample.get('id')}")
        print(f"Skill: {skill}")
        print(f"Category: {category}")
        print(f"Difficulty: {difficulty}")
        print(f"Correct Answer: {correct_letter}")
        print(f"Paragraph (first 150 chars): {paragraph[:150]}...")
        print(f"Question: {question}")
        print()
    
    # --- B: Analyze original question ---
    if verbose:
        print("=" * 70)
        print("B: ANALYZING ORIGINAL QUESTION")
        print("=" * 70)
    
    try:
        original_analysis = solve_rw_question_simple(
            paragraph=paragraph,
            question=question,
            choices=choices,
            skill=skill,
            api_key=api_key,
            model=model,
            verbose=verbose,
        )
        result_bag["_rw_original_analysis"] = original_analysis
        
        if original_analysis.get("error"):
            if verbose:
                print(f"Warning: {original_analysis['error']}")
        else:
            solver_answer = original_analysis.get("final_answer_letter")
            if verbose:
                print(f"Solver's answer: {solver_answer}")
                print(f"Correct answer: {correct_letter}")
                if solver_answer == correct_letter:
                    print("✓ Solver got the correct answer!")
                else:
                    print("✗ Solver's answer differs from correct answer")
        
        # Save reasoning trace
        if out_dir:
            trace_path = out_dir / "original_question_reasoning.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(original_analysis, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"Saved reasoning trace to {trace_path}")
    
    except Exception as e:
        result_bag["error"] = f"Error analyzing original: {e}"
        if verbose:
            print(f"Error: {e}")
            traceback.print_exc()
        return result_bag
    
    # --- C: Generate new question ---
    if verbose:
        print("\n" + "=" * 70)
        print("C: GENERATING NEW R&W QUESTION")
        print("=" * 70)
    
    try:
        from langchain_openai import ChatOpenAI
        llm_kwargs = {"model": model, "api_key": api_key}
        if "gpt-5" not in (model or "").lower():
            llm_kwargs["temperature"] = 0.7
        llm = ChatOpenAI(**llm_kwargs)
        new_question = generate_new_rw_question(
            sample, 
            llm=llm, 
            api_key=api_key,
            model=model,
            verbose=verbose
        )
        result_bag["new_question_item"] = new_question
        
        new_q_block = new_question.get("question", {})
        new_paragraph = (new_q_block.get("paragraph") or "").strip()
        new_question_text = (new_q_block.get("question") or "").strip()
        new_choices = new_q_block.get("choices") or []
        new_correct_answer = new_q_block.get("correct_answer")
        new_explanation = (new_q_block.get("explanation") or "").strip()
        
        # Store new_question_text in result_bag for consistency with Math flow
        result_bag["new_question_text"] = new_question_text
        
        if isinstance(new_correct_answer, list) and len(new_correct_answer) > 0:
            new_correct_letter = new_correct_answer[0].strip().upper()
        elif isinstance(new_correct_answer, str):
            new_correct_letter = new_correct_answer.strip().upper()
        else:
            new_correct_letter = "A"
        
        if verbose:
            print(f"Generated new question:")
            print(f"  Skill: {new_question.get('skill')}")
            print(f"  Difficulty: {new_question.get('difficulty')}")
            print(f"  Correct Answer: {new_correct_letter}")
            print(f"  Paragraph (first 150 chars): {new_paragraph[:150]}...")
            print(f"  Question: {new_question_text}")
            print()
    
    except Exception as e:
        result_bag["error"] = f"Error generating question: {e}"
        if verbose:
            print(f"Error: {e}")
            traceback.print_exc()
        return result_bag
    
    # --- D: Validate new question ---
    if verbose:
        print("\n" + "=" * 70)
        print("D: VALIDATING NEW QUESTION")
        print("=" * 70)
    
    try:
        validation_result = solve_rw_question_simple(
            paragraph=new_paragraph,
            question=new_question_text,
            choices=new_choices,
            skill=skill,
            api_key=api_key,
            model=model,
            verbose=verbose,
        )
        # Store as answer_result for consistency with Math flow
        result_bag["answer_result"] = validation_result
        
        if validation_result.get("error"):
            if verbose:
                print(f"Warning: {validation_result['error']}")
        else:
            validator_answer = validation_result.get("final_answer_letter")
            validator_result = validation_result.get("final_result")
            validator_answer_text = validation_result.get("answer_text")
            
            if verbose:
                print(f"Validator's answer: {validator_answer}")
                print(f"Generated correct answer: {new_correct_letter}")
                if validator_result:
                    print(f"Validator's choice text (first 100 chars): {validator_result[:100]}...")
                if validator_answer_text:
                    print(f"Answer text: {validator_answer_text[:150]}...")
                if validator_answer == new_correct_letter:
                    print("✓ Validator confirms the generated answer is correct!")
                else:
                    print("✗ Validator's answer differs - may need review")
        
        # Save validation result
        if out_dir:
            val_path = out_dir / "new_question_validation.json"
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(validation_result, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"Saved validation result to {val_path}")
    
    except Exception as e:
        result_bag["error"] = (result_bag["error"] or "") + f"; Validation error: {e}"
        if verbose:
            print(f"Error: {e}")
            traceback.print_exc()
    
    return result_bag


# ---------------------------------------------------------------------------
# Unified Flow - Routes to Math or R&W based on question type
# ---------------------------------------------------------------------------

def run_flow(
    sample: Dict[str, Any],
    *,
    steps_json_path: str = "steps_function_and_meaning.json",
    out_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
    open_ai_api_key: Optional[str] = None,
    creative_mode: bool = True,
) -> Dict[str, Any]:
    """
    Unified flow that automatically routes to Math or R&W generation based on question type.
    
    Args:
        sample: Question from questions_practice_test.json
        steps_json_path: Path for steps JSON (Math questions only)
        out_dir: Output directory
        api_key: OpenAI API key
        model: LLM model name
        verbose: Print detailed logs
        creative_mode: If True, generate new scenarios testing same skill. If False, only change numbers.
    
    Returns:
        Dict with results (structure depends on question type)
    """
    # Detect question type
    is_rw = is_reading_writing_question(sample)
    
    if verbose:
        q_type = "Reading & Writing" if is_rw else "Math"
        print(f"\n{'=' * 70}")
        print(f"DETECTED QUESTION TYPE: {q_type}")
        print(f"{'=' * 70}\n")
    
    # Route to appropriate flow
    if is_rw:
        return run_rw_flow(
            sample=sample,
            out_dir=out_dir,
            api_key=api_key,
            model=model,
            verbose=verbose,
        )
    else:
        return run_math_flow(
            sample=sample,
            steps_json_path=steps_json_path,
            out_dir=out_dir,
            api_key=api_key,
            model=model,
            verbose=verbose,
            open_ai_api_key=open_ai_api_key,
            creative_mode=creative_mode,
        )


# ---------------------------------------------------------------------------
# Batch Flow - Generate multiple questions from one sample
# ---------------------------------------------------------------------------

def run_flow_batch(
    sample: Dict[str, Any],
    count: int = 1,
    *,
    steps_json_path: str = "steps_function_and_meaning.json",
    out_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
    open_ai_api_key: Optional[str] = None,
    creative_mode: bool = True,
):
    """
    Generator that runs run_flow `count` times on the same sample.
    Yields dicts: { "index": i, "total": count, "result": <run_flow result> }
    so callers can stream each result as it finishes.

    Args:
        sample: Single source question.
        count: How many new questions to generate.
        (all other kwargs forwarded to run_flow)
    """
    base_out = Path(out_dir) if out_dir else Path("output")
    for i in range(count):
        run_out_dir = base_out / f"q_{i + 1}"
        run_out_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"\n{'#' * 70}")
            print(f"# Batch {i + 1}/{count}")
            print(f"{'#' * 70}")
        result = run_flow(
            sample,
            steps_json_path=steps_json_path,
            out_dir=str(run_out_dir),
            api_key=api_key,
            model=model,
            verbose=verbose,
            open_ai_api_key=open_ai_api_key,
            creative_mode=creative_mode,
        )
        yield {"index": i, "total": count, "result": result}


def main():

    ap = argparse.ArgumentParser(
        description="Unified flow for generating SAT questions (Math and Reading & Writing). "
                    "Automatically detects question type and routes to appropriate generation pipeline."
    )
    ap.add_argument("--sample-index", type=int, default=0, help="Index câu mẫu trong file questions (mặc định 0)")
    ap.add_argument("--question-id", type=str, default=None, help="Lấy câu mẫu theo id thay vì index")
    ap.add_argument("--questions-path", type=str, default="questions_practice_test.json", help="Đường dẫn file danh sách câu hỏi")
    ap.add_argument("--out-dir", type=str, default=None, help="Thư mục ghi kết quả (mặc định: output/)")
    ap.add_argument("--steps-json", type=str, default="steps_function_and_meaning.json", help="Tên file steps JSON (Math questions only)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Model LLM")
    ap.add_argument("--open-ai-api-key", type=str, default=None, help="OpenAI API key (hoặc dùng biến môi trường OPENAI_API_KEY)")
    ap.add_argument("--conservative-mode", action="store_true", help="Chỉ thay đổi số liệu, giữ nguyên context (mặc định: tạo scenario mới với cùng skill)")
    ap.add_argument("--quiet", action="store_true", help="Giảm log")
    ap.add_argument("--save-result", type=str, default=None, help="Lưu kết quả flow ra file JSON")
    ap.add_argument("--count", type=int, default=1, help="Số câu hỏi mới cần tạo trong một lần chạy (mặc định: 1)")
    args = ap.parse_args()

    sample = load_sample_question(
        questions_path=args.questions_path,
        index=args.sample_index,
        question_id=args.question_id,
    )

    # Set default output directory
    out_dir = args.out_dir if args.out_dir else "output"
    count = max(1, args.count)

    batch_kwargs = dict(
        steps_json_path=args.steps_json,
        out_dir=out_dir,
        model=args.model,
        verbose=not args.quiet,
        open_ai_api_key=args.open_ai_api_key,
        creative_mode=not args.conservative_mode,
    )

    results = []
    had_error = False
    for item in run_flow_batch(sample, count=count, **batch_kwargs):
        idx = item["index"]
        total = item["total"]
        result = item["result"]
        results.append(result)

        # tqdm-style progress bar
        done = idx + 1
        bar_len = 30
        filled = int(bar_len * done / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        status = "✓" if not result.get("error") else "✗"
        print(f"\n[{bar}] {done}/{total}  {status}")

        if result.get("error"):
            print(f"  Lỗi câu {done}: {result['error']}")
            had_error = True

        if args.save_result:
            base, ext = os.path.splitext(args.save_result)
            save_path = f"{base}_{done}{ext}" if count > 1 else args.save_result
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Đã lưu kết quả: {save_path}")

    if not args.quiet:
        print("\n" + "=" * 70)
        ok = sum(1 for r in results if not r.get("error"))
        print(f"✓ Hoàn tất: {ok}/{count} câu được tạo thành công.")
        print("=" * 70)

    return 1 if had_error else 0


if __name__ == "__main__":
    exit(main())
