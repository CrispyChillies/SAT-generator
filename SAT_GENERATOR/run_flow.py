#!/usr/bin/env python3
"""
Unified flow for generating SAT questions (Math and Reading & Writing).

Math Flow:
  A (question / explanation / correct_answer)
   ├─→ B: Agent sinh steps_function_and_meaning.json
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
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# Import modules for Math flow
# ---------------------------------------------------------------------------
from generate_question_langchain import generate_new_question, load_sample_question
from agent import LangGraphMathAgent
from sat_math_solver import solve_with_steps
from mathml_parser import MathMLParser

# HuggingFace solver (optional alternative to OpenAI)
try:
    from huggingface_math_solver import HuggingFaceMathSolver, solve_with_steps_hf
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Import modules for R&W flow
# ---------------------------------------------------------------------------
from generate_rw_question import generate_new_rw_question
from rw_question_solver import solve_rw_question_simple

# ---------------------------------------------------------------------------
# Preprocess: multiple-choice A/B/C/D → giá trị đáp án (nội dung choice)
# ---------------------------------------------------------------------------

CHOICE_LETTERS = ("A", "B", "C", "D")


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


# ---------------------------------------------------------------------------
# Helper functions for Math flow
# ---------------------------------------------------------------------------

def extract_solving_method(trace: Any, verbose: bool = False) -> str:
    """
    Extract the solving method/approach from the execution trace.
    
    Args:
        trace: ExecutionTrace from solver
        verbose: Print detailed logs
    
    Returns:
        String describing the solving method/approach
    """
    try:
        # Strategy 1: Look for explicit "Solving Method/Approach" section in response
        if hasattr(trace, 'steps') and trace.steps:
            for step in trace.steps:
                if hasattr(step, 'thought') and step.thought:
                    # Search for the structured "Solving Method/Approach" section
                    import re
                    method_match = re.search(
                        r'\*\*Solving Method[:/]Approach[:\*]*\*?\*?\s*(.+?)(?:\*\*|$)',
                        step.thought,
                        re.IGNORECASE | re.DOTALL
                    )
                    
                    if method_match:
                        method_text = method_match.group(1).strip()
                        # Clean up - remove "Step-by-Step Solution" marker if present
                        method_text = re.sub(r'Step-by-Step Solution.*', '', method_text, flags=re.IGNORECASE | re.DOTALL)
                        method_text = method_text.strip()
                        
                        if method_text and len(method_text) > 20:  # Ensure it's substantial
                            if verbose:
                                print(f"✓ Extracted solving method from structured section ({len(method_text)} chars)")
                            return method_text
        
        # Strategy 2: Extract first few meaningful lines (fallback)
        if hasattr(trace, 'steps') and trace.steps:
            method_parts = []
            
            for step in trace.steps:
                if hasattr(step, 'thought') and step.thought:
                    thought_lines = step.thought.strip().split('\n')
                    for line in thought_lines[:15]:
                        if line.strip() and any(keyword in line.lower() for keyword in 
                               ['method', 'approach', 'strategy', 'use', 'apply', 'theorem', 'formula', 
                                'identify', 'calculate', 'solve for', 'given', 'find']):
                            method_parts.append(line.strip())
                    
                    if method_parts:
                        break
            
            if method_parts:
                solving_method = '\n'.join(method_parts[:6])  # Limit to 6 key lines
                if verbose:
                    print(f"✓ Extracted solving method from thought patterns ({len(solving_method)} chars)")
                return solving_method
        
        # Fallback: use general description
        if verbose:
            print("⚠️ Could not extract specific solving method, using fallback")
        return "Apply the same mathematical approach as the original problem."
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Error extracting solving method: {e}")
        return "Apply the same mathematical approach as the original problem."


def generate_distracting_choices(
    question: str,
    correct_answer: Any,
    api_key: str,
    model: str = "gpt-4o-mini",
    verbose: bool = False
) -> list:
    """
    Generate 3 distracting choices for a math question given the correct answer.
    
    Args:
        question: The question text
        correct_answer: The correct answer
        api_key: OpenAI API key
        model: LLM model name
        verbose: Print detailed logs
    
    Returns:
        List of 3 distracting choices (wrong answers that look plausible)
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are helping create a multiple-choice SAT math question.

Question: {question}

Correct Answer: {correct_answer}

Generate 3 plausible but INCORRECT answer choices that:
1. Are mathematically distinct from the correct answer
2. Represent common mistakes or misconceptions
3. Are in the same format as the correct answer
4. Look realistic and challenging for students

Provide ONLY 3 choices, one per line, with no labels or explanations."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert SAT math question writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        distractors = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Take first 3 non-empty lines
        distractors = distractors[:3]
        
        # Pad with generic distractors if needed
        while len(distractors) < 3:
            distractors.append(f"Option {len(distractors) + 1}")
        
        if verbose:
            print(f"Generated {len(distractors)} distracting choices")
        
        return distractors
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Error generating distractors: {e}")
        # Return generic distractors as fallback
        return ["Option A", "Option B", "Option C"]


def generate_short_explanation(
    question: str,
    correct_answer: Any,
    sample_explanation: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    verbose: bool = False
) -> str:
    """
    Generate a short explanation for the math question solution.
    
    Args:
        question: The question text
        correct_answer: The correct answer
        sample_explanation: Example explanation from original question for style reference
        api_key: OpenAI API key
        model: LLM model name
        verbose: Print detailed logs
    
    Returns:
        Short explanation text (similar to sample format)
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are writing a concise explanation for an SAT math question.

Question: {question}

Correct Answer: {correct_answer}

Example explanation style (from a similar question):
{sample_explanation}

IMPORTANT - MathML Formatting Rules:
1. Wrap ALL mathematical expressions in <math>...</math> tags
2. Use <mn>number</mn> for numbers: <math><mn>5</mn></math>
3. Use <mi>variable</mi> for variables: <math><mi>x</mi></math>
4. Use <mo>operator</mo> for operators: <math><mn>3</mn><mo>+</mo><mn>4</mn></math>
5. For fractions: <math><mfrac><mn>numerator</mn><mn>denominator</mn></mfrac></math>
6. Match the exact MathML style shown in the example above

Examples of correct MathML:
- "The value is <math><mn>42</mn></math>."
- "Solve for <math><mi>x</mi></math>: <math><mi>x</mi><mo>=</mo><mn>7</mn></math>"
- "Calculate <math><mo>(</mo><mn>3</mn><mo>+</mo><mn>4</mn><mo>)</mo><mo>×</mo><mn>2</mn><mo>=</mo><mn>14</mn></math>"

Write a SHORT, clear explanation (2-4 sentences) showing the key steps to solve this problem.
Wrap ALL numbers, variables, and mathematical operations in proper MathML tags as shown above."""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert SAT math tutor. Write clear, concise explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        
        explanation = response.choices[0].message.content.strip()
        
        if verbose:
            print(f"Generated explanation ({len(explanation)} chars)")
        
        return explanation
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Error generating explanation: {e}")
        return f"The correct answer is {correct_answer}."


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
    use_hf_solver: bool = False,
    hf_api_key: Optional[str] = None,
    open_ai_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chạy luồng đầy đủ theo flow.md.

    - A: Lấy question, explanation, correct_answer từ sample.
    - B: Agent giải bài → sinh file steps_function_and_meaning.json.
    - C: Gen câu hỏi mới, explanation và đáp án từ cùng sample.
    - D: Dùng steps JSON + câu hỏi mới → sat_math_solver sinh đáp án cho câu mới.

    Args:
        sample: Một item từ questions_practice_test.json (có question, explanation, correct_answer).
        steps_json_path: Đường dẫn file JSON cho bước B (và dùng lại ở D).
        out_dir: Thư mục ghi file (steps JSON, câu hỏi mới, kết quả). None = dùng thư mục hiện tại.
        api_key: OpenAI API key. None = lấy từ OPENAI_API_KEY.
        model: Tên model cho LLM.
        verbose: In log chi tiết.
        use_hf_solver: Nếu True, dùng HuggingFace solver thay vì OpenAI.
        hf_api_key: HuggingFace API key (nếu dùng HF solver). None = lấy từ HF_API_KEY.

    Returns:
        Dict gồm:
          - steps_json_path: Đường dẫn file steps đã ghi.
          - new_question_item: Câu hỏi mới (dict), gồm question, explanation, correct_answer.
          - new_question_text: Nội dung câu hỏi mới (HTML/string).
          - answer_result: Kết quả từ sat_math_solver (final_result, steps_detail, error, ...).
          - error: Lỗi tổng (nếu có).
    """
    # Check API keys based on solver choice
    if use_hf_solver:
        if not HF_AVAILABLE:
            return {
                "error": "HuggingFace solver not available. Check huggingface_math_solver.py import.",
                "steps_json_path": None,
                "new_question_item": None,
                "new_question_text": None,
                "answer_result": None,
            }
        hf_api_key = hf_api_key or os.getenv("HF_API_KEY")
        if not hf_api_key:
            return {
                "error": "Cần đặt HF_API_KEY trong môi trường hoặc truyền hf_api_key.",
                "steps_json_path": None,
                "new_question_item": None,
                "new_question_text": None,
                "answer_result": None,
            }
    else:
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

    # question_text = parser.parse(question_html) if question_html else question_html
    parsed = parser.parse(question_html) if question_html else question_html
    question_text = parsed['text']
    # graph = parsed['graph']

    result_bag = {
        "steps_json_path": str(steps_path),
        "new_question_item": None,
        "new_question_text": None,
        "answer_result": None,
        "solving_method": None,  # Store solving method from step B
        "error": None,
    }

    # --- B: Agent sinh steps_function_and_meaning.json ---
    if verbose:
        print("\n" + "=" * 70)
        solver_name = "HuggingFace Solver" if use_hf_solver else "LangGraph Agent"
        print(f"B: {solver_name} sinh steps_function_and_meaning.json")
        print("=" * 70)
    try:
        if use_hf_solver:
            # Use HuggingFace solver
            agent = HuggingFaceMathSolver(
                api_key=hf_api_key,
                openai_api_key=open_ai_api_key or os.getenv("OPENAI_API_KEY"), 
                model="zai-org/GLM-Z1-9B-0414:featherless-ai",
                verbose=verbose
            )
            trace = agent.solve(
                question=question_html,
                mathml_explanation=explanation,
                correct_answer=correct_answer,
                steps_json_path=str(steps_path),
            )
        else:
            # Use original OpenAI-based LangGraph agent
            agent = LangGraphMathAgent(api_key=api_key, model=model, verbose=verbose)
            trace = agent.solve(
                question=question_text,
                mathml_explanation=explanation,
                correct_answer=correct_answer,
                steps_json_path=str(steps_path),
            )
        
        if trace.error:
            result_bag["error"] = f"Agent: {trace.error}"
            if verbose:
                print("Agent error:", trace.error)
            return result_bag
        
        # Extract solving method from trace
        solving_method = extract_solving_method(trace, verbose=verbose)
        result_bag["solving_method"] = solving_method
        
        if verbose:
            print(f"\n📋 Extracted solving method:")
            print(f"  {solving_method[:200]}..." if len(solving_method) > 200 else f"  {solving_method}")
    
    except Exception as e:
        result_bag["error"] = f"Agent: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag

    # --- C: Gen câu hỏi mới (question only, no choices/explanation yet) ---
    if verbose:
        print("\n" + "=" * 70)
        print("C: Gen câu hỏi mới")
        print("=" * 70)
    try:
        new_question_item = generate_new_question(sample)
        new_q_block = new_question_item.get("question") or {}
        new_question_text = (new_q_block.get("question") or "").strip()
        
        if verbose and new_question_text:
            print("Câu hỏi mới (đoạn đầu):", new_question_text[:200] + "..." if len(new_question_text) > 200 else new_question_text)
    except Exception as e:
        result_bag["error"] = f"Gen câu hỏi: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag

    # --- D: Sinh đáp án cho câu hỏi mới (áp dụng phương pháp từ B) ---
    if verbose:
        print("\n" + "=" * 70)
        solver_name = "HuggingFace Solver" if use_hf_solver else "sat_math_solver"
        print(f"D: {solver_name} áp dụng phương pháp giải từ bước B")
        print("=" * 70)
    try:
        # Get the effective API key for later use
        effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
        solving_method = result_bag.get("solving_method", "")
        
        # Create guided question with solving method
        if solving_method:
            guided_question = f"""Question: {new_question_text}

IMPORTANT - Use this solving method/approach:
{solving_method}

Apply the method described above to solve this NEW question. Follow the same strategy but with the new numbers and context provided in the question."""
            if verbose:
                print("\n📋 Using solving method from step B")
                print(f"  Method: {solving_method[:150]}..." if len(solving_method) > 150 else f"  Method: {solving_method}")
        else:
            guided_question = new_question_text
        
        if use_hf_solver:
            answer_result = solve_with_steps_hf(
                question=guided_question,
                steps_path=str(steps_path),
                new_correct_answer=None,  # No expected answer yet
                api_key=hf_api_key,
                openai_api_key=open_ai_api_key or os.getenv("OPENAI_API_KEY"),
                model="zai-org/GLM-Z1-9B-0414:featherless-ai",
                parser=parser,
                verbose=verbose,
            )
        else:
            # Use original OpenAI-based solver
            answer_result = solve_with_steps(
                question=guided_question,
                steps_path=str(steps_path),
                api_key=api_key,
                model=model,
                parser=parser,
                verbose=verbose,
            )
        
        if answer_result.get("error"):
            result_bag["error"] = f"Sinh đáp án: {answer_result['error']}"
            if verbose:
                print(f"⚠️ Error: {result_bag['error']}")
            return result_bag
        
        computed_answer = answer_result.get("final_result")
        if computed_answer is None:
            result_bag["error"] = "Solver không trả về đáp án"
            return result_bag
        
        if verbose:
            print(f"✓ Computed answer: {computed_answer}")
    
    except Exception as e:
        result_bag["error"] = f"Sinh đáp án: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag
    
    # --- E: Generate distracting choices ---
    if verbose:
        print("\n" + "=" * 70)
        print("E: Generate distracting choices")
        print("=" * 70)
    
    try:
        distractors = generate_distracting_choices(
            question=new_question_text,
            correct_answer=computed_answer,
            api_key=effective_api_key,
            model=model,
            verbose=verbose
        )
        
        # Combine correct answer with distractors and shuffle
        import random
        all_choices = [str(computed_answer)] + distractors
        random.shuffle(all_choices)
        
        # Find which position has the correct answer
        correct_letter_idx = all_choices.index(str(computed_answer))
        correct_letter = CHOICE_LETTERS[correct_letter_idx]
        
        if verbose:
            print(f"✓ Generated 4 choices, correct answer is {correct_letter}")
            for i, choice in enumerate(all_choices):
                marker = " ← correct" if i == correct_letter_idx else ""
                print(f"  {CHOICE_LETTERS[i]}: {choice}{marker}")
    
    except Exception as e:
        result_bag["error"] = (result_bag["error"] or "") + f"; Gen choices: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag
    
    # --- F: Generate short explanation ---
    if verbose:
        print("\n" + "=" * 70)
        print("F: Generate short explanation")
        print("=" * 70)
    
    try:
        # Get sample explanation for style reference
        q_block = sample.get("question") or {}
        sample_explanation = (q_block.get("explanation") or "").strip()
        
        new_explanation = generate_short_explanation(
            question=new_question_text,
            correct_answer=computed_answer,
            sample_explanation=sample_explanation,
            api_key=effective_api_key,
            model=model,
            verbose=verbose
        )
        
        if verbose:
            print(f"✓ Generated explanation: {new_explanation[:150]}..." if len(new_explanation) > 150 else f"✓ Generated explanation: {new_explanation}")
    
    except Exception as e:
        result_bag["error"] = (result_bag["error"] or "") + f"; Gen explanation: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag
    
    # --- Package final result ---
    final_question_item = {
        "id": new_question_item.get("id"),
        "section": new_question_item.get("section"),
        "skill": new_question_item.get("skill"),
        "category": new_question_item.get("category"),
        "difficulty": new_question_item.get("difficulty"),
        "question": {
            "question": new_question_text,
            "choices": all_choices,
            "correct_answer": [correct_letter],
            "explanation": new_explanation
        }
    }
    
    result_bag["new_question_item"] = final_question_item
    result_bag["new_question_text"] = new_question_text
    result_bag["answer_result"] = {
        "final_result": computed_answer,
        "correct_choice_letter": correct_letter,
        "all_choices": all_choices,
        "explanation": new_explanation
    }

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
        llm = ChatOpenAI(model=model, temperature=0.7, api_key=api_key)
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
    use_hf_solver: bool = False,
    hf_api_key: Optional[str] = None,
    open_ai_api_key: Optional[str] = None,
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
        use_hf_solver: Use HuggingFace solver instead of OpenAI (Math only)
        hf_api_key: HuggingFace API key (if using HF solver)
    
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
            use_hf_solver=use_hf_solver,
            hf_api_key=hf_api_key,
            open_ai_api_key=open_ai_api_key,
        )


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
    ap.add_argument("--use-hf", action="store_true", help="Dùng HuggingFace solver thay vì OpenAI (chỉ cho Math questions)")
    ap.add_argument("--hf-api-key", type=str, default=None, help="HuggingFace API key (hoặc dùng biến môi trường HF_API_KEY)")
    ap.add_argument("--open-ai-api-key", type=str, default=None, help="OpenAI API key (hoặc dùng biến môi trường OPENAI_API_KEY)")
    ap.add_argument("--quiet", action="store_true", help="Giảm log")
    ap.add_argument("--save-result", type=str, default=None, help="Lưu kết quả flow ra file JSON")
    args = ap.parse_args()

    sample = load_sample_question(
        questions_path=args.questions_path,
        index=args.sample_index,
        question_id=args.question_id,
    )

    # Set default output directory
    out_dir = args.out_dir if args.out_dir else "output"

    result = run_flow(
        sample,
        steps_json_path=args.steps_json,
        out_dir=out_dir,
        model=args.model,
        verbose=not args.quiet,
        use_hf_solver=args.use_hf,
        hf_api_key=args.hf_api_key,
        open_ai_api_key=args.open_ai_api_key,
    )

    if args.save_result:
        # Save complete result (structure varies by question type)
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả: {args.save_result}")

    if result.get("error"):
        print("Lỗi:", result["error"])
        return 1
    
    # Success message
    if not args.quiet:
        print("\n" + "=" * 70)
        print("✓ Luồng hoàn tất thành công!")
        print("=" * 70)
        
        # Show what was generated
        is_rw = is_reading_writing_question(sample)
        if is_rw:
            print("Generated: Reading & Writing question")
            if result.get("new_question_item"):
                print(f"  Output: {out_dir}/new_question.json")
                print(f"  Validation: {out_dir}/new_question_validation.json")
        else:
            print("Generated: Math question")
            if result.get("steps_json_path"):
                print(f"  Steps: {result['steps_json_path']}")
            if result.get("new_question_item"):
                print(f"  Question: {out_dir}/new_question.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
