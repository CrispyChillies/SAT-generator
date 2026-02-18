#!/usr/bin/env python3
"""
Luồng sinh câu hỏi và đáp án theo flow.md:

  A (question / explanation / correct_answer)
   ├─→ B: Agent sinh steps_function_and_meaning.json
   ├─→ C: Gen câu hỏi mới, explanation và đáp án
   └─→ D: Sinh đáp án cho câu hỏi mới (dựa vào file JSON từ B và câu hỏi từ C)

Chạy: python run_flow.py [--sample-index N] [--question-id ID] [--questions-path PATH] [--out-dir DIR]
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Import các module theo flow
# ---------------------------------------------------------------------------
from generate_question_langchain import generate_new_question, load_sample_question
from agent import LangGraphMathAgent
from sat_math_solver import solve_with_steps
from mathml_parser import MathMLParser

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
        return raw

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
# Luồng chính
# ---------------------------------------------------------------------------

def run_flow(
    sample: Dict[str, Any],
    *,
    steps_json_path: str = "steps_function_and_meaning.json",
    out_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4.1",
    verbose: bool = True,
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

    Returns:
        Dict gồm:
          - steps_json_path: Đường dẫn file steps đã ghi.
          - new_question_item: Câu hỏi mới (dict), gồm question, explanation, correct_answer.
          - new_question_text: Nội dung câu hỏi mới (HTML/string).
          - answer_result: Kết quả từ sat_math_solver (final_result, steps_detail, error, ...).
          - error: Lỗi tổng (nếu có).
    """
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
    graph = parsed['graph']

    result_bag = {
        "steps_json_path": str(steps_path),
        "new_question_item": None,
        "new_question_text": None,
        "answer_result": None,
        "error": None,
    }

    # --- B: Agent sinh steps_function_and_meaning.json ---
    if verbose:
        print("\n" + "=" * 70)
        print("B: Agent sinh steps_function_and_meaning.json")
        print("=" * 70)
    try:
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
    except Exception as e:
        result_bag["error"] = f"Agent: {e}"
        if verbose:
            import traceback
            traceback.print_exc()
        return result_bag

    # --- C: Gen câu hỏi mới + explanation + đáp án ---
    if verbose:
        print("\n" + "=" * 70)
        print("C: Gen câu hỏi mới, explanation và đáp án")
        print("=" * 70)
    try:
        new_question_item = generate_new_question(sample)
        result_bag["new_question_item"] = new_question_item
        new_q_block = new_question_item.get("question") or {}
        new_question_text = (new_q_block.get("question") or "").strip()
        new_explanation = (new_q_block.get("explanation") or "").strip()
        new_correct_answer = new_q_block.get("correct_answer")
        new_choices = new_q_block.get("choices") or []
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
        print("D: Sinh đáp án cho câu hỏi mới (dựa vào file JSON)")
        print("=" * 70)
    try:
        answer_result = solve_with_steps(
            question=new_question_text,
            steps_path=str(steps_path),
            api_key=api_key,
            model=model,
            parser=parser,
            verbose=verbose,
        )
        result_bag["answer_result"] = answer_result
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


def main():
    ap = argparse.ArgumentParser(description="Chạy luồng theo flow.md: Agent → steps JSON, Gen câu hỏi mới → Sinh đáp án câu mới.")
    ap.add_argument("--sample-index", type=int, default=0, help="Index câu mẫu trong file questions (mặc định 0)")
    ap.add_argument("--question-id", type=str, default=None, help="Lấy câu mẫu theo id thay vì index")
    ap.add_argument("--questions-path", type=str, default="questions_practice_test.json", help="Đường dẫn file danh sách câu hỏi")
    ap.add_argument("--out-dir", type=str, default=None, help="Thư mục ghi steps JSON và file kết quả (mặc định: thư mục hiện tại)")
    ap.add_argument("--steps-json", type=str, default="steps_function_and_meaning.json", help="Tên file steps JSON (ghi trong out-dir)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Model LLM")
    ap.add_argument("--quiet", action="store_true", help="Giảm log")
    ap.add_argument("--save-result", type=str, default=None, help="Lưu kết quả flow ra file JSON")
    args = ap.parse_args()

    sample = load_sample_question(
        questions_path=args.questions_path,
        index=args.sample_index,
        question_id=args.question_id,
    )

    result = run_flow(
        sample,
        steps_json_path=args.steps_json,
        out_dir=args.out_dir,
        model=args.model,
        verbose=not args.quiet,
    )

    if args.save_result:
        # Chuẩn hóa để ghi JSON (bỏ object phức tạp nếu cần)
        to_save = {
            "steps_json_path": result.get("steps_json_path"),
            "new_question_item": result.get("new_question_item"),
            "new_question_text": result.get("new_question_text"),
            "answer_result": result.get("answer_result"),
            "error": result.get("error"),
        }
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả: {args.save_result}")

    if result.get("error"):
        print("Lỗi:", result["error"])
        return 1
    print("\nLuồng hoàn tất.")
    return 0


if __name__ == "__main__":
    exit(main())
