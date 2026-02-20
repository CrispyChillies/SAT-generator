#!/usr/bin/env python3
"""
Web demo: nhập question ID → xem câu gốc (question + explanation + đáp án đúng)
→ chạy run_flow → hiển thị câu hỏi mới và đáp án.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from generate_question_langchain import load_sample_question
from run_flow import run_flow, preprocess_correct_answer

app = Flask(__name__)
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "questions_practice_test.json")
BASE_DIR = Path(__file__).resolve().parent

# Regex: đóng thẻ MathML (vd </msup>) rồi có text (không chứa <) rồi </math> hoặc cuối chuỗi → chèn </math> ngay sau thẻ để text nằm ngoài <math> (trình duyệt không render text trong <math>).
_MATH_TRAILING_TEXT_BEFORE_CLOSE = re.compile(
    r"(</(?:msup|mrow|mfrac|mn|mi|mo|mtext|mfenced|msqrt|msub)\s*>)([^<]+)(</math>)"
)
_MATH_TRAILING_TEXT_AT_END = re.compile(
    r"(</(?:msup|mrow|mfrac|mn|mi|mo|mtext|mfenced|msqrt|msub)\s*>)([^<]+)$"
)


def _normalize_math_html(html: str) -> str:
    """Đưa text nằm sai trong <math> ra ngoài để trình duyệt hiển thị (vd: '. What is the measure...')."""
    if not (html or "").strip():
        return html or ""
    s = _MATH_TRAILING_TEXT_BEFORE_CLOSE.sub(r"\1</math>\2", html)
    s = _MATH_TRAILING_TEXT_AT_END.sub(r"\1</math>\2", s)
    return s


def _answers_match_llm(answer_from_c: str, answer_from_d: str) -> bool:
    """So sánh đáp án từ C và từ D bằng LLM: hai đáp án có cùng giá trị/ý nghĩa không."""
    if not answer_from_c and not answer_from_d:
        return True
    if not answer_from_c or not answer_from_d:
        return False
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""You are a math answer checker. Compare these two answers to the same SAT math question. Decide if they represent the same correct answer (same value, same meaning). Ignore formatting differences (HTML, commas in numbers, etc.).

Answer from generated question (C): 
{answer_from_c}

Answer from solver (D):
{answer_from_d}

Are these two answers the same or equivalent? Reply with exactly one word: YES or NO."""
        msg = llm.invoke([HumanMessage(content=prompt)])
        text = (msg.content or "").strip().upper()
        return "YES" in text
    except Exception:
        return False


DATA_DIR = BASE_DIR / "data"


@app.route("/")
def index():
    return render_template("demo.html")


@app.route("/api/question/<question_id>")
def get_question(question_id: str):
    """Lấy câu hỏi theo ID từ questions_practice_test.json."""
    try:
        sample = load_sample_question(
            questions_path=str(BASE_DIR / QUESTIONS_PATH),
            question_id=question_id.strip(),
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    q_block = sample.get("question") or {}
    # print("question block:", q_block)
    correct_answer_raw = q_block.get("correct_answer") or sample.get("correct_answer")
    correct_answer_display = preprocess_correct_answer(sample)



    return jsonify({
        "id": sample.get("id"),
        "subject": sample.get("subject"),
        "section": sample.get("section"),
        "category": sample.get("category"),
        "difficulty": sample.get("difficulty"),
        "type": sample.get("type"),
        "paragraph_html": (q_block.get("paragraph") or "").strip(),  # Thêm dòng này
        "question_html": _normalize_math_html((q_block.get("question") or "").strip()),
        "explanation_html": (q_block.get("explanation") or "").strip(),
        "correct_answer_letter": (
            correct_answer_raw[0]
            if isinstance(correct_answer_raw, (list, tuple)) and correct_answer_raw
            else correct_answer_raw
        ),
        "correct_answer_html": correct_answer_display if isinstance(correct_answer_display, str) else str(correct_answer_display),
        "choices": q_block.get("choices"),
    })


@app.route("/api/run-flow", methods=["POST"])
def api_run_flow():
    """Chạy run_flow với question_id trong body. Trả về new_question_text + answer_result."""
    data = request.get_json() or {}
    question_id = (data.get("question_id") or "").strip()
    if not question_id:
        return jsonify({"error": "Thiếu question_id"}), 400

    try:
        sample = load_sample_question(
            questions_path=str(BASE_DIR / QUESTIONS_PATH),
            question_id=question_id,
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    result = run_flow(
        sample,
        out_dir=str(BASE_DIR),
        steps_json_path="steps_function_and_meaning.json",
        verbose=True,
    )

    # Chuẩn hóa để JSON (bỏ object không serialize được nếu có)
    new_item = result.get("new_question_item") or {}
    new_q_block = new_item.get("question") or {}
    choices = new_q_block.get("choices") or []
    correct_answer_raw = new_q_block.get("correct_answer")
    # Multiple-choice: correct_answer là ["C"] → lấy nội dung từ choices[index]
    if choices and isinstance(correct_answer_raw, (list, tuple)) and len(correct_answer_raw) > 0:
        letter = (correct_answer_raw[0] or "").strip().upper()
        idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(letter)
        new_correct_answer_html = _normalize_math_html((choices[idx] or "").strip()) if idx is not None and idx < len(choices) else ""
    elif isinstance(correct_answer_raw, str):
        new_correct_answer_html = _normalize_math_html(correct_answer_raw.strip())
    else:
        new_correct_answer_html = ""
    # Chuẩn hóa choices cho frontend (HTML từng option)
    new_choices_html = [_normalize_math_html((c or "").strip()) for c in choices] if choices else []
    out = {
        "steps_json_path": result.get("steps_json_path"),
        "new_question_text": _normalize_math_html(result.get("new_question_text") or ""),
        "new_question_item": new_item,
        "new_explanation_html": _normalize_math_html((new_q_block.get("explanation") or "").strip()),
        "new_correct_answer_html": new_correct_answer_html,
        "new_choices_html": new_choices_html,
        "new_correct_answer_letter": (correct_answer_raw[0] if isinstance(correct_answer_raw, (list, tuple)) and correct_answer_raw else correct_answer_raw) if correct_answer_raw else None,
        "answer_result": None,
        "error": result.get("error"),
    }
    ar = result.get("answer_result")
    if ar:
        out["answer_result"] = {
            "final_result": ar.get("final_result"),
            "answer_text": ar.get("answer_text"),
            "steps_detail": ar.get("steps_detail"),
            "error": ar.get("error"),
        }

    # So sánh đáp án C vs D bằng LLM
    answer_from_c = ""
    if choices and isinstance(correct_answer_raw, (list, tuple)) and correct_answer_raw:
        idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get((correct_answer_raw[0] or "").strip().upper())
        if idx is not None and idx < len(choices):
            answer_from_c = (choices[idx] or "").strip()
    elif isinstance(correct_answer_raw, str):
        answer_from_c = correct_answer_raw.strip()
    answer_from_d = str(ar.get("final_result") or "").strip() if ar else ""
    out["answers_match"] = _answers_match_llm(answer_from_c, answer_from_d)

    if result.get("error"):
        return jsonify(out), 200  # vẫn 200 để client đọc error trong body
    return jsonify(out)


@app.route("/api/save-question", methods=["POST"])
def api_save_question():
    """Lưu câu hỏi (question + explanation + choices + correct_answer) vào file JSON trong thư mục data/."""
    data = request.get_json() or {}
    new_question_item = data.get("new_question_item")
    if not new_question_item or not isinstance(new_question_item, dict):
        return jsonify({"error": "Thiếu new_question_item trong body"}), 400
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_{timestamp}.json"
    filepath = DATA_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(new_question_item, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True, "saved_path": str(filepath), "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
