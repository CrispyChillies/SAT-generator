#!/usr/bin/env python3
"""
Sinh câu hỏi mới từ câu hỏi mẫu bằng LangChain.
- Giữ nguyên category, section, type, difficulty của câu gốc.
- Sinh câu hỏi mới, explanation và đáp án (correct_answer); cùng format HTML + MathML, chỉ đổi số.
- Câu hỏi mới phải đúng format so với câu gốc (HTML + MathML).
"""

import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền llm.")

# ---------------------------------------------------------------------------
# Cấu trúc output từ LLM: câu hỏi + explanation + đáp án
# ---------------------------------------------------------------------------

class GeneratedQuestionContent(BaseModel):
    """Câu hỏi mới, explanation và đáp án đúng; cùng format HTML+MathML, chỉ đổi số so với mẫu (dùng khi không phải multiple-choice hoặc không có 4 choices)."""
    question: str = Field(description="New question content in the same HTML and MathML format as the sample, with only numerical values changed")
    explanation: str = Field(description="New explanation in the same HTML and MathML format as the sample, with only numerical values changed to match the new question")
    correct_answer: str = Field(description="The correct answer for the new question, in the same format as the sample (e.g. HTML/MathML string of the right choice or value)")


class GeneratedMultipleChoiceContent(BaseModel):
    """Câu hỏi multiple-choice: câu hỏi + explanation + đúng 4 lựa chọn (A,B,C,D) + chữ cái đáp án đúng."""
    question: str = Field(description="New question content, same HTML+MathML format with only numerical values changed")
    explanation: str = Field(description="New explanation, same format with only numbers changed to match the new question")
    choices: List[str] = Field(description="Exactly 4 answer choices in order A, B, C, D; each is HTML+MathML string with only numbers changed")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="The letter of the correct answer (A, B, C, or D)")

    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]


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


def _build_prompt(
    original_question_html: str,
    original_explanation: str,
    original_correct_answer: str,
    category: str,
    section: str,
    q_type: str,
    difficulty: str,
) -> str:
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

Return a JSON object with keys: question, explanation, correct_answer. Each value: same string as sample with only numbers substituted; numbers must be consistent across all three."""


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


def _build_prompt_multiple_choice(
    original_question_html: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Prompt cho multiple-choice: sinh question, explanation, 4 choices, và correct_answer_letter."""
    choices_text = "\n".join(
        f"Choice {letter}: {c}" for letter, c in zip(["A", "B", "C", "D"], original_choices)
    )
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
---

Return a JSON object with keys: question, explanation, choices, correct_answer_letter.
- question: new question string (only numbers changed).
- explanation: new explanation string (only numbers changed, consistent with new question).
- choices: list of exactly 4 strings, in order A, B, C, D (only numbers changed in each).
- correct_answer_letter: one of "A", "B", "C", "D" (the correct choice for the new question; typically the same as the sample, {correct_letter})."""


def generate_new_question(
    sample: Dict[str, Any],
    llm: Optional[ChatOpenAI] = None,
) -> Dict[str, Any]:
    """
    Sinh câu hỏi mới, explanation và đáp án từ câu mẫu (cùng category, đúng format, chỉ đổi số liệu).

    Args:
        sample: Một item từ questions_practice_test.json (có id, category, question, explanation, correct_answer, ...).
        llm: LangChain ChatOpenAI. Nếu None sẽ tạo mới từ OPENAI_API_KEY.

    Returns:
        Câu hỏi mới dạng dict, cùng cấu trúc với questions_practice_test.json,
        question.question, question.explanation, question.correct_answer đều được sinh; choices có thể null.
    """
    if llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền llm.")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    category = sample.get("category", "Algebra")
    section = sample.get("section", "Math")
    q_type = sample.get("type", "multiple-choice")
    difficulty = sample.get("difficulty", "Easy")

    original_html = _get_question_html(sample)
    if not original_html:
        raise ValueError("Câu mẫu không có nội dung question (HTML).")
    original_explanation = _get_explanation(sample)
    original_choices = _get_choices(sample)
    correct_letter = _get_correct_answer_letter(sample)
    original_correct_answer = _get_correct_answer_content(sample)
    is_multiple_choice = (q_type == "multiple-choice") and len(original_choices) == 4 and correct_letter and original_explanation
    generate_full = bool(original_explanation and original_correct_answer)

    if is_multiple_choice:
        # Multiple-choice: sinh question, explanation, đúng 4 choices, và correct_answer_letter
        prompt_text = _build_prompt_multiple_choice(
            original_html,
            original_explanation,
            original_choices,
            correct_letter,
            category,
            section,
            difficulty,
        )
        structured_llm = llm.with_structured_output(GeneratedMultipleChoiceContent)
        result_mc: GeneratedMultipleChoiceContent = structured_llm.invoke(
            [HumanMessage(content=prompt_text)]
        )
        new_question_text = (result_mc.question or "").strip()
        new_explanation = (result_mc.explanation or "").strip()
        new_choices = result_mc.choices or []
        if len(new_choices) != 4:
            raise ValueError(f"LLM phải trả về đúng 4 choices, nhận được {len(new_choices)}.")
        new_choices = [str(c).strip() for c in new_choices[:4]]
        new_letter = (result_mc.correct_answer_letter or "").strip().upper()
        if new_letter not in ("A", "B", "C", "D"):
            raise ValueError(f"correct_answer_letter phải là A, B, C hoặc D, nhận được: {result_mc.correct_answer_letter!r}")
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
        prompt_text = _build_prompt(
            original_html,
            original_explanation,
            original_correct_answer,
            category,
            section,
            q_type,
            difficulty,
        )
        structured_llm = llm.with_structured_output(GeneratedQuestionContent)
        result: GeneratedQuestionContent = structured_llm.invoke(
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
            question: str = Field(description="New question content, same format with only numbers changed")
        prompt_question_only = f"""You are an SAT question writer. Change ONLY the numerical values in the sample question below. Do NOT change wording or structure. Output the same HTML + MathML with only numbers substituted.

Sample:
---
{original_html}
---

Return only the new question string (same format, numbers changed)."""
        QuestionOnlyModel = llm.with_structured_output(QuestionOnly)
        res = QuestionOnlyModel.invoke([HumanMessage(content=prompt_question_only)])
        new_question_text = (res.question or "").strip()
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
    # Truyền trực tiếp nội dung câu hỏi mẫu (str, HTML + MathML)
    sample_question = """A drone is <math><mn>120</mn></math> m horizontally from the base of a tower. The angle of elevation from the drone to the top of the tower is <math><mn>30</mn><mo>°</mo></math>. Approximately how tall is the tower (to the nearest meter)?"""
    main(sample_question, count=1, output_path="new_question.json")
