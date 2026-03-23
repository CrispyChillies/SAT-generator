"""
OpenAI Basic Math Solver - Direct chat-completions inference with GPT models
for SAT math solving without tool-calling orchestration.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from mathml_parser import MathMLParser

load_dotenv()


def _supports_custom_temperature(model_name: str) -> bool:
    """Some models (e.g., GPT-5 family) only support default temperature."""
    return "gpt-5" not in (model_name or "").lower()


class ToolExecutionStep(BaseModel):
    step_number: int
    thought: str
    tool_name: str = ""
    tool_input: Dict[str, Any] = {}
    tool_output: Any = None
    param_explanation: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExecutionTrace(BaseModel):
    problem_description: str
    expected_answer: Any
    steps: List[ToolExecutionStep] = []
    final_result: Optional[Any] = None
    is_correct: Optional[bool] = None
    error: Optional[str] = None
    total_steps: int = 0

    def add_step(
        self,
        thought: str,
        tool_name: str = "",
        tool_input: Optional[Dict[str, Any]] = None,
        tool_output: Any = None,
        param_explanation: str = "",
    ):
        step = ToolExecutionStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output=tool_output,
            param_explanation=param_explanation,
        )
        self.steps.append(step)
        self.total_steps = len(self.steps)

    def export_steps_json(self, filepath: str) -> None:
        steps_data = []
        for s in self.steps:
            params_meaning = []
            if s.param_explanation:
                try:
                    parsed = json.loads(s.param_explanation)
                    if isinstance(parsed, list):
                        params_meaning = parsed
                except json.JSONDecodeError:
                    pass

            steps_data.append(
                {
                    "step_number": s.step_number,
                    "function_name": s.tool_name or "reasoning",
                    "params_meaning": params_meaning,
                }
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"steps": steps_data}, f, indent=2, ensure_ascii=False)


class OpenAIBasicMathSolver:
    """Direct GPT-based math solver with response parsing and optional answer verification."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        verification_api_key: Optional[str] = None,
        verification_model: str = "gpt-4o-mini",
        verbose: bool = False,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY in environment or pass api_key parameter")

        self.model = model
        self.verbose = verbose
        self.parser = MathMLParser()
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

        verification_key = verification_api_key or os.getenv("OPENAI_API_KEY")
        self.verification_model = verification_model
        self.verification_client = OpenAI(api_key=verification_key) if verification_key else None

    def solve(
        self,
        mathml_explanation: str,
        correct_answer: Any,
        question: str = "",
        max_iterations: int = 1,
        steps_json_path: Optional[str] = None,
        temperature: float = 0.1,
    ) -> ExecutionTrace:
        def contains_svg(text: str) -> bool:
            return bool(text and "<svg" in text.lower())

        if contains_svg(mathml_explanation):
            parsed_explanation = self.parser.parse(mathml_explanation)
            readable = parsed_explanation["text"]
        else:
            readable = mathml_explanation

        if contains_svg(question):
            parsed_question = self.parser.parse(question)
            question_text = parsed_question["text"]
        else:
            question_text = question

        trace = ExecutionTrace(problem_description=readable, expected_answer=correct_answer)

        if self.verbose:
            print("\n" + "=" * 70)
            print("B-TRACE: OPENAI BASIC SOLVER")
            print("=" * 70)
            if question_text:
                print(f"Question: {question_text[:300]}{'...' if len(question_text) > 300 else ''}")
            print(f"Expected answer: {correct_answer}")

        system_prompt = """You are a math expert. Analyze the provided HTML/MathML content, extract equations, and solve step-by-step.

For each problem:
1. Identify what is being asked
2. Extract the key information and equations
3. Show your step-by-step solution with numbered steps
4. Provide the final answer

    IMPORTANT - Output Format:
    Return ONLY valid JSON with this exact structure:
    {
      "reasoning_steps": ["step 1", "step 2", "..."],
      "final_answer": "-1/3"
    }

    Rules for final_answer:
    - Use a simple plain-text canonical answer.
    - For fractions, use "a/b" format (example: "-1/3"), NOT LaTeX.
    - For decimals, use plain number string (example: "0.25").
    - For coordinates, use "(x, y)".
    - Do NOT include markdown, code fences, or extra keys.

Be precise with calculations and show all intermediate steps."""

        problem_content = f"""Question: {question_text or "(No specific question)"}

Problem/Explanation:
{readable}

Expected Answer: {correct_answer}

Please solve this problem step by step and return ONLY the required JSON object."""

        _ = max_iterations

        try:
            request_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_content},
                ],
                "response_format": {"type": "json_object"},
            }
            if _supports_custom_temperature(self.model):
                request_kwargs["temperature"] = temperature
            completion = self.client.chat.completions.create(**request_kwargs)
            response_text = completion.choices[0].message.content or ""

            if self.verbose:
                print("\n--- Step B Raw Reasoning ---")
                print(response_text)

            _, final_answer = self._parse_response(response_text)
            final_answer = self._to_answer_string(final_answer)

            if self.verbose:
                print("\n--- Step B Parsed Final ---")
                print(final_answer)
            trace.add_step(
                thought=response_text,
                tool_name="reasoning",
                tool_input={},
                tool_output=final_answer,
                param_explanation="",
            )

            trace.final_result = final_answer
            trace.is_correct = self._check_answer(final_answer, correct_answer)

            if self.verbose:
                print("--- Step B Verdict ---")
                print(f"is_correct={trace.is_correct}")
        except Exception as e:
            trace.error = str(e)
            if self.verbose:
                print("--- Step B Error ---")
                print(trace.error)

        if steps_json_path:
            try:
                trace.export_steps_json(steps_json_path)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: failed to save steps JSON: {e}")

        return trace

    def _normalize_text(self, s: str) -> str:
        return (
            (s or "")
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .strip()
        )

    def _to_answer_string(self, value: Any) -> Optional[str]:
        """Convert parsed answer to canonical string form (prefer fraction strings)."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)

        s = self._normalize_text(str(value))
        s = re.sub(r"\\+frac", r"\\frac", s)

        latex_frac = re.fullmatch(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
        if latex_frac:
            sign = "-" if latex_frac.group(1) == "-" else ""
            return f"{sign}{latex_frac.group(2).strip()}/{latex_frac.group(3).strip()}"

        simple_frac = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", s)
        if simple_frac:
            return f"{simple_frac.group(1)}/{simple_frac.group(2)}"

        try:
            n = float(s)
            if n.is_integer():
                return str(int(n))
            return str(n)
        except ValueError:
            return s

    def _parse_numeric_value(self, value: Any) -> Optional[float]:
        """Parse numbers from int/float/decimal-string/fraction-string/latex fraction."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        s = self._normalize_text(str(value))
        s = re.sub(r"\\+frac", r"\\frac", s)

        latex_frac = re.fullmatch(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
        if latex_frac:
            try:
                sign = -1.0 if latex_frac.group(1) == "-" else 1.0
                n = float(latex_frac.group(2))
                d = float(latex_frac.group(3))
                if d != 0:
                    return sign * (n / d)
                return None
            except ValueError:
                return None

        frac = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", s)
        if frac:
            try:
                n = float(frac.group(1))
                d = float(frac.group(2))
                if d != 0:
                    return n / d
                return None
            except ValueError:
                return None

        try:
            return float(s)
        except ValueError:
            return None

    def _parse_response(self, response: str) -> tuple[str, Optional[Any]]:
        final_answer: Optional[Any] = None

        def normalize_text(s: str) -> str:
            # Normalize common unicode minus/dash characters to ASCII '-'
            return self._normalize_text(s)

        def try_parse_fraction_or_number(s: str) -> Optional[Any]:
            s = normalize_text(s)
            frac_match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", s)
            if frac_match:
                try:
                    n = int(frac_match.group(1))
                    d = int(frac_match.group(2))
                    if d == 0:
                        return None
                    val = n / d
                    if float(val).is_integer():
                        return int(val)
                    return val
                except ValueError:
                    return None
            try:
                num = float(s)
                if num.is_integer():
                    return int(num)
                return num
            except ValueError:
                return None

        # Preferred path for OpenAI: strict JSON output with final_answer field.
        try:
            payload = json.loads(response)
            if isinstance(payload, dict):
                raw_final = payload.get("final_answer")
                if raw_final is not None:
                    if isinstance(raw_final, (int, float)):
                        return response, self._to_answer_string(raw_final)
                    raw_s = str(raw_final).strip()
                    return response, self._to_answer_string(raw_s)
        except Exception:
            pass

        boxed_pattern = r"\\+boxed\{([^{}]+)\}"
        all_matches = re.findall(boxed_pattern, response)
        if all_matches:
            if len(all_matches) > 1:
                parsed_items = [self._parse_boxed_content(m.strip()) for m in all_matches]
                final_answer = ",".join(str(x) for x in parsed_items if x is not None)
            else:
                final_answer = self._parse_boxed_content(all_matches[0].strip())

        if final_answer is None:
            nested_pattern = r"\\+boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
            match = re.search(nested_pattern, response)
            if match:
                final_answer = self._parse_boxed_content(match.group(1).strip())

        if final_answer is None:
            final_answer_section = re.search(
                r"\*\*Final\s+Answer[:\*]*\*?\*?\s*(.+?)(?:\n\n|\Z)",
                response,
                re.IGNORECASE | re.DOTALL,
            )
            if final_answer_section:
                section_text = final_answer_section.group(1)
                boxed_match = re.search(r"\\+boxed\{([^{}]+)\}", section_text)
                if boxed_match:
                    final_answer = self._parse_boxed_content(boxed_match.group(1).strip())
                else:
                    section_norm = normalize_text(section_text)
                    frac = re.search(r"([+-]?\d+\s*/\s*[+-]?\d+)", section_norm)
                    if frac:
                        final_answer = try_parse_fraction_or_number(frac.group(1))
                    else:
                        nums = re.findall(r"([+-]?\d+\.?\d*)", section_norm)
                        if nums:
                            final_answer = try_parse_fraction_or_number(nums[-1])

        if final_answer is None:
            patterns = [
                r"final\s+answer\s*(?:is\s*)?[:\s]*([+-]?\d+\s*/\s*[+-]?\d+|[+-]?\d+\.?\d*)",
                r"answer\s*[:\s]*([+-]?\d+\s*/\s*[+-]?\d+|[+-]?\d+\.?\d*)",
                r"=\s*([+-]?\d+\s*/\s*[+-]?\d+|[+-]?\d+\.?\d*)\s*$",
                r"≈\s*([+-]?\d+\s*/\s*[+-]?\d+|[+-]?\d+\.?\d*)",
            ]
            response_norm = normalize_text(response)
            for pattern in patterns:
                match = re.search(pattern, response_norm, re.IGNORECASE | re.MULTILINE)
                if match:
                    candidate = try_parse_fraction_or_number(match.group(1))
                    if candidate is not None:
                        final_answer = candidate
                        break

        if final_answer is None:
            response_norm = normalize_text(response)
            frac = re.findall(r"([+-]?\d+\s*/\s*[+-]?\d+)", response_norm)
            if frac:
                final_answer = try_parse_fraction_or_number(frac[-1])
            numbers = re.findall(r"\b([+-]?\d+\.?\d*)\b", response_norm)
            if numbers:
                n = try_parse_fraction_or_number(numbers[-1])
                if n is not None:
                    final_answer = n

        return response, final_answer

    def _parse_boxed_content(self, content: str) -> Optional[Any]:
        content = (
            content.strip()
            .replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
        )
        content = re.sub(r"\\+frac", r"\\frac", content)

        tuple_match = re.match(r"\(([^,]+),\s*([^)]+)\)", content)
        if tuple_match:
            val1, val2 = tuple_match.group(1).strip(), tuple_match.group(2).strip()
            try:
                num1 = float(val1)
                num2 = float(val2)
                if num1.is_integer():
                    num1 = int(num1)
                if num2.is_integer():
                    num2 = int(num2)
                return (num1, num2)
            except ValueError:
                return f"({val1}, {val2})"

        frac_match = re.match(r"([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}", content)
        if frac_match:
            try:
                sign = -1.0 if frac_match.group(1) == "-" else 1.0
                numerator = float(frac_match.group(2))
                denominator = float(frac_match.group(3))
                if denominator != 0:
                    result = sign * (numerator / denominator)
                    if result.is_integer():
                        return int(result)
                    return result
            except ValueError:
                sign = "-" if frac_match.group(1) == "-" else ""
                return f"{sign}{frac_match.group(2)}/{frac_match.group(3)}"

        simple_frac_match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", content)
        if simple_frac_match:
            try:
                numerator = int(simple_frac_match.group(1))
                denominator = int(simple_frac_match.group(2))
                if denominator != 0:
                    value = numerator / denominator
                    if float(value).is_integer():
                        return int(value)
                    return value
            except ValueError:
                pass

        try:
            num = float(content)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            pass

        assign_match = re.match(r"[a-zA-Z_]\w*\s*=\s*([+-]?[0-9]+\.?[0-9]*)", content)
        if assign_match:
            try:
                num = float(assign_match.group(1))
                if num.is_integer():
                    return int(num)
                return num
            except ValueError:
                pass

        return content if content else None

    def _check_answer_with_llm(self, computed: Any, expected: Any) -> Optional[bool]:
        if computed is None or expected is None or not self.verification_client:
            return None

        try:
            prompt = f"""Compare these two answers and determine if they are mathematically or semantically equivalent.

Computed Answer: {computed}
Expected Answer: {expected}

Consider these as equivalent:
- Different numerical formats: 0.5 and 1/2, 15 and 15.0
- Equivalent mathematical expressions: 6 and 2*3
- Same coordinates in different formats: (15, 3) and x=15, y=3
- Small rounding differences (within 1%)
- Semantically identical text answers

Respond with ONLY one word: "CORRECT" if they are equivalent, or "INCORRECT" if they are different."""

            completion = self.verification_client.chat.completions.create(
                model=self.verification_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise mathematical answer verification expert. Compare answers for equivalence.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
            )
            response = (completion.choices[0].message.content or "").strip().upper()
            # Check INCORRECT first because the token contains "CORRECT" as a substring.
            if "INCORRECT" in response:
                return False
            if re.search(r"\bCORRECT\b", response):
                return True
            return None
        except Exception:
            return None

    def _check_answer_numeric(self, computed: Any, expected: Any) -> Optional[bool]:
        try:
            if isinstance(expected, str) and "<math>" in expected:
                expected_text = self.parser.parse(expected)["text"]
                # Prefer fraction detection (e.g., "-1/3") over loose numeric token extraction.
                frac = re.search(r"([+-]?\d+)\s*/\s*([+-]?\d+)", expected_text)
                if frac:
                    n = int(frac.group(1))
                    d = int(frac.group(2))
                    if d != 0:
                        expected = n / d
                else:
                    nums = re.findall(r"([+-]?[0-9]+\.?[0-9]*)", expected_text)
                    if nums:
                        expected = float(nums[0])

            computed_f = self._parse_numeric_value(computed)
            expected_f = self._parse_numeric_value(expected)
            if computed_f is None or expected_f is None:
                return None

            rel_error = abs(computed_f - expected_f) / max(abs(expected_f), 1e-9)
            abs_error = abs(computed_f - expected_f)
            return rel_error < 0.01 or abs_error < 0.01
        except (ValueError, TypeError):
            if str(computed).strip() == str(expected).strip():
                return True
            return None

    def _check_answer(self, computed: Any, expected: Any) -> Optional[bool]:
        if computed is None or expected is None:
            return None
        numeric_result = self._check_answer_numeric(computed, expected)
        if numeric_result is not None:
            return numeric_result
        return self._check_answer_with_llm(computed, expected)


def solve_with_steps_openai_basic(
    question: str,
    steps_path: Optional[Union[str, Path]] = None,
    *,
    new_correct_answer: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    parser: Optional[MathMLParser] = None,
    verbose: bool = False,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "final_result": None,
            "steps_detail": [],
            "answer_text": None,
            "is_correct": None,
            "error": "OPENAI_API_KEY not set in environment or not provided",
        }

    parser = parser or MathMLParser()

    def contains_svg(text: str) -> bool:
        return bool(text and "<svg" in text.lower())

    if contains_svg(question):
        parsed_question = parser.parse(question)
        question_text = parsed_question["text"]
    else:
        question_text = question

    try:
        solver = OpenAIBasicMathSolver(api_key=api_key, model=model, verbose=verbose)
        trace = solver.solve(
            mathml_explanation=question_text,
            correct_answer=new_correct_answer,
            question=question_text,
            temperature=temperature,
            steps_json_path=str(steps_path) if steps_path else None,
        )

        steps_detail = []
        for step in trace.steps:
            steps_detail.append(
                {
                    "step_number": step.step_number,
                    "thought": step.thought,
                    "tool_name": step.tool_name or "reasoning",
                    "tool_input": step.tool_input or {},
                    "tool_output": step.tool_output,
                    "params_meaning": [],
                }
            )

        answer_text = (
            f"The result is {trace.final_result}."
            if trace.final_result is not None
            else None
        )

        return {
            "final_result": trace.final_result,
            "steps_detail": steps_detail,
            "answer_text": answer_text,
            "is_correct": trace.is_correct,
            "error": trace.error,
        }
    except Exception as e:
        return {
            "final_result": None,
            "steps_detail": [],
            "answer_text": None,
            "is_correct": None,
            "error": str(e),
        }
