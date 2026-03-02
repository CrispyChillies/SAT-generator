"""
HuggingFace Math Solver - Uses GLM-Z1-9B model via HuggingFace router
to solve SAT math problems with step-by-step reasoning.

Compatible with existing agent.py and sat_math_solver.py interfaces.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from mathml_parser import MathMLParser
from dotenv import load_dotenv

load_dotenv()

# Import ExecutionTrace classes from agent.py for compatibility
try:
    from agent import ExecutionTrace, ToolExecutionStep
except ImportError:
    # Fallback: define minimal compatible classes
    from pydantic import BaseModel, Field
    
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
        
        def add_step(self, thought: str, tool_name: str = "", tool_input: Dict[str, Any] = None, 
                     tool_output: Any = None, param_explanation: str = ""):
            step = ToolExecutionStep(
                step_number=len(self.steps) + 1,
                thought=thought,
                tool_name=tool_name,
                tool_input=tool_input or {},
                tool_output=tool_output,
                param_explanation=param_explanation
            )
            self.steps.append(step)
            self.total_steps = len(self.steps)
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                "problem_description": self.problem_description,
                "expected_answer": self.expected_answer,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "thought": s.thought,
                        "tool_name": s.tool_name,
                        "tool_input": s.tool_input,
                        "tool_output": s.tool_output,
                        "param_explanation": s.param_explanation,
                        "timestamp": s.timestamp
                    }
                    for s in self.steps
                ],
                "final_result": self.final_result,
                "is_correct": self.is_correct,
                "error": self.error,
                "total_steps": self.total_steps
            }
        
        def export_steps_json(self, filepath: str) -> None:
            """Export steps with function names and parameter meanings"""
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
                
                steps_data.append({
                    "step_number": s.step_number,
                    "function_name": s.tool_name or "reasoning",
                    "params_meaning": params_meaning,
                })
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"steps": steps_data}, f, indent=2, ensure_ascii=False)
        
        def print_summary(self):
            print(f"\n{'='*70}")
            print("📋 EXECUTION TRACE SUMMARY (HuggingFace Solver)")
            print(f"{'='*70}")
            print(f"Problem: {self.problem_description}")
            print(f"Expected: {self.expected_answer}")
            print(f"Total Steps: {self.total_steps}")
            
            for step in self.steps:
                print(f"\n  {'─'*66}")
                print(f"  Step {step.step_number}:")
                print(f"  💭 Reasoning:")
                for line in step.thought.split('\n')[:5]:  # Limit output
                    if line.strip():
                        print(f"     {line.strip()}")
                
                if step.tool_output:
                    print(f"  📤 Result: {step.tool_output}")
            
            print(f"\n{'─'*70}")
            print(f"Final Result: {self.final_result}")
            status = "✅ CORRECT" if self.is_correct is True else ("❌ INCORRECT" if self.is_correct is False else "(no verdict)")
            print(f"Status: {status}")
            
            if self.error:
                print(f"Error: {self.error}")
            
            print(f"{'='*70}\n")


class HuggingFaceMathSolver:
    """
    Math solver using HuggingFace's GLM-Z1-9B model.
    
    Compatible with LangGraphMathAgent interface.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "zai-org/GLM-Z1-9B-0414:featherless-ai",
        base_url: str = "https://router.huggingface.co/v1",
        openai_api_key: Optional[str] = None,
        chatgpt_model: str = "gpt-4o-mini",
        verbose: bool = False
    ):
        """
        Initialize HuggingFace solver.
        
        Args:
            api_key: HuggingFace API key (or use HF_API_KEY env var)
            model: Model name (default: GLM-Z1-9B)
            base_url: HuggingFace router URL
            openai_api_key: OpenAI API key for ChatGPT verification (or use OPENAI_API_KEY env var)
            chatgpt_model: ChatGPT model for answer verification (default: gpt-4o-mini)
            verbose: Print detailed logs
        """
        self.api_key = api_key or os.getenv("HF_API_KEY")
        if not self.api_key:
            raise ValueError("Set HF_API_KEY in environment or pass api_key parameter")
        
        self.model = model
        self.base_url = base_url
        self.verbose = verbose
        self.parser = MathMLParser()
        
        # HuggingFace client for problem solving
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # ChatGPT client for answer verification
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.chatgpt_model = chatgpt_model
        if self.openai_api_key:
            self.chatgpt_client = OpenAI(
                api_key=self.openai_api_key,
            )
        else:
            self.chatgpt_client = None
            if self.verbose:
                print("⚠️ OPENAI_API_KEY not set - LLM verification will be unavailable")
    
    def solve(
        self, 
        mathml_explanation: str, 
        correct_answer: Any,
        question: str = "",
        max_iterations: int = 1,  # HF model does one-shot reasoning
        steps_json_path: Optional[str] = None,
        temperature: float = 0.1,
    ) -> ExecutionTrace:
        """
        Solve math problem using HuggingFace LLM.
        
        Compatible with agent.LangGraphMathAgent.solve() interface.
        
        Args:
            mathml_explanation: MathML explanation/problem description
            correct_answer: Expected answer
            question: Optional question text
            max_iterations: Not used (HF model does one-shot reasoning)
            steps_json_path: Path to save steps JSON
            temperature: LLM temperature (lower = more deterministic)
            
        Returns:
            ExecutionTrace with complete solving history
        """
        # Parse MathML to readable text
        # readable = self.parser.parse(mathml_explanation)['text'] if mathml_explanation else mathml_explanation
        readable = mathml_explanation
        # question_text = self.parser.parse(question)['text'] if question else question
        question_text = question
        
        trace = ExecutionTrace(
            problem_description=readable,
            expected_answer=correct_answer
        )
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("🤖 HUGGINGFACE SOLVER (GLM-Z1-9B)")
            print(f"{'='*70}")
            if question_text:
                print(f"❓ Question: {question_text}")
            print(f"📝 Explanation: {readable}")
            print(f"🎯 Expected Answer: {correct_answer}")
            print(f"{'='*70}\n")
        
        # Build prompt
        system_prompt = """You are a math expert. Analyze the provided HTML/MathML content, extract the equations, and solve them step-by-step using clear reasoning.

For each problem:
1. Identify what is being asked
2. Extract the key information and equations
3. Show your step-by-step solution with numbered steps
4. Provide the final answer

IMPORTANT - Output Format:
- Structure your response with "**Step-by-Step Solution:**" followed by numbered steps
- End with "**Final Answer:**" section
- Put the final answer in \\boxed{} format: \\boxed{your_answer}
- For single numbers: \\boxed{42}
- For coordinates/tuples: \\boxed{(15, 3)}
- For fractions: \\boxed{\\frac{1}{2}} or the decimal equivalent
- For multiple solutions/answers: Use a SINGLE \\boxed{} with comma-separated values: \\boxed{15,2} (NOT \\boxed{15}, \\boxed{2})
- For multiple equations: Keep them in a single \\boxed{}: \\boxed{m=4c, c+m=25} or use line breaks within one \\boxed{}

Be precise with calculations and show all intermediate steps."""
        
        problem_content = f"""Question: {question_text or "(No specific question)"}

Problem/Explanation:
{readable}

Expected Answer: {correct_answer}

Please solve this problem step by step. Show your reasoning clearly and provide the final answer in \\boxed{{}} format."""
        
        try:
            # Call HuggingFace LLM
            if self.verbose:
                print("🔄 Calling HuggingFace LLM...")
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": problem_content}
                ],
                temperature=temperature
            )
            
            response_text = completion.choices[0].message.content
            
            if self.verbose:
                print(f"\n📝 LLM Response:\n{response_text}\n")
            
            # Parse response to extract steps and final answer
            steps_text, final_answer = self._parse_response(response_text)
            
            # Add main reasoning as a step
            trace.add_step(
                thought=response_text,
                tool_name="reasoning",
                tool_input={},
                tool_output=final_answer,
                param_explanation=""
            )
            
            trace.final_result = final_answer
            
            # Check correctness
            trace.is_correct = self._check_answer(final_answer, correct_answer)
            
            if self.verbose:
                print(f"✅ Final Answer: {final_answer}")
                print(f"📊 Correctness: {trace.is_correct}")
            
        except Exception as e:
            trace.error = str(e)
            if self.verbose:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Export steps if requested
        if steps_json_path:
            try:
                trace.export_steps_json(steps_json_path)
                if self.verbose:
                    print(f"💾 Saved steps to: {steps_json_path}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to save steps: {e}")
        
        return trace
    
    def _parse_response(self, response: str) -> tuple[str, Optional[Any]]:
        """
        Parse LLM response to extract reasoning steps and final answer.
        
        Recognizes patterns:
        - **Step-by-Step Solution:** with numbered steps
        - Final answer in \\boxed{...} format
        - Standard "final answer is X" patterns
        
        Returns:
            (steps_text, final_answer)
        """
        final_answer = None
        
        # Priority 1: Extract from \boxed{} - most reliable LLM answer format
        # First try to find all \boxed{} occurrences
        boxed_pattern = r'\\boxed\{([^{}]+)\}'
        all_matches = re.findall(boxed_pattern, response)
        
        if all_matches:
            # If multiple \boxed{} found (e.g., \boxed{15}, \boxed{2}), combine them
            if len(all_matches) > 1:
                # Join multiple boxed contents with comma
                final_answer = ','.join(match.strip() for match in all_matches)
            else:
                # Single \boxed{} found
                final_answer = all_matches[0].strip()
        
        # If no simple boxed found, try nested braces pattern
        if final_answer is None:
            nested_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            match = re.search(nested_pattern, response)
            if match:
                final_answer = match.group(1).strip()
        
        # Priority 2: Look for "Final Answer:" section
        if final_answer is None:
            final_answer_section = re.search(
                r'\*\*Final\s+Answer[:\*]*\*?\*?\s*(.+?)(?:\n\n|\Z)',
                response, 
                re.IGNORECASE | re.DOTALL
            )
            if final_answer_section:
                section_text = final_answer_section.group(1)
                # Try to extract boxed content from the section
                boxed_match = re.search(r'\\boxed\{([^{}]+)\}', section_text)
                if boxed_match:
                    final_answer = self._parse_boxed_content(boxed_match.group(1).strip())
                else:
                    # Extract number from the section
                    nums = re.findall(r'([+-]?[0-9]+\.?[0-9]*)', section_text)
                    if nums:
                        try:
                            final_answer = float(nums[-1])
                            if final_answer.is_integer():
                                final_answer = int(final_answer)
                        except ValueError:
                            pass
        
        # Priority 3: Standard patterns - "final answer is X", "answer: X"
        if final_answer is None:
            patterns = [
                r'final\s+answer\s*(?:is\s*)?[:\s]*([+-]?[0-9]+\.?[0-9]*)',
                r'answer\s*[:\s]*([+-]?[0-9]+\.?[0-9]*)',
                r'=\s*([+-]?[0-9]+\.?[0-9]*)\s*$',
                r'≈\s*([+-]?[0-9]+\.?[0-9]*)',
                r'therefore[,\s]+.*?([+-]?[0-9]+\.?[0-9]*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        final_answer = float(match.group(1))
                        if final_answer.is_integer():
                            final_answer = int(final_answer)
                        break
                    except (ValueError, AttributeError):
                        continue
        
        # Priority 4: Fallback - extract last number in text
        if final_answer is None:
            numbers = re.findall(r'\b([+-]?[0-9]+\.?[0-9]*)\b', response)
            if numbers:
                try:
                    final_answer = float(numbers[-1])
                    if final_answer.is_integer():
                        final_answer = int(final_answer)
                except ValueError:
                    pass
        
        return response, final_answer
    
    def _parse_boxed_content(self, content: str) -> Optional[Any]:
        """
        Parse content extracted from \\boxed{...}.
        
        Handles:
        - Single numbers: 15, 3.14, -5
        - Tuples/coordinates: (15, 3), (x, y)
        - Fractions: \\frac{1}{2}
        - Variables with values: x = 15
        - Text answers: "yes", "no"
        
        Returns:
            Parsed value (number, tuple, string, etc.)
        """
        content = content.strip()
        
        # Handle tuple/coordinate format: (15, 3) or (x, y)
        tuple_match = re.match(r'\(([^,]+),\s*([^)]+)\)', content)
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
                # Return as string tuple if not numeric
                return f"({val1}, {val2})"
        
        # Handle fraction: \frac{a}{b}
        frac_match = re.match(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', content)
        if frac_match:
            try:
                numerator = float(frac_match.group(1))
                denominator = float(frac_match.group(2))
                if denominator != 0:
                    result = numerator / denominator
                    if result.is_integer():
                        return int(result)
                    return result
            except ValueError:
                return f"{frac_match.group(1)}/{frac_match.group(2)}"
        
        # Handle simple number
        try:
            num = float(content)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            pass
        
        # Handle variable assignment: x = 15
        assign_match = re.match(r'[a-zA-Z_]\w*\s*=\s*([+-]?[0-9]+\.?[0-9]*)', content)
        if assign_match:
            try:
                num = float(assign_match.group(1))
                if num.is_integer():
                    return int(num)
                return num
            except ValueError:
                pass
        
        # Return as string for text answers
        return content if content else None
    
    def _check_answer_with_llm(self, computed: Any, expected: Any) -> Optional[bool]:
        """
        Use ChatGPT to verify if computed answer matches expected answer.
        
        This is useful when answers may be in different formats but semantically equivalent:
        - Fractions vs decimals: 1/2 vs 0.5
        - Different mathematical expressions: 2*3 vs 6
        - Coordinates in different formats: (15, 3) vs x=15, y=3
        - Equivalent text answers
        
        Args:
            computed: The computed answer from the solver
            expected: The expected correct answer
            
        Returns:
            True if ChatGPT judges answers as equivalent, False if different, None if error
        """
        if computed is None or expected is None:
            return None
        
        if not self.chatgpt_client:
            if self.verbose:
                print("⚠️ ChatGPT client not available - skipping LLM verification")
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

            completion = self.chatgpt_client.chat.completions.create(
                model=self.chatgpt_model,
                messages=[
                    {"role": "system", "content": "You are a precise mathematical answer verification expert. Compare answers for equivalence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content
            if not response:
                return None
            
            response = response.strip().upper()
            
            if "CORRECT" in response:
                return True
            elif "INCORRECT" in response:
                return False
            else:
                # If ChatGPT response is unclear, return None
                if self.verbose:
                    print(f"⚠️ ChatGPT verification unclear: {response}")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️ ChatGPT verification failed: {e}")
            return None
    
    def _check_answer(self, computed: Any, expected: Any) -> Optional[bool]:
        """
        Check if computed answer matches expected answer.
        
        Uses hybrid approach:
        1. First attempts numeric comparison with tolerance (fast, deterministic)
        2. Falls back to LLM-based verification for non-numeric or complex answers
        
        Args:
            computed: The computed answer
            expected: The expected answer
            
        Returns:
            True if answers match, False if different, None if cannot determine
        """
        if computed is None or expected is None:
            return None
        
        # Strategy 1: Try numeric comparison first
        numeric_result = self._check_answer_numeric(computed, expected)
        if numeric_result is not None:
            if self.verbose:
                print(f"✓ Numeric comparison: {numeric_result}")
            return numeric_result
        
        # Strategy 2: Fall back to ChatGPT verification for non-numeric or unclear cases
        if self.verbose:
            print("⚙️ Using ChatGPT verification for answer comparison...")
        
        llm_result = self._check_answer_with_llm(computed, expected)
        if llm_result is not None:
            if self.verbose:
                print(f"✓ ChatGPT verification: {llm_result}")
            return llm_result
        
        # If both methods fail, return None (cannot determine)
        if self.verbose:
            print("⚠️ Could not determine answer correctness")
        return None
    
    def _check_answer_numeric(self, computed: Any, expected: Any) -> Optional[bool]:
        """
        Check if answers match using numeric comparison.
        
        Handles numeric comparisons with tolerance for floating point errors.
        Returns None if answers cannot be compared numerically.
        
        Args:
            computed: The computed answer
            expected: The expected answer
            
        Returns:
            True if numerically equivalent, False if different, None if not numeric
        """
        try:
            # Parse expected if it's MathML/HTML
            if isinstance(expected, str) and '<math>' in expected:
                expected_text = self.parser.parse(expected)['text']
                # Extract number from text
                nums = re.findall(r'([0-9]+\.?[0-9]*)', expected_text)
                if nums:
                    expected = float(nums[0])
            
            # Convert to numbers
            if isinstance(expected, str):
                expected = float(expected)
            if isinstance(computed, str):
                computed = float(computed)
            
            # Numeric comparison with tolerance
            computed_f = float(computed)
            expected_f = float(expected)
            
            # Allow 1% relative error or 0.01 absolute error
            rel_error = abs(computed_f - expected_f) / max(abs(expected_f), 1e-9)
            abs_error = abs(computed_f - expected_f)
            
            is_close = rel_error < 0.01 or abs_error < 0.01
            return is_close
            
        except (ValueError, TypeError):
            # Not numeric - try string comparison as last resort
            if str(computed).strip() == str(expected).strip():
                return True
            # Cannot compare numerically, return None to trigger LLM verification
            return None


def solve_with_steps_hf(
    question: str,
    steps_path: Optional[Union[str, Path]] = None,
    *,
    new_correct_answer: Optional[Any] = None,
    api_key: Optional[str] = None,
    model: str = "zai-org/GLM-Z1-9B-0414:featherless-ai",
    parser: Optional[MathMLParser] = None,
    verbose: bool = False,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """
    Solve SAT math problem using HuggingFace LLM (one-shot reasoning).
    
    Compatible with sat_math_solver.solve_with_steps() interface.
    
    NOTE: This version doesn't use a steps JSON file like the original.
    The HF model does one-shot reasoning rather than following predefined steps.
    
    Args:
        question: Question content (may be HTML/MathML)
        steps_path: Path to save steps JSON (optional, will create minimal format)
        new_correct_answer: Expected correct answer for verification (optional)
        api_key: HuggingFace API key (or use HF_API_KEY env var)
        model: Model name
        parser: MathMLParser to convert question to text
        verbose: Log details
        temperature: LLM temperature
        
    Returns:
        Dict with:
        - final_result: Numeric result
        - steps_detail: List of reasoning steps
        - answer_text: Text answer
        - is_correct: Whether answer matches expected (if new_correct_answer provided)
        - error: Error string if any
    """
    api_key = api_key or os.getenv("HF_API_KEY")
    if not api_key:
        return {
            "final_result": None,
            "steps_detail": [],
            "answer_text": None,
            "is_correct": None,
            "error": "HF_API_KEY not set in environment or not provided",
        }
    
    parser = parser or MathMLParser()
    # question_text = parser.parse(question)['text'] if question else str(question)
    question_text = question 
    
    try:
        # Initialize solver
        solver = HuggingFaceMathSolver(
            api_key=api_key,
            model=model,
            verbose=verbose
        )
        
        # Solve (no expected answer in this interface)
        trace = solver.solve(
            mathml_explanation=question_text,
            correct_answer=new_correct_answer,  # Pass expected answer for verification
            question=question_text,
            temperature=temperature,
            steps_json_path=str(steps_path) if steps_path else None,
        )
        
        # Convert to sat_math_solver format
        steps_detail = []
        for step in trace.steps:
            steps_detail.append({
                "step_number": step.step_number,
                "thought": step.thought,
                "tool_name": step.tool_name or "reasoning",
                "tool_input": step.tool_input or {},
                "tool_output": step.tool_output,
                "params_meaning": [],
            })
        
        answer_text = f"The result is {trace.final_result}." if trace.final_result is not None else None
        
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


def main():
    """Example usage of HuggingFace solver"""
    # Example 1: MathML question
    question = """A drone is <math><mn>85</mn></math> m horizontally from the base of a tower. The angle of elevation from the drone to the top of the tower is <math><mn>45</mn><mo>°</mo></math>. Approximately how tall is the tower (to the nearest meter)?"""
    
    explanation = """Use <math><mi>tan</mi><mi>θ</mi><mo>=</mo><mfrac><mi>opp</mi><mi>adj</mi></mfrac></math>. Here <math><mi>tan</mi><mn>45</mn><mo>°</mo><mo>=</mo><mfrac><mi>h</mi><mn>85</mn></mfrac></math>, so <math><mi>h</mi><mo>=</mo><mn>85</mn><mi>tan</mi><mn>45</mn><mo>°</mo></math>. Since <math><mi>tan</mi><mn>45</mn><mo>°</mo><mo>=</mo><mn>1</mn></math>, <math><mi>h</mi><mo>=</mo><mn>85</mn></math> m."""
    
    correct_answer = "85"
    
    # Check for API key
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("⚠️ Set HF_API_KEY environment variable to run this example")
        print("Example: export HF_API_KEY='hf_...'")
        return
    
    print("\n" + "="*70)
    print("🧪 TESTING HUGGINGFACE MATH SOLVER")
    print("="*70)
    
    # Test 1: Using solve() method (agent-compatible)
    print("\n📝 Test 1: Using solve() method (agent.py compatible)")
    print("-" * 70)
    
    solver = HuggingFaceMathSolver(api_key=api_key, verbose=True)
    trace = solver.solve(
        question=question,
        mathml_explanation=explanation,
        correct_answer=correct_answer,
        steps_json_path="hf_steps_function_and_meaning.json",
    )
    
    trace.print_summary()
    
    # Test 2: Using solve_with_steps_hf() (sat_math_solver compatible)
    print("\n" + "="*70)
    print("📝 Test 2: Using solve_with_steps_hf() (sat_math_solver.py compatible)")
    print("="*70 + "\n")
    
    result = solve_with_steps_hf(
        question=question,
        steps_path="hf_steps_simple.json",
        api_key=api_key,
        verbose=True,
    )
    
    if result["error"]:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"\n✅ Final Result: {result['final_result']}")
        print(f"📝 Answer Text: {result['answer_text']}")
        print(f"📊 Steps: {len(result['steps_detail'])}")
    
    print("\n" + "="*70)
    print("✅ TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
