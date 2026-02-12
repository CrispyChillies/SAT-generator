"""
SAT Math Solver - LLM answers SAT math questions by using in sequence
the tools specified in steps_function_and_meaning.json.
LLM chooses parameter values for each tool based on the problem.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from tools import math_tools
from mathml_parser import MathMLParser
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Set OPENAI_API_KEY in the environment or pass llm.")

def load_steps(steps_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read JSON file containing list of steps and parameter meanings."""
    with open(steps_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("steps", [])


def get_tool_by_name(name: str):
    """Get tool from math_tools by name (add, multiply, divide, ...)."""
    for tool in math_tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool not found with name: {name}")


def _build_step_prompt(
    step: Dict[str, Any],
    step_index: int,
    question_text: str,
    previous_outputs: List[Any],
    previous_steps: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build prompt for one step: tool description + param meanings + values from previous step (if any)."""
    function_name = step["function_name"]
    params_meaning = step.get("params_meaning", [])
    step_num = step.get("step_number", step_index + 1)
    previous_steps = previous_steps or []
    lines = []

    if previous_outputs and previous_steps:
        lines.append("Previous steps (param, meaning, output of each step):")
        for k, (prev_step, out) in enumerate(zip(previous_steps, previous_outputs), 1):
            lines.append(f"  **Step {k}**:")
            for p in prev_step.get("params_meaning", []):
                param = p.get("param", "?")
                meaning = p.get("meaning", "")
                lines.append(f"    - **{param}**: {meaning}")
            lines.append(f"    → output = **{out}**")
        lines.append("")

    lines.extend([
        f"**Step {step_num}** – You must call exactly one tool: `{function_name}`.",
        "",
        "Parameters (param, meaning – choose value from the problem or use output from a previous step above):",
    ])
    for p in params_meaning:
        param = p.get("param", "?")
        meaning = p.get("meaning", "")
        lines.append(f"- **{param}**: {meaning}")

    lines.extend([
        "",
        "Call the tool with the correct name and the (numeric) parameters you chose. Call only once, do not call any other tool.",
    ])
    return "\n".join(lines)


def run_llm_step(
    llm_with_tool,
    question_text: str,
    step: Dict[str, Any],
    step_index: int,
    previous_outputs: List[Any],
    system_prompt: str,
    previous_steps: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict], Optional[Any], str]:
    """
    Give LLM the prompt for the current step; LLM chooses parameters and calls the tool.
    Returns (tool_call_dict, tool_output, thought).
    """
    step_prompt = _build_step_prompt(
        step, step_index, question_text, previous_outputs, previous_steps
    )
    user_content = (
        f"**Question:**\n{question_text}\n\n"
        f"**Task:**\n{step_prompt}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm_with_tool.invoke(messages)
    thought = (response.content or "").strip()

    if not getattr(response, "tool_calls", None) or not response.tool_calls:
        return None, None, thought

    tool_call = response.tool_calls[0]
    return tool_call, None, thought


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Any:
    """Call tool by name with args, return the result."""
    tool = get_tool_by_name(tool_name)
    return tool.invoke(args)


def solve_with_steps(
    question: str,
    steps_path: Union[str, Path],
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    parser: Optional[MathMLParser] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    LLM answers SAT question by calling tools in sequence from the JSON file.
    LLM chooses parameter values for each tool based on the problem.

    - question: Question content (may be HTML/MathML).
    - steps_path: Path to steps_function_and_meaning.json.
    - api_key: OpenAI API key (if None, use env OPENAI_API_KEY).
    - model: Model name.
    - parser: MathMLParser to convert question to text (if None, use default).
    - verbose: Log each step.

    Returns:
      - final_result: Numeric result of the last step.
      - steps_detail: List of each step (thought, tool_name, tool_input, tool_output).
      - answer_text: Text answer (if any).
      - error: Error string if any.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    steps_path = Path(steps_path)
    if not steps_path.exists():
        return {
            "final_result": None,
            "steps_detail": [],
            "answer_text": None,
            "error": f"File not found: {steps_path}",
        }

    steps = load_steps(steps_path)
    if not steps:
        return {
            "final_result": None,
            "steps_detail": [],
            "answer_text": None,
            "error": "JSON file contains no steps.",
        }

    parser = parser or MathMLParser()
    question_text = parser.parse(question) if question else ""
    if not question_text:
        question_text = question

    llm = ChatOpenAI(model=model, temperature=0)
    system_prompt = (
        "You are an SAT math assistant. You must answer the question by calling IN THE CORRECT ORDER "
        "the tools specified for each step. For each step, you may call exactly one such tool with "
        "numeric parameters (float/int); for any parameter that requires 'use previous step result', "
        "use the value given; for the rest, derive from the problem (e.g. wholesale price, percentage, ...). "
        "Answer by calling the tool, not just stating the answer in words during the computation step."
    )

    steps_detail: List[Dict[str, Any]] = []
    previous_outputs: List[Any] = []
    final_result = None
    err_msg: Optional[str] = None

    for step_index, step in enumerate(steps):
        function_name = step["function_name"]
        tool = get_tool_by_name(function_name)
        llm_with_tool = llm.bind_tools([tool])

        if verbose:
            print(f"\n--- Step {step_index + 1}: {function_name} ---")

        tool_call, _, thought = run_llm_step(
            llm_with_tool,
            question_text,
            step,
            step_index,
            previous_outputs,
            system_prompt,
            previous_steps=steps[:step_index],
        )

        if tool_call is None:
            err_msg = (
                f"Step {step_index + 1}: LLM did not call tool. Thought: {thought}"
            )
            if verbose:
                print(err_msg)
            break

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        if tool_name != function_name:
            err_msg = f"Step {step_index + 1}: LLM called tool '{tool_name}', required '{function_name}'."
            if verbose:
                print(err_msg)
            break

        try:
            output = execute_tool(tool_name, tool_args)
        except Exception as e:
            err_msg = f"Step {step_index + 1}: Error running tool: {e}"
            if verbose:
                print(err_msg)
            break

        previous_outputs.append(output)
        final_result = output
        steps_detail.append({
            "step_number": step_index + 1,
            "thought": thought,
            "tool_name": tool_name,
            "tool_input": tool_args,
            "tool_output": output,
            "params_meaning": step.get("params_meaning", []),
        })
        if verbose:
            print(f"  Thought: {thought[:200]}...")
            print(f"  Tool: {tool_name}({tool_args}) = {output}")

    # Optionally call LLM once more to state the answer in text
    answer_text = None
    if not err_msg and steps_detail and final_result is not None:
        answer_text = f"The result is {final_result}."  # can replace with a short LLM call

    return {
        "final_result": final_result,
        "steps_detail": steps_detail,
        "answer_text": answer_text,
        "error": err_msg,
    }


def main():
    """Example: solve souvenir question with steps_function_and_meaning.json."""
    steps_path = Path(__file__).parent / "steps_function_and_meaning.json"
    question = """A drone is <math><mn>85</mn></math> m horizontally from the base of a tower. The angle of elevation from the drone to the top of the tower is <math><mn>45</mn><mo>°</mo></math>. Approximately how tall is the tower (to the nearest meter)?"""
    # Can use HTML/MathML version from questions_practice_test if available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY in the environment or pass api_key to solve_with_steps.")
        return

    result = solve_with_steps(
        question=question,
        steps_path=steps_path,
        model="gpt-4.1-mini",
        verbose=True,
    )

    if result["error"]:
        print("Error:", result["error"])
    else:
        print("\nFinal result:", result["final_result"])
        print("Step details:")
        for d in result["steps_detail"]:
            print(f"  Step {d['step_number']}: {d['tool_name']}({d['tool_input']}) = {d['tool_output']}")


if __name__ == "__main__":
    main()
