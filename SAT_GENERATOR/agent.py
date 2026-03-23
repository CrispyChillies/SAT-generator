import os
import json
from typing import Dict, Any, List, Optional, Annotated, TypedDict
from datetime import datetime
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import math_tools
from mathml_parser import MathMLParser
from pydantic import BaseModel, ConfigDict, Field


def _build_chat_model(model: str):
    kwargs: Dict[str, Any] = {"model": model}
    if "gpt-5" not in (model or "").lower():
        kwargs["temperature"] = 0
    return ChatOpenAI(**kwargs)

# ============================================================================
# TOOL EXECUTION TRACKER
# ============================================================================

class ToolExecutionStep(BaseModel):
    """Structure for a single tool execution step"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_number: int
    thought: str  # LLM's reasoning
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    param_explanation: str = ""  # NEW: LLM explains param meanings
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ExecutionTrace(BaseModel):
    """Complete execution trace"""
    problem_description: str
    expected_answer: Any
    steps: List[ToolExecutionStep] = []
    final_result: Optional[Any] = None
    is_correct: Optional[bool] = None
    error: Optional[str] = None
    total_steps: int = 0
    
    def add_step(self, thought: str, tool_name: str, tool_input: Dict[str, Any], 
                 tool_output: Any, param_explanation: str = ""):
        """Add a new execution step"""
        step = ToolExecutionStep(
            step_number=len(self.steps) + 1,
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            param_explanation=param_explanation
        )
        self.steps.append(step)
        self.total_steps = len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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

    def _steps_function_and_meaning(self) -> List[Dict[str, Any]]:
        """Chỉ trả về tên hàm và meaning của từng param cho mỗi step."""
        out = []
        for s in self.steps:
            params_meaning: List[Dict[str, str]] = []
            if s.param_explanation:
                try:
                    parsed = json.loads(s.param_explanation)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "meaning" in item:
                                p = item.get("param")
                                m = item.get("meaning", "")
                                params_meaning.append({"param": p, "meaning": m})
                except json.JSONDecodeError:
                    pass
            out.append({
                "step_number": s.step_number,
                "function_name": s.tool_name,
                "params_meaning": params_meaning,
            })
        return out

    def export_steps_json(self, filepath: str) -> None:
        """Lưu tên hàm và meaning của các params của tất cả các step ra file JSON."""
        data = {"steps": self._steps_function_and_meaning()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Print human-readable summary"""
        print(f"\n{'='*70}")
        print("📋 EXECUTION TRACE SUMMARY")
        print(f"{'='*70}")
        print(f"Problem: {self.problem_description}")
        print(f"Expected: {self.expected_answer}")
        print(f"Total Steps: {self.total_steps}")
        
        for step in self.steps:
            print(f"\n  {'─'*66}")
            print(f"  Step {step.step_number}:")
            
            print(f"  💭 LLM's Thought:")
            for line in step.thought.split('\n'):
                if line.strip():
                    print(f"     {line.strip()}")
            
            print(f"\n  🔧 Action: {step.tool_name}({step.tool_input})")
            print(f"  📤 Observation: {step.tool_output}")
            
            if step.param_explanation:
                print(f"\n  📝 Parameter Explanation:")
                try:
                    items = json.loads(step.param_explanation)
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and "param" in item and "meaning" in item:
                                val = item.get("value", "")
                                print(f"     {{'param': {item['param']!r}, 'value': {repr(val)}, 'meaning': {item['meaning']!r}}}")
                            else:
                                print(f"     {item}")
                    else:
                        print(f"     {step.param_explanation}")
                except json.JSONDecodeError:
                    for line in step.param_explanation.split('\n'):
                        if line.strip():
                            print(f"     {line.strip()}")
        
        print(f"\n{'─'*70}")
        print(f"Final Result: {self.final_result}")
        status = "✅ CORRECT" if self.is_correct is True else ("❌ INCORRECT" if self.is_correct is False else "(no verdict)")
        print(f"Status: {status}")
        
        if self.error:
            print(f"Error: {self.error}")
        
        print(f"{'='*70}\n")

# ============================================================================
# LANGGRAPH STATE
# ============================================================================

class AgentState(TypedDict):
    """State for the math reasoning agent"""
    messages: Annotated[List, operator.add]  # Conversation history
    problem_description: str
    question_text: str  # Câu hỏi gốc (để giải thích tham số trong ngữ cảnh)
    expected_answer: Any
    trace: ExecutionTrace
    current_step: int
    max_iterations: int
    verbose: bool

# ============================================================================
# LANGGRAPH MATH AGENT
# ============================================================================

class LangGraphMathAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", verbose: bool = False):
        """
        Initialize Math Agent using LangGraph
        
        Args:
            api_key: OpenAI API key
            model: Model name
            verbose: Print detailed execution logs
        """
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = _build_chat_model(model)
        self.llm_with_tools = self.llm.bind_tools(math_tools)
        self.parser = MathMLParser()
        self.tools = math_tools
        self.verbose = verbose
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("llm", self._llm_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("explain_params", self._explain_params_node)
        workflow.add_node("final", self._final_node)
        
        # Set entry point
        workflow.set_entry_point("llm")
        
        # Add edges
        workflow.add_conditional_edges(
            "llm",
            self._should_continue,
            {
                "continue": "tools",
                "end": "final"
            }
        )
        workflow.add_edge("tools", "explain_params")
        workflow.add_edge("explain_params", "llm")
        workflow.add_edge("final", END)
        
        return workflow.compile()
    
    def _llm_node(self, state: AgentState) -> AgentState:
        """Node 1: LLM thinks and chooses tools"""
        messages = state["messages"]
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("🤔 LLM THINKING...")
            print(f"{'='*70}")
        
        # Call LLM with tools
        response = self.llm_with_tools.invoke(messages)
        
        if self.verbose and response.content:
            print(f"💭 Thought: {response.content}")
        
        return {
            **state,
            "messages": [response]
        }
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """Node 2: Execute tools"""
        last_message = state["messages"][-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        # Chỉ thực thi 1 tool đầu tiên mỗi lần LLM trả lời (tránh duplicate/thừa)
        first_tool_call = last_message.tool_calls[0]
        tool_calls_to_run = [first_tool_call]
        # Cập nhật lại messages để lịch sử chỉ còn 1 tool call (đồng bộ với số ToolMessage trả về)
        messages = list(state["messages"])
        messages[-1] = AIMessage(
            content=last_message.content,
            tool_calls=[first_tool_call]
        )
        state = {**state, "messages": messages}
        
        tool_messages = []
        for tool_call in tool_calls_to_run:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            # Find and execute tool
            tool_func = next((t for t in self.tools if t.name == tool_name), None)
            if tool_func:
                try:
                    result = tool_func.invoke(tool_input)
                    
                    if self.verbose:
                        print(f"\n{'─'*66}")
                        print(f"🔧 Executing: {tool_name}({tool_input})")
                        print(f"📤 Result: {result}")
                    
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                    
                    # Store thought from previous LLM response
                    thought = last_message.content if last_message.content else f"Using {tool_name}"
                    
                    # Temporarily store step info (will be completed in explain_params)
                    state["trace"].steps.append(ToolExecutionStep(
                        step_number=len(state["trace"].steps) + 1,
                        thought=thought,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=result,
                        param_explanation=""  # Will be filled in next node
                    ))
                    
                except Exception as e:
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                    )
        
        return {
            **state,
            "messages": tool_messages,
            "current_step": state["current_step"] + 1
        }
    
    def _explain_params_node(self, state: AgentState) -> AgentState:
        """Node 3: LLM explains the meaning of parameters. Chỉ giải thích cho step hiện tại (step vừa chạy)."""
        if not state["trace"].steps:
            return state
        
        trace = state["trace"]
        # Chỉ xử lý step cuối (step hiện tại vừa được tools node thực thi)
        step = trace.steps[-1]
        if step.param_explanation:
            return state
        
        question = state.get("question_text") or state.get("problem_description") or "(No question)"
        problem_desc = state.get("problem_description") or ""
        # Lịch sử các bước đã chạy (trước bước hiện tại) để LLM có ngữ cảnh đưa ra meaning chính xác
        previous_steps_text = ""
        if len(trace.steps) > 1:
            prev_steps = trace.steps[:-1]
            lines = ["**Previous steps in this solution:**"]
            for s in prev_steps:
                lines.append(f"- Step {s.step_number}: {s.tool_name}({json.dumps(s.tool_input)}) → {s.tool_output}")
                if s.param_explanation:
                    lines.append(f"  (params meaning: {s.param_explanation[:200]}{'...' if len(s.param_explanation) > 200 else ''})")
            previous_steps_text = "\n".join(lines) + "\n\n"

        system_content = (
            "You are a helpful math tutor. Explain each parameter in GENERAL terms only: "
            "what it represents (e.g. 'monthly payment', 'number of months'). "
            "Do NOT include specific numbers or values in the 'meaning' field. "
            "The 'param' field MUST be the exact parameter NAME (key) from the tool call, e.g. 'a', 'b', not a number. "
            "Use the problem context and previous steps (if any) to give accurate, context-aware meanings. "
            "Respond with valid JSON only: an array of objects with keys param, value, meaning."
        )
        
        problem_context_block = ""
        if problem_desc and problem_desc != question:
            problem_context_block = f"**Problem / explanation:** {problem_desc}\n\n"

        explain_prompt = f"""**Question:** {question}
{problem_context_block}{previous_steps_text}**Current step:** The tool '{step.tool_name}' was just called with these parameters (use these exact keys as "param"):
{json.dumps(step.tool_input, indent=2)}

Result: {step.tool_output}

Explain what each parameter REPRESENTS in general. Use only generic descriptions. Do NOT put specific numbers in "meaning". The "param" must be the parameter NAME from the list above (e.g. "a", "b"), not a number.

Respond with a JSON array only. Each element: "param" (key from above), "value" (same value from above), "meaning" (short generic description).

Example:
[
  {{"param": "a", "value": 36, "meaning": "the number of months in the lease period"}},
  {{"param": "b", "value": 1000, "meaning": "the fixed initial cost in dollars"}}
]

Output ONLY the JSON array:"""
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=explain_prompt)
        ]

        response = self.llm.invoke(messages)
        raw = (response.content or "").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                fixed = []
                for item in parsed:
                    if isinstance(item, dict) and "meaning" in item:
                        p = item.get("param")
                        v = item.get("value")
                        if p is not None and isinstance(step.tool_input, dict):
                            if not isinstance(p, str) or p not in step.tool_input:
                                for k, val in step.tool_input.items():
                                    if val == p or (v is not None and val == v):
                                        p, v = k, val
                                        break
                            if isinstance(p, str) and p in step.tool_input:
                                v = step.tool_input[p]
                        fixed.append({"param": p, "value": v, "meaning": item.get("meaning", "")})
                explanation = json.dumps(fixed, indent=2, ensure_ascii=False) if fixed else raw
            else:
                explanation = raw
        except json.JSONDecodeError:
            explanation = raw
        
        trace.steps[-1].param_explanation = explanation
        if self.verbose:
            print(f"\n📝 Parameter Explanation (step {step.step_number}):")
            print(f"   {explanation}")
        
        trace.total_steps = len(trace.steps)
        combined = f"**Parameter explanation(s):** {explanation}"
        return {
            **state,
            "messages": [AIMessage(content=combined)]
        }
    
    def _final_node(self, state: AgentState) -> AgentState:
        """Node 4: Final verdict"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'content'):
            final_answer = last_message.content
            state["trace"].final_result = final_answer
            state["trace"].is_correct = self._parse_verdict(final_answer)
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"🎯 FINAL ANSWER: {final_answer}")
                print(f"{'='*70}")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue or end"""
        last_message = state["messages"][-1]
        
        # Check max iterations
        if state["current_step"] >= state["max_iterations"]:
            return "end"
        
        # If LLM called tools, continue
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end
        return "end"
    
    def _parse_verdict(self, text: str) -> Optional[bool]:
        """Parse CORRECT/INCORRECT from LLM output"""
        if not text:
            return None
        upper = text.upper()
        if "INCORRECT" in upper:
            return False
        if "CORRECT" in upper:
            return True
        return None
    
    def solve(self, mathml_explanation: str, correct_answer: Any, 
              question: str = "", max_iterations: int = 25,
              steps_json_path: Optional[str] = None) -> ExecutionTrace:
        """
        Solve math problem using LangGraph workflow
        
        Args:
            mathml_explanation: MathML explanation
            correct_answer: Expected answer
            question: Optional question text
            max_iterations: Maximum number of iterations
            steps_json_path: Nếu có, lưu tên hàm + meaning params của từng step ra file JSON này
            
        Returns:
            ExecutionTrace with complete solving history
        """
        readable = self.parser.parse(mathml_explanation)['text']
        question_text = self.parser.parse(question)['text'] if question else question

        trace = ExecutionTrace(
            problem_description=readable,
            expected_answer=correct_answer
        )

        print(f"\n{'='*70}")
        if question_text:
            print(f"❓ Question: {question_text}")
        print(f"📝 Explanation: {readable}")
        print(f"🎯 Expected Answer: {correct_answer}")
        print(f"{'='*70}\n")

        # Initial system message
        system_prompt = f"""You are a mathematical reasoning agent. You MUST use the available tools to do any calculation—never do arithmetic in your head or in text.

**Current Problem:**
Question: {question_text or "(No question provided)"}
Explanation: {readable}
Expected Answer: {correct_answer}

**Rules (mandatory):**
1. You MUST call tools for calculations. Do NOT just reason to the answer without calling at least one tool (e.g. use add, multiply, etc. to verify or compute values).
2. In each message, call at most ONE tool. Do not call multiple tools in the same response.
3. After you get tool results, you may do one more tool call if needed, or conclude. Only after using tools, compare with the expected answer and end with exactly "CORRECT" or "INCORRECT".

**Important:**
- For trig functions, input must be in RADIANS
- Be precise with calculations
- π (pi) = 3.14159265359
- e = 2.71828182846

Begin by calling one tool to perform or verify the needed calculation. Do not skip tools."""

        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please solve this problem step by step. You must use the math tools (e.g. add, multiply) for calculations—do not give the final answer without calling at least one tool.")
            ],
            "problem_description": readable,
            "question_text": question_text or "",
            "expected_answer": correct_answer,
            "trace": trace,
            "current_step": 0,
            "max_iterations": max_iterations,
            "verbose": self.verbose
        }

        # In graph ra trước (Mermaid có mũi tên), làm sạch cho terminal
        print("\n📊 WORKFLOW GRAPH (trước khi chạy):")
        mermaid = self.graph.get_graph().draw_mermaid(with_styles=False)
        mermaid = mermaid.replace("&nbsp;", " ").replace(" -. ", " -[").replace(" .-> ", "]-> ")
        for line in mermaid.strip().split("\n"):
            print(line.strip())
        print()

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            trace = final_state["trace"]
            
        except Exception as e:
            trace.error = str(e)
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()

        if steps_json_path:
            trace.export_steps_json(steps_json_path)
        return trace

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    agent = LangGraphMathAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4.1-mini",
        verbose=True
    )
    
    # question = "A drone is <math><mn>120</mn></math> m horizontally from the base of a tower. The angle of elevation from the drone to the top of the tower is <math><mn>30</mn><mo>°</mo></math>. Approximately how tall is the tower (to the nearest meter)?"
    
    # explanation = "Use <math><mi>tan</mi><mi>θ</mi><mo>=</mo><mfrac><mi>opp</mi><mi>adj</mi></mfrac></math>. Here <math><mi>tan</mi><mn>30</mn><mo>°</mo><mo>=</mo><mfrac><mi>h</mi><mn>120</mn></mfrac></math>, so <math><mi>h</mi><mo>=</mo><mn>120</mn><mi>tan</mi><mn>30</mn><mo>°</mo></math>. Since <math><mi>tan</mi><mn>30</mn><mo>°</mo><mo>≈</mo><mn>0.577</mn></math>, <math><mi>h</mi><mo>≈</mo><mn>120</mn><mo>×</mo><mn>0.577</mn><mo>≈</mo><mn>69</mn></math> m."
    
    # correct_answer = "<math><mn>69</mn></math>"

    question = "The line graph shows the percent of cars for sale at a used car lot on a given day by model year. The line graph: Begins at 2010, 12% Remains level to 2011, 12% Remains level to 2012, 12% Falls sharply to 2013, 8% Falls sharply to 2014, 4% Rises sharply to 2015, 9% Rises gradually to 2016, 10% Remains level to 2017, 10% Rises gradually to 2018, 11% Remains level to 2019, 11% For what model year is the percent of cars for sale the smallest?"

    explanation = "Choice C is correct. For the given line graph, the percent of cars for sale at a used car lot on a given day is represented on the vertical axis. The percent of cars for sale is the smallest when the height of the line graph is the lowest. The lowest height of the line graph occurs for cars with a model year of 2014. Choice A is incorrect and may result from conceptual errors. Choice B is incorrect and may result from conceptual errors. Choice D is incorrect and may result from conceptual errors."

    correct_answer = "2014"
    
    print("\n🧪 TEST WITH LANGGRAPH AGENT")
    trace = agent.solve(
        question=question,
        mathml_explanation=explanation,
        correct_answer=correct_answer,
        steps_json_path="steps_function_and_meaning.json",
    )
    
    trace.print_summary()
    print("\n📄 Đã lưu tên hàm và meaning params ra: steps_function_and_meaning.json")
    
    # Export to dict if needed
    print("\n📦 TRACE AS DICT:")
    import json
    print(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False))