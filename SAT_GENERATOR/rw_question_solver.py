"""
SAT Reading & Writing Solver - LLM solves R&W questions using reasoning tools.
Unlike math solver which uses computational tools, this uses logical analysis tools.
"""

import os
import re
import json
from pathlib import Path
import string
from typing import Dict, Any, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from rw_reasoning_tools import rw_reasoning_tools, get_rw_tool_by_name
from mathml_parser import MathMLParser

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Set OPENAI_API_KEY in environment")

def solve_rw_question(
    paragraph: str,
    question: str,
    choices: List[str],
    *,
    skill: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve an SAT Reading & Writing question using reasoning tools.
    
    Args:
        paragraph: The paragraph providing context
        question: The question being asked
        choices: List of 4 answer choices
        skill: The skill being tested (e.g., "Inferences", "Command of Evidence")
        api_key: OpenAI API key (if None, use OPENAI_API_KEY from env)
        model: Model name
        verbose: Print reasoning steps
    
    Returns:
        Dict with:
        - final_answer_letter: The selected answer (A, B, C, or D)
        - final_result: The actual choice text (for comparison)
        - answer_text: Formatted answer string (for display)
        - reasoning_steps: List of reasoning steps taken
        - explanation: Text explanation of the answer
        - error: Error message if any
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Build system prompt
    system_prompt = f"""You are an expert SAT Reading & Writing tutor. Your task is to solve the question by analyzing the paragraph and evaluating each answer choice systematically.

Skill being tested: {skill or "Not specified"}

You have access to reasoning tools to help you analyze the text and choices. Use them to:
1. Understand the paragraph's main claims and structure
2. Evaluate each answer choice against the paragraph
3. Identify the best answer and explain why

Be thorough and logical in your reasoning."""

    # Build user prompt
    # Do not include A., B., C., D. prefixes — UI handles labeling
    choices_text = "\n".join([f"{choice}" for choice in choices])
    
    user_prompt = f"""Please solve this SAT Reading & Writing question.

**Paragraph:**
{paragraph}

**Question:**
{question}

**Answer Choices:**
{choices_text}

Use the available reasoning tools to analyze the paragraph and choices systematically. Then provide your final answer."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(rw_reasoning_tools)
    
    reasoning_steps = []
    error_msg = None
    final_answer_letter = None
    explanation = None
    
    # Multi-step reasoning loop (allow up to 10 tool calls)
    max_iterations = 10
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            error_msg = f"Error calling LLM: {e}"
            if verbose:
                print(error_msg)
            break
        
        # Check if LLM provided a text response (final answer)
        if response.content:
            if verbose:
                print(f"LLM response: {response.content[:200]}...")
            
            # Check if this looks like a final answer
            content_upper = response.content.upper()
            if any(phrase in content_upper for phrase in ["FINAL ANSWER", "THE ANSWER IS", "CORRECT ANSWER", "CHOICE"]):
                explanation = response.content
                
                # Extract answer letter
                for letter in ["A", "B", "C", "D"]:
                    if letter in content_upper:
                        # Check if it's actually indicating this as the answer
                        if any(phrase in content_upper for phrase in [
                            f"ANSWER IS {letter}",
                            f"CHOICE {letter}",
                            f"SELECT {letter}",
                            f"{letter} IS CORRECT",
                            f"THE CORRECT ANSWER: {letter}",
                        ]):
                            final_answer_letter = letter
                            break
                
                if final_answer_letter:
                    reasoning_steps.append({
                        "step_number": iteration + 1,
                        "action": "final_answer",
                        "response": response.content,
                    })
                    break
        
        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                
                if verbose:
                    print(f"  Tool: {tool_name}")
                    print(f"  Args: {tool_args}")
                
                try:
                    # Execute tool
                    tool = get_rw_tool_by_name(tool_name)
                    result = tool.invoke(tool_args)
                    
                    if verbose:
                        print(f"  Result: {result[:200]}...")
                    
                    reasoning_steps.append({
                        "step_number": iteration + 1,
                        "tool_name": tool_name,
                        "tool_input": tool_args,
                        "tool_output": result,
                    })
                    
                    # Add tool result to messages
                    messages.append(AIMessage(content="", tool_calls=[tool_call]))
                    messages.append(HumanMessage(
                        content=f"Tool '{tool_name}' returned:\n{result}\n\nContinue your analysis or provide final answer."
                    ))
                    
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {e}"
                    if verbose:
                        print(error_msg)
                    reasoning_steps.append({
                        "step_number": iteration + 1,
                        "tool_name": tool_name,
                        "tool_input": tool_args,
                        "error": str(e),
                    })
                    break
            
            if error_msg:
                break
        else:
            # No tool calls and no clear final answer
            if iteration == max_iterations - 1:
                # Last iteration, force an answer
                explanation = response.content or "No clear answer provided"
                # Try to extract any letter mentioned
                content_upper = (response.content or "").upper()
                for letter in ["A", "B", "C", "D"]:
                    if letter in content_upper:
                        final_answer_letter = letter
                        break
            else:
                # Ask for final answer
                messages.append(AIMessage(content=response.content or ""))
                messages.append(HumanMessage(
                    content="Please provide your final answer. State which choice (A, B, C, or D) is correct and explain why."
                ))
    
    if not final_answer_letter and not error_msg:
        error_msg = "Could not determine final answer from reasoning"
    
    # Extract the actual choice text for final_result
    final_result = None
    answer_text = None
    
    if final_answer_letter and final_answer_letter in "ABCD":
        choice_index = ord(final_answer_letter) - ord('A')
        if 0 <= choice_index < len(choices):
            final_result = choices[choice_index]
            answer_text = f"The answer is {final_answer_letter}: {final_result}"
    
    return {
        "final_answer_letter": final_answer_letter,
        "final_result": final_result,
        "answer_text": answer_text,
        "reasoning_steps": reasoning_steps,
        "explanation": explanation,
        "error": error_msg,
    }


def solve_rw_question_simple(
    paragraph: str,
    question: str,
    choices: List[str],
    *,
    skill: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Simplified solver that uses one tool call (compare_choices) to get answer directly.
    Faster and more reliable than multi-step reasoning for most questions.
    
    Args:
        paragraph: The paragraph providing context
        question: The question being asked
        choices: List of 4 answer choices
        skill: The skill being tested
        api_key: OpenAI API key
        model: Model name
        verbose: Print reasoning
    
    Returns:
        Dict with:
        - final_answer_letter: The answer letter (A/B/C/D)
        - final_result: The actual choice text (for comparison with generated answer)
        - answer_text: Formatted answer string (for display)
        - reasoning_steps: List of reasoning steps
        - explanation: Full explanation
        - error: Error message if any
    """
    parser = MathMLParser()
    parsed_paragraph = parser.parse_paragraph(paragraph)
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Use compare_choices tool directly
    tool = get_rw_tool_by_name("compare_choices")
    
    if verbose:
        print("Using compare_choices tool to analyze all options...")    
    try:
        paragraph_content = parsed_paragraph['text']
        if parsed_paragraph['has_graph']:
            graph = parsed_paragraph['graph']
            paragraph_content = f"{paragraph_content}\n\nGraph: {graph.raw_long_description}"
        print(paragraph_content)
        result = tool.invoke({
            "paragraph": paragraph_content,
            "question": question,
            "choices": choices,
            "skill": skill or "General Reading",
        })
        
        if verbose:
            print(f"Analysis result:\n{result}\n")
        
        # Extract answer letter from result
        final_answer_letter = None
        
        # Check for explicit best answer statements (most reliable)
        import re
        
        # STRATEGY 1: Look for "ANSWER: X" format (standardized output from compare_choices)
        # This should appear at the END of the response
        answer_match = re.search(r'^ANSWER:\s*([A-D])\s*$', result, re.MULTILINE | re.IGNORECASE)
        if answer_match:
            final_answer_letter = answer_match.group(1).upper()
        
        # STRATEGY 2: Fallback - extract from conclusion section
        if not final_answer_letter:
            conclusion_patterns = [
                r'(?:best answer|final answer)[:\s]+(?:is\s+)?\*?\*?([A-D])\*?\*?',
                r'ranking.*?best\s+to\s+worst.*?([A-D])',
            ]
            
            for pattern in conclusion_patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    final_answer_letter = match.group(1).upper()
                    break
        
        # Extract the actual choice text for final_result
        final_result = None
        answer_text = None
        
        if final_answer_letter and final_answer_letter in "ABCD":
            choice_index = ord(final_answer_letter) - ord('A')
            if 0 <= choice_index < len(choices):
                final_result = choices[choice_index]
                answer_text = f"The answer is {final_answer_letter}: {final_result}"
        
        reasoning_steps = [{
            "step_number": 1,
            "tool_name": "compare_choices",
            "tool_input": {
                "paragraph": paragraph,
                "question": question,
                "choices": f"[{len(choices)} choices]",
                "skill": skill,
            },
            "tool_output": result,
        }]
        
        return {
            "final_answer_letter": final_answer_letter,
            "final_result": final_result,  # The actual choice text
            "answer_text": answer_text,    # Formatted answer for display
            "reasoning_steps": reasoning_steps,
            "explanation": result,
            "error": None if final_answer_letter else "Could not extract answer letter from analysis",
        }
        
    except Exception as e:
        error_msg = f"Error running compare_choices tool: {e}"
        if verbose:
            print(error_msg)
        
        return {
            "final_answer_letter": None,
            "final_result": None,
            "answer_text": None,
            "reasoning_steps": [],
            "explanation": None,
            "error": error_msg,
        }


def main():
    """Example: solve a sample R&W question."""
    # Example question (Inferences)
    paragraph = """Marta Coll and colleagues' 2010 Mediterranean Sea biodiversity census reported approximately 17,000 species, nearly double the number reported in Carlo Bianchi and Carla Morri's 2000 census. Much of this increase is likely due to the use of more sophisticated methods in the 2010 census, but another factor is that the morphological variability of microorganisms is poorly understood. Researchers' decisions on such matters therefore can be highly consequential. Indeed, the two censuses reported similar counts of vertebrate, plant, and algal species, suggesting that ______"""
    
    question = "Which choice most logically completes the text?"
    
    choices = [
        "Coll and colleagues reported a much higher number of species than Bianchi and Morri did largely due to the inclusion of invertebrate species.",
        "some differences observed in microorganisms may have been treated as variations within species by Bianchi and Morri but treated as indicative of distinct species by Coll and colleagues.",
        "Bianchi and Morri may have been less sensitive to the degree of morphological variation displayed within a typical species of microorganism than Coll and colleagues were.",
        "the absence of clarity regarding how to differentiate among species of microorganisms may have resulted in Coll and colleagues underestimating the total number of species in the Mediterranean Sea.",
    ]
    
    print("Solving sample R&W question...\n")
    
    # Use simple solver
    result = solve_rw_question_simple(
        paragraph=paragraph,
        question=question,
        choices=choices,
        skill="Inferences",
        verbose=True,
    )
    
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"\nFinal Answer: {result['final_answer_letter']}")
        print(f"\nExplanation:\n{result['explanation']}")


if __name__ == "__main__":
    main()
