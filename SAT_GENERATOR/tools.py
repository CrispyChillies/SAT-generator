import os
from langchain_classic.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, Union, Literal,  List, Dict, Union
import numpy as np
import sympy as sp

# Lazy-initialized LLM for step-generation tool (uses OPENAI_API_KEY from env)
_llm_step = None

def _get_step_llm():
    global _llm_step
    if _llm_step is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set to use llm_generate_step tool")
        _llm_step = ChatOpenAI(model="gpt-4.1", temperature=0)
    return _llm_step

# Pydantic models for tool inputs
class BinaryOpInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class UnaryOpInput(BaseModel):
    x: float = Field(description="Input number")

class PowerInput(BaseModel):
    x: float = Field(description="Base number")
    n: float = Field(description="Exponent")

class IntegralInput(BaseModel):
    expr_str: str = Field(description="Expression as string, e.g., 'x**2 + 2*x'")
    var: str = Field(description="Variable to integrate with respect to, e.g., 'x'")
    lower: Optional[float] = Field(default=None, description="Lower bound for definite integral")
    upper: Optional[float] = Field(default=None, description="Upper bound for definite integral")

class DerivativeInput(BaseModel):
    expr_str: str = Field(description="Expression as string, e.g., 'x**2 + 2*x'")
    var: str = Field(description="Variable to differentiate with respect to, e.g., 'x'")

class TrigInput(BaseModel):
    func: Literal["sin", "cos", "tan"] = Field(description="Trigonometric function: 'sin', 'cos', or 'tan'")
    x: float = Field(description="Input angle in radians")

class StepRequirementInput(BaseModel):
    requirement: str = Field(
        description="Non-computational requirement for the current step (e.g. write an equation that models a constraint, state which formula applies, or give short reasoning to choose an answer). If the question has multiple steps, pass exactly one step's requirement per call."
    )
class MinWithLabelsInput(BaseModel):
    values: List[float] = Field(description="List of values (e.g., percentages) to find minimum.")
    labels: List[Union[int, float, str]] = Field(description="List of labels (e.g., years like [2015, 2016, 2017]) corresponding to each value. Must have same length as values.")

class MaxWithLabelsInput(BaseModel):
    values: List[float] = Field(description="List of values (e.g., percentages) to find maximum.")
    labels: List[Union[int, float, str]] = Field(description="List of labels (e.g., years like [2015, 2016, 2017]) corresponding to each value. Must have same length as values.")

class GetValueAtLabelInput(BaseModel):
    values: List[float] = Field(description="List of values from the graph (e.g., [30, 62, 36, 50, ...])")
    labels: List[Union[int, float, str]] = Field(description="List of labels from the graph (e.g., [1, 2, 3, 4, ...] for Group 1, Group 2, etc.)")
    target_label: Union[int, float, str] = Field(description="The specific label to look up (e.g., 1 for Group 1, 2015 for year 2015)")

class SumValuesInput(BaseModel):
    values: List[float] = Field(description="List of values to sum")

class AverageValuesInput(BaseModel):
    values: List[float] = Field(description="List of values to calculate average")

class MaxIncreasePeriodInput(BaseModel):
    values: List[float] = Field(description="List of values in order (e.g., temperatures)")
    labels: List[Union[int, float, str]] = Field(description="List of labels (e.g., days [1, 2, 3])")

# Tool functions
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def subtract_numbers(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide_numbers(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def power_or_exponential(base: Optional[float], exponent: float) -> float:
    """Calculate base^exponent. If base is not provided, calculates e^exponent (natural exponential)."""
    if base is None:
        return float(np.exp(exponent))
    return float(base ** exponent)

def natural_log(x: float) -> float:
    """Calculate natural logarithm ln(x)."""
    if x <= 0:
        raise ValueError("Logarithm undefined for non-positive numbers")
    return float(np.log(x))

def square_root(x: float) -> float:
    """Calculate square root of x."""
    if x < 0:
        raise ValueError("Square root undefined for negative numbers")
    return float(np.sqrt(x))

def trig_func(func: str, x: float) -> float:
    """Calculate sin(x), cos(x), or tan(x) (x in radians)."""
    if func == "sin":
        return float(np.sin(x))
    if func == "cos":
        return float(np.cos(x))
    if func == "tan":
        return float(np.tan(x))
    raise ValueError(f"Unknown trig function: {func}")

def derivative_func(expr_str: str, var: str) -> str:
    """
    Calculate the derivative of an expression.
    Returns the derivative as a string.
    """
    try:
        var_symbol = sp.Symbol(var)
        expr = sp.sympify(expr_str)
        result = sp.diff(expr, var_symbol)
        return str(result)
    except Exception as e:
        raise ValueError(f"Error computing derivative: {str(e)}")

def integral_func(expr_str: str, var: str, lower: Optional[float] = None, upper: Optional[float] = None) -> Union[str, float]:
    """
    Calculate the integral of an expression.
    If lower and upper bounds are provided, returns definite integral (float).
    Otherwise returns indefinite integral (string).
    """
    try:
        var_symbol = sp.Symbol(var)
        expr = sp.sympify(expr_str)
        result = sp.integrate(expr, var_symbol)
        
        if lower is not None and upper is not None:
            value = float(result.subs(var_symbol, upper) - result.subs(var_symbol, lower))
            return value
        return str(result)
    except Exception as e:
        raise ValueError(f"Error computing integral: {str(e)}")

def find_min_with_labels(values: List[float], labels: List[Union[int, float, str]]) -> Union[int, float, str]:
    """
    ALWAYS USE THIS TOOL for finding minimum/maximum values in a list with labels.
    This is the PRIMARY tool for graph/chart questions asking "which year/label has the smallest/largest value".
    
    Returns the label (e.g., year) corresponding to the minimum value directly.
    
    Example:
        Input: values=[15, 14, 3, 7], labels=[2010, 2011, 2012, 2013]
        Output: 2012
        
        The answer is the label (2012), NOT the index or min_value.
    """
    if not values:
        raise ValueError("Input values list is empty.")
    if len(values) != len(labels):
        raise ValueError(f"values and labels must have same length. Got {len(values)} values and {len(labels)} labels.")
    
    min_value = min(values)
    index = values.index(min_value)
    return labels[index]

def find_max_with_labels(values: List[float], labels: List[Union[int, float, str]]) -> Union[int, float, str]:
    """
    ALWAYS USE THIS TOOL for finding minimum/maximum values in a list with labels.
    This is the PRIMARY tool for graph/chart questions asking "which year/label has the smallest/largest value".
    
    Returns the label (e.g., year) corresponding to the maximum value directly.
    
    Example:
        Input: values=[15, 14, 3, 7], labels=[2010, 2011, 2012, 2013]
        Output: 2010
        
        The answer is the label (2010), NOT the index or max_value.
    """
    if not values:
        raise ValueError("Input values list is empty.")
    if len(values) != len(labels):
        raise ValueError(f"values and labels must have same length. Got {len(values)} values and {len(labels)} labels.")
    
    max_value = max(values)
    index = values.index(max_value)
    return labels[index]

def get_value_at_label(values: List[float], labels: List[Union[int, float, str]], target_label: Union[int, float, str]) -> float:
    """
    ALWAYS USE THIS TOOL for reading a specific value from a graph/chart at a given label.
    This is the PRIMARY tool for questions like "How many books were collected by group 1?" or "What was the value in 2015?".
    
    Returns the value at the specified label.
    
    Example:
        Input: values=[30, 62, 36, 50], labels=[1, 2, 3, 4], target_label=1
        Output: 30.0
        
        The answer is the value (30.0) at Group 1.
    """
    if not values:
        raise ValueError("Input values list is empty.")
    if len(values) != len(labels):
        raise ValueError(f"values and labels must have same length. Got {len(values)} values and {len(labels)} labels.")
    
    # Try to find exact match first
    try:
        index = labels.index(target_label)
        return float(values[index])
    except ValueError:
        # If target_label is numeric, try converting labels to same type
        try:
            if isinstance(target_label, (int, float)):
                for i, label in enumerate(labels):
                    if isinstance(label, (int, float)) and label == target_label:
                        return float(values[i])
                    elif str(label) == str(target_label):
                        return float(values[i])
            raise ValueError(f"Label '{target_label}' not found in labels list: {labels}")
        except:
            raise ValueError(f"Label '{target_label}' not found in labels list: {labels}")

def sum_values(values: List[float]) -> float:
    """
    Calculate the sum of all values in a list.
    Use for questions like "What is the total number of books collected?"
    
    Example:
        Input: values=[30, 62, 36, 50, 46, 40, 54, 60, 16, 20]
        Output: 414.0
    """
    if not values:
        raise ValueError("Input values list is empty.")
    return float(sum(values))

def average_values(values: List[float]) -> float:
    """
    Calculate the average (mean) of all values in a list.
    Use for questions like "What is the average number of books per group?"
    
    Example:
        Input: values=[30, 62, 36, 50, 46, 40, 54, 60, 16, 20]
        Output: 41.4
    """
    if not values:
        raise ValueError("Input values list is empty.")
    return float(sum(values) / len(values))


def find_max_increase_period(values: List[float], labels: List[Union[int, float, str]]) -> str:
    """
    Find the time period with the greatest increase (positive change).
    Returns the period label where the maximum increase occurred.
    
    Example:
        Input: values=[69, 60, 73, 67], labels=[1, 2, 3, 4]
        Output: "2 to 3" (increase of 13)
    """
    if len(values) != len(labels):
        raise ValueError(f"values and labels must have same length")
    if len(values) < 2:
        raise ValueError("Need at least 2 data points")
    
    max_increase = float('-inf')
    max_period = None
    
    for i in range(len(values) - 1):
        increase = values[i+1] - values[i]
        if increase > max_increase:
            max_increase = increase
            max_period = f"{labels[i]} to {labels[i+1]}"
    
    return max_period
# ---------------------------------------------------------------------------
# LLM-based step output for non-computational requirements (equations, reasoning, etc.)
# ---------------------------------------------------------------------------

def llm_generate_step(requirement: str) -> str:
    """
    Use an LLM to generate the output for a step whose requirement is NOT computational. For multi-step questions, pass one step's requirement
    per call. Do not use this for numeric calculation—use add, multiply, etc. instead.
    """
    requirement = (requirement or "").strip()
    if not requirement:
        return "Error: requirement cannot be empty."
    llm = _get_step_llm()
    prompt = (
        "You are a precise math assistant. For the current step, the requirement is:\n\n"
        f"{requirement}\n\n"
        "Produce ONLY the output that satisfies this requirement. Use standard math notation. "
        "No extra preamble or explanation—just the required output."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response)
    if hasattr(response, "content") and response.content:
        return response.content.strip()
    return str(response)

# Create LangChain tools
math_tools = [
    StructuredTool.from_function(
        func=add_numbers,
        name="add",
        description="Add two numbers together. Use this for addition operations.",
        args_schema=BinaryOpInput
    ),
    StructuredTool.from_function(
        func=subtract_numbers,
        name="subtract",
        description="Subtract the second number from the first. Use this for subtraction operations.",
        args_schema=BinaryOpInput
    ),
    StructuredTool.from_function(
        func=multiply_numbers,
        name="multiply",
        description="Multiply two numbers. Use this for multiplication operations.",
        args_schema=BinaryOpInput
    ),
    StructuredTool.from_function(
        func=divide_numbers,
        name="divide",
        description="Divide the first number by the second. Use this for division operations.",
        args_schema=BinaryOpInput
    ),
    StructuredTool.from_function(
        func=power_or_exponential,
        name="power",
        description="Compute exponentiation: use base=null (or omit base) for e^exponent (natural exponential); use base and exponent for base^exponent. Use for both e^x and x^n.",
        args_schema=PowerInput
    ),
    StructuredTool.from_function(
        func=natural_log,
        name="log",
        description="Calculate natural logarithm (ln) of x. Use this for logarithmic operations.",
        args_schema=UnaryOpInput
    ),
    StructuredTool.from_function(
        func=square_root,
        name="sqrt",
        description="Calculate square root of x. Use this for square root operations.",
        args_schema=UnaryOpInput
    ),
    StructuredTool.from_function(
        func=trig_func,
        name="trig",
        description="Compute sin(x), cos(x), or tan(x). Set func to 'sin', 'cos', or 'tan'; x is the angle in radians.",
        args_schema=TrigInput
    ),
    StructuredTool.from_function(
        func=derivative_func,
        name="derivative",
        description="Calculate the derivative of a mathematical expression. Returns symbolic result.",
        args_schema=DerivativeInput
    ),
    StructuredTool.from_function(
        func=integral_func,
        name="integral",
        description="Calculate integral of a mathematical expression. Can compute definite or indefinite integrals.",
        args_schema=IntegralInput
    ),
    StructuredTool.from_function(
        func=llm_generate_step,
        name="llm_generate_step",
        description="Use when the requirement is not computational. Input: requirement (string) for this step only; one step per call for multi-step questions. Do not use for numeric computation—use add, multiply, sqrt, etc. instead.",
        args_schema=StepRequirementInput
    ),
]

math_tools.extend([
    StructuredTool.from_function(
        func=find_min_with_labels,
        name="find_min_with_labels",
        description="PREFERRED for graph problems with years/labels. Pass values (e.g., percentages) and labels (e.g., years [2015, 2016, 2017]). Returns the label (year) directly with min value. No index calculation needed.",
        args_schema=MinWithLabelsInput
    ),
    StructuredTool.from_function(
        func=find_max_with_labels,
        name="find_max_with_labels",
        description="PREFERRED for graph problems with years/labels. Pass values (e.g., percentages) and labels (e.g., years [2015, 2016, 2017]). Returns the label (year) directly with max value. No index calculation needed.",
        args_schema=MaxWithLabelsInput
    ),
    StructuredTool.from_function(
        func=get_value_at_label,
        name="get_value_at_label",
        description="PREFERRED for reading a specific value from a graph at a given label. Use for questions like 'How many books were collected by group 1?' Pass values, labels, and target_label (e.g., 1 for Group 1). Returns the value at that label.",
        args_schema=GetValueAtLabelInput
    ),
    StructuredTool.from_function(
        func=sum_values,
        name="sum_values",
        description="Calculate the sum of all values in a list. Use for questions asking for the total (e.g., 'What is the total number of books?'). Returns the sum.",
        args_schema=SumValuesInput
    ),
    StructuredTool.from_function(
        func=average_values,
        name="average_values",
        description="Calculate the average (mean) of all values in a list. Use for questions asking for the average/mean (e.g., 'What is the average per group?'). Returns the average.",
        args_schema=AverageValuesInput
    ),
    StructuredTool.from_function(
        func=find_max_increase_period,
        name="find_max_increase_period",
        description="Find the time period with the greatest increase (positive change). Use for questions asking which period had the largest increase (e.g., 'Which year had the greatest increase?'). Returns the period label where the maximum increase occurred.",
        args_schema=MaxIncreasePeriodInput
    ),
])