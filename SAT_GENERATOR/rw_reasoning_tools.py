"""
Reasoning tools for SAT Reading & Writing questions.
Unlike math tools that perform calculations, these tools analyze text and logical relationships.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Lazy-initialized LLM for reasoning tools
_llm_reasoning = None

def _get_reasoning_llm():
    """Get or initialize LLM for reasoning tasks."""
    global _llm_reasoning
    if _llm_reasoning is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        _llm_reasoning = ChatOpenAI(model="gpt-4o", temperature=0)
    return _llm_reasoning


# ============================================================================
# Pydantic Input Schemas
# ============================================================================

class AnalyzeClaimsInput(BaseModel):
    paragraph: str = Field(description="The paragraph text to analyze for main claims and arguments")

class AnalyzeEvidenceInput(BaseModel):
    paragraph: str = Field(description="The paragraph text containing evidence")
    claim: str = Field(description="The claim to find supporting or contradicting evidence for")

class EvaluateInferenceInput(BaseModel):
    paragraph: str = Field(description="The paragraph providing context")
    inference: str = Field(description="The proposed inference or conclusion to evaluate")

class EvaluateChoiceInput(BaseModel):
    paragraph: str = Field(description="The paragraph providing context")
    question: str = Field(description="The question being asked")
    choice: str = Field(description="The answer choice to evaluate")
    skill: str = Field(description="The skill being tested (e.g., 'Inferences', 'Command of Evidence', 'Central Ideas and Details')")

class CompareChoicesInput(BaseModel):
    paragraph: str = Field(description="The paragraph providing context")
    question: str = Field(description="The question being asked")
    choices: List[str] = Field(description="List of 4 answer choices to compare")
    skill: str = Field(description="The skill being tested")

class IdentifyDistractorInput(BaseModel):
    paragraph: str = Field(description="The paragraph providing context")
    question: str = Field(description="The question being asked")
    choice: str = Field(description="The answer choice to analyze")
    skill: str = Field(description="The skill being tested")

class IdentifyCentralIdeaInput(BaseModel):
    paragraph: str = Field(description="The paragraph to extract the central idea from")

class EvaluateWordChoiceInput(BaseModel):
    paragraph: str = Field(description="The paragraph with a blank to fill")
    question: str = Field(description="The question asking for word choice")
    word: str = Field(description="The word or phrase to evaluate for the blank")


# ============================================================================
# Reasoning Tool Functions
# ============================================================================

def analyze_claims(paragraph: str) -> str:
    """
    Extract main claims, arguments, and logical structure from a paragraph.
    Returns a structured analysis of the key points.
    """
    llm = _get_reasoning_llm()
    prompt = f"""Analyze this paragraph and extract:
1. Main claim(s) or argument(s)
2. Supporting evidence or data
3. Logical structure (e.g., claim → evidence → conclusion)
4. Key relationships between ideas

Paragraph:
{paragraph}

Provide a clear, structured analysis."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def analyze_evidence(paragraph: str, claim: str) -> str:
    """
    Identify evidence in the paragraph that supports or contradicts a given claim.
    Returns analysis of how the evidence relates to the claim.
    """
    llm = _get_reasoning_llm()
    prompt = f"""Given this paragraph and claim, identify:
1. Evidence that SUPPORTS the claim
2. Evidence that CONTRADICTS the claim
3. Evidence that is NEUTRAL or IRRELEVANT

Paragraph:
{paragraph}

Claim:
{claim}

Provide a structured analysis of the evidence."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def evaluate_inference(paragraph: str, inference: str) -> str:
    """
    Evaluate whether an inference logically follows from the paragraph.
    Returns assessment of logical validity and strength.
    """
    llm = _get_reasoning_llm()
    prompt = f"""Evaluate whether this inference logically follows from the paragraph.

Paragraph:
{paragraph}

Proposed Inference:
{inference}

Assess:
1. Does this inference logically follow? (Yes/No/Partially)
2. What evidence supports this inference?
3. What evidence contradicts or weakens it?
4. Overall strength: Strong/Moderate/Weak/Invalid

Provide clear reasoning."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def evaluate_choice(paragraph: str, question: str, choice: str, skill: str) -> str:
    """
    Evaluate a single answer choice based on the paragraph, question, and skill being tested.
    Returns detailed assessment of the choice's validity.
    """
    llm = _get_reasoning_llm()
    prompt = f"""You are evaluating an SAT Reading & Writing answer choice.

Skill being tested: {skill}

Paragraph:
{paragraph}

Question:
{question}

Answer Choice:
{choice}

Evaluate this choice:
1. Does it correctly answer the question?
2. Is it supported by the paragraph?
3. Does it demonstrate understanding of the tested skill?
4. Any logical flaws or weaknesses?
5. Rating: Correct / Partially Correct / Flawed / Irrelevant

Provide detailed reasoning."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def compare_choices(paragraph: str, question: str, choices: List[str], skill: str) -> str:
    """
    Compare all answer choices and rank them by logical strength.
    Returns analysis identifying the best answer and explaining why others are wrong.
    """
    llm = _get_reasoning_llm()
    # Do not include A., B., C., D. prefixes — UI handles labeling
    choices_text = "\n".join([f"{choice}" for choice in choices])
    
    prompt = f"""You are solving an SAT Reading & Writing question.

Skill being tested: {skill}

Paragraph:
{paragraph}

Question:
{question}

Answer Choices:
{choices_text}

Analyze each choice systematically and determine the best answer.

IMPORTANT: You MUST end your response with EXACTLY this format on the last line:
ANSWER: [letter]

Where [letter] is A, B, C, or D.

Example ending:
Therefore, choice D is correct because it accurately reflects the data.

ANSWER: D"""
    
    print(prompt)
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def identify_distractor_type(paragraph: str, question: str, choice: str, skill: str) -> str:
    """
    Classify a distractor (incorrect answer) by type:
    - Partially correct but incomplete
    - Logically flawed
    - Irrelevant
    Returns the distractor type and explanation.
    """
    llm = _get_reasoning_llm()
    prompt = f"""Classify this incorrect answer choice by distractor type.

Skill being tested: {skill}

Paragraph:
{paragraph}

Question:
{question}

Incorrect Choice:
{choice}

What type of distractor is this?
1. PARTIALLY CORRECT BUT INCOMPLETE: Has some valid element but missing key point(s)
2. LOGICALLY FLAWED: Contains reasoning error, misinterpretation, or contradiction
3. IRRELEVANT: Unrelated to the question or uses information not in paragraph

Classify and explain why."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def identify_central_idea(paragraph: str) -> str:
    """
    Extract the central idea or main point of a paragraph.
    Returns the central idea with supporting explanation.
    """
    llm = _get_reasoning_llm()
    prompt = f"""Identify the central idea (main point) of this paragraph.

Paragraph:
{paragraph}

Provide:
1. The central idea in one concise sentence
2. Key details that support this central idea
3. Any secondary points that elaborate on the main idea

Be clear and specific."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def evaluate_word_choice(paragraph: str, question: str, word: str) -> str:
    """
    Evaluate whether a word or phrase logically and precisely fits in context.
    Used for "Words in Context" questions.
    Returns assessment of the word's appropriateness.
    """
    llm = _get_reasoning_llm()
    prompt = f"""Evaluate whether this word/phrase is the most logical and precise choice for the blank.

Paragraph (with blank indicated):
{paragraph}

Question:
{question}

Word/Phrase to evaluate:
{word}

Assess:
1. Does it fit grammatically?
2. Does it fit the meaning/context?
3. Is it precise and logical?
4. Rating: Perfect / Good / Acceptable / Poor / Wrong

Explain your reasoning."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ============================================================================
# Create LangChain Tools
# ============================================================================

rw_reasoning_tools = [
    StructuredTool.from_function(
        func=analyze_claims,
        name="analyze_claims",
        description="Extract main claims, arguments, and logical structure from a paragraph. Use this to understand what the paragraph is arguing or explaining.",
        args_schema=AnalyzeClaimsInput
    ),
    StructuredTool.from_function(
        func=analyze_evidence,
        name="analyze_evidence",
        description="Identify evidence in the paragraph that supports or contradicts a claim. Use for Command of Evidence questions.",
        args_schema=AnalyzeEvidenceInput
    ),
    StructuredTool.from_function(
        func=evaluate_inference,
        name="evaluate_inference",
        description="Evaluate whether an inference logically follows from the paragraph. Use for Inference questions.",
        args_schema=EvaluateInferenceInput
    ),
    StructuredTool.from_function(
        func=evaluate_choice,
        name="evaluate_choice",
        description="Evaluate a single answer choice for validity. Use to assess whether a choice correctly answers the question.",
        args_schema=EvaluateChoiceInput
    ),
    StructuredTool.from_function(
        func=compare_choices,
        name="compare_choices",
        description="Compare all answer choices and identify the best one. Use this as the main tool to solve the question.",
        args_schema=CompareChoicesInput
    ),
    StructuredTool.from_function(
        func=identify_distractor_type,
        name="identify_distractor_type",
        description="Classify why an incorrect choice is wrong (partially correct, flawed, or irrelevant). Use to understand distractor patterns.",
        args_schema=IdentifyDistractorInput
    ),
    StructuredTool.from_function(
        func=identify_central_idea,
        name="identify_central_idea",
        description="Extract the central idea or main point from a paragraph. Use for Central Ideas and Details questions.",
        args_schema=IdentifyCentralIdeaInput
    ),
    StructuredTool.from_function(
        func=evaluate_word_choice,
        name="evaluate_word_choice",
        description="Evaluate whether a word/phrase is the most logical and precise choice for a blank. Use for Words in Context questions.",
        args_schema=EvaluateWordChoiceInput
    ),
]


# ============================================================================
# Helper function to get tool by name
# ============================================================================

def get_rw_tool_by_name(name: str):
    """Get reasoning tool by name."""
    for tool in rw_reasoning_tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool not found: {name}")
