#!/usr/bin/env python3
"""
Generate new SAT Reading & Writing questions from sample questions.
- Preserves category, section, skill, difficulty
- Creates completely new scenario testing the same reasoning skill
- Generates paragraph, question, 4 choices, correct answer, and explanation
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal
import argparse
import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from mathml_parser import MathMLParser, TableSpec

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY must be set in environment")


# ============================================================================
# Pydantic Output Schemas
# ============================================================================

class GeneratedRWQuestionContent(BaseModel):
    """Generated R&W question with paragraph, question, 4 choices, correct answer, and explanation."""
    paragraph_text: str = Field(description="New paragraph providing context (completely new scenario, different topic)")
    question: str = Field(description="New question text (e.g., 'Which choice most logically completes the text?')")
    choices: List[str] = Field(description="Exactly 4 answer choices (A, B, C, D) - one correct, others are distractors")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="Letter of the correct answer")
    explanation: str = Field(description="Clear explanation of WHY the correct answer works logically and why others are wrong")
    
    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v):
        if len(v) != 4:
            raise ValueError("Must have exactly 4 choices")
        return v


class GeneratedRWQuestionWithGraph(BaseModel):
    """Generated R&W question with embedded graph (preserves original SVG, updates text only)."""
    paragraph_text: str = Field(description="New paragraph text WITHOUT the graph/SVG (new scenario, different data)")
    question: str = Field(description="New question text")
    choices: List[str] = Field(description="Exactly 4 answer choices")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="Letter of the correct answer")
    explanation: str = Field(description="Explanation of the correct answer")
    
    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v):
        if len(v) != 4:
            raise ValueError("Must have exactly 4 choices")
        return v


class GeneratedRWQuestionWithTable(BaseModel):
    """Generated R&W question with HTML table (new scenario with new data)."""
    paragraph_text: str = Field(description="New paragraph text WITHOUT the table (describes new scenario)")
    table_caption: str = Field(description="New table caption/title")
    table_headers: List[str] = Field(description="Column headers (same number as original)")
    table_row_labels: List[str] = Field(description="Row labels (same number as original)")
    table_data: List[List[str]] = Field(description="Table data cells (same structure as original)")
    question: str = Field(description="New question text (usually same as original)")
    choices: List[str] = Field(description="Exactly 4 answer choices")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="Letter of the correct answer")
    explanation: str = Field(description="Explanation of the correct answer")
    
    @field_validator("choices")
    @classmethod
    def validate_choices(cls, v):
        if len(v) != 4:
            raise ValueError("Must have exactly 4 choices")
        return v


# ============================================================================
# Helper Functions
# ============================================================================

def _has_embedded_graph(paragraph: str) -> bool:
    """Check if paragraph contains embedded SVG or figure."""
    if not paragraph:
        return False
    return "<svg" in paragraph.lower() or ("<figure" in paragraph.lower() and "<svg" in paragraph.lower())


def _has_embedded_table(paragraph: str) -> bool:
    """Check if paragraph contains embedded HTML table."""
    if not paragraph:
        return False
    return "<table" in paragraph.lower()


def _extract_paragraph_without_graph(paragraph: str) -> str:
    """Extract paragraph text without SVG/figure elements."""
    import re
    # Remove SVG and figure tags
    text = re.sub(r'<figure[^>]*>.*?</figure>', '', paragraph, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<svg[^>]*>.*?</svg>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<div[^>]*>.*?long description.*?</div>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _get_paragraph(sample: Dict[str, Any]) -> str:
    """Extract paragraph from sample question."""
    q_block = sample.get("question", {})
    return (q_block.get("paragraph") or "").strip()


def _get_question_text(sample: Dict[str, Any]) -> str:
    """Extract question text from sample."""
    q_block = sample.get("question", {})
    return (q_block.get("question") or "").strip()


def _get_choices(sample: Dict[str, Any]) -> List[str]:
    """Extract answer choices from sample."""
    q_block = sample.get("question", {})
    choices = q_block.get("choices") or []
    return [str(c).strip() for c in choices]


def _get_correct_answer_letter(sample: Dict[str, Any]) -> str:
    """Extract correct answer letter from sample."""
    q_block = sample.get("question", {})
    answer = q_block.get("correct_answer")
    if isinstance(answer, list) and len(answer) > 0:
        return answer[0].strip().upper()
    if isinstance(answer, str):
        return answer.strip().upper()
    return "A"  # Default fallback


def _get_explanation(sample: Dict[str, Any]) -> str:
    """Extract explanation from sample."""
    q_block = sample.get("question", {})
    return (q_block.get("explanation") or "").strip()


def _infer_reasoning_type(skill: str, question: str, paragraph: str) -> str:
    """
    Infer the reasoning type based on skill and question pattern.
    Examples: claim + data + evaluation, cause-effect, comparison, definition, etc.
    """
    skill_lower = skill.lower()
    question_lower = question.lower()
    
    # Pattern matching based on skill
    if "inference" in skill_lower:
        if "complete" in question_lower:
            return "claim + supporting data + logical conclusion"
        return "evidence → inference"
    elif "command of evidence" in skill_lower:
        return "hypothesis + data analysis + evidence evaluation"
    elif "central ideas" in skill_lower:
        return "passage structure + main point extraction"
    elif "words in context" in skill_lower:
        return "contextual meaning + precise word choice"
    elif "rhetorical synthesis" in skill_lower:
        return "information synthesis + audience/purpose alignment"
    else:
        return "logical reasoning structure"


def _infer_logical_schema(skill: str, paragraph: str, choices: List[str]) -> str:
    """
    Infer the logical schema (how the question tests reasoning).
    Examples: elimination, comparison, interpretation, synthesis, etc.
    """
    skill_lower = skill.lower()
    
    if "inference" in skill_lower:
        return "Given premises in paragraph → evaluate which conclusion logically follows"
    elif "command of evidence" in skill_lower:
        return "Given hypothesis + data → identify evidence that supports/weakens"
    elif "central ideas" in skill_lower:
        return "Given passage → identify main point vs. supporting details"
    elif "words in context" in skill_lower:
        return "Given context + blank → select most precise/logical word"
    elif "rhetorical" in skill_lower:
        return "Given information + goal → synthesize appropriate statement"
    else:
        return "Logical evaluation of choices against paragraph evidence"


def _build_generation_prompt(
    paragraph: str,
    question: str,
    choices: List[str],
    correct_letter: str,
    explanation: str,
    skill: str,
    category: str,
    difficulty: str,
    has_graph: bool = False,
) -> str:
    """Build prompt for LLM to generate new R&W question."""
    
    reasoning_type = _infer_reasoning_type(skill, question, paragraph)
    logical_schema = _infer_logical_schema(skill, paragraph, choices)
    
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    correct_choice = choices[ord(correct_letter) - ord('A')] if correct_letter in "ABCD" else choices[0]
    
    # Identify distractor patterns
    distractors = []
    for i, choice in enumerate(choices):
        letter = chr(65 + i)
        if letter != correct_letter:
            distractors.append(f"{letter}. {choice}")
    
    graph_note = ""
    if has_graph:
        graph_note = "\n**Note**: Original has embedded graph/table. For now, create paragraph WITHOUT graph. We'll handle graph regeneration separately."
    
    prompt = f"""You are an SAT Reading & Writing question designer.

Your task is NOT to rewrite the text with small changes.
Your task is to generate a completely NEW scenario that tests the SAME reasoning skill.

**Original Question Analysis:**

Category: {category}
Skill: {skill}
Difficulty: {difficulty}
Reasoning Type: {reasoning_type}
Logical Schema: {logical_schema}

**Original Paragraph:**
{paragraph}

**Original Question:**
{question}

**Original Choices:**
{choices_text}

**Correct Answer:** {correct_letter}. {correct_choice}

**Original Explanation:**
{explanation}

**Distractor Patterns in Original:**
{chr(10).join(distractors)}
{graph_note}

---

**YOUR TASK:**

1. Generate a **NEW academic-style scenario** (different topic, different context, different subject matter).
   - If original is about biology, try physics, history, literature, sociology, etc.
   - Change all specific details (names, numbers, fields, phenomena)

2. **Preserve the reasoning structure**:
   - Same logical schema: {logical_schema}
   - Same type of claim/evidence/conclusion relationship
   - Same pattern of what makes the answer logically correct

3. **Preserve the question type**:
   - Keep same question format (e.g., "Which choice most logically completes the text?")
   - Test the same skill: {skill}

4. **Create 4 answer choices** with SAME distractor logic patterns:
   - 1 fully correct (follows the logical schema perfectly)
   - 1 partially correct but incomplete (has valid element but misses key point)
   - 1 logically flawed (reasoning error, misinterpretation, or contradiction)
   - 1 irrelevant (unrelated or uses information not supported by paragraph)

5. **Match the distractor patterns**: Study how the original distractors were wrong, and create new distractors that are wrong in analogous ways.

6. **Write clear explanation**: Explain WHY the correct answer works logically and WHY each other choice fails.

**Important Guidelines:**
- Do NOT copy phrasing from the original
- Do NOT reuse the topic/field
- Keep SAT academic tone (formal, educational)
- Similar difficulty level: {difficulty}
- Paragraph should be similar length to original
- Ensure new scenario is realistic and educational

**Output the new question in JSON format with these exact fields:**
- paragraph_text: (string) The new paragraph
- question: (string) The new question text  
- choices: (array of 4 strings) The four answer choices
- correct_answer_letter: (string) "A", "B", "C", or "D"
- explanation: (string) Clear explanation of why correct answer works and why others don't

Generate a completely new, high-quality SAT R&W question now."""

    return prompt


def _build_table_generation_prompt(
    paragraph: str,
    table_spec: TableSpec,
    question: str,
    choices: List[str],
    correct_letter: str,
    explanation: str,
    skill: str,
    category: str,
    difficulty: str,
) -> str:
    """Build prompt for LLM to generate new R&W question with table."""
    
    reasoning_type = _infer_reasoning_type(skill, question, paragraph)
    logical_schema = _infer_logical_schema(skill, paragraph, choices)
    
    choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    correct_choice = choices[ord(correct_letter) - ord('A')] if correct_letter in "ABCD" else choices[0]
    
    # Format table structure for LLM
    table_structure = f"""**Original Table Structure:**
Caption: {table_spec.caption}

Headers: {table_spec.headers}

Row Labels: {table_spec.row_labels}

Data (rows):
"""
    for i, row in enumerate(table_spec.rows or []):
        label = table_spec.row_labels[i] if table_spec.row_labels and i < len(table_spec.row_labels) else f"Row {i+1}"
        table_structure += f"  {label}: {row}\n"
    
    prompt = f"""You are an SAT Reading & Writing question designer.

Your task: Generate a NEW scenario with a NEW DATA TABLE that tests the SAME reasoning skill.

**Original Question Analysis:**

Category: {category}
Skill: {skill}
Difficulty: {difficulty}
Reasoning Type: {reasoning_type}
Logical Schema: {logical_schema}

{table_structure}

**Original Paragraph (without table):**
{paragraph}

**Original Question:**
{question}

**Original Choices:**
{choices_text}

**Correct Answer:** {correct_letter}. {correct_choice}

**Original Explanation:**
{explanation}

---

**YOUR TASK:**

1. **Create a COMPLETELY NEW scenario** (different topic, different context):
   - If original is about museums, try: products, cities, schools, books, etc.
   - Change ALL specific details (names, numbers, categories)

2. **Generate NEW TABLE DATA**:
   - Same structure: {len(table_spec.headers or [])} columns, {len(table_spec.rows or [])} rows
   - New caption (related to your new scenario)
   - New column headers (appropriate for new scenario)
   - New row labels (items/entities in your scenario)
   - New numerical data (realistic for your context)
   - Data should have similar patterns (e.g., if original shows top item has highest value, new data should too)

3. **Write NEW paragraph** describing the scenario (WITHOUT the table - table will be added separately)

4. **Keep or adapt the question** (usually "Which choice best describes data in the table that support the researchers' conclusion?")

5. **Create 4 answer choices** with SAME distractor logic:
   - 1 correct choice that identifies the key data pattern
   - 3 wrong choices with similar errors as original (wrong data, wrong comparison, wrong group, etc.)

6. **Write explanation** of why correct answer works and why others don't

**Important:**
- DO NOT reuse the topic/field from original
- Keep SAT academic tone
- Table data must be REALISTIC and LOGICAL for your new scenario
- Same difficulty level: {difficulty}

**Output JSON with these fields:**
- paragraph_text: New paragraph WITHOUT table
- table_caption: New caption
- table_headers: List of {len(table_spec.headers or [])} column headers
- table_row_labels: List of {len(table_spec.rows or [])} row labels
- table_data: List of {len(table_spec.rows or [])} data rows (each row has {len(table_spec.rows[0])} values)
- question: Question text
- choices: Array of 4 answer choices
- correct_answer_letter: "A", "B", "C", or "D"
- explanation: Clear explanation

Generate a completely new, high-quality SAT R&W question with table now."""

    return prompt


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_new_rw_question(
    sample: Dict[str, Any],
    llm: Optional[ChatOpenAI] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate a new R&W question from a sample question.
    
    Args:
        sample: Sample question from questions_practice_test.json (R&W question)
        llm: Optional ChatOpenAI instance (if None, creates new one)
        api_key: OpenAI API key (if None, uses environment variable)
        model: LLM model name (only used if llm is None)
        verbose: Print progress logs
    
    Returns:
        Dict with same structure as input sample, but with new content
    """
    if verbose:
        print("[generate_rw_question] Starting question generation...")
    
    if llm is None:
        # Use provided api_key or fall back to environment variable
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not provided and not found in environment")
        if verbose:
            print(f"[generate_rw_question] Creating LLM with model: {model}")
        llm = ChatOpenAI(model=model, temperature=0.7, api_key=key)
    
    # Extract metadata
    category = sample.get("category", "")
    skill = sample.get("skill", "")
    difficulty = sample.get("difficulty", "Medium")
    section = sample.get("section", "Reading and Writing")
    
    if verbose:
        print(f"[generate_rw_question] Skill: {skill}, Difficulty: {difficulty}")
    
    # Extract content
    paragraph = _get_paragraph(sample)
    question_text = _get_question_text(sample)
    choices = _get_choices(sample)
    correct_letter = _get_correct_answer_letter(sample)
    explanation = _get_explanation(sample)
    
    if verbose:
        print(f"[generate_rw_question] Extracted question components")
        print(f"  Paragraph length: {len(paragraph)} chars")
        print(f"  Choices: {len(choices)}")
    
    if not paragraph or not question_text or not choices:
        raise ValueError("Sample question missing required fields (paragraph, question, or choices)")
    
    # Check for embedded table (higher priority than graph)
    has_table = _has_embedded_table(paragraph)
    has_graph = _has_embedded_graph(paragraph) if not has_table else False
    
    if verbose:
        print(f"[generate_rw_question] Has embedded table: {has_table}")
        print(f"[generate_rw_question] Has embedded graph: {has_graph}")
    
    # Handle table case
    if has_table:
        # Parse table structure
        parser = MathMLParser()
        parsed = parser.parse_paragraph(paragraph)
        
        if not parsed.get("has_table"):
            raise ValueError("Table detection mismatch - expected table but parser didn't find it")
        
        table_spec = parsed["table"]
        paragraph_for_prompt = parsed["text"]
        
        if verbose:
            print(f"[generate_rw_question] Parsed table: {table_spec.caption}")
            print(f"  Headers: {len(table_spec.headers or [])}, Rows: {len(table_spec.rows or [])}")
            print(f"  Paragraph without table: {len(paragraph_for_prompt)} chars")
        
        # Build table generation prompt
        prompt = _build_table_generation_prompt(
            paragraph=paragraph_for_prompt,
            table_spec=table_spec,
            question=question_text,
            choices=choices,
            correct_letter=correct_letter,
            explanation=explanation,
            skill=skill,
            category=category,
            difficulty=difficulty,
        )
        
        # Use table-specific schema
        llm_with_structure = llm.with_structured_output(GeneratedRWQuestionWithTable)
        
        if verbose:
            print(f"[generate_rw_question] Calling LLM for table generation... (prompt length: {len(prompt)} chars)")
        
        try:
            generated = llm_with_structure.invoke([HumanMessage(content=prompt)])
            if verbose:
                print("[generate_rw_question] LLM table generation completed successfully")
                print(f"  New table caption: {generated.table_caption}")
        except Exception as e:
            if verbose:
                print(f"[generate_rw_question] Error during LLM generation: {e}")
            raise
        
        # Rebuild table HTML from generated data
        new_table_spec = TableSpec(
            caption=generated.table_caption,
            headers=generated.table_headers,
            rows=generated.table_data,
            row_labels=generated.table_row_labels,
            table_class=table_spec.table_class,  # Preserve original class
            original_html=""  # Not needed for new generation
        )
        
        table_html = new_table_spec.to_html()
        
        # Combine paragraph + table
        final_paragraph = generated.paragraph_text + "\n\n" + table_html
        
        if verbose:
            print(f"[generate_rw_question] Rebuilt table HTML ({len(table_html)} chars)")
        
        # Build new question
        new_question = {
            "id": str(uuid.uuid4()),
            "subject": sample.get("subject", "SAT"),
            "pool": sample.get("pool", "generated"),
            "section": section,
            "category": category,
            "skill": skill,
            "difficulty": difficulty,
            "type": "multiple-choice",
            "question": {
                "paragraph": final_paragraph,
                "question": generated.question,
                "choices": generated.choices,
                "correct_answer": [generated.correct_answer_letter],
                "explanation": generated.explanation,
            },
            "image_url": None,
        }
        
        return new_question
    
    # Handle graph case
    if has_graph:
        paragraph_for_prompt = _extract_paragraph_without_graph(paragraph)
        if verbose:
            print(f"[generate_rw_question] Extracted paragraph without graph: {len(paragraph_for_prompt)} chars")
    else:
        paragraph_for_prompt = paragraph
    
    # Build generation prompt
    if verbose:
        print("[generate_rw_question] Building generation prompt...")
    
    prompt = _build_generation_prompt(
        paragraph=paragraph_for_prompt,
        question=question_text,
        choices=choices,
        correct_letter=correct_letter,
        explanation=explanation,
        skill=skill,
        category=category,
        difficulty=difficulty,
        has_graph=has_graph,
    )
    
    # Call LLM with structured output
    if verbose:
        print(f"[generate_rw_question] Calling LLM for generation... (prompt length: {len(prompt)} chars)")
    
    llm_with_structure = llm.with_structured_output(GeneratedRWQuestionContent)
    
    try:
        generated = llm_with_structure.invoke([HumanMessage(content=prompt)])
        if verbose:
            print("[generate_rw_question] LLM generation completed successfully")
    except Exception as e:
        if verbose:
            print(f"[generate_rw_question] Error during LLM generation: {e}")
        print(f"Error generating question: {e}")
        raise
    
    # Build new question in same format as sample
    new_question = {
        "id": str(uuid.uuid4()),
        "subject": sample.get("subject", "SAT"),
        "pool": sample.get("pool", "generated"),
        "section": section,
        "category": category,
        "skill": skill,
        "difficulty": difficulty,
        "type": "multiple-choice",
        "question": {
            "paragraph": generated.paragraph_text,
            "question": generated.question,
            "choices": generated.choices,
            "correct_answer": [generated.correct_answer_letter],
            "explanation": generated.explanation,
        },
        "image_url": None,
    }
    
    return new_question


# ============================================================================
# Utility Functions
# ============================================================================

def load_sample_question(
    questions_path: str = "questions_practice_test.json",
    index: int = 0,
    question_id: Optional[str] = None,
    skill: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load a sample R&W question from the questions file.
    
    Args:
        questions_path: Path to questions JSON file
        index: Index of question to load (if question_id not provided)
        question_id: Specific question ID to load
        skill: Filter by skill (e.g., "Inferences")
    
    Returns:
        Question dict
    """
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    # Filter R&W questions
    rw_questions = [q for q in questions if q.get("section") == "Reading and Writing"]
    
    if skill:
        rw_questions = [q for q in rw_questions if q.get("skill") == skill]
    
    if question_id:
        for q in rw_questions:
            if q.get("id") == question_id:
                return q
        raise ValueError(f"Question ID not found: {question_id}")
    
    if index >= len(rw_questions):
        raise ValueError(f"Index {index} out of range (only {len(rw_questions)} R&W questions)")
    
    return rw_questions[index]


def main():
    """CLI for generating R&W questions."""
    parser = argparse.ArgumentParser(description="Generate new SAT Reading & Writing questions")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of sample question")
    parser.add_argument("--question-id", type=str, default=None, help="Specific question ID to use as sample")
    parser.add_argument("--skill", type=str, default=None, help="Filter by skill (e.g., 'Inferences')")
    parser.add_argument("--questions-path", type=str, default="questions_practice_test.json", help="Path to questions file")
    parser.add_argument("--output", type=str, default=None, help="Output file path (JSON)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--count", type=int, default=1, help="Number of questions to generate")
    
    args = parser.parse_args()
    
    # Load sample
    sample = load_sample_question(
        questions_path=args.questions_path,
        index=args.sample_index,
        question_id=args.question_id,
        skill=args.skill,
    )
    
    print(f"Loaded sample question:")
    print(f"  ID: {sample.get('id')}")
    print(f"  Skill: {sample.get('skill')}")
    print(f"  Difficulty: {sample.get('difficulty')}")
    print(f"  Category: {sample.get('category')}")
    print()
    
    # Generate questions
    llm = ChatOpenAI(model=args.model, temperature=0.7)
    generated_questions = []
    
    for i in range(args.count):
        print(f"Generating question {i+1}/{args.count}...")
        new_q = generate_new_rw_question(sample, llm=llm)
        generated_questions.append(new_q)
        
        print(f"Generated question {i+1}:")
        print(f"  Paragraph (first 100 chars): {new_q['question']['paragraph'][:100]}...")
        print(f"  Question: {new_q['question']['question']}")
        print(f"  Correct answer: {new_q['question']['correct_answer'][0]}")
        print()
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_questions, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(generated_questions)} question(s) to {output_path}")
    else:
        # Print first generated question
        print("Generated Question (full):")
        print(json.dumps(generated_questions[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
