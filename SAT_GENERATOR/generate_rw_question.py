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
import io
import re

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from mathml_parser import MathMLParser, TableSpec
from openai import LengthFinishReasonError

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


class GeneratedRWQuestionWithGroupedBarChart(BaseModel):
    """Generated R&W question with grouped bar chart (new scenario with new data)."""
    paragraph_text: str = Field(description="New paragraph text WITHOUT the graph (describes new scenario)")
    graph_title: str = Field(description="New graph title")
    graph_y_label: str = Field(description="Y-axis label (appropriate for new data)")
    graph_groups: List[str] = Field(description="Group names (same number as original, e.g., ['condition A', 'condition B'])")
    graph_categories: List[str] = Field(description="Category names on X-axis (same number as original)")
    # Flatten the data structure - each category's data as a separate field
    graph_data_flat: List[float] = Field(description="Flattened data: for each category, provide values for each group in order")
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


class GeneratedRWQuestionWithBarChart(BaseModel):
    """Generated R&W question with simple bar chart (new scenario with new data)."""
    paragraph_text: str = Field(description="New paragraph text WITHOUT the graph (describes new scenario)")
    graph_title: str = Field(description="New graph title")
    graph_y_label: str = Field(description="Y-axis label (appropriate for new data)")
    graph_x_label: str = Field(description="X-axis label (e.g., 'Glacier', 'Product', 'City')")
    graph_categories: List[str] = Field(description="Category names on X-axis (same number as original)")
    graph_values: List[float] = Field(description="Values for each category (Y-axis data)")
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
    
    @field_validator("graph_values")
    @classmethod
    def validate_values_match_categories(cls, v, info):
        if 'graph_categories' in info.data:
            categories = info.data['graph_categories']
            if len(v) != len(categories):
                raise ValueError(f"Number of values ({len(v)}) must match number of categories ({len(categories)})")
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


def _calculate_clean_y_axis_range(max_value: float) -> tuple:
    """
    Calculate clean, SAT-style y-axis range for graphs.
    
    Args:
        max_value: Maximum data value
    
    Returns:
        Tuple of (y_min, y_max, y_increment) with clean, rounded values
    
    Examples:
        - max_value=150 → (0, 200, 50)
        - max_value=45 → (0, 95, 10)
        - max_value=750 → (0, 800, 100)
    """
    import math
    
    if max_value <= 0:
        return (0, 100, 20)
    
    # Set y_max to max_value + 50 for consistent height
    y_max = max_value + 50
    
    # Determine appropriate increment based on y_max magnitude
    if y_max <= 50:
        increment = 10
    elif y_max <= 100:
        increment = 10
    elif y_max <= 200:
        increment = 20
    elif y_max <= 500:
        increment = 50
    elif y_max <= 1000:
        increment = 100
    elif y_max <= 2000:
        increment = 200
    elif y_max <= 5000:
        increment = 500
    else:
        increment = 1000
    
    # Round y_max UP to nearest multiple of increment for clean divisions
    y_max_rounded = math.ceil(y_max / increment) * increment
    
    return (0, int(y_max_rounded), int(increment))


def _generate_bar_chart_svg(
    title: str,
    y_label: str,
    x_label: str,
    categories: List[str],
    values: List[float],
    y_range: Optional[tuple] = None,
) -> str:
    """
    Generate simple bar chart SVG using matplotlib.
    
    Args:
        title: Graph title
        y_label: Y-axis label
        x_label: X-axis label
        categories: List of category names (X-axis labels)
        values: List of values for each category
        y_range: Optional (min, max, increment) for y-axis
    
    Returns:
        SVG string
    """
    # Set up the figure with SAT-style formatting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for bars
    num_categories = len(categories)
    x = np.arange(num_categories)
    
    # Colors: dark gray, light gray, black (matching SAT style)
    colors = ['#666666', '#CCCCCC', '#000000']
    bar_colors = [colors[i % len(colors)] for i in range(num_categories)]
    
    # Plot bars
    ax.bar(x, values, color=bar_colors, edgecolor='black', linewidth=0.9)
    
    # Customize axes
    ax.set_ylabel(y_label, fontfamily='serif', fontsize=12)
    ax.set_xlabel(x_label, fontfamily='serif', fontsize=12)
    ax.set_title(title, fontfamily='serif', fontsize=13, wrap=True)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontfamily='serif', fontsize=11)
    
    # Set y-axis range if provided
    if y_range:
        y_min, y_max, y_inc = y_range
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + y_inc, y_inc))
    
    # Format y-axis
    ax.tick_params(axis='y', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # Grid and styling
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to SVG string
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg', bbox_inches='tight')
    plt.close(fig)
    
    svg_string = svg_buffer.getvalue()
    svg_buffer.close()
    
    return svg_string


def _generate_grouped_bar_chart_svg(
    title: str,
    y_label: str,
    groups: List[str],
    categories: List[str],
    data: Dict[str, Dict[str, float]],
    y_range: Optional[tuple] = None,
) -> str:
    """
    Generate grouped bar chart SVG using matplotlib.
    
    Args:
        title: Graph title
        y_label: Y-axis label
        groups: List of group names (e.g., ["Group A", "Group B"])
        categories: List of category names (X-axis labels)
        data: Nested dict {category: {group: value}}
        y_range: Optional (min, max, increment) for y-axis
    
    Returns:
        SVG string
    """
    # Set up the figure with SAT-style formatting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for grouped bars
    num_groups = len(groups)
    num_categories = len(categories)
    x = np.arange(num_categories)
    width = 0.35  # Width of bars
    
    # Colors: light gray and dark gray (matching SAT style)
    colors = ['#B3B3B3', '#333333']
    
    # Plot bars for each group
    for i, group in enumerate(groups):
        values = [data.get(cat, {}).get(group, 0) for cat in categories]
        offset = (i - num_groups/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=group, color=colors[i % len(colors)],
               edgecolor='black', linewidth=0.9)
    
    # Customize axes
    ax.set_ylabel(y_label, fontfamily='serif', fontsize=12)
    ax.set_title(title, fontfamily='serif', fontsize=13, wrap=True)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontfamily='serif', fontsize=11, rotation=-40, ha='right')
    
    # Set y-axis range if provided
    if y_range:
        y_min, y_max, y_inc = y_range
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + y_inc, y_inc))
    
    # Format y-axis
    ax.tick_params(axis='y', labelsize=11)
    for label in ax.get_yticklabels():
        label.set_fontfamily('serif')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fontsize=11, prop={'family': 'serif'})
    
    # Grid and styling
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save to SVG string
    svg_buffer = io.StringIO()
    plt.savefig(svg_buffer, format='svg', bbox_inches='tight')
    plt.close(fig)
    
    svg_string = svg_buffer.getvalue()
    svg_buffer.close()
    
    return svg_string


def _build_bar_chart_long_description_html(
    title: str,
    categories: List[str],
    values: List[float],
    y_unit: Optional[str] = None,
) -> str:
    """
    Build the sr-only long description HTML for simple bar chart.
    
    Args:
        title: Graph title
        categories: List of category names
        values: List of values for each category
        y_unit: Optional unit for values (e.g., "square kilometers")
    
    Returns:
        HTML string for sr-only div with long description
    """
    html = f'<div aria-label="Long description for bar graph titled {title}" class="sr-only" role="region">\n'
    html += '<ul>\n'
    
    # Data by category
    html += f'<li>The data for the {len(categories)} categories are as follows: <br/>\n<ul>\n'
    for i, category in enumerate(categories):
        value = values[i] if i < len(values) else 0
        # Format numbers with commas for readability
        formatted_value = f"{int(value):,}" if value == int(value) else f"{value:,.1f}"
        if y_unit:
            html += f'<li>{category}: {formatted_value} {y_unit}</li>\n'
        else:
            html += f'<li>{category}: {formatted_value}</li>\n'
    
    html += '</ul>\n</li>\n'
    html += '</ul>\n'
    html += '</div>'
    
    return html


def _build_long_description_html(
    title: str,
    groups: List[str],
    categories: List[str],
    data: Dict[str, Dict[str, float]],
) -> str:
    """
    Build the sr-only long description HTML for grouped bar chart.
    
    Args:
        title: Graph title
        groups: List of group names
        categories: List of category names
        data: Nested dict {category: {group: value}}
    
    Returns:
        HTML string for sr-only div with long description
    """
    html = f'<div aria-label="Long description for bar graph titled {title}" class="sr-only" role="region">\n'
    html += '<ul>\n'
    
    # Section 1: Group names
    html += '<li>For each data category, the following bars are shown: <br/>\n<ul>\n'
    for group in groups:
        html += f'<li>{group}</li>\n'
    html += '</ul>\n</li>\n'
    
    # Section 2: Data by category
    html += f'<li>The data for the {len(categories)} categories are as follows: <br/>\n<ul>\n'
    for category in categories:
        html += f'<li>{category}:\n<ul>\n'
        if category in data:
            for group, value in data[category].items():
                # Format numbers with commas for readability
                formatted_value = f"{int(value):,}" if value == int(value) else f"{value:,.1f}"
                html += f'<li>{group}: {formatted_value}</li>\n'
        html += '</ul>\n</li>\n'
    
    html += '</ul>\n</li>\n'
    html += '</ul>\n'
    html += '</div>'
    
    return html


def _build_figure_block_with_graph(
    svg_string: str,
    long_description_html: str,
    paragraph_text: str,
) -> str:
    """
    Build complete figure block with SVG and long description, then combine with paragraph.
    
    Args:
        svg_string: SVG content (full <svg>...</svg>)
        long_description_html: Long description div (<div class="sr-only">...</div>)
        paragraph_text: Paragraph text (without graph)
    
    Returns:
        Complete paragraph HTML with embedded figure
    """
    # Extract just the SVG tag content (remove XML declaration if present)
    svg_match = re.search(r'(<svg[^>]*>.*?</svg>)', svg_string, re.DOTALL)
    if svg_match:
        svg_content = svg_match.group(1)
    else:
        svg_content = svg_string
    
    # Build figure block
    figure_html = f'<figure class="image">\n{svg_content}\n</figure>\n{long_description_html}'
    
    # Combine with paragraph
    combined = f'<p>{paragraph_text}</p>\n\n{figure_html}'
    
    return combined


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
    
    # Do not include A., B., C., D. prefixes — UI handles labeling
    choices_text = "\n".join([f"{c}" for c in choices])
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
    
    # Do not include A., B., C., D. prefixes — UI handles labeling
    choices_text = "\n".join([f"{c}" for c in choices])
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


def _build_bar_chart_generation_prompt(
    paragraph: str,
    graph_spec,  # GraphSpec with x_values and y_values
    question: str,
    choices: List[str],
    correct_letter: str,
    explanation: str,
    skill: str,
    category: str,
    difficulty: str,
) -> str:
    """Build prompt for LLM to generate new R&W question with simple bar chart."""
    
    reasoning_type = _infer_reasoning_type(skill, question, paragraph)
    logical_schema = _infer_logical_schema(skill, paragraph, choices)
    
    # Do not include A., B., C., D. prefixes — UI handles labeling
    choices_text = "\n".join([f"{c}" for c in choices])
    correct_choice = choices[ord(correct_letter) - ord('A')] if correct_letter in "ABCD" else choices[0]
    
    # Format graph structure for LLM
    if graph_spec.y_axis_range:
        y_axis_range_str = f"Y-axis Range: {graph_spec.y_axis_range[0]} to {graph_spec.y_axis_range[1]} (increments of {graph_spec.y_axis_range[2]})"
    else:
        y_axis_range_str = "Y-axis Range: Not specified (infer from data)"
    
    graph_structure = f"""**Original Bar Chart Structure:**

Title: {graph_spec.title}
Y-axis Label: {graph_spec.y_label}
X-axis Label: {graph_spec.x_label or 'Category'}
{y_axis_range_str}

Categories: {graph_spec.x_values}
(These are shown on the X-axis)

Data:
"""
    
    for i, category in enumerate(graph_spec.x_values or []):
        value = graph_spec.y_values[i] if i < len(graph_spec.y_values or []) else 0
        graph_structure += f"  {category}: {value}\n"
    
    # Analyze data patterns
    patterns = []
    if graph_spec.y_values:
        max_val = max(graph_spec.y_values)
        min_val = min(graph_spec.y_values)
        max_idx = graph_spec.y_values.index(max_val)
        min_idx = graph_spec.y_values.index(min_val)
        
        max_category = graph_spec.x_values[max_idx] if max_idx < len(graph_spec.x_values) else "unknown"
        min_category = graph_spec.x_values[min_idx] if min_idx < len(graph_spec.x_values) else "unknown"
        
        patterns.append(f"- Highest value: {max_category} ({max_val})")
        patterns.append(f"- Lowest value: {min_category} ({min_val})")
        patterns.append("- Preserve the ranking pattern (highest to lowest)")
    
    patterns_text = "\n".join(patterns) if patterns else "- Analyze the original data pattern and preserve it"
    
    prompt = f"""You are an SAT Reading & Writing question designer.

Your task: Generate a NEW scenario with a NEW BAR CHART that tests the SAME reasoning skill.

**Original Question Analysis:**

Category: {category}
Skill: {skill}
Difficulty: {difficulty}
Reasoning Type: {reasoning_type}
Logical Schema: {logical_schema}

{graph_structure}

**Data Patterns to Preserve:**
{patterns_text}

**Original Paragraph (without graph):**
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
   - If original is about glaciers, try: products, buildings, species, countries, etc.
   - Change ALL specific details (names, context, description)

2. **Generate NEW BAR CHART DATA**:
   - Same structure: {len(graph_spec.x_values or [])} categories
   - New graph title (related to your new scenario)
   - New Y-axis label (appropriate for your new measurement type)
   - New X-axis label (appropriate for your new categories)
   - New category names ({len(graph_spec.x_values or [])} items)
   - New numerical data (realistic for your context)
   - **PRESERVE THE RANKING PATTERN**: If original has highest→middle→lowest, preserve that order!

3. **Write NEW paragraph** describing the scenario (WITHOUT the graph - graph will be shown separately):
   - Describe the measurement/study context
   - Mention what is being compared
   - Set up the question naturally
   - Keep academic tone

4. **Keep or adapt the question** (usually asking to complete sentence using graph data)

5. **Create 4 answer choices** with SAME distractor logic:
   - 1 correct choice that accurately describes the data pattern
   - 3 wrong choices with similar errors as original (wrong comparison, reversed order, incorrect values, etc.)

6. **Write explanation** of why correct answer works and why others don't

**Important:**
- DO NOT reuse the topic/field from original
- Keep SAT academic tone
- Data must be REALISTIC and follow the same ranking pattern as original
- Same difficulty level: {difficulty}
- The data should support the same type of comparison/conclusion

**Output JSON with these fields:**
- paragraph_text: New paragraph WITHOUT graph
- graph_title: New graph title
- graph_y_label: Y-axis label
- graph_x_label: X-axis label (e.g., "Glacier", "Product", "Species")
- graph_categories: List of {len(graph_spec.x_values or [])} category names
- graph_values: List of {len(graph_spec.x_values or [])} values (numbers for Y-axis)
- question: Question text
- choices: Array of 4 answer choices
- correct_answer_letter: "A", "B", "C", or "D"
- explanation: Clear explanation

Generate a completely new, high-quality SAT R&W question with bar chart now."""

    return prompt


def _build_grouped_bar_chart_generation_prompt(
    paragraph: str,
    graph_spec,  # GraphSpec with grouped_data
    question: str,
    choices: List[str],
    correct_letter: str,
    explanation: str,
    skill: str,
    category: str,
    difficulty: str,
) -> str:
    """Build prompt for LLM to generate new R&W question with grouped bar chart."""
    
    reasoning_type = _infer_reasoning_type(skill, question, paragraph)
    logical_schema = _infer_logical_schema(skill, paragraph, choices)
    
    # Do not include A., B., C., D. prefixes — UI handles labeling
    choices_text = "\n".join([f"{c}" for c in choices])
    correct_choice = choices[ord(correct_letter) - ord('A')] if correct_letter in "ABCD" else choices[0]
    
    # Format graph structure for LLM
    # Handle y_axis_range - it may be None if not specified in the graph
    if graph_spec.y_axis_range:
        y_axis_range_str = f"Y-axis Range: {graph_spec.y_axis_range[0]} to {graph_spec.y_axis_range[1]} (increments of {graph_spec.y_axis_range[2]})"
    else:
        y_axis_range_str = "Y-axis Range: Not specified (infer from data)"
    
    graph_structure = f"""**Original Grouped Bar Chart Structure:**

Title: {graph_spec.title}
Y-axis Label: {graph_spec.y_label}
{y_axis_range_str}

Groups: {graph_spec.groups}
(These are the bars shown for each category)

Categories: {graph_spec.categories}
(These are shown on the X-axis)

Data:
"""
    
    for category in graph_spec.categories:
        graph_structure += f"\n  {category}:\n"
        if category in graph_spec.grouped_data:
            for group, value in graph_spec.grouped_data[category].items():
                graph_structure += f"    {group}: {value}\n"
    
    # Analyze data patterns
    patterns = []
    
    # Check if groups have similar values across categories
    if graph_spec.grouped_data:
        first_category = graph_spec.categories[0]
        if first_category in graph_spec.grouped_data:
            group1_values = []
            group2_values = []
            
            for cat in graph_spec.categories:
                if cat in graph_spec.grouped_data:
                    values = list(graph_spec.grouped_data[cat].values())
                    if len(values) >= 2:
                        group1_values.append(values[0])
                        group2_values.append(values[1])
            
            if group1_values and group2_values:
                avg_diff = sum(abs(v1 - v2) for v1, v2 in zip(group1_values, group2_values)) / len(group1_values)
                if avg_diff < 50:  # Relatively small differences
                    patterns.append("- Both groups show SIMILAR values across all categories (small differences)")
                else:
                    patterns.append("- Groups show DIFFERENT values across categories (one group consistently higher/lower)")
    
    patterns_text = "\n".join(patterns) if patterns else "- Analyze the original data pattern and preserve it"
    
    prompt = f"""You are an SAT Reading & Writing question designer.

Your task: Generate a NEW scenario with a NEW GROUPED BAR CHART that tests the SAME reasoning skill.

**Original Question Analysis:**

Category: {category}
Skill: {skill}
Difficulty: {difficulty}
Reasoning Type: {reasoning_type}
Logical Schema: {logical_schema}

{graph_structure}

**Data Patterns to Preserve:**
{patterns_text}

**Original Paragraph (without graph):**
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
   - If original is about municipalities/politics, try: companies, schools, products, countries, etc.
   - Change ALL specific details (names, context, study description)

2. **Generate NEW GROUPED BAR CHART DATA**:
   - Same structure: {len(graph_spec.groups)} groups, {len(graph_spec.categories)} categories
   - New graph title (related to your new scenario)
   - New Y-axis label (appropriate for your new data type)
   - New group names (the two conditions/treatments in your scenario)
   - New category names (the X-axis categories)
   - New numerical data (realistic for your context)
   - **PRESERVE THE DATA PATTERN**: If original groups have similar values, your groups should too!

3. **Write NEW paragraph** describing the research/study scenario (WITHOUT the graph - graph will be shown separately):
   - Describe the study setup
   - Explain what the researchers measured
   - State the hypothesis being tested
   - Keep academic/research tone

4. **Keep or adapt the question** (usually "Which choice best describes data from the graph that weaken/support the hypothesis?")

5. **Create 4 answer choices** with SAME distractor logic:
   - 1 correct choice that identifies the key pattern (e.g., "groups are similar" or "groups differ")
   - 3 wrong choices with similar errors as original (mentions only one group, wrong comparison, irrelevant data, etc.)

6. **Write explanation** of why correct answer works and why others don't

**Important:**
- DO NOT reuse the topic/field from original
- Keep SAT academic/research tone
- Data must be REALISTIC and follow the same pattern as original
- Same difficulty level: {difficulty}
- Graph data should make the same argumentative point (weaken/support hypothesis in same way)

**Output JSON with these fields:**
- paragraph_text: New paragraph WITHOUT graph
- graph_title: New graph title
- graph_y_label: Y-axis label  
- graph_groups: List of {len(graph_spec.groups)} group names
- graph_categories: List of {len(graph_spec.categories)} category names
- graph_data_flat: List of {len(graph_spec.categories) * len(graph_spec.groups)} values (for each category, provide values for each group in order)
  Example: If 3 categories and 2 groups, provide [cat1_group1, cat1_group2, cat2_group1, cat2_group2, cat3_group1, cat3_group2]
- question: Question text
- choices: Array of 4 answer choices
- correct_answer_letter: "A", "B", "C", or "D"
- explanation: Clear explanation

Generate a completely new, high-quality SAT R&W question with grouped bar chart now."""

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
        llm = ChatOpenAI(model=model, temperature=0.7, api_key=key, max_tokens=8192)
    
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
        except LengthFinishReasonError as e:
            if verbose:
                print(f"[generate_rw_question] Error: LLM hit token limit during generation")
                print(f"  Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}")
                print(f"  Prompt tokens: {e.completion.usage.prompt_tokens if e.completion.usage else 'unknown'}")
            raise ValueError(
                f"LLM response exceeded token limit. Try using a model with higher limits or simplifying the prompt. "
                f"Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}"
            ) from e
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
    # Handle graph case - check if it's a grouped bar chart first
    elif has_graph:
        # Parse graph to see if it's a grouped bar chart
        parser = MathMLParser()
        parsed = parser.parse_paragraph(paragraph)
        
        if parsed.get("has_graph") and parsed.get("graph"):
            graph_spec = parsed["graph"]
            
            # Check if this is a grouped bar chart
            if graph_spec.graph_type == "grouped_bar" and graph_spec.grouped_data:
                paragraph_for_prompt = parsed["text"]
                
                if verbose:
                    print(f"[generate_rw_question] Detected grouped bar chart: {graph_spec.title}")
                    print(f"  Groups: {len(graph_spec.groups or [])} | Categories: {len(graph_spec.categories or [])}")
                    print(f"  Y-axis: {graph_spec.y_label} ({graph_spec.y_axis_range})")
                    print(f"  Paragraph without graph: {len(paragraph_for_prompt)} chars")
                
                # Build grouped bar chart generation prompt
                prompt = _build_grouped_bar_chart_generation_prompt(
                    paragraph=paragraph_for_prompt,
                    graph_spec=graph_spec,
                    question=question_text,
                    choices=choices,
                    correct_letter=correct_letter,
                    explanation=explanation,
                    skill=skill,
                    category=category,
                    difficulty=difficulty,
                )
                
                # Use grouped bar chart specific schema
                llm_with_structure = llm.with_structured_output(GeneratedRWQuestionWithGroupedBarChart)
                if verbose:
                    print(f"[generate_rw_question] Calling LLM for grouped bar chart generation... (prompt length: {len(prompt)} chars)")
                
                try:
                    generated = llm_with_structure.invoke([HumanMessage(content=prompt)])
                    
                    if verbose:
                        print("[generate_rw_question] LLM grouped bar chart generation completed successfully")
                        print(f"  New graph title: {generated.graph_title}")
                except LengthFinishReasonError as e:
                    if verbose:
                        print(f"[generate_rw_question] Error: LLM hit token limit during generation")
                        print(f"  Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}")
                        print(f"  Prompt tokens: {e.completion.usage.prompt_tokens if e.completion.usage else 'unknown'}")
                    raise ValueError(
                        f"LLM response exceeded token limit. Try using a model with higher limits or simplifying the prompt. "
                        f"Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}"
                    ) from e
                except Exception as e:
                    if verbose:
                        print(f"[generate_rw_question] Error during LLM generation: {e}")
                    raise
                
                # Convert flattened data back to nested dict structure
                # graph_data_flat is: [cat1_grp1, cat1_grp2, cat2_grp1, cat2_grp2, ...]
                graph_data = {}
                num_groups = len(generated.graph_groups)
                data_idx = 0
                
                for category in generated.graph_categories:
                    graph_data[category] = {}
                    for i, group in enumerate(generated.graph_groups):
                        if data_idx < len(generated.graph_data_flat):
                            graph_data[category][group] = generated.graph_data_flat[data_idx]
                            data_idx += 1
                
                if verbose:
                    print(f"[generate_rw_question] Converted flattened data to nested structure")
                    print(f"  Categories: {len(graph_data)}, Data points: {sum(len(v) for v in graph_data.values())}")
                
                # Generate SVG graph using matplotlib
                if verbose:
                    print(f"[generate_rw_question] Generating grouped bar chart SVG...")
                
                try:
                    # Always calculate y-axis range from new data (not from original graph)
                    all_values = [v for cat_data in graph_data.values() for v in cat_data.values()]
                    max_val = max(all_values) if all_values else 100
                    y_range = _calculate_clean_y_axis_range(max_val)
                    
                    svg_string = _generate_grouped_bar_chart_svg(
                        title=generated.graph_title,
                        y_label=generated.graph_y_label,
                        groups=generated.graph_groups,
                        categories=generated.graph_categories,
                        data=graph_data,
                        y_range=y_range,
                    )
                    
                    if verbose:
                        print(f"[generate_rw_question] SVG generated ({len(svg_string)} chars)")
                except Exception as e:
                    if verbose:
                        print(f"[generate_rw_question] Warning: SVG generation failed: {e}")
                        print(f"[generate_rw_question] Falling back to text description")
                    svg_string = None
                
                # Build long description HTML (sr-only div)
                long_desc_html = _build_long_description_html(
                    title=generated.graph_title,
                    groups=generated.graph_groups,
                    categories=generated.graph_categories,
                    data=graph_data,
                )
                
                if verbose:
                    print(f"[generate_rw_question] Built long description HTML ({len(long_desc_html)} chars)")
                
                # Build complete paragraph with figure block
                if svg_string:
                    final_paragraph = _build_figure_block_with_graph(
                        svg_string=svg_string,
                        long_description_html=long_desc_html,
                        paragraph_text=generated.paragraph_text,
                    )
                    if verbose:
                        print(f"[generate_rw_question] Built complete figure block with SVG")
                else:
                    # Fallback: text description only
                    graph_data_description = f"\n\n[Graph showing: {generated.graph_title}. Y-axis: {generated.graph_y_label}. Data comparison between {' and '.join(generated.graph_groups)} across {len(generated.graph_categories)} categories.]\n\n"
                    final_paragraph = generated.paragraph_text + graph_data_description + "\n\n" + long_desc_html
                    if verbose:
                        print(f"[generate_rw_question] Using fallback text description")
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
                    # Store graph data for potential future SVG generation
                    "graph_data": {
                        "type": "grouped_bar",
                        "title": generated.graph_title,
                        "y_label": generated.graph_y_label,
                        "groups": generated.graph_groups,
                        "categories": generated.graph_categories,
                        "data": graph_data,  # Use the converted dict
                    }
                }
                
                return new_question
            elif graph_spec.graph_type == "bar" and graph_spec.x_values and graph_spec.y_values:
                paragraph_for_prompt = parsed["text"]
                
                if verbose:
                    print(f"[generate_rw_question] Detected simple bar chart: {graph_spec.title}")
                    print(f"  Categories: {len(graph_spec.x_values or [])}")
                    print(f"  Y-axis: {graph_spec.y_label} ({graph_spec.y_axis_range})")
                    print(f"  Paragraph without graph: {len(paragraph_for_prompt)} chars")
                
                # Build bar chart generation prompt
                prompt = _build_bar_chart_generation_prompt(
                    paragraph=paragraph_for_prompt,
                    graph_spec=graph_spec,
                    question=question_text,
                    choices=choices,
                    correct_letter=correct_letter,
                    explanation=explanation,
                    skill=skill,
                    category=category,
                    difficulty=difficulty,
                )
                
                # Use bar chart specific schema
                llm_with_structure = llm.with_structured_output(GeneratedRWQuestionWithBarChart)
                
                if verbose:
                    print(f"[generate_rw_question] Calling LLM for bar chart generation... (prompt length: {len(prompt)} chars)")
                
                try:
                    generated = llm_with_structure.invoke([HumanMessage(content=prompt)])
                    
                    if verbose:
                        print("[generate_rw_question] LLM bar chart generation completed successfully")
                        print(f"  New graph title: {generated.graph_title}")
                except LengthFinishReasonError as e:
                    if verbose:
                        print(f"[generate_rw_question] Error: LLM hit token limit during generation")
                        print(f"  Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}")
                        print(f"  Prompt tokens: {e.completion.usage.prompt_tokens if e.completion.usage else 'unknown'}")
                    raise ValueError(
                        f"LLM response exceeded token limit. Try using a model with higher limits or simplifying the prompt. "
                        f"Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}"
                    ) from e
                except Exception as e:
                    if verbose:
                        print(f"[generate_rw_question] Error during LLM generation: {e}")
                    raise
                
                # Generate SVG graph using matplotlib
                if verbose:
                    print(f"[generate_rw_question] Generating bar chart SVG...")
                
                try:
                    # Always calculate y-axis range from new data (not from original graph)
                    max_val = max(generated.graph_values) if generated.graph_values else 100
                    y_range = _calculate_clean_y_axis_range(max_val)
                    
                    svg_string = _generate_bar_chart_svg(
                        title=generated.graph_title,
                        y_label=generated.graph_y_label,
                        x_label=generated.graph_x_label,
                        categories=generated.graph_categories,
                        values=generated.graph_values,
                        y_range=y_range,
                    )
                    
                    if verbose:
                        print(f"[generate_rw_question] SVG generated ({len(svg_string)} chars)")
                except Exception as e:
                    if verbose:
                        print(f"[generate_rw_question] Warning: SVG generation failed: {e}")
                        print(f"[generate_rw_question] Falling back to text description")
                    svg_string = None
                
                # Extract Y-axis unit from y_label if possible
                y_unit = None
                if "(" in generated.graph_y_label and ")" in generated.graph_y_label:
                    # Extract unit from label like "Area (square km)"
                    import re
                    unit_match = re.search(r'\(([^)]+)\)', generated.graph_y_label)
                    if unit_match:
                        y_unit = unit_match.group(1)
                
                # Build long description HTML (sr-only div)
                long_desc_html = _build_bar_chart_long_description_html(
                    title=generated.graph_title,
                    categories=generated.graph_categories,
                    values=generated.graph_values,
                    y_unit=y_unit,
                )
                
                if verbose:
                    print(f"[generate_rw_question] Built long description HTML ({len(long_desc_html)} chars)")
                
                # Build complete paragraph with figure block
                if svg_string:
                    final_paragraph = _build_figure_block_with_graph(
                        svg_string=svg_string,
                        long_description_html=long_desc_html,
                        paragraph_text=generated.paragraph_text,
                    )
                    if verbose:
                        print(f"[generate_rw_question] Built complete figure block with SVG")
                else:
                    # Fallback: text description only
                    graph_data_description = f"\n\n[Bar chart showing: {generated.graph_title}. Y-axis: {generated.graph_y_label}. Categories: {', '.join(generated.graph_categories)}.]\n\n"
                    final_paragraph = generated.paragraph_text + graph_data_description + "\n\n" + long_desc_html
                    if verbose:
                        print(f"[generate_rw_question] Using fallback text description")
                
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
                    # Store graph data for potential future SVG generation
                    "graph_data": {
                        "type": "bar",
                        "title": generated.graph_title,
                        "y_label": generated.graph_y_label,
                        "x_label": generated.graph_x_label,
                        "categories": generated.graph_categories,
                        "values": generated.graph_values,
                    }
                }
                
                return new_question
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
      except LengthFinishReasonError as e:
          if verbose:
              print(f"[generate_rw_question] Error: LLM hit token limit during generation")
              print(f"  Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}")
              print(f"  Prompt tokens: {e.completion.usage.prompt_tokens if e.completion.usage else 'unknown'}")
          raise ValueError(
              f"LLM response exceeded token limit. Try using a model with higher limits or simplifying the prompt. "
              f"Completion tokens: {e.completion.usage.completion_tokens if e.completion.usage else 'unknown'}"
          ) from e
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
        question_id="0a2b60f3-73a9-48bf-8ed8-02ca96d39cb4",
        skill=args.skill,
    )
    
    print(f"Loaded sample question:")
    print(f"  ID: {sample.get('id')}")
    print(f"  Skill: {sample.get('skill')}")
    print(f"  Difficulty: {sample.get('difficulty')}")
    print(f"  Category: {sample.get('category')}")
    print()
    
    # Generate questions
    llm = ChatOpenAI(model=args.model, temperature=0.7, max_tokens=8192)
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
