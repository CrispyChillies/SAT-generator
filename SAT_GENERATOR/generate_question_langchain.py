#!/usr/bin/env python3
"""
Sinh câu hỏi mới từ câu hỏi mẫu bằng LangChain.
- Giữ nguyên category, section, type, difficulty của câu gốc.
- Sinh câu hỏi mới, explanation và đáp án (correct_answer); cùng format HTML + MathML, chỉ đổi số.
- Câu hỏi mới phải đúng format so với câu gốc (HTML + MathML).
"""

import os
import json
import re
import uuid
import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Cần đặt OPENAI_API_KEY trong môi trường hoặc truyền llm.")

# ---------------------------------------------------------------------------
# Import parser để lấy thông tin Graph
# ---------------------------------------------------------------------------

from mathml_parser import MathMLParser

# ---------------------------------------------------------------------------
# Cấu trúc output từ LLM: câu hỏi + explanation + đáp án
# ---------------------------------------------------------------------------

class GeneratedQuestionContent(BaseModel):
    """Câu hỏi mới, explanation và đáp án đúng; cùng format HTML+MathML, chỉ đổi số so với mẫu (dùng khi không phải multiple-choice hoặc không có 4 choices)."""
    question: str = Field(description="New question content in the same HTML and MathML format as the sample, with only numerical values changed")
    explanation: str = Field(description="New explanation in the same HTML and MathML format as the sample, with only numerical values changed to match the new question")
    correct_answer: str = Field(description="The correct answer for the new question, in the same format as the sample (e.g. HTML/MathML string of the right choice or value)")
    new_paragraph: Optional[str] = Field(default=None, description="If the sample has a paragraph, generate a new paragraph with updated numbers that match the new question. If sample has no paragraph, this should be null.")


class GeneratedMultipleChoiceContent(BaseModel):
    """Câu hỏi multiple-choice: câu hỏi + explanation + đúng 4 lựa chọn (A,B,C,D) + chữ cái đáp án đúng."""
    question: str = Field(description="New question content, same HTML+MathML format with only numerical values changed")
    explanation: str = Field(description="New explanation, same format with only numbers changed to match the new question")
    choices: List[str] = Field(description="Exactly 4 answer choices in order A, B, C, D; each is HTML+MathML string with only numbers changed")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="The letter of the correct answer (A, B, C, or D)")
    new_paragraph: Optional[str] = Field(default=None, description="If the sample has a paragraph, generate a new paragraph with updated numbers that match the new question. If sample has no paragraph, this should be null.")

    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]


class GeneratedGraphQuestionContent(BaseModel):
    """Output cho câu hỏi có đồ thị: LLM chỉ sinh text mới + số liệu đồ thị mới (không sinh SVG)."""
    question_text: str = Field(description="New question text (without SVG), same format with only numbers changed")
    explanation: str = Field(description="New explanation, same format with only numbers changed")
    choices: List[str] = Field(description="Exactly 4 answer choices in order A, B, C, D; each with only numbers changed")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="The letter of the correct answer")
    new_x_values: List[Union[int, float]] = Field(description="New x-axis values for the graph (e.g., years [2015, 2016, ...] or days [1.0, 2.0, 3.0, ...])")
    new_y_values: List[float] = Field(description="New y-axis values for the graph (e.g., percentages or temperatures)")
    new_long_description: str = Field(description="New long description for the graph in HTML format (<ul><li>...</li></ul>), matching the new x/y values. MUST preserve the same HTML structure as the original.")
    new_paragraph: Optional[str] = Field(default=None, description="If the sample has a paragraph, generate a new paragraph with updated numbers that match the new question. If sample has no paragraph, this should be null.")

    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]
    
    @field_validator("new_y_values")
    @classmethod
    def validate_xy_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure new_x_values and new_y_values have the same length."""
        if hasattr(info, 'data') and 'new_x_values' in info.data:
            x_values = info.data['new_x_values']
            if len(x_values) != len(v):
                raise ValueError(f"new_x_values and new_y_values must have the same length. Got {len(x_values)} x-values and {len(v)} y-values.")
        return v


class GeneratedGraphFreeResponseContent(BaseModel):
    """Output cho câu hỏi tự luận có đồ thị: LLM sinh text mới + số liệu đồ thị mới + correct_answer (không sinh SVG)."""
    question_text: str = Field(description="New question text (without SVG), same format with only numbers changed")
    explanation: str = Field(description="New explanation, same format with only numbers changed")
    correct_answer: str = Field(description="The correct answer for the new question, in the same format as the sample (e.g. HTML/MathML string of the right value)")
    new_x_values: List[Union[int, float]] = Field(description="New x-axis values for the graph (e.g., years [2015, 2016, ...] or days [1.0, 2.0, 3.0, ...])")
    new_y_values: List[float] = Field(description="New y-axis values for the graph (e.g., percentages or temperatures)")
    new_long_description: str = Field(description="New long description for the graph in HTML format (<ul><li>...</li></ul>), matching the new x/y values. MUST preserve the same HTML structure as the original.")
    new_paragraph: Optional[str] = Field(default=None, description="If the sample has a paragraph, generate a new paragraph with updated numbers that match the new question. If sample has no paragraph, this should be null.")
    
    @field_validator("new_y_values")
    @classmethod
    def validate_xy_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure new_x_values and new_y_values have the same length."""
        if hasattr(info, 'data') and 'new_x_values' in info.data:
            x_values = info.data['new_x_values']
            if len(x_values) != len(v):
                raise ValueError(f"new_x_values and new_y_values must have the same length. Got {len(x_values)} x-values and {len(v)} y-values.")
        return v


class GeneratedParagraphGraphMultipleChoiceContent(BaseModel):
    """Output cho câu hỏi multiple-choice có đồ thị trong paragraph: LLM sinh text mới + số liệu đồ thị mới cho paragraph."""
    question: str = Field(description="New question content (just the question text, no graph)")
    explanation: str = Field(description="New explanation, same format with only numbers changed")
    choices: List[str] = Field(description="Exactly 4 answer choices in order A, B, C, D")
    correct_answer_letter: Literal["A", "B", "C", "D"] = Field(description="The letter of the correct answer")
    paragraph_text: str = Field(description="New paragraph text without SVG and long description, only the intro/context text")
    new_x_values: List[Union[int, float]] = Field(description="New x-axis values for the paragraph graph")
    new_y_values: List[float] = Field(description="New y-axis values for the paragraph graph")
    new_long_description: str = Field(description="New long description for the paragraph graph in HTML format, preserving original structure")
    
    @field_validator("choices")
    @classmethod
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]
    
    @field_validator("new_y_values")
    @classmethod
    def validate_xy_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure new_x_values and new_y_values have the same length."""
        if hasattr(info, 'data') and 'new_x_values' in info.data:
            x_values = info.data['new_x_values']
            if len(x_values) != len(v):
                raise ValueError(f"new_x_values and new_y_values must have the same length. Got {len(x_values)} x-values and {len(v)} y-values.")
        return v


# ---------------------------------------------------------------------------
# Utility functions cho xử lý đồ thị
# ---------------------------------------------------------------------------
    def choices_must_be_four(cls, v: List[str]) -> List[str]:
        if v is None or len(v) != 4:
            raise ValueError("choices phải có đúng 4 phần tử (A, B, C, D)")
        return [str(x).strip() for x in v]
    
    @field_validator("new_y_values")
    @classmethod
    def validate_xy_length_match(cls, v: List[float], info) -> List[float]:
        """Ensure new_x_values and new_y_values have the same length."""
        if hasattr(info, 'data') and 'new_x_values' in info.data:
            x_values = info.data['new_x_values']
            if len(x_values) != len(v):
                raise ValueError(f"new_x_values and new_y_values must have the same length. Got {len(x_values)} x-values and {len(v)} y-values.")
        return v


# ---------------------------------------------------------------------------
# Utility functions cho xử lý đồ thị
# ---------------------------------------------------------------------------

def _remove_svg_from_html(html: str) -> str:
    """Loại bỏ toàn bộ SVG element khỏi HTML."""
    return re.sub(r'<svg\b.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)


def _remove_svg_and_long_desc_from_html(html: str) -> str:
    """Loại bỏ toàn bộ SVG element và long description (sr-only div) khỏi HTML."""
    # Remove SVG
    result = re.sub(r'<svg\b.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Remove sr-only div containing long description
    result = re.sub(r'<div[^>]*class="sr-only"[^>]*>.*?</div>', '', result, flags=re.DOTALL | re.IGNORECASE)
    return result


def _generate_line_graph_svg(
    x_values: List[Union[int, float]],
    y_values: List[float],
    x_label: str = "Model year",
    y_label: str = "Percent of cars for sale",
    y_unit: str = "%",
    width: float = 8,
    height: float = 5,
) -> str:
    """
    Tạo line graph SVG bằng matplotlib.
    
    Args:
        x_values: Danh sách giá trị trục x (vd: năm)
        y_values: Danh sách giá trị trục y (vd: percent)
        x_label: Nhãn trục x
        y_label: Nhãn trục y
        y_unit: Đơn vị y (vd: "%")
        width: Chiều rộng đồ thị (inches)
        height: Chiều cao đồ thị (inches)
    
    Returns:
        SVG string của đồ thị
    """
    # Tạo figure với kích thước phù hợp
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Vẽ line graph với markers
    ax.plot(x_values, y_values, marker='o', markersize=8, linewidth=2, color='black')
    
    # Thiết lập labels
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    
    # Thiết lập trục x - hiển thị tất cả năm, xoay nếu cần
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values], rotation=45, ha='right')
    
    # Thiết lập trục y - từ 0 đến max + buffer, với grid lines
    y_max = max(y_values)
    y_axis_max = int((y_max // 5 + 1) * 5)  # Round up to nearest 5
    ax.set_ylim(0, y_axis_max)
    ax.set_yticks(range(0, y_axis_max + 1, 5))
    if y_unit == "%":
        ax.set_yticklabels([f"{y}%" for y in range(0, y_axis_max + 1, 5)])
    
    # Thêm grid
    ax.grid(True, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tight layout để tránh cắt labels
    plt.tight_layout()
    
    # Export to SVG string
    svg_buffer = io.BytesIO()
    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
    plt.close(fig)  # Đóng figure để giải phóng memory
    
    svg_buffer.seek(0)
    svg_string = svg_buffer.getvalue().decode('utf-8')
    
    # Loại bỏ XML declaration và DOCTYPE nếu có
    svg_string = re.sub(r'<\?xml[^>]*\?>', '', svg_string)
    svg_string = re.sub(r'<!DOCTYPE[^>]*>', '', svg_string)
    svg_string = svg_string.strip()
    
    return svg_string


def _generate_bar_graph_svg(
    x_values: List[Union[int, float]],
    y_values: List[float],
    x_label: str = "Group",
    y_label: str = "Number of books collected",
    y_unit: str = "",
    width: float = 8,
    height: float = 5,
) -> str:
    """
    Tạo bar graph SVG bằng matplotlib.
    
    Args:
        x_values: Danh sách giá trị trục x (vd: group 1, 2, 3, 4)
        y_values: Danh sách giá trị trục y (vd: số lượng)
        x_label: Nhãn trục x
        y_label: Nhãn trục y
        y_unit: Đơn vị y (vd: "", "books")
        width: Chiều rộng đồ thị (inches)
        height: Chiều cao đồ thị (inches)
    
    Returns:
        SVG string của đồ thị
    """
    # Tạo figure với kích thước phù hợp
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Vẽ bar graph
    bars = ax.bar(x_values, y_values, color='gray', edgecolor='black', linewidth=1.5, width=0.6)
    
    # Thiết lập labels
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    
    # Thiết lập trục x - hiển thị tất cả giá trị
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values])
    
    # Thiết lập trục y - từ 0 đến max + buffer
    y_max = max(y_values)
    # Round up to nearest 10 for bar graphs (usually larger numbers)
    if y_max <= 20:
        y_axis_max = int((y_max // 5 + 1) * 5)
        tick_interval = 5
    else:
        y_axis_max = int((y_max // 10 + 1) * 10)
        tick_interval = 10
    
    ax.set_ylim(0, y_axis_max)
    ax.set_yticks(range(0, y_axis_max + 1, tick_interval))
    if y_unit:
        ax.set_yticklabels([f"{y}{y_unit}" for y in range(0, y_axis_max + 1, tick_interval)])
    
    # Thêm grid (chỉ trục y)
    ax.grid(True, axis='y', linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tight layout để tránh cắt labels
    plt.tight_layout()
    
    # Export to SVG string
    svg_buffer = io.BytesIO()
    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
    plt.close(fig)  # Đóng figure để giải phóng memory
    
    svg_buffer.seek(0)
    svg_string = svg_buffer.getvalue().decode('utf-8')
    
    # Loại bỏ XML declaration và DOCTYPE nếu có
    svg_string = re.sub(r'<\?xml[^>]*\?>', '', svg_string)
    svg_string = re.sub(r'<!DOCTYPE[^>]*>', '', svg_string)
    svg_string = svg_string.strip()
    
    return svg_string


def _generate_scatter_plot_svg(
    x_values: List[float],
    y_values: List[float],
    x_label: str = "Time (days since June 1)",
    y_label: str = "Temperature (°F)",
    y_unit: str = "°F",
    width: float = 8,
    height: float = 5,
) -> str:
    """
    Tạo scatter plot SVG bằng matplotlib.
    
    Args:
        x_values: Danh sách giá trị trục x (vd: days [1, 2, 3, 4, 5, 6, 7])
        y_values: Danh sách giá trị trục y (vd: temperatures [69, 60, 73, ...])
        x_label: Nhãn trục x
        y_label: Nhãn trục y
        y_unit: Đơn vị y (vd: "°F")
        width: Chiều rộng đồ thị (inches)
        height: Chiều cao đồ thị (inches)
    
    Returns:
        SVG string của đồ thị
    """
    # Tạo figure với kích thước phù hợp
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Vẽ scatter plot
    ax.scatter(x_values, y_values, s=60, color='black', marker='o', zorder=3)
    
    # Thiết lập labels
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    
    # Thiết lập trục x - hiển thị tất cả giá trị
    if all(isinstance(x, (int, float)) for x in x_values):
        x_min, x_max = min(x_values), max(x_values)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(int(x)) if x == int(x) else str(x) for x in x_values])
    
    # Thiết lập trục y
    y_min, y_max = min(y_values), max(y_values)
    y_range = y_max - y_min
    # Add 10% padding
    y_axis_min = max(0, y_min - y_range * 0.1)
    y_axis_max = y_max + y_range * 0.1
    
    # Round to nice numbers
    if y_axis_max <= 100:
        tick_interval = 10
    else:
        tick_interval = 20
    
    y_axis_min = int(y_axis_min // tick_interval) * tick_interval
    y_axis_max = int((y_axis_max // tick_interval) + 1) * tick_interval
    
    ax.set_ylim(y_axis_min, y_axis_max)
    ax.set_yticks(range(y_axis_min, y_axis_max + 1, tick_interval))
    if y_unit:
        ax.set_yticklabels([f"{y}{y_unit}" for y in range(y_axis_min, y_axis_max + 1, tick_interval)])
    
    # Thêm grid
    ax.grid(True, linestyle='-', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # Tight layout để tránh cắt labels
    plt.tight_layout()
    
    # Export to SVG string
    svg_buffer = io.BytesIO()
    fig.savefig(svg_buffer, format='svg', bbox_inches='tight')
    plt.close(fig)  # Đóng figure để giải phóng memory
    
    svg_buffer.seek(0)
    svg_string = svg_buffer.getvalue().decode('utf-8')
    
    # Loại bỏ XML declaration và DOCTYPE nếu có
    svg_string = re.sub(r'<\?xml[^>]*\?>', '', svg_string)
    svg_string = re.sub(r'<!DOCTYPE[^>]*>', '', svg_string)
    svg_string = svg_string.strip()
    
    return svg_string


def _generate_aria_label(
    x_values: List[Union[int, float]],
    y_values: List[float],
    x_label: str = "Model year",
    y_label: str = "Percent of cars for sale",
    y_unit: str = "%",
    graph_type: str = "line",
) -> str:
    """
    Tạo aria-label cho đồ thị (accessibility).
    """
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Determine tick interval based on graph type and y_max
    if graph_type == "bar":
        y_axis_min = 0
        if y_max <= 20:
            y_axis_max = int((y_max // 5 + 1) * 5)
            tick_interval = 5
        else:
            y_axis_max = int((y_max // 10 + 1) * 10)
            tick_interval = 10
    elif graph_type == "scatter":
        # Scatter plots use different y-axis range
        y_range = y_max - y_min
        y_axis_min = max(0, y_min - y_range * 0.1)
        y_axis_max = y_max + y_range * 0.1
        
        if y_axis_max <= 100:
            tick_interval = 10
        else:
            tick_interval = 20
        
        y_axis_min = int(y_axis_min // tick_interval) * tick_interval
        y_axis_max = int((y_axis_max // tick_interval) + 1) * tick_interval
    else:  # line graph
        y_axis_min = 0
        y_axis_max = int((y_max // 5 + 1) * 5)
        tick_interval = 5
    
    if graph_type == "scatter":
        graph_type_text = "scatter plot"
    elif graph_type == "bar":
        graph_type_text = "bar graph"
    else:
        graph_type_text = "line graph"
    
    # Format for scatter plot is slightly different
    if graph_type == "scatter":
        return (
            f"A {graph_type_text}. The horizontal axis is labeled {x_label}. "
            f"It ranges from {x_min} to {x_max} in increments of 1. "
            f"The vertical axis is labeled {y_label}. "
            f"It ranges from {y_axis_min}{y_unit} to {y_axis_max}{y_unit} in increments of {tick_interval}. "
            f"Refer to long description."
        )
    else:
        return (
            f"A {graph_type_text}. The horizontal axis is labeled {x_label}. "
            f"It ranges from {x_min} to {x_max} in increments of 1. "
            f"The vertical axis is labeled {y_label}. "
            f"It ranges from 0{y_unit} to {y_axis_max}{y_unit} in increments of 1, "
            f"with values marked every {tick_interval} grid lines. Refer to long description."
        )


def _update_graph_in_html(
    original_html: str,
    old_x_values: List[Union[int, float]],
    old_y_values: List[float],
    new_x_values: List[Union[int, float]],
    new_y_values: List[float],
    new_long_description: str,
    x_label: str = "Model year",
    y_label: str = "Percent of cars for sale",
    y_unit: str = "%",
    graph_type: str = "line",
) -> str:
    """
    Thay thế SVG cũ bằng SVG mới được tạo từ matplotlib và cập nhật long description.
    
    Args:
        original_html: HTML gốc chứa SVG và long description
        old_x_values: Giá trị x cũ (không dùng trực tiếp, giữ cho compatibility)
        old_y_values: Giá trị y cũ (không dùng trực tiếp, giữ cho compatibility)
        new_x_values: Giá trị x mới
        new_y_values: Giá trị y mới
        new_long_description: Mô tả đồ thị mới (sr-only text)
        x_label: Nhãn trục x
        y_label: Nhãn trục y
        y_unit: Đơn vị y
        graph_type: Loại đồ thị ("line" hoặc "bar")
    
    Returns:
        HTML với SVG mới và long description đã cập nhật
    """
    result = original_html
    
    # 1. Tạo SVG mới bằng matplotlib (tùy loại đồ thị)
    if graph_type == "bar":
        new_svg = _generate_bar_graph_svg(
            x_values=new_x_values,
            y_values=new_y_values,
            x_label=x_label,
            y_label=y_label,
            y_unit=y_unit,
        )
    elif graph_type == "scatter":
        new_svg = _generate_scatter_plot_svg(
            x_values=new_x_values,
            y_values=new_y_values,
            x_label=x_label,
            y_label=y_label,
            y_unit=y_unit,
        )
    else:  # line graph (default)
        new_svg = _generate_line_graph_svg(
            x_values=new_x_values,
            y_values=new_y_values,
            x_label=x_label,
            y_label=y_label,
            y_unit=y_unit,
        )
    
    # 2. Tạo aria-label mới cho accessibility
    new_aria_label = _generate_aria_label(
        x_values=new_x_values,
        y_values=new_y_values,
        x_label=x_label,
        y_label=y_label,
        y_unit=y_unit,
        graph_type=graph_type,
    )
    
    # 3. Thêm aria-label vào SVG mới
    # Tìm <svg và thêm role="img" aria-label="..."
    new_svg = re.sub(
        r'<svg\b',
        f'<svg role="img" aria-label="{new_aria_label}"',
        new_svg,
        count=1
    )
    
    # 4. Thay thế SVG cũ bằng SVG mới
    # Pattern: <svg ...>...</svg>
    result = re.sub(
        r'<svg\b[^>]*>.*?</svg>',
        new_svg,
        result,
        count=1,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 5. Cập nhật long description trong <div class="sr-only">
    long_desc_pattern = r'(<div[^>]*class="sr-only"[^>]*>)(.*?)(</div>)'
    result = re.sub(
        long_desc_pattern,
        lambda m: m.group(1) + new_long_description + m.group(3),
        result,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    return result


def _verify_graph_correct_answer(
    question_text: str,
    choices: List[str],
    x_values: List[Union[int, float]],
    y_values: List[float],
    llm_answer_letter: str,
) -> str:
    """
    Verify and calculate the correct answer based on the question and graph data.
    
    Args:
        question_text: The question text
        choices: List of 4 choices (A, B, C, D)
        x_values: X-axis values from the graph
        y_values: Y-axis values from the graph
        llm_answer_letter: The answer letter suggested by LLM
    
    Returns:
        The correct answer letter (A, B, C, or D)
    """
    # Parse question to understand what it's asking
    q_lower = question_text.lower()
    
    # Determine what we're looking for
    looking_for_min = any(keyword in q_lower for keyword in ["smallest", "lowest", "minimum", "least"])
    looking_for_max = any(keyword in q_lower for keyword in ["largest", "highest", "maximum", "greatest", "most"])
    
    if not looking_for_min and not looking_for_max:
        # Can't determine - trust LLM
        return llm_answer_letter
    
    # Find the correct x_value
    if looking_for_min:
        min_idx = y_values.index(min(y_values))
        correct_x = x_values[min_idx]
    else:  # looking_for_max
        max_idx = y_values.index(max(y_values))
        correct_x = x_values[max_idx]
    
    # Map to choice - choices should contain the x_value
    # Extract numeric values from choices
    correct_letter = llm_answer_letter  # default
    
    for i, choice in enumerate(choices):
        choice_str = str(choice).strip()
        # Try to extract year/number from choice
        import re
        numbers = re.findall(r'\d+', choice_str)
        if numbers:
            choice_value = int(numbers[0]) if numbers[0].isdigit() else float(numbers[0])
            if choice_value == correct_x or abs(choice_value - correct_x) < 0.01:
                correct_letter = ["A", "B", "C", "D"][i]
                break
    
    return correct_letter


def _update_explanation_for_corrected_answer(
    original_explanation: str,
    old_letter: str,
    new_letter: str,
    choices: List[str],
) -> str:
    """
    Update explanation when the correct answer letter changes.
    Replace references to the old letter with the new letter.
    
    Args:
        original_explanation: The explanation from LLM
        old_letter: The old (incorrect) answer letter
        new_letter: The new (correct) answer letter
        choices: List of 4 choices
    
    Returns:
        Updated explanation
    """
    explanation = original_explanation
    
    # Replace "Choice X is correct" -> "Choice Y is correct"
    explanation = re.sub(
        rf'\bChoice {old_letter} is correct\b',
        f'Choice {new_letter} is correct',
        explanation,
        flags=re.IGNORECASE
    )
    
    # Replace "Choice X is the best answer" -> "Choice Y is the best answer"
    explanation = re.sub(
        rf'\bChoice {old_letter} is the best answer\b',
        f'Choice {new_letter} is the best answer',
        explanation,
        flags=re.IGNORECASE
    )
    
    # Replace other letters as incorrect
    for letter in ["A", "B", "C", "D"]:
        if letter == new_letter:
            continue
        explanation = re.sub(
            rf'\bChoice {letter} is correct\b',
            f'Choice {letter} is incorrect',
            explanation,
            flags=re.IGNORECASE
        )
    
    return explanation


def _build_prompt_graph_multiple_choice(
    question_text_no_svg: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
    original_paragraph: Optional[str] = None,
) -> str:
    """Prompt cho câu hỏi multiple-choice có đồ thị: KHÔNG truyền SVG, chỉ truyền text + GraphSpec."""
    choices_text = "\n".join(
        f"Choice {letter}: {c}" for letter, c in zip(["A", "B", "C", "D"], original_choices)
    )
    
    # Extract long_description_html for the prompt
    long_desc_html = graph_spec.get("long_description_html", "")
    
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)
    
    # Handle paragraph
    paragraph_section = ""
    if original_paragraph:
        paragraph_section = f"""

Sample PARAGRAPH (context before the question):
---
{original_paragraph}
---

IMPORTANT: You MUST also generate new_paragraph with updated numbers that match your new question data.
"""
    
    return f"""You are an SAT question writer. This is a MULTIPLE-CHOICE question with a GRAPH/CHART.

Task: Generate new numerical values for the graph and update all related text accordingly.

IMPORTANT:
- The question contains a graph (SVG and long description will be handled separately by code).
- You must generate NEW x_values and y_values for the graph.
- Update the question text, explanation, choices to match the new graph data.
- Keep the same structure and wording, only change numbers.

CRITICAL - CORRECT ANSWER CALCULATION (READ CAREFULLY):
- DO NOT COPY the sample's correct_answer_letter ({correct_letter}). The sample letter is {correct_letter}, but your answer WILL BE DIFFERENT if the new data changes which choice is correct.
- YOU MUST follow these steps IN ORDER:
  1. Generate new_x_values and new_y_values
  2. Read the question carefully to understand what it's asking (e.g., "which year has the smallest value", "which period has the greatest increase", etc.)
  3. Using your NEW data (new_x_values and new_y_values), calculate which answer is correct
  4. Set correct_answer_letter to the letter (A, B, C, or D) that corresponds to the correct answer based on your NEW data
- EXAMPLE: If the question asks "In which year was the percentage the smallest?" and your new_y_values = [15, 8, 12, 10] with new_x_values = [2020, 2021, 2022, 2023], the correct answer is 2021 (smallest value is 8). If 2021 is choice B, then correct_answer_letter = "B", even if the sample was "{correct_letter}".

- The 4 choices should be the same type as the original (e.g., if original choices are years, new choices should also be years from new_x_values).
- DO NOT include the long description (<ul><li>...) in question_text. It will be added separately to the figure block.
- CRITICAL: The new_long_description MUST use the EXACT same HTML structure as the original (with <ul>, <li>, <br> tags). Only change the numbers.

Original GraphSpec:
{graph_spec_json}

Original Long Description HTML Structure (YOU MUST PRESERVE THIS EXACT HTML FORMAT for new_long_description):
---
{long_desc_html}
---

Sample question TEXT (without SVG and without long description):
---
{question_text_no_svg}
---

Sample explanation:
---
{original_explanation}
---

Sample 4 choices (correct answer in the sample is {correct_letter}, but you may need to change it based on new data):
---
{choices_text}
---

Category: {category}. Section: {section}. Difficulty: {difficulty}.{paragraph_section}

Return a JSON object with:
- question_text: new question text (without SVG, without long description, only numbers changed in the intro and question sentences)
- explanation: new explanation (numbers changed to match new graph and correct answer)
- choices: list of 4 strings (A, B, C, D order, numbers changed). If the question asks about years/labels, choices should be 4 different x_values from new_x_values.
- correct_answer_letter: The letter (A, B, C, or D) of the correct answer BASED ON YOUR NEW DATA. DO NOT just copy "{correct_letter}" from the sample. Calculate which choice is actually correct using your new_x_values and new_y_values, then return that letter.
- new_x_values: list of new x-axis values (e.g., [2015, 2016, 2017, ...] or [1.0, 2.0, 3.0, ...])
- new_y_values: list of new y-axis values (e.g., [10.0, 15.0, 8.0, ...])
- new_long_description: new graph description in HTML format, MUST use the same <ul><li>...</li></ul> structure as the original, only changing the numbers
- new_paragraph: {"If sample has paragraph, generate new paragraph with numbers matching the new question. Same HTML format, only numbers changed." if original_paragraph else "null (no paragraph in sample)"}
"""


def _build_prompt_graph_free_response(
    question_text_no_svg: str,
    original_explanation: str,
    original_correct_answer: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
    original_paragraph: Optional[str] = None,
) -> str:
    """Prompt cho câu hỏi tự luận có đồ thị: KHÔNG truyền SVG, chỉ truyền text + GraphSpec."""
    
    # Extract long_description_html for the prompt
    long_desc_html = graph_spec.get("long_description_html", "")
    graph_type = graph_spec.get("graph_type", "unknown")
    
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)
    
    # Handle paragraph
    paragraph_section = ""
    if original_paragraph:
        paragraph_section = f"""

Sample PARAGRAPH (context before the question):
---
{original_paragraph}
---

IMPORTANT: You MUST also generate new_paragraph with updated numbers that match your new question data.
"""
    
    return f"""You are an SAT question writer. This is a FREE-RESPONSE question with a GRAPH/CHART.

Task: Generate new numerical values for the graph and update all related text accordingly.

IMPORTANT:
- The question contains a graph (SVG and long description will be handled separately by code).
- You must generate NEW x_values and y_values for the graph.
- Update the question text, explanation, and correct answer to match the new graph data.
- Keep the same structure and wording, only change numbers.
- CRITICAL: You MUST calculate the correct answer based on the NEW data.
- The correct answer should match the format of the sample (e.g., if it's a number, provide a number; if it's HTML/MathML, provide HTML/MathML).
- DO NOT include the long description (<ul><li>...) in question_text. It will be added separately to the figure block.
- CRITICAL: The new_long_description MUST use the EXACT same HTML structure as the original (with <ul>, <li>, <br> tags). Only change the numbers.

Original GraphSpec:
{graph_spec_json}

Original Long Description HTML Structure (YOU MUST PRESERVE THIS EXACT HTML FORMAT for new_long_description):
---
{long_desc_html}
---

Sample question TEXT (without SVG and without long description):
---
{question_text_no_svg}
---

Sample explanation:
---
{original_explanation}
---

Sample correct answer:
---
{original_correct_answer}
---

Category: {category}. Section: {section}. Difficulty: {difficulty}.

Return a JSON object with:
- question_text: new question text (without SVG, without long description, only numbers changed in the intro and question sentences)
- explanation: new explanation (numbers changed to match new graph and correct answer)
- correct_answer: the correct answer for the new question, in the same format as the sample
- new_x_values: list of new x-axis values (e.g., [2015, 2016, 2017, ...])
- new_y_values: list of new y-axis values (e.g., [10.0, 15.0, 8.0, ...])
- new_long_description: new graph description in HTML format, MUST use the same <ul><li>...</li></ul> structure as the original, only changing the numbers
"""


def _get_question_html(sample: Dict[str, Any]) -> str:
    """Lấy nội dung câu hỏi mẫu (HTML + MathML) nguyên bản."""
    q = sample.get("question") or {}
    return (q.get("question") or "").strip()


def _get_explanation(sample: Dict[str, Any]) -> str:
    """Lấy explanation mẫu (HTML + MathML)."""
    q = sample.get("question") or {}
    return (q.get("explanation") or "").strip()


def _build_prompt_paragraph_graph_multiple_choice(
    original_question: str,
    paragraph_text_no_svg: str,
    original_explanation: str,
    original_choices: List[str],
    correct_letter: str,
    graph_spec: Dict[str, Any],
    category: str,
    section: str,
    difficulty: str,
) -> str:
    """Prompt cho câu hỏi multiple-choice có đồ thị trong paragraph."""
    choices_text = "\n".join(
        f"Choice {letter}: {c}" for letter, c in zip(["A", "B", "C", "D"], original_choices)
    )
    
    long_desc_html = graph_spec.get("long_description_html", "")
    graph_spec_json = json.dumps(graph_spec, default=str, ensure_ascii=False, indent=2)
    
    return f"""You are an SAT question writer. This is a MULTIPLE-CHOICE question with a GRAPH in the PARAGRAPH section.

Task: Generate new numerical values for the paragraph graph and update all related text accordingly.

IMPORTANT:
- The PARAGRAPH contains a graph (SVG and long description will be handled separately by code).
- You must generate NEW x_values and y_values for the paragraph graph.
- Update the paragraph text, question, explanation, and choices to match the new graph data.
- Keep the same structure and wording, only change numbers.

CRITICAL - CORRECT ANSWER CALCULATION (READ CAREFULLY):
- DO NOT COPY the sample's correct_answer_letter ({correct_letter}). Calculate the correct answer based on your NEW data.
- Follow these steps:
  1. Generate new_x_values and new_y_values for the paragraph graph
  2. Read the question carefully to understand what it asks
  3. Using your NEW graph data, calculate which answer is correct
  4. Set correct_answer_letter to the correct letter (A, B, C, or D)

Original Paragraph GraphSpec:
{graph_spec_json}

Original Long Description HTML Structure (YOU MUST PRESERVE THIS EXACT HTML FORMAT):
---
{long_desc_html}
---

Sample PARAGRAPH text (without SVG and long description):
---
{paragraph_text_no_svg}
---

Sample QUESTION:
---
{original_question}
---

Sample explanation:
---
{original_explanation}
---

Sample 4 choices (correct answer in sample is {correct_letter}, but you may need to change it):
---
{choices_text}
---

Category: {category}. Section: {section}. Difficulty: {difficulty}.

Return a JSON object with:
- question: new question text (just the question, not the paragraph)
- explanation: new explanation matching new data
- choices: list of 4 strings (A, B, C, D order)
- correct_answer_letter: The letter of the correct answer BASED ON YOUR NEW DATA
- paragraph_text: new paragraph text without SVG and long description (only the context text)
- new_x_values: list of new x-axis values for the paragraph graph
- new_y_values: list of new y-axis values for the paragraph graph
- new_long_description: new graph description in HTML format, preserving the exact <ul><li> structure
"""


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
    original_paragraph: Optional[str] = None,
) -> str:
    # Handle paragraph
    paragraph_section = ""
    if original_paragraph:
        paragraph_section = f"""

        Sample PARAGRAPH (context before the question):
        ---
        {original_paragraph}
        ---

        IMPORTANT: You MUST also generate new_paragraph with updated numbers that match your new question data.
        """
    
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
---{paragraph_section}

Return a JSON object with keys: question, explanation, correct_answer, new_paragraph. Each value: same string as sample with only numbers substituted; numbers must be consistent across all three.
- new_paragraph: {"Generate new paragraph with numbers matching the new question. Same HTML format, only numbers changed." if original_paragraph else "null (no paragraph in sample)"}"""


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
    original_paragraph: Optional[str] = None,
) -> str:
    """Prompt cho multiple-choice: sinh question, explanation, 4 choices, và correct_answer_letter."""
    choices_text = "\n".join(
        f"Choice {letter}: {c}" for letter, c in zip(["A", "B", "C", "D"], original_choices)
    )
    
    # Handle paragraph
    paragraph_section = ""
    if original_paragraph:
        paragraph_section = f"""

        Sample PARAGRAPH (context before the question):
        ---
        {original_paragraph}
        ---

        IMPORTANT: You MUST also generate new_paragraph with updated numbers that match your new question data.
        """
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
---{paragraph_section}

Return a JSON object with keys: question, explanation, choices, correct_answer_letter, new_paragraph.
- question: new question string (only numbers changed).
- explanation: new explanation string (only numbers changed, consistent with new question).
- choices: list of exactly 4 strings, in order A, B, C, D (only numbers changed in each).
- correct_answer_letter: one of "A", "B", "C", "D" (the correct choice for the new question; typically the same as the sample, {correct_letter}).
- new_paragraph: {"Generate new paragraph with numbers matching the new question. Same HTML format, only numbers changed." if original_paragraph else "null (no paragraph in sample)"}"""



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
    
    parser = MathMLParser()
    parsed = parser.parse(original_html)
    graph_spec = parsed.get("graph")

    if not original_html:
        raise ValueError("Câu mẫu không có nội dung question (HTML).")
    original_explanation = _get_explanation(sample)
    original_choices = _get_choices(sample)
    correct_letter = _get_correct_answer_letter(sample)
    original_correct_answer = _get_correct_answer_content(sample)
    original_paragraph = sample.get("question", {}).get("paragraph")
    
    # Parse paragraph to check for graph
    paragraph_graph_spec = None
    paragraph_text_no_svg = None
    if original_paragraph:
        paragraph_parsed = parser.parse_paragraph(original_paragraph)
        if paragraph_parsed.get("has_graph"):
            paragraph_graph_spec = paragraph_parsed.get("graph")
            paragraph_text_no_svg = paragraph_parsed.get("text")
    
    is_multiple_choice = (q_type == "multiple-choice") and len(original_choices) == 4 and correct_letter and original_explanation
    generate_full = bool(original_explanation and original_correct_answer)

    if is_multiple_choice:
        # Kiểm tra nếu PARAGRAPH có đồ thị → dùng luồng xử lý riêng
        if paragraph_graph_spec is not None and hasattr(paragraph_graph_spec, 'x_values') and paragraph_graph_spec.x_values:
            # ========== LUỒNG XỬ LÝ CÂU HỎI CÓ ĐỒ THỊ TRONG PARAGRAPH ==========
            print("Detected graph in paragraph. Using paragraph graph flow.")
            
            # Validate required data
            if not paragraph_text_no_svg:
                raise ValueError("Paragraph graph detected but paragraph text could not be extracted.")
            if not correct_letter:
                raise ValueError("Paragraph graph detected but correct_letter is missing.")
            
            # Convert GraphSpec to dict
            graph_spec_dict = {
                "graph_type": paragraph_graph_spec.graph_type,
                "x_label": paragraph_graph_spec.x_label,
                "y_label": paragraph_graph_spec.y_label,
                "x_values": paragraph_graph_spec.x_values,
                "y_values": paragraph_graph_spec.y_values,
                "y_unit": paragraph_graph_spec.y_unit,
                "raw_long_description": paragraph_graph_spec.raw_long_description,
                "long_description_html": paragraph_graph_spec.long_description_html,
            }
            
            # Build prompt for paragraph graph
            prompt_text = _build_prompt_paragraph_graph_multiple_choice(
                original_html,
                paragraph_text_no_svg,
                original_explanation,
                original_choices,
                correct_letter,
                graph_spec_dict,
                category,
                section,
                difficulty,
            )
            
            structured_llm = llm.with_structured_output(GeneratedParagraphGraphMultipleChoiceContent)
            result_pg: GeneratedParagraphGraphMultipleChoiceContent = structured_llm.invoke(
                [HumanMessage(content=prompt_text)]
            )
            
            # Validate results
            new_choices = result_pg.choices or []
            if len(new_choices) != 4:
                raise ValueError(f"LLM phải trả về đúng 4 choices, nhận được {len(new_choices)}.")
            new_choices = [str(c).strip() for c in new_choices[:4]]
            new_letter = (result_pg.correct_answer_letter or "").strip().upper()
            if new_letter not in ("A", "B", "C", "D"):
                raise ValueError(f"correct_answer_letter phải là A, B, C hoặc D, nhận được: {result_pg.correct_answer_letter!r}")
            
            new_question_text = (result_pg.question or "").strip()
            new_explanation = (result_pg.explanation or "").strip()
            new_paragraph_text = (result_pg.paragraph_text or "").strip()
            
            if not new_question_text:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            if not new_paragraph_text:
                raise ValueError("LLM không trả về paragraph text.")
            
            # VERIFY CORRECT ANSWER based on paragraph graph data
            verified_letter = _verify_graph_correct_answer(
                new_question_text,
                new_choices,
                result_pg.new_x_values,
                result_pg.new_y_values,
                new_letter,
            )
            
            if verified_letter != new_letter:
                print(f"⚠️  WARNING (Paragraph Graph): LLM returned correct_answer_letter={new_letter}, but based on graph data, the correct answer should be {verified_letter}")
                print(f"   Question asks for: {new_question_text[:100]}...")
                print(f"   Auto-correcting to: {verified_letter}")
                
                # Update explanation to match corrected answer
                new_explanation = _update_explanation_for_corrected_answer(
                    new_explanation,
                    new_letter,
                    verified_letter,
                    new_choices,
                )
                
                new_letter = verified_letter
            
            # Regenerate paragraph with new graph
            updated_paragraph_html = _update_graph_in_html(
                original_paragraph,
                old_x_values=paragraph_graph_spec.x_values,
                old_y_values=paragraph_graph_spec.y_values,
                new_x_values=result_pg.new_x_values,
                new_y_values=result_pg.new_y_values,
                new_long_description=result_pg.new_long_description,
                x_label=paragraph_graph_spec.x_label or "Year",
                y_label=paragraph_graph_spec.y_label or "Percent",
                y_unit=paragraph_graph_spec.y_unit or "%",
                graph_type=paragraph_graph_spec.graph_type or "line",
            )
            
            # Extract figure block from updated paragraph
            figure_match = re.search(
                r"<figure[^>]*>.*?</figure>",
                updated_paragraph_html,
                flags=re.DOTALL | re.IGNORECASE
            )
            
            # Reconstruct paragraph: figure + long description + text
            if figure_match:
                figure_block = figure_match.group(0)
                # Add long description div after figure (if not already included)
                long_desc_match = re.search(
                    r'<div[^>]*class="sr-only"[^>]*>.*?</div>',
                    updated_paragraph_html,
                    flags=re.DOTALL | re.IGNORECASE
                )
                long_desc_div = long_desc_match.group(0) if long_desc_match else ""
                
                # Combine: figure + long_desc + paragraph_text
                new_paragraph = f"{figure_block}{long_desc_div}\n<p>{new_paragraph_text}</p>"
            else:
                # Fallback: just use text
                new_paragraph = f"<p>{new_paragraph_text}</p>"
            
            new_question_content = {
                "paragraph": new_paragraph,
                "question": new_question_text,
                "choices": new_choices,
                "correct_answer": [new_letter],
                "explanation": new_explanation,
            }
        # Kiểm tra nếu câu hỏi có đồ thị → dùng luồng xử lý riêng (không truyền SVG vào prompt)
        elif graph_spec is not None and hasattr(graph_spec, 'x_values') and graph_spec.x_values:
            # ========== LUỒNG XỬ LÝ CÂU HỎI CÓ ĐỒ THỊ ==========
            # Loại bỏ SVG và long description khỏi HTML để giảm token
            # Long description sẽ được xử lý riêng và chèn vào figure block
            question_text_no_svg = _remove_svg_and_long_desc_from_html(original_html)

            print("Question text without SVG:", question_text_no_svg)
            
            # Convert GraphSpec to dict for JSON serialization
            graph_spec_dict = {
                "graph_type": graph_spec.graph_type,
                "x_label": graph_spec.x_label,
                "y_label": graph_spec.y_label,
                "x_values": graph_spec.x_values,
                "y_values": graph_spec.y_values,
                "y_unit": graph_spec.y_unit,
                "raw_long_description": graph_spec.raw_long_description,
                "long_description_html": graph_spec.long_description_html,
            }
            
            # Dùng prompt riêng cho câu hỏi có đồ thị
            prompt_text = _build_prompt_graph_multiple_choice(
                question_text_no_svg,
                original_explanation,
                original_choices,
                correct_letter,
                graph_spec_dict,
                category,
                section,
                difficulty,
                original_paragraph,
            )
            
            structured_llm = llm.with_structured_output(GeneratedGraphQuestionContent)
            result_graph: GeneratedGraphQuestionContent = structured_llm.invoke(
                [HumanMessage(content=prompt_text)]
            )
            
            # Validate kết quả
            new_choices = result_graph.choices or []
            if len(new_choices) != 4:
                raise ValueError(f"LLM phải trả về đúng 4 choices, nhận được {len(new_choices)}.")
            new_choices = [str(c).strip() for c in new_choices[:4]]
            new_letter = (result_graph.correct_answer_letter or "").strip().upper()
            if new_letter not in ("A", "B", "C", "D"):
                raise ValueError(f"correct_answer_letter phải là A, B, C hoặc D, nhận được: {result_graph.correct_answer_letter!r}")
            
            new_question_text_no_svg = (result_graph.question_text or "").strip()
            new_explanation = (result_graph.explanation or "").strip()
            
            if not new_question_text_no_svg:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            
            # VERIFY CORRECT ANSWER based on graph data
            verified_letter = _verify_graph_correct_answer(
                new_question_text_no_svg,
                new_choices,
                result_graph.new_x_values,
                result_graph.new_y_values,
                new_letter,
            )
            
            if verified_letter != new_letter:
                print(f"⚠️  WARNING: LLM returned correct_answer_letter={new_letter}, but based on graph data, the correct answer should be {verified_letter}")
                print(f"   Question asks for: {new_question_text_no_svg[:100]}...")
                print(f"   Auto-correcting to: {verified_letter}")
                
                # Update explanation to match corrected answer
                new_explanation = _update_explanation_for_corrected_answer(
                    new_explanation,
                    new_letter,
                    verified_letter,
                    new_choices,
                )
                
                new_letter = verified_letter
            
            # Tạo SVG mới bằng matplotlib và cập nhật long description
            updated_html_with_svg = _update_graph_in_html(
                original_html,
                old_x_values=graph_spec.x_values,
                old_y_values=graph_spec.y_values,
                new_x_values=result_graph.new_x_values,
                new_y_values=result_graph.new_y_values,
                new_long_description=result_graph.new_long_description,
                x_label=graph_spec.x_label or "Model year",
                y_label=graph_spec.y_label or "Percent",
                y_unit=graph_spec.y_unit or "%",
                graph_type=graph_spec.graph_type or "line",
            )
            
            # Trích xuất figure block (SVG + long_description) từ HTML đã cập nhật
            # Pattern: <figure...>...<svg>...</svg>...<div class="sr-only">...</div>...</figure>
            figure_match = re.search(
                r"<figure[^>]*>.*?</figure>",
                updated_html_with_svg,
                flags=re.DOTALL | re.IGNORECASE
            )
            
            if figure_match:
                figure_block = figure_match.group(0)
                # Ghép: text_intro (trước câu hỏi chính) + figure + text_question (câu hỏi chính)
                # Giả sử new_question_text_no_svg có format: "<p>intro...</p>\n<p>question...</p>"
                # Chèn figure vào giữa
                
                # Loại bỏ long description từ LLM output (nếu có) để tránh trùng lặp
                # vì figure_block đã chứa long description
                clean_question_text = _remove_svg_and_long_desc_from_html(new_question_text_no_svg)
                
                # Tách text thành 2 phần: intro (mô tả đồ thị) và question (câu hỏi)
                # Tìm câu hỏi cuối cùng (thường bắt đầu bằng "For what..." hoặc kết thúc bằng "?")
                parts = re.split(r'(<p[^>]*>.*?</p>)', clean_question_text, flags=re.DOTALL)
                parts = [p for p in parts if p.strip()]  # Loại bỏ empty strings
                
                if len(parts) >= 2:
                    # Giả sử phần cuối là câu hỏi
                    intro_parts = parts[:-1]
                    question_part = parts[-1]
                    intro_text = ''.join(intro_parts)
                    new_question_text = f'{intro_text}\n<p style="text-align: center;">{figure_block}</p>\n{question_part}'
                else:
                    # Nếu chỉ có 1 phần, đặt figure ở đầu
                    new_question_text = f'<p style="text-align: center;">{figure_block}</p>\n{clean_question_text}'
            else:
                # Fallback: chỉ dùng text không có SVG
                new_question_text = new_question_text_no_svg
            
            new_question_content = {
                "paragraph": result_graph.new_paragraph,
                "question": new_question_text,
                "choices": new_choices,
                "correct_answer": [new_letter],
                "explanation": new_explanation,
            }
        else:
            prompt_text = _build_prompt_multiple_choice(
                original_html,
                original_explanation,
                original_choices,
                correct_letter,
                category,
                section,
                difficulty,
                original_paragraph,
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
                "paragraph": result_mc.new_paragraph,
                "question": new_question_text,
                "choices": new_choices,
                "correct_answer": [new_letter],
                "explanation": new_explanation,
            }

    elif generate_full:
        # Không phải multiple-choice hoặc thiếu 4 choices: sinh question, explanation, correct_answer (nội dung)
        # Kiểm tra nếu câu hỏi có đồ thị → dùng luồng xử lý riêng (không truyền SVG vào prompt)
        if graph_spec is not None and hasattr(graph_spec, 'x_values') and graph_spec.x_values:
            # ========== LUỒNG XỬ LÝ CÂU HỎI TỰ LUẬN CÓ ĐỒ THỊ ==========
            # Loại bỏ SVG và long description khỏi HTML để giảm token
            question_text_no_svg = _remove_svg_and_long_desc_from_html(original_html)

            print("Free-response question text without SVG:", question_text_no_svg)
            
            # Convert GraphSpec to dict for JSON serialization
            graph_spec_dict = {
                "graph_type": graph_spec.graph_type,
                "x_label": graph_spec.x_label,
                "y_label": graph_spec.y_label,
                "x_values": graph_spec.x_values,
                "y_values": graph_spec.y_values,
                "y_unit": graph_spec.y_unit,
                "raw_long_description": graph_spec.raw_long_description,
                "long_description_html": graph_spec.long_description_html,
            }
            
            # Dùng prompt riêng cho câu hỏi tự luận có đồ thị
            prompt_text = _build_prompt_graph_free_response(
                question_text_no_svg,
                original_explanation,
                original_correct_answer,
                graph_spec_dict,
                category,
                section,
                difficulty,
                original_paragraph,
            )
            
            structured_llm = llm.with_structured_output(GeneratedGraphFreeResponseContent)
            result_free_response: GeneratedGraphFreeResponseContent = structured_llm.invoke(
                [HumanMessage(content=prompt_text)]
            )
            
            # Validate kết quả
            new_question_text_no_svg = (result_free_response.question_text or "").strip()
            new_explanation = (result_free_response.explanation or "").strip()
            new_correct_answer = (result_free_response.correct_answer or "").strip()
            
            if not new_question_text_no_svg:
                raise ValueError("LLM không trả về nội dung câu hỏi.")
            if not new_explanation:
                raise ValueError("LLM không trả về explanation.")
            if not new_correct_answer:
                raise ValueError("LLM không trả về correct_answer.")
            
            # Tạo SVG mới bằng matplotlib và cập nhật long description
            updated_html_with_svg = _update_graph_in_html(
                original_html,
                old_x_values=graph_spec.x_values,
                old_y_values=graph_spec.y_values,
                new_x_values=result_free_response.new_x_values,
                new_y_values=result_free_response.new_y_values,
                new_long_description=result_free_response.new_long_description,
                x_label=graph_spec.x_label or "Model year",
                y_label=graph_spec.y_label or "Percent",
                y_unit=graph_spec.y_unit or "%",
                graph_type=graph_spec.graph_type or "line",
            )
            
            # Trích xuất figure block (SVG + long_description) từ HTML đã cập nhật
            figure_match = re.search(
                r"<figure[^>]*>.*?</figure>",
                updated_html_with_svg,
                flags=re.DOTALL | re.IGNORECASE
            )
            
            if figure_match:
                figure_block = figure_match.group(0)
                
                # Loại bỏ long description từ LLM output (nếu có) để tránh trùng lặp
                clean_question_text = _remove_svg_and_long_desc_from_html(new_question_text_no_svg)
                
                # Tách text thành 2 phần: intro (mô tả đồ thị) và question (câu hỏi)
                parts = re.split(r'(<p[^>]*>.*?</p>)', clean_question_text, flags=re.DOTALL)
                parts = [p for p in parts if p.strip()]  # Loại bỏ empty strings
                
                if len(parts) >= 2:
                    # Giả sử phần cuối là câu hỏi
                    intro_parts = parts[:-1]
                    question_part = parts[-1]
                    intro_text = ''.join(intro_parts)
                    new_question_text = f'{intro_text}\n<p style="text-align: center;">{figure_block}</p>\n{question_part}'
                else:
                    # Nếu chỉ có 1 phần, đặt figure ở đầu
                    new_question_text = f'<p style="text-align: center;">{figure_block}</p>\n{clean_question_text}'
            else:
                # Fallback: chỉ dùng text không có SVG
                new_question_text = new_question_text_no_svg
            
            new_question_content = {
                "paragraph": result_free_response.new_paragraph,
                "question": new_question_text,
                "choices": None,
                "correct_answer": new_correct_answer,
                "explanation": new_explanation,
            }
        else:
            # ========== LUỒNG XỬ LÝ CÂU HỎI TỰ LUẬN KHÔNG CÓ ĐỒ THỊ ==========
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
                "paragraph": result.new_paragraph,
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
            "paragraph": None,  # Question-only mode doesn't regenerate paragraph
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
    with open("graph.json", "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
    new_q = generate_new_question(graph_data, llm=llm)  # Truyền full object
    
    with open("new_question.json", "w", encoding="utf-8") as f:
        json.dump(new_q, f, ensure_ascii=False, indent=2)
