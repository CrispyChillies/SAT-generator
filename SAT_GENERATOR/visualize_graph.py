#!/usr/bin/env python3
"""
Visualize the LangGraph agent dependencies/flow.

Usage:
    python visualize_graph.py [--output-png agent_graph.png]
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

from agent import LangGraphMathAgent


def visualize_graph(output_file: str = "agent_graph.png", ascii_only: bool = False):
    """
    Visualize the LangGraph agent structure.
    
    Args:
        output_file: Path to save the PNG image
        ascii_only: If True, only print ASCII representation (no image file)
    """
    # Create agent instance
    print("Creating agent...")
    api_key = os.getenv("OPENAI_API_KEY")
    agent = LangGraphMathAgent(api_key=api_key, verbose=False)
    
    # Get the compiled graph
    graph = agent.graph.get_graph()
    
    # Print ASCII representation
    print("\n" + "="*70)
    print("AGENT GRAPH STRUCTURE (ASCII)")
    print("="*70)
    print(graph.draw_ascii())
    print("="*70 + "\n")
    
    if not ascii_only:
        try:
            # Generate PNG image using Mermaid
            print(f"Generating graph image...")
            mermaid_png = graph.draw_mermaid_png()
            
            # Save to file
            output_path = Path(output_file)
            with open(output_path, "wb") as f:
                f.write(mermaid_png)
            
            print(f"✅ Graph image saved to: {output_path.absolute()}")
            
        except Exception as e:
            print(f"⚠️  Could not generate PNG image: {e}")
            print("ASCII representation is shown above.")
            print("\nTo enable PNG generation, install: pip install pygraphviz")
    
    # Print node information
    print("\nNODE DESCRIPTIONS:")
    print("-" * 70)
    print("• llm           - LLM thinks and decides which tools to use")
    print("• tools         - Execute the chosen mathematical tools")
    print("• explain_params - LLM explains the meaning of tool parameters")
    print("• final         - Final processing and result validation")
    print("-" * 70)
    
    # Print edge information  
    print("\nFLOW:")
    print("-" * 70)
    print("1. START → llm")
    print("2. llm → tools (if tool needed) OR llm → final (if done)")
    print("3. tools → explain_params")
    print("4. explain_params → llm (loop continues)")
    print("5. final → END")
    print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LangGraph agent structure")
    parser.add_argument(
        "--output-png",
        "-o",
        default="agent_graph.png",
        help="Output PNG file path (default: agent_graph.png)"
    )
    parser.add_argument(
        "--ascii-only",
        "-a",
        action="store_true",
        help="Only show ASCII diagram, don't generate PNG"
    )
    
    args = parser.parse_args()
    
    visualize_graph(output_file=args.output_png, ascii_only=args.ascii_only)
