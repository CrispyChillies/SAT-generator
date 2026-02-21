#!/usr/bin/env python3
import json
import os

def generate_html():
    # Read the questions from question.json
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_file = os.path.join(script_dir, '/home/aaronpham5504/Coding/SAT-generator/SAT_GENERATOR/output/rw_test/new_rw_question.json')
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    # Start building HTML content
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAT Math Questions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .question-container {
            background-color: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question-header {
            background-color: #4a90e2;
            color: white;
            padding: 15px;
            margin: -25px -25px 20px -25px;
            border-radius: 8px 8px 0 0;
        }
        .question-id {
            font-weight: bold;
            font-size: 14px;
        }
        .question-meta {
            font-size: 12px;
            margin-top: 5px;
            opacity: 0.9;
        }
        .question-content {
            margin: 20px 0;
            line-height: 1.6;
        }
        .paragraph {
            background-color: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #6c757d;
            border-radius: 4px;
            font-style: italic;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .question-image {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .choices {
            margin: 20px 0;
        }
        .choice {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #4a90e2;
        }
        .correct-answer {
            background-color: #d4edda;
            border-left-color: #28a745;
            font-weight: bold;
        }
        .explanation {
            margin-top: 20px;
            padding: 15px;
            background-color: #e7f3ff;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .explanation h4 {
            margin-top: 0;
            color: #007bff;
        }
        math {
            font-size: 1.1em;
        }
        .stats {
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #ffc107;
        }
    </style>
</head>
<body>
    <h1>SAT Math Questions</h1>
"""
    
    # Add statistics
    total_questions = len(questions)
    difficulty_counts = {}
    category_counts = {}
    
    for q in questions:
        diff = q.get('difficulty', 'Unknown')
        cat = q.get('category', 'Unknown')
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    html_content += f"""
    <div class="stats">
        <h3>Question Statistics</h3>
        <p><strong>Total Questions:</strong> {total_questions}</p>
        <p><strong>Difficulty Distribution:</strong> {', '.join([f'{k}: {v}' for k, v in difficulty_counts.items()])}</p>
        <p><strong>Category Distribution:</strong> {', '.join([f'{k}: {v}' for k, v in category_counts.items()])}</p>
    </div>
"""
    
    # Add each question
    for i, question in enumerate(questions, 1):
        q_data = question.get('question', {})
        choices = q_data.get('choices', []) or []  # Handle None values
        correct_answer = q_data.get('correct_answer', []) or []  # Handle None values
        
        html_content += f"""
    <div class="question-container">
        <div class="question-header">
            <div class="question-id">Question {i}: {question.get('id', 'N/A')}</div>
            <div class="question-meta">
                Subject: {question.get('subject', 'N/A')} | 
                Section: {question.get('section', 'N/A')} | 
                Category: {question.get('category', 'N/A')} | 
                Difficulty: {question.get('difficulty', 'N/A')} | 
                Type: {question.get('type', 'N/A')}
            </div>
        </div>
        
        <div class="question-content">
"""
        
        # Add image if it exists
        image_url = question.get('image_url', '')
        if image_url and image_url != 'null' and image_url is not None:
            html_content += f"""
            <div class="image-container">
                <img src="{image_url}" alt="Question image" class="question-image">
            </div>
"""
        
        # Add paragraph if it exists
        paragraph = q_data.get('paragraph', '')
        if paragraph and paragraph != 'null' and paragraph is not None:
            html_content += f"""
            <div class="paragraph">
                {paragraph}
            </div>
"""
        
        # Add the main question
        html_content += f"""
            {q_data.get('question', 'No question text available')}
        </div>
        
        <div class="choices">
            <h4>Choices:</h4>
"""
        
        # Add choices (only if there are choices)
        if choices:
            choice_labels = ['A', 'B', 'C', 'D']
            for j, choice in enumerate(choices):
                is_correct = choice_labels[j] in correct_answer if correct_answer else False
                css_class = 'correct-answer' if is_correct else ''
                html_content += f"""
            <div class="choice {css_class}">
                <strong>{choice_labels[j]}.</strong> {choice}
            </div>
"""
        else:
            # For grid-in questions or questions without choices
            html_content += """
            <div class="choice">
                <strong>Grid-in question:</strong> Enter your answer in the grid.
            </div>
"""
        
        html_content += """
        </div>
"""
        
        # Add explanation if available
        explanation = q_data.get('explanation', '')
        if explanation:
            html_content += f"""
        <div class="explanation">
            <h4>Explanation:</h4>
            {explanation}
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # Write HTML file
    html_file = os.path.join(script_dir, 'temp.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated HTML file: {html_file}")
    print(f"Displayed {total_questions} questions")
    print(f"Open {html_file} in your browser to view the questions")

if __name__ == "__main__":
    generate_html()
