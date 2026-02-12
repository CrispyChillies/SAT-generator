import json
from pathlib import Path

OUTPUT_DIR = Path("static_site")
OUTPUT_DIR.mkdir(exist_ok=True)

with open("questions_practice_test.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<link rel="stylesheet" href="styles.css">
</head>
<body>
<div class="container">
{content}
</div>
</body>
</html>
"""

def render_question(q):
    question_html = f"<h2>{q['subject']} – {q['section']}</h2>"
    question_html += f"<p><strong>Question:</strong> {q['question']['question']}</p>"

    # Hiển thị choices nếu có (không phải null)
    if q["question"]["choices"]:
        question_html += "<ul>"
        for i, c in enumerate(q["question"]["choices"]):
            question_html += f"<li>{chr(65+i)}. {c}</li>"
        question_html += "</ul>"

    # Luôn hiển thị explanation và correct_answer
    question_html += "<h3>Explanation</h3>"
    question_html += q["question"]["explanation"]  # ✅ dùng HTML gốc

    # Xử lý correct_answer (có thể là mảng hoặc giá trị đơn)
    correct_answer = q["question"].get("correct_answer")
    if correct_answer:
        if isinstance(correct_answer, list) and len(correct_answer) > 0:
            # Nếu có nhiều đáp án đúng, hiển thị tất cả dưới dạng danh sách
            if len(correct_answer) > 1:
                question_html += "<p><strong>Correct Answer:</strong></p>"
                question_html += "<ul>"
                for ans in correct_answer:
                    question_html += f"<li>{ans}</li>"
                question_html += "</ul>"
            else:
                question_html += (
                    f"<p><strong>Correct Answer:</strong> "
                    f"{correct_answer[0]}</p>"
                )
        else:
            question_html += (
                f"<p><strong>Correct Answer:</strong> "
                f"{correct_answer}</p>"
            )

    return question_html

# generate pages
index_links = []

for i, q in enumerate(questions):
    question_id = q['id']
    print("index: ", i, "id:", question_id)
    content = render_question(q)
    html = BASE_HTML.format(
        title=f"Question {i+1}",
        content=content
    )

    filename = f"{question_id}.html"
    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write(html)

    index_links.append(f"<li><a href='{filename}'>{question_id}</a></li>")

# index page
index_html = BASE_HTML.format(
    title="SAT Practice",
    content="<h1>SAT Math Practice</h1><ul>" + "".join(index_links) + "</ul>"
)

with open(OUTPUT_DIR / "index.html", "w", encoding="utf-8") as f:
    f.write(index_html)

print("✅ Static site generated in static_site/")
