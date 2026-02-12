import json
from collections import Counter
from operator import itemgetter

def print_table(headers, rows, title=""):
    """In bảng với độ rộng cột tự động"""
    if not rows:
        return
    
    # Tính độ rộng tối đa cho mỗi cột
    num_cols = len(headers)
    col_widths = [len(str(h)) for h in headers]
    
    for row in rows:
        for i, cell in enumerate(row):
            cell_len = len(str(cell))
            if cell_len > col_widths[i]:
                col_widths[i] = cell_len
    
    # Thêm padding
    col_widths = [w + 2 for w in col_widths]
    
    # Tính tổng độ rộng
    total_width = sum(col_widths) + num_cols - 1
    
    # In tiêu đề
    if title:
        print("\n" + "=" * total_width)
        print(title.center(total_width))
        print("=" * total_width)
    
    # In header
    header_row = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_row)
    print("-" * total_width)
    
    # In các dòng dữ liệu
    for row in rows:
        data_row = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(data_row)
    
    print("=" * total_width)

# Đọc file JSON
with open('questions_practice_test.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

# Thống kê các bộ 3 section-category-skill
section_category_skill_triples = Counter()

for question in questions:
    section = question.get('section', 'N/A')
    category = question.get('category', 'N/A')
    skill = question.get('skill', '') or '(empty)'  # Thay thế chuỗi rỗng bằng '(empty)'
    triple = (section, category, skill)
    section_category_skill_triples[triple] += 1

# Sắp xếp theo số lượng giảm dần
sorted_triples = sorted(section_category_skill_triples.items(), key=itemgetter(1), reverse=True)

# Chuẩn bị dữ liệu cho bảng
print(f"\nTổng số câu hỏi: {len(questions)}")
print(f"Tổng số bộ 3 section-category-skill duy nhất: {len(section_category_skill_triples)}")

table_rows = []
for idx, ((section, category, skill), count) in enumerate(sorted_triples, 1):
    table_rows.append([idx, section, category, skill, count])

print_table(['STT', 'Section', 'Category', 'Skill', 'Số lượng'], table_rows, 
            "THỐNG KÊ CÁC BỘ 3 SECTION-CATEGORY-SKILL")

# Thống kê theo section
section_stats = Counter()
for question in questions:
    section = question.get('section', 'N/A')
    section_stats[section] += 1

sorted_sections = sorted(section_stats.items(), key=itemgetter(1), reverse=True)

table_rows_section = []
for idx, (section, count) in enumerate(sorted_sections, 1):
    table_rows_section.append([idx, section, count])

print_table(['STT', 'Section', 'Số lượng'], table_rows_section,
            "THỐNG KÊ THEO SECTION")

# Thống kê theo category
category_stats = Counter()
for question in questions:
    category = question.get('category', 'N/A')
    category_stats[category] += 1

sorted_categories = sorted(category_stats.items(), key=itemgetter(1), reverse=True)

table_rows_category = []
for idx, (category, count) in enumerate(sorted_categories, 1):
    table_rows_category.append([idx, category, count])

print_table(['STT', 'Category', 'Số lượng'], table_rows_category,
            "THỐNG KÊ THEO CATEGORY")

# Thống kê theo skill
skill_stats = Counter()
for question in questions:
    skill = question.get('skill', '') or '(empty)'
    skill_stats[skill] += 1

sorted_skills = sorted(skill_stats.items(), key=itemgetter(1), reverse=True)

table_rows_skill = []
for idx, (skill, count) in enumerate(sorted_skills, 1):
    table_rows_skill.append([idx, skill, count])

print_table(['STT', 'Skill', 'Số lượng'], table_rows_skill,
            "THỐNG KÊ THEO SKILL")

# Lưu kết quả vào file
output_data = {
    'total_questions': len(questions),
    'total_unique_triples': len(section_category_skill_triples),
    'section_category_skill_triples': [
        {
            'section': section,
            'category': category,
            'skill': skill,
            'count': count
        }
        for (section, category, skill), count in sorted_triples
    ],
    'section_stats': [
        {
            'section': section,
            'count': count
        }
        for section, count in sorted_sections
    ],
    'category_stats': [
        {
            'category': category,
            'count': count
        }
        for category, count in sorted_categories
    ],
    'skill_stats': [
        {
            'skill': skill,
            'count': count
        }
        for skill, count in sorted_skills
    ]
}

with open('thong_ke_category_skill.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 80)
print("Kết quả đã được lưu vào file: thong_ke_category_skill.json")
print("=" * 80)
