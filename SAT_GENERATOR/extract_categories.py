import json
import re
import csv
from collections import defaultdict

# Read the questions JSON file
with open('questions_updated_1224.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Read the lectures JSON file
try:
    with open('ques-gen/updated/lectures_updated_1224.json', 'r', encoding='utf-8') as f:
        lectures_data = json.load(f)
except FileNotFoundError:
    # Try alternative path
    try:
        with open('lectures_updated_1224.json', 'r', encoding='utf-8') as f:
            lectures_data = json.load(f)
    except FileNotFoundError:
        lectures_data = []
        print("Warning: Could not find lectures file. Lessons table will be empty.")

# Create a nested dictionary structure: subject -> pool -> section -> category -> difficulty -> count
hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))

# Extract all unique combinations and count questions by difficulty
for item in data:
    subject = item.get('subject', '')
    pool = item.get('pool', '')
    section = item.get('section', '')
    category = item.get('category', '')
    difficulty = item.get('difficulty', '')
    
    if subject and pool and section and category:
        # Map difficulty to short form: Easy -> E, Medium -> M, Hard -> H
        diff_short = difficulty[0] if difficulty else ''
        hierarchy[subject][pool][section][category][diff_short] += 1

# Prepare data for questions table - one row per category
table_data = []
for subject in sorted(hierarchy.keys()):
    for pool in sorted(hierarchy[subject].keys()):
        for section in sorted(hierarchy[subject][pool].keys()):
            categories_dict = hierarchy[subject][pool][section]
            # Create a separate row for each category
            for category in sorted(categories_dict.keys()):
                difficulty_counts = categories_dict[category]
                total = sum(difficulty_counts.values())
                e_count = difficulty_counts.get('E', 0)
                m_count = difficulty_counts.get('M', 0)
                h_count = difficulty_counts.get('H', 0)
                
                table_data.append({
                    'subject': subject,
                    'pool': pool,
                    'section': section,
                    'category': category,
                    'count': total,
                    'easy': e_count,
                    'medium': m_count,
                    'hard': h_count
                })

# Extract lessons data: section -> set of domains
lessons_hierarchy = defaultdict(set)
for item in lectures_data:
    section = item.get('section', '')
    domain = item.get('domain', '')
    if section and domain:
        lessons_hierarchy[section].add(domain)

# Prepare data for lessons table
lessons_table_data = []
for section in sorted(lessons_hierarchy.keys()):
    domains = sorted(lessons_hierarchy[section])
    lessons_table_data.append({
        'section': section,
        'domain': ', '.join(domains)
    })

# Count questions by unit_number, section, and domain (using category as domain)
unit_counts = defaultdict(int)  # (unit_number, section, domain) -> count
for item in data:
    unit_number = item.get('unit_number')
    section = item.get('section', '')
    category = item.get('category', '')
    
    if unit_number is not None and section:
        # Use category as domain (since questions don't have a separate domain field)
        # If category is empty, try to get domain from lectures mapping
        domain = category if category else 'Unknown'
        
        key = (unit_number, section, domain)
        unit_counts[key] += 1

# Prepare data for units table
units_table_data = []
for (unit_number, section, domain), count in sorted(unit_counts.items()):
    units_table_data.append({
        'unit_number': unit_number,
        'section': section,
        'domain': domain,
        'count': count
    })

# Print markdown table for questions
print("| Subject | Pool | Section | Category | Number of Questions |")
print("|---------|------|---------|----------|---------------------|")
for row in table_data:
    # Escape pipe characters in category if any
    category = row['category'].replace('|', '\\|')
    # Format: total (E: easy - M: medium - H: hard)
    count_str = f"{row['count']} (E: {row['easy']} - M: {row['medium']} - H: {row['hard']})"
    print(f"| {row['subject']} | {row['pool']} | {row['section']} | {category} | {count_str} |")

print("\n\n---\n\n")

# Print markdown table for lessons
print("| Section | Domain |")
print("|---------|--------|")
for row in lessons_table_data:
    # Escape pipe characters in domain if any
    domain = row['domain'].replace('|', '\\|')
    print(f"| {row['section']} | {domain} |")

print("\n\n---\n\n")

# Print markdown table for units
print("| Unit Number | Section | Domain | Number of Questions |")
print("|-------------|---------|--------|---------------------|")
for row in units_table_data:
    section = row['section'].replace('|', '\\|')
    domain = row['domain'].replace('|', '\\|')
    print(f"| {row['unit_number']} | {section} | {domain} | {row['count']} |")

print("\n\n---\n\n")

# Also create an HTML table for better visualization
html_output = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
            position: relative;
        }
        th:hover {
            background-color: #45a049;
        }
        th.sortable::after {
            content: ' ↕';
            font-size: 0.8em;
            opacity: 0.5;
        }
        th.sort-asc::after {
            content: ' ↑';
            opacity: 1;
        }
        th.sort-desc::after {
            content: ' ↓';
            opacity: 1;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e8f5e9;
        }
        .subject-header {
            background-color: #2196F3;
            color: white;
            font-weight: bold;
        }
        .pool-header {
            background-color: #FF9800;
            color: white;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Question Categories by Subject, Pool, and Section</h1>
    <p><em>Click on column headers to sort</em></p>
    <table id="categoriesTable">
        <thead>
            <tr>
                <th class="sortable" onclick="sortTable(0)">Subject</th>
                <th class="sortable" onclick="sortTable(1)">Pool</th>
                <th class="sortable" onclick="sortTable(2)">Section</th>
                <th class="sortable" onclick="sortTable(3)">Category</th>
                <th class="sortable" onclick="sortTable(4)">Number of Questions</th>
            </tr>
        </thead>
        <tbody>
"""

for row in table_data:
    category = row['category'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Format: total (E: easy - M: medium - H: hard)
    count_str = f"{row['count']} (E: {row['easy']} - M: {row['medium']} - H: {row['hard']})"
    html_output += f"""
            <tr>
                <td>{row['subject']}</td>
                <td>{row['pool']}</td>
                <td>{row['section']}</td>
                <td>{category}</td>
                <td>{count_str}</td>
            </tr>
"""

html_output += """
        </tbody>
    </table>
    
    <h2>Lessons by Section and Domain</h2>
    <p><em>Click on column headers to sort</em></p>
    <table id="lessonsTable">
        <thead>
            <tr>
                <th class="sortable" onclick="sortLessonsTable(0)">Section</th>
                <th class="sortable" onclick="sortLessonsTable(1)">Domain</th>
            </tr>
        </thead>
        <tbody>
"""

for row in lessons_table_data:
    domain = row['domain'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    html_output += f"""
            <tr>
                <td>{row['section']}</td>
                <td>{domain}</td>
            </tr>
"""

html_output += """
        </tbody>
    </table>
    
    <h2>Questions by Unit Number</h2>
    <p><em>Click on column headers to sort</em></p>
    <table id="unitsTable">
        <thead>
            <tr>
                <th class="sortable" onclick="sortUnitsTable(0)">Unit Number</th>
                <th class="sortable" onclick="sortUnitsTable(1)">Section</th>
                <th class="sortable" onclick="sortUnitsTable(2)">Domain</th>
                <th class="sortable" onclick="sortUnitsTable(3)">Number of Questions</th>
            </tr>
        </thead>
        <tbody>
"""

for row in units_table_data:
    section = row['section'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    domain = row['domain'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    html_output += f"""
            <tr>
                <td>{row['unit_number']}</td>
                <td>{section}</td>
                <td>{domain}</td>
                <td>{row['count']}</td>
            </tr>
"""

html_output += """
        </tbody>
    </table>
    <script>
        let sortDirection = {};
        let lessonsSortDirection = {};
        let unitsSortDirection = {};
        
        function sortTable(columnIndex) {
            const table = document.getElementById('categoriesTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            
            // Determine sort direction
            if (!sortDirection[columnIndex] || sortDirection[columnIndex] === 'desc') {
                sortDirection[columnIndex] = 'asc';
            } else {
                sortDirection[columnIndex] = 'desc';
            }
            
            // Remove sort indicators from all headers
            headers.forEach((header, index) => {
                header.classList.remove('sort-asc', 'sort-desc');
                if (index === columnIndex) {
                    header.classList.add('sort-' + sortDirection[columnIndex]);
                }
            });
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.cells[columnIndex].textContent.trim();
                const bText = b.cells[columnIndex].textContent.trim();
                
                // For column 4 (Number of Questions), extract the total number before the parenthesis
                if (columnIndex === 4) {
                    const aMatch = aText.match(/^(\\d+)/);
                    const bMatch = bText.match(/^(\\d+)/);
                    const aNum = aMatch ? parseInt(aMatch[1]) : 0;
                    const bNum = bMatch ? parseInt(bMatch[1]) : 0;
                    if (sortDirection[columnIndex] === 'asc') {
                        return aNum - bNum;
                    } else {
                        return bNum - aNum;
                    }
                }
                
                // Compare as strings for other columns
                if (sortDirection[columnIndex] === 'asc') {
                    return aText.localeCompare(bText);
                } else {
                    return bText.localeCompare(aText);
                }
            });
            
            // Remove all rows from tbody
            rows.forEach(row => tbody.removeChild(row));
            
            // Add sorted rows back
            rows.forEach(row => tbody.appendChild(row));
        }
        
        function sortLessonsTable(columnIndex) {
            const table = document.getElementById('lessonsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            
            // Determine sort direction
            if (!lessonsSortDirection[columnIndex] || lessonsSortDirection[columnIndex] === 'desc') {
                lessonsSortDirection[columnIndex] = 'asc';
            } else {
                lessonsSortDirection[columnIndex] = 'desc';
            }
            
            // Remove sort indicators from all headers
            headers.forEach((header, index) => {
                header.classList.remove('sort-asc', 'sort-desc');
                if (index === columnIndex) {
                    header.classList.add('sort-' + lessonsSortDirection[columnIndex]);
                }
            });
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.cells[columnIndex].textContent.trim();
                const bText = b.cells[columnIndex].textContent.trim();
                
                // Compare as strings
                if (lessonsSortDirection[columnIndex] === 'asc') {
                    return aText.localeCompare(bText);
                } else {
                    return bText.localeCompare(aText);
                }
            });
            
            // Remove all rows from tbody
            rows.forEach(row => tbody.removeChild(row));
            
            // Add sorted rows back
            rows.forEach(row => tbody.appendChild(row));
        }
        
        function sortUnitsTable(columnIndex) {
            const table = document.getElementById('unitsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            
            // Determine sort direction
            if (!unitsSortDirection[columnIndex] || unitsSortDirection[columnIndex] === 'desc') {
                unitsSortDirection[columnIndex] = 'asc';
            } else {
                unitsSortDirection[columnIndex] = 'desc';
            }
            
            // Remove sort indicators from all headers
            headers.forEach((header, index) => {
                header.classList.remove('sort-asc', 'sort-desc');
                if (index === columnIndex) {
                    header.classList.add('sort-' + unitsSortDirection[columnIndex]);
                }
            });
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.cells[columnIndex].textContent.trim();
                const bText = b.cells[columnIndex].textContent.trim();
                
                // For column 0 (Unit Number) and column 3 (Number of Questions), sort numerically
                if (columnIndex === 0 || columnIndex === 3) {
                    const aNum = parseInt(aText) || 0;
                    const bNum = parseInt(bText) || 0;
                    if (unitsSortDirection[columnIndex] === 'asc') {
                        return aNum - bNum;
                    } else {
                        return bNum - aNum;
                    }
                }
                
                // For other columns (Section, Domain), sort as strings
                if (unitsSortDirection[columnIndex] === 'asc') {
                    return aText.localeCompare(bText);
                } else {
                    return bText.localeCompare(aText);
                }
            });
            
            // Remove all rows from tbody
            rows.forEach(row => tbody.removeChild(row));
            
            // Add sorted rows back
            rows.forEach(row => tbody.appendChild(row));
        }
    </script>
</body>
</html>
"""

# Write HTML file
with open('categories_table.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("HTML table saved to 'categories_table.html'")

# Write CSV file for units table
csv_filename = 'questions_by_unit_number.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Unit Number', 'Section', 'Domain', 'Number of Questions']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    for row in units_table_data:
        writer.writerow({
            'Unit Number': row['unit_number'],
            'Section': row['section'],
            'Domain': row['domain'],
            'Number of Questions': row['count']
        })

print(f"CSV file saved to '{csv_filename}'")
