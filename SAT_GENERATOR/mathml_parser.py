import re
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GraphSpec:
    graph_type: str  # "line" | "bar" | "grouped_bar" | "scatter" ...
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_values: Optional[List[Any]] = None
    y_values: Optional[List[float]] = None
    y_unit: Optional[str] = None
    raw_long_description: Optional[str] = None  # Plain text version
    long_description_html: Optional[str] = None  # HTML structure preserved
    
    # Grouped bar chart specific fields
    title: Optional[str] = None  # Graph title
    groups: Optional[List[str]] = None  # Group names (e.g., ["before election", "after election"])
    categories: Optional[List[str]] = None  # Category names (e.g., ["no response", "responded to inquiry"])
    grouped_data: Optional[Dict[str, Dict[str, float]]] = None  # {category: {group: value}}
    y_axis_range: Optional[tuple] = None  # (min, max, increment) e.g., (0, 1300, 100)


@dataclass
class TableSpec:
    """Structured representation of HTML table for template-based regeneration."""
    caption: Optional[str] = None  # Table caption/title
    headers: Optional[List[str]] = None  # Column headers
    rows: Optional[List[List[str]]] = None  # Data rows (list of lists)
    row_labels: Optional[List[str]] = None  # Row labels (first column if it's <th>)
    original_html: Optional[str] = None  # Original HTML for reference
    table_class: Optional[str] = None  # CSS class (e.g., "gdr")
    
    def to_html(self) -> str:
        """
        Build HTML table from TableSpec.
        Recreates the same structure as SAT tables.
        """
        table_class_attr = f' class="{self.table_class}"' if self.table_class else ''
        
        parts = []
        parts.append('<p>')
        parts.append('<figure class="table">')
        parts.append(f'<table{table_class_attr}>')
        
        # Caption
        if self.caption:
            parts.append('<caption style="caption-side: top;">')
            parts.append(f'<p style="text-align: center;">{self.caption}</p>')
            parts.append('</caption>')
        
        # Headers
        if self.headers:
            parts.append('<thead>')
            parts.append('<tr>')
            for header in self.headers:
                parts.append(f'<th scope="col" style="text-align: center;vertical-align: bottom;">{header}</th>')
            parts.append('</tr>')
            parts.append('</thead>')
        
        # Data rows
        if self.rows:
            parts.append('<tbody>')
            for i, row in enumerate(self.rows):
                parts.append('<tr>')
                
                # Row label (first column as <th>)
                if self.row_labels and i < len(self.row_labels) and self.row_labels[i]:
                    parts.append(f'<th scope="row" style="text-align: left;">{self.row_labels[i]}</th>')
                
                # Data cells
                for cell in row:
                    parts.append(f'<td style="text-align: center;">{cell}</td>')
                
                parts.append('</tr>')
            parts.append('</tbody>')
        
        parts.append('</table>')
        parts.append('</figure>')
        parts.append('</p>')
        
        return ''.join(parts)


class MathMLParser:
    """
    - Parse HTML + MathML -> clean text
    - Extract graph long description (sr-only) -> GraphSpec
    """

    _HTML_CONTAINER_TAGS = frozenset(
        {'p', 'div', 'span', 'body', 'html', 'section', 'article', 'figure', 'ul', 'li'}
    )

    def parse(self, html_or_mathml: str) -> Dict[str, Any]:
        """
        Return:
        {
          "text": "...",
          "graph": GraphSpec | None,
          "table": TableSpec | None
        }
        """
        s = (html_or_mathml or "").strip()
        if not s:
            return {"text": "", "graph": None, "table": None}

        # Remove MathML namespace
        s = s.replace('xmlns="http://www.w3.org/1998/Math/MathML"', '')

        # --- Step 1: Extract table if present ---
        table = self._extract_table(s)
        
        # --- Step 2: Extract graph long description BEFORE XML parsing ---
        # (Because SVG may contain invalid XML or huge payload)
        long_desc_result = self._extract_graph_long_description(s)
        graph = self._long_desc_to_graphspec(long_desc_result, full_html=s) if long_desc_result else None

        # --- Step 3: Remove SVG and table entirely to save parsing cost ---
        s_wo_svg = self._remove_svg_blocks(s)
        s_wo_table = self._remove_table_blocks(s_wo_svg)

        # --- Step 4: Parse remaining HTML/MathML ---
        try:
            root = ET.fromstring(f"<root>{s_wo_table}</root>")
            text = self._parse_children(root).strip()
        except Exception:
            # fallback: remove tags roughly
            text = self._strip_tags_fallback(s_wo_table).strip()

        return {"text": text, "graph": graph, "table": table}

    def parse_paragraph(self, paragraph_html: str) -> Dict[str, Any]:
        """
        Parse paragraph HTML that may contain a graph or table.
        
        Return:
        {
          "text": "...",  # Plain text of paragraph (no SVG/graph/table)
          "graph": GraphSpec | None,  # Graph info if present
          "table": TableSpec | None,  # Table info if present
          "has_graph": bool,  # Whether paragraph contains a graph
          "has_table": bool,  # Whether paragraph contains a table
          "original_html": str  # Original HTML for reference
        }
        """
        s = (paragraph_html or "").strip()
        if not s:
            return {"text": "", "graph": None, "has_graph": False, "original_html": ""}

        # Remove MathML namespace
        s = s.replace('xmlns="http://www.w3.org/1998/Math/MathML"', '')

        # --- Step 1: Extract table if present ---
        table = self._extract_table(s)
        has_table = table is not None
        
        # --- Step 2: Extract graph long description ---
        long_desc_result = self._extract_graph_long_description(s)
        graph = self._long_desc_to_graphspec(long_desc_result, full_html=s) if long_desc_result else None
        has_graph = graph is not None

        # --- Step 3: Remove SVG, table, and sr-only div entirely to get paragraph text ---
        s_wo_svg = self._remove_svg_blocks(s)
        s_wo_table = self._remove_table_blocks(s_wo_svg)
        s_wo_table = self._remove_sr_only(s_wo_table)  # Remove long description div

        # --- Step 4: Parse remaining HTML to get text ---
        try:
            root = ET.fromstring(f"<root>{s_wo_table}</root>")
            text = self._parse_children(root).strip()
        except Exception:
            # fallback: remove tags roughly
            text = self._strip_tags_fallback(s_wo_table).strip()

        return {
            "text": text,
            "graph": graph,
            "table": table,
            "has_graph": has_graph,
            "has_table": has_table,
            "original_html": paragraph_html
        }

    # ----------------------------
    # Text + MathML parsing
    # ----------------------------
    def _parse_children(self, elem) -> str:
        parts = []
        if elem.text:
            parts.append(elem.text)

        for child in elem:
            parts.append(self._parse_element(child))
            if child.tail:
                parts.append(child.tail)

        return ''.join(parts)

    def _parse_element(self, elem) -> str:
        if elem is None:
            return ''
        tag = elem.tag.split('}')[-1] if isinstance(elem.tag, str) else ''

        if tag in ('mn', 'mi', 'mo'):
            return (elem.text or '')

        if tag == 'mrow':
            return ' '.join(self._parse_element(c) for c in elem)

        if tag == 'mfrac':
            num = self._parse_element(elem[0]) if len(elem) > 0 else ""
            den = self._parse_element(elem[1]) if len(elem) > 1 else ""
            return f"({num}) / ({den})"

        if tag == 'mfenced':
            inner = ' '.join(self._parse_element(c) for c in elem)
            return f"({inner})"

        if tag == 'msqrt':
            inner = self._parse_element(elem[0]) if len(elem) > 0 else ""
            return f"sqrt({inner})"

        if tag == 'msup':
            base = self._parse_element(elem[0]) if len(elem) > 0 else ""
            exp = self._parse_element(elem[1]) if len(elem) > 1 else ""
            return f"({base})^({exp})"

        if tag == 'msub':
            base = self._parse_element(elem[0]) if len(elem) > 0 else ""
            sub = self._parse_element(elem[1]) if len(elem) > 1 else ""
            return f"{base}_{sub}"

        if tag == 'math':
            return ' '.join(self._parse_element(c) for c in elem)

        if tag == 'mtext':
            return self._parse_children(elem)

        # HTML wrapper tags
        if tag in self._HTML_CONTAINER_TAGS or not tag:
            return self._parse_children(elem)

        return ''

    # ----------------------------
    # Table extraction
    # ----------------------------
    def _extract_table(self, html: str) -> Optional[TableSpec]:
        """
        Extract HTML table and parse into structured TableSpec.
        Handles SAT-style tables with <figure><table>...</table></figure>
        """
        # Find table block (may be wrapped in <figure>)
        table_match = re.search(
            r'<figure[^>]*>.*?(<table[^>]*>.*?</table>).*?</figure>',
            html,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # Fallback: table without figure wrapper
        if not table_match:
            table_match = re.search(
                r'(<table[^>]*>.*?</table>)',
                html,
                flags=re.DOTALL | re.IGNORECASE
            )
        
        if not table_match:
            return None
        
        table_html = table_match.group(1) if len(table_match.groups()) > 0 else table_match.group(0)
        
        # Parse table structure
        return self._parse_html_table(table_html)
    
    def _parse_html_table(self, table_html: str) -> Optional[TableSpec]:
        """
        Parse HTML table into TableSpec with headers, rows, and data.
        """
        try:
            # Extract caption
            caption = None
            caption_match = re.search(r'<caption[^>]*>(.*?)</caption>', table_html, flags=re.DOTALL | re.IGNORECASE)
            if caption_match:
                caption_html = caption_match.group(1)
                caption = self._strip_tags_fallback(caption_html).strip()
            
            # Extract table class
            table_class = None
            class_match = re.search(r'<table[^>]*class=["\']([^"\']+)["\']', table_html, re.IGNORECASE)
            if class_match:
                table_class = class_match.group(1)
            
            # Extract headers from <thead>
            headers = []
            thead_match = re.search(r'<thead[^>]*>(.*?)</thead>', table_html, flags=re.DOTALL | re.IGNORECASE)
            if thead_match:
                thead_html = thead_match.group(1)
                # Find all <th> in header row
                header_cells = re.findall(r'<th[^>]*>(.*?)</th>', thead_html, flags=re.DOTALL | re.IGNORECASE)
                headers = [self._strip_tags_fallback(cell).strip() for cell in header_cells]
            
            # Extract data rows from <tbody>
            rows = []
            row_labels = []
            tbody_match = re.search(r'<tbody[^>]*>(.*?)</tbody>', table_html, flags=re.DOTALL | re.IGNORECASE)
            if tbody_match:
                tbody_html = tbody_match.group(1)
                # Find all <tr> rows
                tr_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', tbody_html, flags=re.DOTALL | re.IGNORECASE)
                
                for tr_html in tr_matches:
                    row_data = []
                    
                    # Check if first cell is <th> (row label)
                    th_match = re.search(r'<th[^>]*>(.*?)</th>', tr_html, flags=re.DOTALL | re.IGNORECASE)
                    if th_match:
                        row_label = self._strip_tags_fallback(th_match.group(1)).strip()
                        row_labels.append(row_label)
                    else:
                        row_labels.append(None)
                    
                    # Extract all <td> cells
                    td_matches = re.findall(r'<td[^>]*>(.*?)</td>', tr_html, flags=re.DOTALL | re.IGNORECASE)
                    row_data = [self._strip_tags_fallback(cell).strip() for cell in td_matches]
                    
                    if row_data:  # Only add non-empty rows
                        rows.append(row_data)
            
            # Only create TableSpec if we have meaningful data
            if not headers and not rows:
                return None
            
            return TableSpec(
                caption=caption,
                headers=headers,
                rows=rows,
                row_labels=row_labels if any(row_labels) else None,
                original_html=table_html,
                table_class=table_class
            )
        
        except Exception as e:
            # If parsing fails, return None
            return None
    
    # ----------------------------
    # Graph extraction
    # ----------------------------
    def _extract_graph_long_description(self, html: str) -> Optional[Dict[str, str]]:
        """
        Extract sr-only long description from SAT HTML.
        Example marker:
          aria-label="Long description for line graph"
        
        Returns:
            Dict with 'html' (original HTML structure), 'text' (plain text for parsing),
            and 'aria_label' (the aria-label text for graph type detection)
        """
        # More flexible regex: match <div> with aria-label and class="sr-only" in any order
        # Pattern 1: aria-label first
        m = re.search(
            r'<div[^>]*aria-label="(Long description[^"]*)"[^>]*class="sr-only"[^>]*>'
            r'(.*?)</div>',
            html,
            flags=re.DOTALL | re.IGNORECASE
        )
        
        # Pattern 2: class first
        if not m:
            m = re.search(
                r'<div[^>]*class="sr-only"[^>]*aria-label="(Long description[^"]*)"[^>]*>'
                r'(.*?)</div>',
                html,
                flags=re.DOTALL | re.IGNORECASE
            )
        if not m:
            return None

        aria_label = m.group(1)
        inner_html = m.group(2)
        inner_html = self._remove_svg_blocks(inner_html)
        
        # Keep the HTML structure (just clean up whitespace between tags)
        html_content = inner_html.strip()

        # Also create plain text version for parsing x/y values
        text = self._strip_tags_fallback(inner_html)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            'html': html_content,
            'text': text,
            'aria_label': aria_label
        } if (html_content or text) else None

    def _parse_single_bar_chart_html(self, html: str) -> Optional[Dict[str, Any]]:
        """
        Parse single (non-grouped) bar chart from HTML list structure.
        
        **Differentiation from grouped bar chart:**
        
        SINGLE BAR CHART pattern:
        <ul>
          <li>The data for the N categories are as follows:
            <ul>
              <li>category1: value1 [unit]</li>  ← VALUE DIRECTLY after colon
              <li>category2: value2 [unit]</li>
            </ul>
          </li>
        </ul>
        
        GROUPED BAR CHART pattern:
        <ul>
          <li>For each data category, the following bars are shown:
            <ul>
              <li>group1</li>
              <li>group2</li>
            </ul>
          </li>
          <li>The data for the N categories are as follows:
            <ul>
              <li>category1:               ← Notice the NESTED <ul> after colon
                <ul>
                  <li>group1: value1</li>
                  <li>group2: value2</li>
                </ul>
              </li>
            </ul>
          </li>
        </ul>
        
        Key differences:
        1. Grouped has "For each data category, the following bars are shown" section
        2. Grouped has nested <ul> tags after category name
        3. Single has value DIRECTLY after colon (no nested structure)
        
        Returns:
            Dict with 'categories' and 'values' keys or None
        """
        # Check if this looks like bar chart data
        if "data for the" not in html.lower() or "categories are as follows" not in html.lower():
            return None
        
        # If this has "for each data category" and "following bars are shown", it's grouped
        if "for each data category" in html.lower() and "following bars are shown" in html.lower():
            return None
        
        # Extract data section after "data for the N categories are as follows"
        data_match = re.search(
            r'data for the \d+ categories are as follows:.*?<br\s*/?>?\s*<ul>(.*?)</ul>',
            html,
            re.DOTALL | re.IGNORECASE
        )
        if not data_match:
            return None
        
        data_section = data_match.group(1)
        
        # Key check: If data_section contains nested <ul> tags, it's a grouped bar chart
        # We're looking for direct <li>CATEGORY: VALUE</li>, not <li>CATEGORY:<ul>...</ul></li>
        if re.search(r'<li>[^<]*:<\s*ul\s*>', data_section, re.IGNORECASE):
            return None
        
        # Extract category:value pairs where value is directly after colon
        # Pattern: <li>CATEGORY_NAME: NUMBER</li> (no nested tags)
        
        categories = []
        values = []
        
        # Find all <li>TEXT</li> items where TEXT contains CATEGORY: NUMBER
        # and TEXT does NOT contain nested tags or <ul>
        li_pattern = r'<li>([^<]+)</li>'
        li_items = re.findall(li_pattern, data_section, re.IGNORECASE)
        
        for item_text in li_items:
            # Check if this item matches CATEGORY: NUMBER [UNIT] pattern
            # Examples: "Gorner: 41.2 square kilometers" or "Item A: 1,252"
            # The key is that NUMBER is directly after colon (not nested in another tag)
            # Pattern allows optional unit/text after the number
            match = re.match(r'^([^:]+):\s*([0-9,.]+)(?:\s+.*)?$', item_text.strip())
            if match:
                category = match.group(1).strip()
                value_str = match.group(2).strip()
                # Remove commas from numbers (e.g., "1,252" -> 1252)
                value = float(value_str.replace(',', ''))
                categories.append(category)
                values.append(value)
        
        if not categories:
            return None
        
        return {
            'categories': categories,
            'values': values
        }

    def _parse_grouped_bar_chart_html(self, html: str) -> Optional[Dict[str, Any]]:
        """
        Parse grouped bar chart from nested HTML list structure.
        
        Pattern:
        <ul>
          <li>For each data category, the following bars are shown:
            <ul>
              <li>group1</li>
              <li>group2</li>
            </ul>
          </li>
          <li>The data for the N categories are as follows:
            <ul>
              <li>category1:
                <ul>
                  <li>group1: value1</li>
                  <li>group2: value2</li>
                </ul>
              </li>
              ...
            </ul>
          </li>
        </ul>
        
        Key difference from single: values are NESTED in <ul> after category name
        
        Returns:
            Dict with 'groups', 'categories', 'data' keys or None
        """
        # Check if this is a grouped bar chart
        if "for each data category" not in html.lower() or "following bars are shown" not in html.lower():
            return None
        
        # Extract groups from first section
        groups = []
        groups_match = re.search(r'following bars are shown:.*?<ul>(.*?)</ul>', html, re.DOTALL | re.IGNORECASE)
        if groups_match:
            group_items = re.findall(r'<li>([^<]+)</li>', groups_match.group(1))
            groups = [g.strip() for g in group_items if g.strip()]
        
        if not groups:
            return None
        
        # Extract categories and data from second section
        categories = []
        grouped_data = {}
        
        # Find the data section - more flexible pattern
        # Match from "data for the N categories" to the end of the outer ul
        # Make <br> tag optional (some HTML has newline/whitespace instead)
        data_match = re.search(r'data for the \d+ categories are as follows:\s*(?:<br\s*/?>)?\s*<ul>(.*)', html, re.DOTALL | re.IGNORECASE)
        if not data_match:
            return None
        
        data_section = data_match.group(1)
        
        # Extract each category block - match category name followed by nested ul with data
        category_pattern = r'<li>\s*([^:<]+)\s*:\s*<ul>(.*?)</ul>\s*</li>'
        category_matches = re.findall(category_pattern, data_section, re.DOTALL | re.IGNORECASE)
        
        for category_name, category_data in category_matches:
            category_name = category_name.strip()
            categories.append(category_name)
            
            # Extract group:value pairs within this category
            value_pattern = r'<li>([^:]+):\s*([0-9,]+)</li>'
            value_matches = re.findall(value_pattern, category_data, re.IGNORECASE)
            
            group_values = {}
            for group_name, value_str in value_matches:
                group_name = group_name.strip()
                # Remove commas from numbers (e.g., "1,252" -> 1252)
                value = float(value_str.replace(',', ''))
                group_values[group_name] = value
            
            grouped_data[category_name] = group_values
        
        if not categories or not grouped_data:
            return None
        
        return {
            'groups': groups,
            'categories': categories,
            'data': grouped_data
        }

    def _long_desc_to_graphspec(self, long_desc_result: Dict[str, str], full_html: str = "") -> Optional[GraphSpec]:
        """
        Convert SAT long description to GraphSpec (supports line graph, bar graph, grouped bar, scatter plot).
        
        Args:
            long_desc_result: Dict with 'html', 'text', and 'aria_label' keys
            full_html: Full HTML content (for better label detection from question text)
        """
        if not long_desc_result:
            return None
        
        long_desc_text = long_desc_result.get('text', '')
        long_desc_html = long_desc_result.get('html', '')
        aria_label = long_desc_result.get('aria_label', '').lower()

        # Extract title from aria-label (e.g., "Long description for bar graph titled XYZ")
        title = None
        title_match = re.search(r'titled\s+([^"]+)$', aria_label, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).strip()
        
        # Extract y-axis label and range from aria-label if available
        y_label = None
        y_axis_range = None
        
        # Pattern: "The vertical axis is labeled XYZ. It ranges from A to B in increments of C"
        if full_html:
            y_label_match = re.search(r'vertical axis is labeled ([^.]+)', full_html, re.IGNORECASE)
            if y_label_match:
                y_label = y_label_match.group(1).strip()
            
            y_range_match = re.search(r'ranges from ([0-9,]+) to ([0-9,]+) in increments of ([0-9,]+)', full_html, re.IGNORECASE)
            if y_range_match:
                y_min = float(y_range_match.group(1).replace(',', ''))
                y_max = float(y_range_match.group(2).replace(',', ''))
                y_inc = float(y_range_match.group(3).replace(',', ''))
                y_axis_range = (y_min, y_max, y_inc)

        # detect graph type from aria-label first, fallback to content
        graph_type = "unknown"
        if "line graph" in aria_label or "line graph" in long_desc_text.lower():
            graph_type = "line"
        elif "bar graph" in aria_label or "bar graph" in long_desc_text.lower():
            graph_type = "bar"
        elif "scatter plot" in aria_label or "scatter plot" in long_desc_text.lower() or "scatterplot" in long_desc_text.lower():
            graph_type = "scatter"

        # Pattern 4: Grouped bar chart (check FIRST before other patterns)
        if graph_type == "bar":
            grouped_result = self._parse_grouped_bar_chart_html(long_desc_html)
            if grouped_result:
                return GraphSpec(
                    graph_type="grouped_bar",
                    title=title,
                    y_label=y_label,
                    y_axis_range=y_axis_range,
                    groups=grouped_result['groups'],
                    categories=grouped_result['categories'],
                    grouped_data=grouped_result['data'],
                    raw_long_description=long_desc_text,
                    long_description_html=long_desc_html
                )
            
            # Pattern 5: Single bar chart (if not grouped)
            single_bar_result = self._parse_single_bar_chart_html(long_desc_html)
            if single_bar_result:
                return GraphSpec(
                    graph_type="bar",
                    title=title,
                    y_label=y_label,
                    y_axis_range=y_axis_range,
                    x_values=single_bar_result['categories'],  # Category names on X-axis
                    y_values=single_bar_result['values'],      # Values on Y-axis
                    x_label="Category",
                    raw_long_description=long_desc_text,
                    long_description_html=long_desc_html
                )

        # Try multiple patterns to extract data
        pairs = None
        x_vals = None
        y_vals = None
        x_label = None
        y_unit = None
        
        # Pattern 1: "(X comma Y)" format (scatter plot format)
        # Example: "(1 comma 69)", "(2 comma 60)", "(3 comma 73)"
        scatter_pairs = re.findall(r'\((\d+(?:\.\d+)?)\s+comma\s+(\d+(?:\.\d+)?)\)', long_desc_text, re.IGNORECASE)
        
        if scatter_pairs:
            x_vals = [float(x) for x, _ in scatter_pairs]
            y_vals = [float(y) for _, y in scatter_pairs]
            
            # Detect labels and units from full HTML context (better than just long description)
            context = (full_html or long_desc_text).lower()
            
            # Detect x_label - look for patterns like "times x, in days"
            if re.search(r'times?\s+x\s*,?\s*in\s+days?\s+since', context):
                x_label = "Time (days since June 1)"
            elif "time" in context and "days" in context:
                x_label = "Time (days)"
            elif "days since" in context:
                x_label = "Days since June 1"
            elif "days" in context:
                x_label = "Days"
            else:
                x_label = "x"
            
            # Detect y_label and y_unit - look for patterns like "temperature y, in °F"
            if re.search(r'temperature\s+y\s*,?\s*in\s+°f', context) or re.search(r'temperature\s+y\s*,?\s*in\s+degree', context):
                y_label = "Temperature (°F)"
                y_unit = "°F"
            elif "temperature" in context:
                if "°f" in context or "degree f" in context or "degrees f" in context:
                    y_label = "Temperature (°F)"
                    y_unit = "°F"
                else:
                    y_label = "Temperature"
                    y_unit = None
            else:
                y_label = "y"
                y_unit = None
        else:
            # Pattern 2: "Year, percentage" format (line graph format)
            year_pairs = re.findall(r'(\d{4})\s*,\s*(\d+(?:\.\d+)?)\s*%?', long_desc_text)
            
            if year_pairs:
                x_vals = [int(x) for x, _ in year_pairs]
                y_vals = [float(y) for _, y in year_pairs]
                # heuristic axis labels (SAT dataset style)
                x_label = "Model year" if any(2000 <= x <= 2100 for x in x_vals) else "Year"
                y_label = "Percent" if "%" in long_desc_text else None
                y_unit = "%" if "%" in long_desc_text else None

        # If no patterns matched, return basic GraphSpec with just the description
        if x_vals is None or y_vals is None:
            return GraphSpec(
                graph_type=graph_type,
                raw_long_description=long_desc_text,
                long_description_html=long_desc_html
            )

        return GraphSpec(
            graph_type=graph_type,
            x_label=x_label,
            y_label=y_label,
            x_values=x_vals,
            y_values=y_vals,
            y_unit=y_unit,
            raw_long_description=long_desc_text,
            long_description_html=long_desc_html
        )

    # ----------------------------
    # Utilities
    # ----------------------------
    def _remove_svg_blocks(self, html: str) -> str:
        # Remove <svg>...</svg> completely
        html = re.sub(r'<svg\b.*?</svg>', '', html, flags=re.DOTALL | re.IGNORECASE)
        return html
    
    def _remove_table_blocks(self, html: str) -> str:
        # Remove <figure><table>...</table></figure> completely
        html = re.sub(r'<figure[^>]*>.*?<table[^>]*>.*?</table>.*?</figure>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Also remove standalone tables
        html = re.sub(r'<table[^>]*>.*?</table>', '', html, flags=re.DOTALL | re.IGNORECASE)
        return html

    def _remove_sr_only(self, html: str) -> str:
        # Remove sr-only div containing long description
        html = re.sub(r'<div[^>]*class="sr-only"[^>]*>.*?</div>', '', html, flags=re.DOTALL | re.IGNORECASE)
        return html

    def _strip_tags_fallback(self, html: str) -> str:
        # Remove tags quickly
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</p\s*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</li\s*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<.*?>', '', html, flags=re.DOTALL)
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        return html
