import re
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class GraphSpec:
    graph_type: str  # "line" | "bar" | "scatter" ...
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_values: Optional[List[Any]] = None
    y_values: Optional[List[float]] = None
    y_unit: Optional[str] = None
    raw_long_description: Optional[str] = None  # Plain text version
    long_description_html: Optional[str] = None  # HTML structure preserved


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
          "graph": GraphSpec | None
        }
        """
        s = (html_or_mathml or "").strip()
        if not s:
            return {"text": "", "graph": None}

        # Remove MathML namespace
        s = s.replace('xmlns="http://www.w3.org/1998/Math/MathML"', '')

        # --- Step 1: Extract graph long description BEFORE XML parsing ---
        # (Because SVG may contain invalid XML or huge payload)
        long_desc_result = self._extract_graph_long_description(s)
        graph = self._long_desc_to_graphspec(long_desc_result, full_html=s) if long_desc_result else None

        # --- Step 2: Remove SVG entirely to save parsing cost ---
        s_wo_svg = self._remove_svg_blocks(s)

        # --- Step 3: Parse remaining HTML/MathML ---
        try:
            root = ET.fromstring(f"<root>{s_wo_svg}</root>")
            text = self._parse_children(root).strip()
        except Exception:
            # fallback: remove tags roughly
            text = self._strip_tags_fallback(s_wo_svg).strip()

        return {"text": text, "graph": graph}

    def parse_paragraph(self, paragraph_html: str) -> Dict[str, Any]:
        """
        Parse paragraph HTML that may contain a graph.
        
        Return:
        {
          "text": "...",  # Plain text of paragraph (no SVG/graph)
          "graph": GraphSpec | None,  # Graph info if present
          "has_graph": bool,  # Whether paragraph contains a graph
          "original_html": str  # Original HTML for reference
        }
        """
        s = (paragraph_html or "").strip()
        if not s:
            return {"text": "", "graph": None, "has_graph": False, "original_html": ""}

        # Remove MathML namespace
        s = s.replace('xmlns="http://www.w3.org/1998/Math/MathML"', '')

        # --- Step 1: Extract graph long description ---
        long_desc_result = self._extract_graph_long_description(s)
        graph = self._long_desc_to_graphspec(long_desc_result, full_html=s) if long_desc_result else None
        has_graph = graph is not None

        # --- Step 2: Remove SVG and sr-only div entirely to get paragraph text ---
        s_wo_svg = self._remove_svg_blocks(s)
        s_wo_svg = self._remove_sr_only(s_wo_svg)  # Remove long description div

        # --- Step 3: Parse remaining HTML to get text ---
        try:
            root = ET.fromstring(f"<root>{s_wo_svg}</root>")
            text = self._parse_children(root).strip()
        except Exception:
            # fallback: remove tags roughly
            text = self._strip_tags_fallback(s_wo_svg).strip()

        return {
            "text": text,
            "graph": graph,
            "has_graph": has_graph,
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

    def _long_desc_to_graphspec(self, long_desc_result: Dict[str, str], full_html: str = "") -> Optional[GraphSpec]:
        """
        Convert SAT long description to GraphSpec (supports line graph, bar graph, scatter plot).
        
        Args:
            long_desc_result: Dict with 'html', 'text', and 'aria_label' keys
            full_html: Full HTML content (for better label detection from question text)
        """
        if not long_desc_result:
            return None
        
        long_desc_text = long_desc_result.get('text', '')
        long_desc_html = long_desc_result.get('html', '')
        aria_label = long_desc_result.get('aria_label', '').lower()

        # detect graph type from aria-label first, fallback to content
        graph_type = "unknown"
        if "line graph" in aria_label or "line graph" in long_desc_text.lower():
            graph_type = "line"
        elif "bar graph" in aria_label or "bar graph" in long_desc_text.lower():
            graph_type = "bar"
        elif "scatter plot" in aria_label or "scatter plot" in long_desc_text.lower() or "scatterplot" in long_desc_text.lower():
            graph_type = "scatter"

        # Try multiple patterns to extract data
        pairs = None
        x_vals = None
        y_vals = None
        x_label = None
        y_label = None
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
            # Pattern 2: "Group X: Y" format (bar graph format)
            group_pairs = re.findall(r'Group\s+(\d+)\s*:\s*(\d+(?:\.\d+)?)', long_desc_text, re.IGNORECASE)
            
            if group_pairs:
                x_vals = [f"Group {x}" for x, _ in group_pairs]
                y_vals = [float(y) for _, y in group_pairs]
                x_label = "Group"
                
                # Detect Y label and unit from context
                lower_text = long_desc_text.lower()
                if "books" in lower_text:
                    y_label = "Number of books"
                    y_unit = "books"
                elif "number" in lower_text:
                    y_label = "Number"
                    y_unit = None
                else:
                    y_label = "Value"
                    y_unit = None
            else:
                # Pattern 3: "Year, percentage" format (line graph format)
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
