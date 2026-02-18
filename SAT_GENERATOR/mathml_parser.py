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
        graph = self._long_desc_to_graphspec(long_desc_result) if long_desc_result else None

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
            Dict with 'html' (original HTML structure) and 'text' (plain text for parsing)
        """
        # robust regex: get content inside <div ... aria-label="Long description ..."> ... </div>
        m = re.search(
            r'aria-label="Long description[^"]*"\s+class="sr-only"\s*>'
            r'(.*?)</div>',
            html,
            flags=re.DOTALL | re.IGNORECASE
        )
        if not m:
            return None

        # inside is usually <ul><li>...</li></ul>
        inner_html = m.group(1)
        inner_html = self._remove_svg_blocks(inner_html)
        
        # Keep the HTML structure (just clean up whitespace between tags)
        html_content = inner_html.strip()

        # Also create plain text version for parsing x/y values
        text = self._strip_tags_fallback(inner_html)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {'html': html_content, 'text': text} if (html_content or text) else None

    def _long_desc_to_graphspec(self, long_desc_result: Dict[str, str]) -> Optional[GraphSpec]:
        """
        Convert SAT long description to GraphSpec (currently supports line graph).
        
        Args:
            long_desc_result: Dict with 'html' and 'text' keys
        """
        if not long_desc_result:
            return None
        
        long_desc_text = long_desc_result.get('text', '')
        long_desc_html = long_desc_result.get('html', '')

        # detect graph type
        graph_type = "unknown"
        if "line graph" in long_desc_text.lower():
            graph_type = "line"

        # Extract pairs like: "Begins at 2010, 12%"
        # Works also for: "Falls sharply to 2014, 4%"
        pairs = re.findall(r'(\d{4})\s*,\s*(\d+(?:\.\d+)?)\s*%?', long_desc_text)

        if not pairs:
            return GraphSpec(
                graph_type=graph_type,
                raw_long_description=long_desc_text,
                long_description_html=long_desc_html
            )

        x_vals = [int(x) for x, _ in pairs]
        y_vals = [float(y) for _, y in pairs]

        # heuristic axis labels (SAT dataset style)
        x_label = "Model year" if any(2000 <= x <= 2100 for x in x_vals) else None
        y_label = "Percent" if "%" in long_desc_text else None
        y_unit = "%" if "%" in long_desc_text else None

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

    def _strip_tags_fallback(self, html: str) -> str:
        # Remove tags quickly
        html = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</p\s*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'</li\s*>', '\n', html, flags=re.IGNORECASE)
        html = re.sub(r'<.*?>', '', html, flags=re.DOTALL)
        html = html.replace('&nbsp;', ' ')
        html = html.replace('&amp;', '&')
        return html
