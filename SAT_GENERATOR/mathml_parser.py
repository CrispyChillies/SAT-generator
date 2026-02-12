from xml.etree import ElementTree as ET

class MathMLParser:
    # Thẻ HTML bọc ngoài: giữ nguyên text + nội dung con (để có "The correct answer is", " is ", v.v.)
    _HTML_CONTAINER_TAGS = frozenset({'p', 'div', 'span', 'body', 'html', 'section', 'article'})

    def parse(self, mathml_string: str) -> str:
        """Convert MathML (hoặc HTML chứa MathML) sang text dễ đọc"""
        mathml_string = (mathml_string or '').strip()
        if not mathml_string:
            return ''
        mathml_string = mathml_string.replace('xmlns="http://www.w3.org/1998/Math/MathML"', '')
        
        try:
            root = ET.fromstring(mathml_string)
            out = self._parse_element(root)
            return (out or '').strip()
        except Exception:
            return mathml_string

    def _parse_element(self, elem) -> str:
        if elem is None:
            return ''
        tag = elem.tag.split('}')[-1] if isinstance(elem.tag, str) else ''

        if tag == 'mn':
            return (elem.text or '') or ''
        if tag == 'mi':
            return (elem.text or '') or ''
        if tag == 'mo':
            return (elem.text or '') or ''
        if tag == 'mrow':
            return ' '.join(self._parse_element(c) for c in elem)
        if tag == 'mfrac':
            num = self._parse_element(elem[0])
            den = self._parse_element(elem[1])
            return f"({num}) / ({den})"
        if tag == 'mfenced':
            return ' '.join(self._parse_element(c) for c in elem)
        if tag == 'msqrt':
            return f"sqrt({self._parse_element(elem[0])})"
        if tag == 'msup':
            base = self._parse_element(elem[0])
            exp = self._parse_element(elem[1])
            return f"({base})^({exp})"
        if tag == 'msub':
            base = self._parse_element(elem[0])
            sub = self._parse_element(elem[1])
            return f"{base}_{sub}"
        if tag == 'math':
            return ' '.join(self._parse_element(c) for c in elem)
        # MathML <mtext>: text trong công thức (vd "2" trong cm², hoặc câu hỏi "What is the measure...")
        if tag == 'mtext':
            parts = [elem.text or '']
            for child in elem:
                parts.append(self._parse_element(child))
                parts.append(child.tail or '')
            return ''.join(parts)

        # HTML wrapper (p, div, span, ...): nối text + từng con + tail
        if tag in self._HTML_CONTAINER_TAGS or not tag:
            parts = [elem.text or '']
            for child in elem:
                parts.append(self._parse_element(child))
                parts.append(child.tail or '')
            return ''.join(parts)

        return ''