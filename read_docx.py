import zipfile
import xml.etree.ElementTree as ET

def read_docx(path):
    with zipfile.ZipFile(path) as docx:
        tree = ET.fromstring(docx.read('word/document.xml'))
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for p in tree.findall('.//w:p', namespaces):
            p_text = ''.join([node.text for node in p.findall('.//w:t', namespaces) if node.text])
            if p_text:
                text.append(p_text)
        return '\n'.join(text)

doc_path = 'report/ARM_hallucination_proposal_v5.docx'
output_path = 'output_v5.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(read_docx(doc_path))
