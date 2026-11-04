"""Export the executed notebook with a legible, explicit light code theme."""
from pathlib import Path
from nbconvert import HTMLExporter
import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / 'Harandi_Reproduction_KO.ipynb', as_version=4)
body, _ = HTMLExporter(template_name='lab', theme='light').from_notebook_node(notebook)
# Local Jupyter installations can inject a custom CSS theme. Keep the portable
# preview readable independently of those machine-specific settings.
css = '''<style id="portable-light-code">
.jp-CodeCell .jp-InputArea-editor, .jp-CodeCell .highlight,
.jp-CodeCell .highlight pre {background: #f6f8fa !important; color: #1f2328 !important;}
.highlight span {color: #1f2328 !important;}
.highlight .k, .highlight .kn, .highlight .kc, .highlight .kd,
.highlight .ow {color: #8250df !important;}
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sa,
.highlight .sd {color: #0a3069 !important;}
.highlight .c, .highlight .c1, .highlight .cm {color: #57606a !important;}
.highlight .mi, .highlight .mf {color: #0550ae !important;}
.jp-InputPrompt, .jp-OutputPrompt {color: #57606a !important;}
</style>'''
body = body.replace('</head>', css + '\n</head>').replace('<html lang="en">', '<html lang="ko">')
target = root / 'Harandi_Reproduction_KO.html'
target.write_text(body, encoding='utf-8')
print(target)
