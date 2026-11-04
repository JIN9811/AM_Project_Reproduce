"""Regression checks for formulas crossing Markdown's punctuation parser."""
from html import unescape
from pathlib import Path
import re
import sys

import pytest

mistune = pytest.importorskip('mistune', reason='Install requirements-documents.txt for document tests.')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from latex_report import LatexReport, markdown_text, tex_text


def test_display_math_survives_markdown_unchanged(tmp_path):
    latex = r'\begin{aligned}e&=\|u_x\|_2,\\a^*&=\nabla\!u_x.\end{aligned}'
    report = LatexReport(tmp_path)
    report.equation(latex)
    rendered = mistune.html(''.join(report.md))
    formula = re.search(r'<code class="language-math">(.*?)</code>', rendered, re.S)
    assert formula, 'Display math must bypass ordinary Markdown escaping/emphasis.'
    assert unescape(formula[1]).strip() == latex + '\n' + r'\tag{1}'


def test_inline_math_survives_emphasis_and_escape_processing():
    latex = r'\sigma_{xy}=\mu\!u_{x,y}'
    rendered = mistune.html(markdown_text('전단 $' + latex + '$ 및 $T_0=0$.'))
    formulas = re.findall(r'\$<code>(.*?)</code>\$', rendered)
    assert list(map(unescape, formulas)) == [latex, 'T_0=0']


def test_code_identifiers_are_not_changed_to_math():
    source = '`sig_x_O` 및 results/fem_circle/solution.npz'
    rendered = mistune.html(markdown_text(source))
    assert '<code>sig_x_O</code>' in rendered
    assert '<code>results/fem_circle/solution.npz</code>' in rendered
    assert '<em>' not in rendered


def test_table_math_preserves_absolute_values_without_creating_columns():
    result = markdown_text(r'$\|u_x\|_2+|q_x|$')
    assert '|' not in result
    assert r'\lVert' in result and r'\rVert' in result
    assert r'\vert' in result


def test_html_subscript_keeps_its_base_in_math():
    assert markdown_text('T<sub>0</sub>') == '$`T_{0}`$'
    assert tex_text('T<sub>0</sub>') == '$T_{0}$'


def test_inline_inequality_avoids_github_double_html_escaping():
    assert markdown_text(r'$-1<\nu<1/2$') == r'$`-1\lt \nu\lt 1/2`$'
