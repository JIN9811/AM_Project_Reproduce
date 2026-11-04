"""Verify that GitHub delivers every formula to MathJax without altering TeX.

Uses GitHub's stateless Markdown rendering endpoint; it does not publish files.
Requires an authenticated gh CLI. --published-ref checks a committed blob page.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'JIN9811/AM_Project_Reproduce'
FILES = ('reports/FULL_PROJECT_REPORT_KO.md', 'reports/FEM_REPORT_KO.md', 'docs/AUTHOR_NOTATION_KO.md')


class MathParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.formulas = []
        self.inside = False
        self.parts = []
        self.nested = []

    def handle_starttag(self, tag, attrs):
        if tag == 'math-renderer':
            self.inside = True
            self.parts = []
        elif self.inside:
            self.nested.append(tag)

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'math-renderer':
            self.formulas.append(''.join(self.parts).strip().strip('$').strip())
            self.inside = False


def expected_math(source):
    blocks = re.findall(r'^```math\n(.*?)\n```', source, re.M | re.S)
    rest = re.sub(r'^```[^\n]*\n.*?\n```', '', source, flags=re.M | re.S)
    inline = re.findall(r'\$`([^`\n]+)`\$', rest)
    rest = re.sub(r'\$`[^`\n]+`\$', '', rest)
    if '$' in rest:
        raise ValueError('Unprotected dollar delimiters remain in Markdown.')
    return [x.strip() for x in blocks + inline]


def check(root, output, published_ref=None):
    scratch = root/'tmp/markdown'
    scratch.mkdir(parents=True, exist_ok=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'published_blob' if published_ref else 'GitHub Markdown API',
              'published_ref': published_ref, 'files': [],
              'scope': 'Exact source-to-MathJax formula preservation, not a browser screenshot.',
              'status': 'pass'}
    for name in FILES:
        source = (root/name).read_text(encoding='utf-8')
        expected = expected_math(source)
        if published_ref:
            url = f'https://github.com/{REPOSITORY}/blob/{published_ref}/{name}'
            with urllib.request.urlopen(url) as response:
                rendered = response.read().decode('utf-8')
        else:
            payload = scratch/'request.json'
            payload.write_text(json.dumps({'text': source, 'mode': 'gfm', 'context': REPOSITORY}), encoding='utf-8')
            proc = subprocess.run(['gh', 'api', 'markdown', '--method', 'POST', '--input', str(payload)], capture_output=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr.decode('utf-8', errors='replace'))
            rendered = proc.stdout.decode('utf-8')
        (scratch/(Path(name).stem+'.html')).write_text(rendered, encoding='utf-8')
        parser = MathParser()
        parser.feed(rendered)
        missing = list((Counter(expected)-Counter(parser.formulas)).elements())
        unexpected = list((Counter(parser.formulas)-Counter(expected)).elements())
        row = {'file': name, 'sha256': hashlib.sha256((root/name).read_bytes()).hexdigest(),
               'source_formulas': len(expected), 'rendered_formulas': len(parser.formulas),
               'missing_or_changed': missing, 'unexpected_or_changed': unexpected,
               'nested_markup': parser.nested,
               'passed': not (missing or unexpected or parser.nested)}
        report['files'].append(row)
        if not row['passed']:
            report['status'] = 'fail'
        print(f"{name}: {len(parser.formulas)}/{len(expected)} formulas; {'PASS' if row['passed'] else 'FAIL'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report['status'] == 'pass'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--published-ref')
    args = p.parse_args()
    root = args.root.resolve()
    raise SystemExit(0 if check(root, args.output or root/'validation/REPORT_MATH_QA.json', args.published_ref) else 1)
