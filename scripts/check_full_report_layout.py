"""Automated PDF checks and all-page renders; never claims manual visual review.

Run with the document Python after build_full_korean_report.py. The JSON belongs
in the deliverable folder; page PNGs/contact sheets belong in the scratch workdir.
An automatic pass checks enumerated conditions only, not overall visual quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pdfplumber
from PIL import Image, ImageChops, ImageDraw, ImageStat
from reportlab.pdfbase.ttfonts import TTFont


ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {'\ufffd', '\ufffe', '\uffff', '\x00'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def codepoint(character):
    return f'U+{ord(character):04X}'


def load_font_maps(paths):
    maps, metadata = [], []
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Glyph audit font is missing: {path}; use --font/--bold-font.')
        font = TTFont(f'LayoutAudit{index}', str(path))
        maps.append({value for value, glyph in font.face.charToGlyph.items() if glyph != 0})
        metadata.append({'path':str(path), 'sha256':sha256(path), 'mapped_codepoints':len(maps[-1])})
    return maps, metadata


def inspect_page(page, font_maps, tolerance):
    text = page.extract_text() or ''
    chars = [c for c in page.chars if c.get('text', '').strip()]
    width, height = float(page.width), float(page.height)
    outside = []
    missing = set()
    replacement = set()
    fonts = set()
    for char in chars:
        fontname = str(char.get('fontname', ''))
        fonts.add(fontname)
        bounds = [float(char[key]) for key in ('x0', 'top', 'x1', 'bottom')]
        if (not all(math.isfinite(v) for v in bounds) or bounds[0] < -tolerance
                or bounds[1] < -tolerance or bounds[2] > width+tolerance
                or bounds[3] > height+tolerance):
            outside.append({'text':char['text'], 'bounds_points':[v if math.isfinite(v) else str(v) for v in bounds]})
        # ReportLab embeds Malgun and Malgun-Bold under subset names. Audit the
        # corresponding supplied font map, rather than silently accepting any
        # character merely because a different supplied font contains it.
        mapping = font_maps[1] if 'bold' in fontname.lower() and len(font_maps)>1 else font_maps[0]
        for value in char['text']:
            if value.isspace():
                continue
            if value in REPLACEMENTS:
                replacement.add(codepoint(value))
            if ord(value) not in mapping:
                missing.add((value, codepoint(value), fontname))
    unresolved = sorted(set(re.findall(r'\(cid:\d+\)', text)))
    failures = []
    if not text.strip() or not chars:
        failures.append('empty_extracted_text')
    if outside:
        failures.append('text_outside_page_bounds')
    if replacement:
        failures.append('replacement_or_null_character')
    if missing:
        failures.append('character_absent_from_audit_font_map')
    if unresolved:
        failures.append('unresolved_pdf_character_mapping')
    body_chars = sum(len(c['text'].strip()) for c in chars if float(c['bottom']) < height-55)
    return {'page':page.page_number, 'width_points':width, 'height_points':height,
            'text_char_count':len(text.strip()), 'visible_text_char_count':sum(len(c['text']) for c in chars),
            'body_text_char_count':body_chars, 'fonts':sorted(fonts),
            'out_of_bounds_character_count':len(outside), 'out_of_bounds_examples':outside[:30],
            'replacement_codepoints':sorted(replacement),
            'missing_font_characters':[{'character':a,'codepoint':b,'font':c} for a,b,c in sorted(missing)],
            'unresolved_character_ids':unresolved, 'failures':failures,
            'warnings':['sparse_body_text_requires_visual_review'] if body_chars < 50 else []}


class Renderer:
    def __init__(self, pdf, requested, dpi):
        self.pdf = Path(pdf)
        self.dpi = dpi
        self.document = None
        self.executable = None
        if requested in ('auto', 'pdfium'):
            try:
                import pypdfium2
                self.document = pypdfium2.PdfDocument(str(pdf))
                self.name = 'pypdfium2'
                self.version = version('pypdfium2')
                return
            except ImportError:
                if requested == 'pdfium':
                    raise RuntimeError('pypdfium2 is not installed; install requirements-documents.txt.')
        self.executable = shutil.which('pdftoppm')
        if not self.executable:
            raise RuntimeError('No PDF renderer: install pypdfium2 or put Poppler pdftoppm on PATH.')
        self.name = 'poppler_pdftoppm'
        info = subprocess.run([self.executable,'-v'], capture_output=True, text=True, errors='replace')
        self.version = (info.stderr or info.stdout).splitlines()[0] if (info.stderr or info.stdout) else None

    def render(self, index, destination):
        if self.document is not None:
            page = self.document[index]
            bitmap = None
            try:
                bitmap = page.render(scale=self.dpi/72)
                bitmap.to_pil().convert('RGB').save(destination)
            finally:
                if bitmap is not None:
                    bitmap.close()
                page.close()
        else:
            prefix = destination.with_suffix('')
            result = subprocess.run([self.executable,'-f',str(index+1),'-l',str(index+1),
                                     '-r',str(self.dpi),'-png','-singlefile',str(self.pdf),str(prefix)],
                                    capture_output=True, text=True, errors='replace')
            if result.returncode:
                raise RuntimeError('Poppler render failed: '+result.stderr[-1500:])
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError('Renderer did not produce a nonempty page PNG.')
        with Image.open(destination) as image:
            image.load()
            rgb = image.convert('RGB')
            extrema = rgb.getextrema()
            blank = all(low==high==255 for low,high in extrema)
            stats = ImageStat.Stat(ImageChops.difference(rgb, Image.new('RGB',rgb.size,'white')))
            mean_ink = sum(stats.mean)/3
            return {'png':str(destination), 'png_sha256':sha256(destination),
                    'pixel_width':image.width, 'pixel_height':image.height,
                    'mean_nonwhite_intensity':mean_ink, 'blank_white_raster':blank}

    def close(self):
        if self.document is not None:
            self.document.close()


def contacts(pages, directory):
    rendered = [page for page in pages if page.get('render',{}).get('png')]
    paths = []
    columns, rows, tile_width, tile_height = 3, 3, 440, 665
    for start in range(0,len(rendered),columns*rows):
        sheet = Image.new('RGB',(columns*tile_width, rows*tile_height+46),'#d0d6de')
        draw = ImageDraw.Draw(sheet)
        draw.text((12,10),'AUTOMATED RENDERS - MANUAL VISUAL REVIEW NOT PERFORMED BY THIS TOOL', fill='#172b44')
        for offset, page in enumerate(rendered[start:start+columns*rows]):
            left, top = (offset%columns)*tile_width, (offset//columns)*tile_height+46
            state = 'PASS' if not page['failures'] else 'FAIL'
            draw.text((left+10,top+6), f"Page {page['page']} | automated checks {state}",fill='#172b44')
            with Image.open(page['render']['png']) as source:
                image = source.convert('RGB')
                image.thumbnail((tile_width-20,tile_height-32), Image.Resampling.LANCZOS)
                sheet.paste(image,(left+(tile_width-image.width)//2,top+28))
        path = directory/f'contact-{start//(columns*rows)+1:02d}.png'
        sheet.save(path)
        paths.append({'path':str(path), 'sha256':sha256(path),
                      'pages':[page['page'] for page in rendered[start:start+columns*rows]]})
    return paths


def audit(root, pdf, workdir, output, dpi=110, tolerance=0.5, renderer_name='auto', font=None, bold_font=None):
    root,pdf,workdir,output = (Path(p).resolve() for p in (root,pdf,workdir,output))
    report = {'schema_version':1, 'created_utc':timestamp(), 'status':'fail',
              'automatic_checks_passed':False, 'pdf':str(pdf), 'pdf_sha256':None,
              'page_count':0, 'rendered_page_count':0, 'pages':[], 'contact_sheets':[],
              'automatic_scope':['nonempty extracted text on every page',
                  'extracted character bounding boxes inside physical page',
                  'no Unicode replacement/null or unresolved CID character mappings',
                  'extracted characters present in supplied Malgun font maps',
                  'every page rendered to nonempty, nonwhite PNG'],
              'limitations':['Automatic pass is not manual visual approval.',
                  'This tool does not establish scientific correctness, table readability, text/image overlap, or aesthetic quality.',
                  'Font-map checks cover extracted PDF text, not glyphs or clipping already rasterized inside figures.',
                  'Font-map coverage cannot by itself prove every embedded-font glyph was drawn correctly.'],
              'manual_visual_review':{'performed':False, 'status':'not_performed_by_this_tool',
                  'instruction':'A person or a separate visual reviewer must inspect the rendered pages; do not infer approval from this JSON.'},
              'parameters':{'dpi':dpi,'text_bounds_tolerance_points':tolerance},
              'software':{p:version(p) for p in ('pdfplumber','pypdfium2','Pillow','reportlab')},
              'errors':[], 'warnings':[]}
    renderer = None
    try:
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise FileNotFoundError('PDF is missing or empty: '+str(pdf))
        report['pdf_sha256'] = sha256(pdf)
        report['pdf_size_bytes'] = pdf.stat().st_size
        directory = workdir/report['pdf_sha256'][:16]
        directory.mkdir(parents=True, exist_ok=True)
        report['render_directory'] = str(directory)
        font_maps, font_metadata = load_font_maps([
            font or os.environ.get('FONT_KO','C:/Windows/Fonts/malgun.ttf'),
            bold_font or os.environ.get('FONT_KO_BOLD','C:/Windows/Fonts/malgunbd.ttf')])
        report['audit_fonts'] = font_metadata
        renderer = Renderer(pdf,renderer_name,dpi)
        report['renderer'] = {'name':renderer.name,'version':renderer.version}
        with pdfplumber.open(pdf) as document:
            report['page_count'] = len(document.pages)
            if not document.pages:
                raise ValueError('PDF contains no pages.')
            for index, page in enumerate(document.pages):
                try:
                    row = inspect_page(page,font_maps,tolerance)
                except Exception as error:
                    row = {'page':index+1,'failures':['text_inspection_error'],
                           'text_inspection_error':f'{type(error).__name__}: {error}', 'warnings':[]}
                try:
                    row['render'] = renderer.render(index,directory/f'page-{index+1:03d}.png')
                    report['rendered_page_count'] += 1
                    if row['render']['blank_white_raster']:
                        row['failures'].append('empty_white_render')
                except Exception as error:
                    row['failures'].append('page_render_error')
                    row['render_error'] = f'{type(error).__name__}: {error}'
                row['automatic_checks_passed'] = not row['failures']
                report['pages'].append(row)
                print(f"Page {index+1}/{report['page_count']}: {'pass' if not row['failures'] else 'FAIL'}",flush=True)
        report['contact_sheets'] = contacts(report['pages'],directory)
        if sha256(pdf) != report['pdf_sha256']:
            report['errors'].append('PDF changed during audit; regenerate QA for the stable artifact.')
        failures = [page['page'] for page in report['pages'] if page['failures']]
        if failures:
            report['errors'].append('Page checks failed: '+', '.join(map(str,failures)))
        if report['rendered_page_count'] != report['page_count']:
            report['errors'].append('Not every actual PDF page was rendered.')
        report['warnings'] = [{'page':page['page'],'warnings':page['warnings']}
                              for page in report['pages'] if page['warnings']]
        report['automatic_checks_passed'] = not report['errors']
        report['status'] = 'pass' if report['automatic_checks_passed'] else 'fail'
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    finally:
        if renderer is not None:
            renderer.close()
        report['finished_utc'] = timestamp()
        atomic_json(output,report)
    print(f"Automated PDF QA: {report['status']}; {report['rendered_page_count']}/{report['page_count']} pages rendered; manual visual review not performed by this tool.")
    print(str(output))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--pdf',type=Path,help='Default: ROOT/FULL_PROJECT_REPORT_KO.pdf')
    parser.add_argument('--output',type=Path,help='Default: ROOT/FULL_REPORT_LAYOUT_QA.json')
    parser.add_argument('--workdir',type=Path,help='Scratch output for every page PNG and contact sheet; keep outside archive')
    parser.add_argument('--dpi',type=int,default=110)
    parser.add_argument('--tolerance',type=float,default=.5,help='Allowed physical page-boundary rounding in points')
    parser.add_argument('--renderer',choices=['auto','pdfium','poppler'],default='auto')
    parser.add_argument('--font',type=Path,help='Regular Malgun TrueType font, or FONT_KO')
    parser.add_argument('--bold-font',type=Path,help='Bold Malgun TrueType font, or FONT_KO_BOLD')
    args=parser.parse_args()
    if not 72 <= args.dpi <= 300 or not 0 <= args.tolerance <= 2:
        parser.error('dpi must be 72..300 and tolerance must be 0..2 points')
    root=args.root.resolve()
    report=audit(root, args.pdf or root/'FULL_PROJECT_REPORT_KO.pdf',
        args.workdir or root.parents[1]/'work/full_study/report_layout_qa',
        args.output or root/'FULL_REPORT_LAYOUT_QA.json',args.dpi,args.tolerance,args.renderer,
        args.font,args.bold_font)
    return 0 if report['automatic_checks_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
