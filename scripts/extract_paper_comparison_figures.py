"""Extract complete original figure panels from the user-supplied Harandi PDF.

Only page whitespace/body text outside each figure is removed. Plot content,
annotations, axes and colour bars are preserved; there is no digitisation or
recolouring. Third-party figures retain their original CC BY-NC-ND licence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'figures/paper_reference'
# PDF points, measured from the top-left of the original page.
REGIONS = [
    ('5', 13, (44, 44, 550, 207)),
    ('6', 14, (90, 44, 505, 490)),
    ('7', 15, (50, 44, 545, 205)),
    ('8', 16, (70, 45, 525, 704)),
    ('9', 17, (45, 45, 550, 485)),
    ('10', 18, (90, 45, 505, 494)),
    ('11', 19, (45, 44, 550, 487)),
    ('12', 20, (44, 44, 550, 208)),
    ('13', 21, (76, 57, 519, 687)),
    ('14', 22, (48, 45, 547, 472)),
    ('15', 23, (53, 64, 541, 237)),
    ('16', 23, (70, 302, 525, 692)),
    ('17', 24, (45, 63, 550, 382)),
    ('18', 24, (108, 455, 490, 682)),
    ('19', 25, (57, 45, 537, 496)),
    ('20', 26, (92, 485, 505, 688)),
    ('21', 27, (85, 60, 525, 267)),
    ('23', 28, (74, 190, 522, 692)),
    ('24', 29, (74, 190, 522, 692)),
    ('25', 30, (50, 255, 545, 683)),
    ('A1', 35, (45, 44, 553, 491)),
    ('B1', 36, (81, 57, 513, 687)),
    ('B2', 37, (76, 45, 518, 424)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', type=Path, help='Original 37-page Harandi paper')
    parser.add_argument('--dpi', type=int, default=240)
    args = parser.parse_args()
    doc = pdfium.PdfDocument(args.pdf)
    assert len(doc) == 37, 'Crop coordinates refer to the 37-page Wiley PDF.'
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest = {
        'source_title': 'Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains',
        'authors': 'Ali Harandi, Ahmad Moeineddin, Michael Kaliske, Stefanie Reese, Shahed Rezaei',
        'doi': 'https://doi.org/10.1002/nme.7388',
        'source_filename': args.pdf.name,
        'source_pdf_sha256': hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        'copyright': '© 2023 The Authors. Published by John Wiley & Sons Ltd.',
        'license': 'CC BY-NC-ND 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-nc-nd/4.0/',
        'scope': 'Third-party original figure extracts; not project-generated results and not covered by a project code licence.',
        'processing': 'Rasterised original PDF at stated dpi; complete figure area extracted without changing plot content; proportional display scaling only.',
        'coordinate_system': 'PDF points from top-left; x0, top, x1, bottom',
        'dpi': args.dpi,
        'figures': [],
    }
    for fid, pn, box in REGIONS:
        page = doc[pn - 1]
        width, height = page.get_size()
        assert abs(width - 595) < 2 and abs(height - 782) < 2, (width, height)
        rendered = page.render(scale=args.dpi / 72).to_pil().convert('RGB')
        sx, sy = rendered.width / width, rendered.height / height
        pixel_box = tuple(round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box))
        path = TARGET / f'paper_fig{fid}.png'
        rendered.crop(pixel_box).save(path, optimize=True)
        manifest['figures'].append({
            'paper_figure': fid, 'paper_page': pn, 'bbox_points': list(box),
            'path': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pixel_dimensions': [pixel_box[2] - pixel_box[0], pixel_box[3] - pixel_box[1]],
        })
    (TARGET / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'original_figures': len(manifest['figures']), 'dpi': args.dpi, 'source_sha256': manifest['source_pdf_sha256']}))


if __name__ == '__main__':
    main()
