"""Korean academic PDF/Markdown authoring with editable XeLaTeX source.

Compile locally with XeLaTeX or, on Windows, the Ubuntu WSL installation.
No shell interpolation is used. Figures and numerical results are read only.
"""
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def escape(value):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
                    '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}',
                    '^':r'\textasciicircum{}'}.get(c,c) for c in str(value))


SYMBOLS = {'σxx':r'\sigma_{xx}','σyy':r'\sigma_{yy}','σxy':r'\sigma_{xy}',
           'εzz':r'\varepsilon_{zz}','T0':r'T_0','ux':r'u_x','uy':r'u_y',
           'sxx':r'\sigma_{xx}','syy':r'\sigma_{yy}','sxy':r'\sigma_{xy}',
           'qx':r'q_x','qy':r'q_y','L2':r'L^2',
           'σ':r'\boldsymbol{\sigma}','ε':r'\boldsymbol{\varepsilon}',
           'Ω':r'\Omega','α':r'\alpha','ν':r'\nu','μ':r'\mu','λ':r'\lambda',
           '∇':r'\nabla','²':r'{}^2','×':r'\times','∫':r'\int','∂':r'\partial'}
TOKEN = re.compile(r'(?<![A-Za-z_])(?:ux|uy|sxx|syy|sxy|qx|qy|T0|L2)(?![A-Za-z_])|σxx|σyy|σxy|εzz|[σεΩανμλ∇²×∫∂]')


def plain_tex(text):
    text=html.unescape(text).replace('−','-').replace('–','-').replace('—','-').replace('‑','-')
    pieces=[]; end=0
    # File identifiers and SHA digests must be allowed to wrap in table columns.
    path_pattern=r'(?:[A-Za-z0-9_*.-]+[/\\])+[A-Za-z0-9_*./\\-]+|[A-Za-z0-9_.-]+\.(?:json|py|npz|ipynb|md|pdf|pt|csv|html|tex)|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[0-9a-f]{40,64}'
    for m in re.finditer(path_pattern,text):
        pieces.append(symbol_tex(text[end:m.start()]))
        pieces.append(r'\nolinkurl{'+m.group().replace('\\','/')+'}')
        end=m.end()
    pieces.append(symbol_tex(text[end:]))
    return ''.join(pieces)


def symbol_tex(text):
    pieces=[]; end=0
    for m in TOKEN.finditer(text):
        pieces.extend([escape(text[end:m.start()]),'$'+SYMBOLS[m.group()]+'$']); end=m.end()
    pieces.append(escape(text[end:])); return ''.join(pieces)


def tex_text(text):
    """Convert the report's small HTML vocabulary; preserve explicit $math$."""
    text=str(text)
    pattern=r'(\$[^$]+\$|<link\s+href="[^"]+"[^>]*>.*?</link>|<b>.*?</b>|<(?:sub|super)>.*?</(?:sub|super)>|<br\s*/?>)'
    output=[]
    for part in re.split(pattern,text,flags=re.S):
        if not part: continue
        if part.startswith('$'): output.append(part)
        elif part.startswith('<link'):
            m=re.match(r'<link\s+href="([^"]+)"[^>]*>(.*?)</link>',part,re.S)
            output.append(r'\href{'+m[1].replace('%',r'\%')+'}{'+tex_text(m[2])+'}')
        elif part.startswith('<b>'): output.append(r'\textbf{'+tex_text(part[3:-4])+'}')
        elif part.startswith('<sub>'): output.append('$_{'+escape(part[5:-6])+'}$')
        elif part.startswith('<super>'): output.append('$^{'+escape(part[7:-8])+'}$')
        elif part.startswith('<br'): output.append(r'\newline ')
        else: output.append(plain_tex(part))
    return ''.join(output)


def markdown_text(text):
    text=str(text).replace('<b>','**').replace('</b>','**').replace('<br/>','\n')
    text=re.sub(r'<link href="([^"]+)"[^>]*>(.*?)</link>',r'[\2](\1)',text)
    text=re.sub(r'<sub>(.*?)</sub>',r'$_{\1}$',text)
    text=re.sub(r'<super>(.*?)</super>',r'$^{\1}$',text)
    pieces=re.split(r'(\$[^$]+\$|\[[^]]+\]\([^)]+\)|[A-Za-z0-9_./*-]+\.(?:json|py|npz|ipynb|md|pdf|pt|csv|tex))',text)
    return ''.join(p if p.startswith(('$','[')) or re.search(r'\.(?:json|py|npz|ipynb|md|pdf|pt|csv|tex)$',p)
                   else TOKEN.sub(lambda m:'$'+SYMBOLS[m.group()]+'$',p) for p in pieces)


PREAMBLE = r'''\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,left=18mm,right=18mm,top=22mm,bottom=22mm,headheight=13pt]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}[Scale=0.86,BoldFont={Latin Modern Mono}]
\usepackage{amsmath,amssymb,unicode-math}
\setmathfont{Latin Modern Math}
\setmathfont{Latin Modern Math}[version=bold]
\renewcommand{\boldsymbol}[1]{\symbfit{#1}}
\usepackage{kotex}
\IfFontExistsTF{UnBatang}{\setmainhangulfont{UnBatang}\setsanshangulfont{UnDotum}}{}
\usepackage{graphicx,booktabs,longtable,array,caption,needspace}
\usepackage[table]{xcolor}
\usepackage{xurl,hyperref,fancyhdr,titlesec,tocloft,fvextra}
\definecolor{ink}{HTML}{172B44}
\definecolor{accent}{HTML}{287B7C}
\definecolor{muted}{HTML}{58616D}
\definecolor{tint}{HTML}{E8EDF3}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=accent,bookmarksnumbered=false,pdfauthor={AM Project Reproduce}}
\urlstyle{tt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\linespread{1.22}
\setlength{\emergencystretch}{3em}
\setcounter{secnumdepth}{0}
\setcounter{tocdepth}{1}
\setlength{\cftbeforesecskip}{3pt}
\renewcommand{\cftsecfont}{\small}
\renewcommand{\cftsecpagefont}{\small}
\setlength{\cftbeforetoctitleskip}{0pt}
\titleformat{\section}{\large\bfseries\sffamily\color{ink}}{}{0pt}{}
\titleformat{\subsection}{\normalsize\bfseries\sffamily\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{10pt}{9pt}
\titlespacing*{\subsection}{0pt}{10pt}{5pt}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false,skip=5pt,hypcap=false}
\renewcommand{\figurename}{그림}
\renewcommand{\tablename}{표}
\renewcommand{\contentsname}{목차}
\setlength{\LTpre}{4pt}\setlength{\LTpost}{8pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.25}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\footnotesize\sffamily\color{muted} AM PROJECT / @HEADER@}
\fancyfoot[L]{\scriptsize\sffamily\color{muted} @STATUS@}
\fancyfoot[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0.3pt}
\begin{document}
'''


def compile_tex(root, stem):
    root=Path(root).resolve(); build=root/'tmp/latex'/stem; build.mkdir(parents=True,exist_ok=True)
    executable=shutil.which('xelatex')
    args=['-interaction=nonstopmode','-halt-on-error','-file-line-error',
          '-output-directory=tmp/latex/'+stem,stem+'.tex']
    if executable: cmd=[executable,*args]
    elif os.name=='nt' and shutil.which('wsl'):
        cmd=['wsl','-d',os.environ.get('HARANDI_WSL_DISTRO','Ubuntu'),'--cd',str(root),'--','xelatex',*args]
    else: raise RuntimeError('Install XeLaTeX and Korean support: see docs/LATEX_REPORTS_KO.md')
    for iteration in range(3):
        proc=subprocess.run(cmd,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (build/f'compile-{iteration+1}.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
        if proc.returncode: raise RuntimeError('XeLaTeX failed: '+(proc.stdout+proc.stderr)[-5000:])
    log=(build/(stem+'.log')).read_text(encoding='utf-8',errors='replace')
    missing=[line for line in log.splitlines() if 'Missing character:' in line]
    if missing: raise RuntimeError('LaTeX font coverage failure: '+'; '.join(missing[:10]))
    warnings=[line for line in log.splitlines() if 'Overfull' in line or 'Warning:' in line]
    shutil.copy2(build/(stem+'.pdf'),root/(stem+'.pdf'))
    qa={'engine':'XeLaTeX','passes':3,'missing_characters':missing,'warnings':warnings,
        'tex_sha256':hashlib.sha256((root/(stem+'.tex')).read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256((root/(stem+'.pdf')).read_bytes()).hexdigest()}
    (root/(stem+'_TEX_BUILD.json')).write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    return root/(stem+'.pdf')


class LatexReport:
    def __init__(self,root,partial=False,complete=19,total=19,stem='FULL_PROJECT_REPORT_KO',
                 header='열탄성 혼합 물리정보 신경망의 수치 재현',status=None):
        self.root=Path(root); self.partial=partial; self.complete=complete; self.total=total
        self.stem=stem; self.header=header
        self.status=status or f'{"진행 보고서" if partial else "등록 실험 완료 보고서"} | 검증 완료 {complete}/{total}'
        self.tex=[]; self.md=[]; self.figures=[]; self.table_number=0; self.equation_number=0; self.last_heading='연구 개요'

    def p(self,text,style='body'):
        if style=='equation': raise ValueError('Use equation() with explicit LaTeX source.')
        value=tex_text(text)
        if style=='title':
            self.tex.append(r'{\LARGE\bfseries\sffamily\color{ink} '+value+r'\par}\vspace{5mm}')
        elif style=='small': self.tex.append(r'{\small\color{muted} '+value+r'\par}')
        else: self.tex.append(value+r'\par')
        self.md.append(('# ' if style=='title' else '')+markdown_text(text)+'\n')

    def equation(self,latex):
        self.equation_number+=1
        self.tex.append(r'\begin{equation}'+latex+r'\end{equation}')
        self.md.append('$$\n'+latex+'\n\\tag{'+str(self.equation_number)+'}\n$$\n')

    def h(self,text,level=1):
        self.last_heading=re.sub(r'^\d+(?:\.\d+)*\.?\s*','',text)
        command='section' if level==1 else 'subsection'
        if text=='보고서 읽기와 목차': command='section*'
        self.tex.append('\\'+command+'{'+tex_text(text)+'}')
        self.md.append('#'*(level+1)+' '+markdown_text(text)+'\n')

    def page(self): self.tex.append(r'\clearpage'); self.md.append('\n---\n')

    def toc(self): self.tex.append(r'\tableofcontents\thispagestyle{fancy}'); self.md.append('목차는 PDF에서 자동 생성된다.\n')

    def table(self,headers,rows,widths=None,numeric=False,caption=None):
        if not rows:
            self.p('검증 완료 자료가 없어 수치 표를 작성하지 않는다. 해당 실행은 대기 목록에 남긴다.','small'); return
        self.table_number+=1; widths=widths or [1/len(headers)]*len(headers)
        title=caption or self.last_heading
        cols=''.join(r'>{\raggedright\arraybackslash}p{\dimexpr '+str(w)+r'\linewidth-2\tabcolsep\relax}' for w in widths)
        dense=numeric or len(rows)>14
        self.tex.append(r'\begingroup'+(r'\fontsize{8}{10.5}\selectfont\renewcommand{\arraystretch}{1.1}' if dense else r'\small'))
        heading=r'\rowcolor{tint} '+' & '.join(r'\textbf{'+tex_text(v)+'}' for v in headers)+r'\\\midrule'
        short=len(rows)<=14
        if short:
            self.tex.append(r'\par\begin{minipage}{\linewidth}\captionof{table}{'+tex_text(title)+r'}\begin{tabular}{'+cols+'}')
            self.tex.append(heading)
        else:
            self.tex.append(r'\begin{longtable}{'+cols+'}')
            self.tex.append(r'\caption{'+tex_text(title)+r'}\\')
            self.tex.extend([heading+r'\endfirsthead',heading+r'\endhead',r'\bottomrule\endfoot'])
        for row in rows:
            self.tex.append(' & '.join(tex_text(v) for v in row)+(r'\\\addlinespace[1pt]' if dense else r'\\\addlinespace[2pt]'))
        self.tex.append((r'\bottomrule\end{tabular}\end{minipage}\par\vspace{5pt}' if short else r'\end{longtable}')+r'\endgroup')
        self.md.extend(['**표 '+str(self.table_number)+'. '+markdown_text(title)+'**\n',
            '| '+' | '.join(markdown_text(v) for v in headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
            '\n'.join('| '+' | '.join(markdown_text(v).replace('|','/') for v in row)+' |' for row in rows)+'\n'])

    def figure(self,path,caption,maxheight_mm=150,identifier=None):
        path=Path(path); relative=path.relative_to(self.root).as_posix() if path.is_absolute() else path.as_posix()
        if not (self.root/relative).exists(): raise FileNotFoundError(relative)
        title=re.sub(r'^그림\s*\d+\.\s*','',caption)
        # A minipage keeps a figure and its caption together, without float reordering.
        self.tex.append(r'\par\begin{minipage}{\linewidth}\centering'+'\n'+
            r'\includegraphics[width=\linewidth,height='+str(maxheight_mm)+r'mm,keepaspectratio]{\detokenize{'+relative+'}}\n'+
            r'\captionof{figure}{'+tex_text(title)+r'}\end{minipage}\par')
        self.figures.append(identifier or relative)
        self.md.append(f'![그림 {len(self.figures)}. {title}]({relative})\n')

    def fig(self,identifier,caption,manifest,maxheight=425.2):
        item=next((v for v in manifest.get('figures',[]) if v['id']==identifier),None)
        if item is None or not (self.root/item['png']).exists():
            self.p('그림 대기: '+caption+' 완료 실행과 기준해가 준비되면 추가된다.','small'); return
        self.figure(item['png'],caption,maxheight*25.4/72,identifier)

    def code(self,text):
        self.tex.append('\\begin{Verbatim}[fontsize=\\small,breaklines=true,breakanywhere=true]\n'+text+'\n\\end{Verbatim}')
        self.md.append('```text\n'+text+'\n```\n')

    def build(self):
        pre=PREAMBLE.replace('@HEADER@',tex_text(self.header)).replace('@STATUS@',tex_text(self.status))
        (self.root/(self.stem+'.tex')).write_text(pre+'\n\n'.join(self.tex)+'\n\\end{document}\n',encoding='utf-8')
        (self.root/(self.stem+'.md')).write_text('\n'.join(self.md),encoding='utf-8')
        return compile_tex(self.root,self.stem)
