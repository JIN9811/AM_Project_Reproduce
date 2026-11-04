"""Build a new Korean academic full-study PDF and editable Markdown.

Use the bundled document Python. Completion is independently verified by the
project's torch Python; --partial produces an explicitly labeled progress report.
No training, figure generation, or changes to earlier reports are performed.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from reportlab.lib.units import mm

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('ux','uy','sxx','syy','sxy','T','qx','qy')
WIDTH = 174*mm


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(text):
    return (str(text).replace('−','-').replace('–','-').replace('—','-').replace('‑','-')
            .replace('ᵀ','<super>T</super>').replace('₀','<sub>0</sub>'))


def number(value, percent=False):
    if value is None or not np.isfinite(value):
        return '-'
    return f'{value*100:.3g}' if percent else f'{value:.4g}'


def budget(job):
    c=job['config']
    if job['kind']=='transfer': return len(c['ratios'])*c['steps']
    factor=1 if c['mode']=='coupled' else 2
    return factor*(c['steps'] if job['kind']=='baseline' else c['adam_steps']+c['lbfgs_steps'])


def find_pinn_python(root, chosen=None):
    candidates=[chosen, os.environ.get('HARANDI_PYTHON'),
                root/'.venv/Scripts/python.exe', root/'.venv/bin/python',
                root.parents[1]/'work/pinn-env/Scripts/python.exe', sys.executable]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            check=subprocess.run([str(candidate),'-c','import torch'],capture_output=True)
            if check.returncode==0: return str(candidate)
    raise RuntimeError('Set --pinn-python or HARANDI_PYTHON to the project Python with torch installed.')


def verify_state(root, python):
    source=('import json; from harandi.full_study import catalogue,job_complete; '
            'from pathlib import Path; jobs=catalogue(); '
            'print(json.dumps({"catalogue":jobs,"complete":[j["id"] for j in jobs '
            'if job_complete(j,Path("results/full_study"))]}))')
    result=subprocess.run([python,'-c',source],cwd=root,capture_output=True,text=True,
                          encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    if result.returncode: raise RuntimeError('Completion verification failed: '+result.stderr[-3000:])
    return json.loads(result.stdout)


def metrics(pred,ref):
    result={}
    for j,f in enumerate(FIELDS):
        error=np.abs(pred[:,j]-ref[:,j]); norm=np.linalg.norm(ref[:,j]); peak=np.abs(ref[:,j]).max()
        result[f]={'relative_l2':float(np.linalg.norm(error)/norm) if norm else None,
                   'mean_absolute_error':float(error.mean()),'max_absolute_error':float(error.max()),
                   'mean_peak_percent':float(error.mean()/peak*100) if peak else None,
                   'max_peak_percent':float(error.max()/peak*100) if peak else None}
    return result


def load_results(root,specs,complete,manifest):
    results={}; pending=[]
    for job in specs:
        if job['id'] not in complete or job['kind']=='timing': continue
        folder=root/'results/full_study'/job['id']
        reference=(root/'results/full_study/parametric_fem/ratio_15/solution.npz' if job['kind']=='transfer'
                   else root/'results'/f"fem_{job['config']['geometry']}_n200/solution.npz")
        if not reference.exists(): pending.append(str(reference.relative_to(root))); continue
        with np.load(folder/'prediction.npz') as p,np.load(reference) as r:
            xy=p['xy'].copy(); direct=p['fields'].copy(); derived=direct.copy()
            derived[:,2:5]=p['constitutive_stress']; derived[:,6:]=p['fourier_flux']
            a=np.lexsort((xy[:,0],xy[:,1])); b=np.lexsort((r['xy'][:,0],r['xy'][:,1]))
            if xy.shape!=r['xy'].shape or not np.allclose(xy[a],r['xy'][b],atol=1e-12,rtol=0):
                raise ValueError('Reference grid mismatch: '+job['id'])
            ref=np.empty_like(r['fields']); ref[a]=r['fields'][b]
        if not all(np.isfinite(v).all() for v in (direct,derived,ref)): raise ValueError('Nonfinite result: '+job['id'])
        meta=read_json(reference.parent/'metadata.json')
        expected_ratio=15 if job['kind']=='transfer' else None
        expected_geom='rectangles' if expected_ratio else job['config']['geometry']
        if meta['problem']['geometry']!=expected_geom or meta['problem']['ratio']!=expected_ratio:
            raise ValueError('Reference BVP mismatch: '+job['id'])
        x,y=xy.T
        boundary={'ux_left':float(np.abs(direct[x==0,0]).max()),
                  'T_left':float(np.abs(direct[x==0,5]-1).max()),
                  'normal_u':float(max(np.abs(direct[(x==0)|(x==1),0]).max(),np.abs(direct[(y==0)|(y==1),1]).max())),
                  'temperature':float(max(np.abs(direct[x==0,5]-1).max(),np.abs(direct[x==1,5]).max()))}
        cached=manifest.get('jobs',{}).get(job['id'],{})
        if cached and cached.get('prediction_sha256')!=sha(folder/'prediction.npz'):
            raise ValueError('Figure manifest is stale for '+job['id']+'; regenerate harandi.full_figures.')
        results[job['id']]={'direct':metrics(direct,ref),'derived':metrics(derived,ref),
            'boundary':boundary,'reference':str(reference.relative_to(root)),'reference_meta':meta,
            'prediction_sha256':sha(folder/'prediction.npz'),'reference_sha256':sha(reference),
            'status':read_json(folder/'experiment.json') if (folder/'experiment.json').exists() else read_json(folder/'run_status.json')}
    return results,sorted(set(pending))


from latex_report import LatexReport as Report

FIGURE_NAMES={1:'동시·순차 학습 절차',2:'기하와 재료',3:'경계조건',4:'혼합 네트워크와 손실',
 5:'원형의 학습·평가·FEM 격자',6:'원형 동시 손실 이력',7:'원형 변형 형상',8:'원형 역학 전체장',9:'원형 열 전체장',
 10:'원형 순차 손실 이력',11:'원형 세 방법·두 학습 방식 단면',12:'사각형의 학습·평가·FEM 격자',
 13:'사각형 역학 전체장',14:'사각형 열 전체장',15:'왼쪽 경계점 보강',16:'사각형 동시·순차 단면',
 17:'하드 경계와 최적화기',18:'하드 Adam·L-BFGS 손실',19:'선택한 구조의 오차장',20:'깊이별 오차',21:'너비별 오차',
 22:'매개변수·전이학습 절차',23:'비율 15 물리+데이터',24:'비율 15 데이터만',25:'비율 15 단면',
 'A1':'FEM 격자 수렴','B1':'하드 L-BFGS 역학 전체장','B2':'하드 L-BFGS 열 전체장'}
ISSUES={
 'C01':('길이 단위','도해의 μm와 본문/축의 mm가 충돌한다. 수치 단위 영역임을 명시한다.'),
 'C02':('열 상수 누락','α와 T0의 수치가 표에 없다. α=1, T0=0은 저자 코드에 따른 재구성 선택이다.'),
 'C03':('좌표 미기재','원형 중심과 사각형 원점은 완전히 치수화되지 않았다. 도해 추정/저자 자료를 구분한다.'),
 'C04':('FEM/PINN 도해 기하','그림 12의 좌상단 개재물 위치가 패널마다 달라 보인다. 같은 기하라고 단정하지 않는다.'),
 'C05':('Adam 예산 충돌','Table 1의 100k와 Remark 7의 1M을 분리하여 주 실행과 추가 연속 실행으로 남긴다.'),
 'C06':('시간 산술 불일치','Table 4의 0.3409초×10k는 약 56.8분이며 본문 40분과 다르다. 로컬 비용을 별도 측정한다.'),
 'C07':('그림 20 캡션','본문과 x축은 깊이를 바꾸지만 캡션은 뉴런 수로 적혀 있다. 깊이 1,2,4,5,6을 따른다.'),
 'C08':('그림 25 단면','캡션 x=0.5와 패널/본문 x=0.35가 충돌한다. 표시 단면은 x=0.35이다.'),
 'C09':('학습 방식 순위','동시와 순차가 더 좋다는 상반된 문장이 있다. 실제 성분별 결과를 보고한다.'),
 'C10':('열유속 방향·인용','본문의 qy 오차를 qx로 바꾸지 않는다. 다른 기하를 가리키는 그림 인용도 구분한다.'),
 'C11':('매개변수 입력','그림 22의 matrix/inclusion 비율을 채택한다. 국소 E,k라는 본문과 전역 비율 인코딩을 구분한다.'),
 'C12':('네 가중치 불완전','0,0.1 및 범위 끝 1 외의 값은 미기재다. 0.5는 명시적 추가 선택이다.'),
 'C13':('qy 네트워크 오기','식 20의 qx 반복과 달리 도식은 별도 qy 가지다. 독립 네트워크를 사용한다.'),
 'C14':('Fourier 부호','도식의 양의 부호보다 명시된 방정식 q=-k grad T를 따른다.'),
 'C15':('열변형률 구성식','전변형률과 탄성변형률의 혼용 대신 텐서 구성식으로 일관되게 계산한다.'),
 'C16':('전단 계수',r'텐서 전단과 공학 전단의 계수 차이를 명시하고 $\sigma_{xy}=\mu(u_{x,y}+u_{y,x})$를 사용한다.'),
 'C17':('역학 Neumann 손실','식 56은 0이지만 손실 그림은 0이 아니다. 롤러 접선력 0을 명시한다.'),
 'C18':('비균일점의 적분','균일점의 동일 가중치 설명과 보강점 배치가 충돌한다. 점 평균 에너지와 공간 적분을 구분한다.'),
 'C19':('동결 장의 독립성','정상 열전도는 역학에 의존하지 않는다. 비활성 열손실 증가를 재현하려고 역연성을 만들지 않는다.'),
 'C20':('결론의 L-BFGS 문구','이미 4.3절/부록 B에 있는 L-BFGS를 미래 과제로 쓴 편집상의 충돌이다.'),
 'C21':('하드 실행의 동일성','그림 17과 부록 B의 실행·시드 동일성이 미기재다. 독립 실행 수를 임의로 늘리지 않는다.'),
 'C22':('정성적 FEM 수렴','100에서 거의 수렴이라는 말에 허용오차가 없다. 50/100/200의 정량 차이를 남긴다.')}


def create_report(root,partial,pinn_python):
    manifest_path=root/'FULL_STUDY_RESULTS.json'
    if not manifest_path.exists(): raise FileNotFoundError('First run python -m harandi.full_figures [--partial].')
    manifest=read_json(manifest_path)
    inventory=read_json(root/'PAPER_EXPERIMENT_INVENTORY.json')
    protocol=read_json(root/'FULL_STUDY_PROTOCOL.json')
    state=verify_state(root,pinn_python)
    specs=state['catalogue']; complete=set(state['complete']); pending=[j['id'] for j in specs if j['id'] not in complete]
    if not partial and (pending or manifest.get('status')!='complete' or manifest.get('pending_references')
                        or set(manifest.get('completed_jobs',[]))!=complete or manifest.get('catalogue_jobs')!=len(specs)):
        raise RuntimeError(f'Full completion is not verified: {len(complete)}/{len(specs)} jobs. '
                           'Regenerate full_figures when finished, or use --partial for a progress report.')
    results,pending_refs=load_results(root,specs,complete,manifest)
    if not partial and pending_refs: raise RuntimeError('Missing metric references: '+str(pending_refs))
    codes={j['id']:f'J{i+1:02d}' for i,j in enumerate(specs)}
    jobs={j['id']:j for j in specs}
    r=Report(root,partial,len(complete),len(specs))
    mode='진행 보고서' if partial else '등록된 전체 실험 완료 보고서'
    generated=datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
    r.p('AM PROJECT  |  수치 재현 연구','h2')
    r.p('이질 재료의 열탄성 해석을 위한<br/>혼합 물리정보 신경망의 수치 재현','title')
    r.p('Numerical Reproduction of Mixed Physics-Informed Neural Networks<br/>for Thermoelastic Analysis of Heterogeneous Materials','small')
    r.p('Harandi et al. (2024), DOI 10.1002/nme.7388<br/>'+mode+' · '+generated,'small')
    r.h('국문초록',2)
    abstract='본 연구는 이질 재료의 정상 열탄성 문제를 대상으로 혼합 물리정보 신경망의 수치 재현성과 적용 한계를 검토한다. 원형 및 사각형 개재물에 대해 FEniCS 기준해를 구성하고, 혼합·표준 신경망과 심층 에너지법의 예측을 동일 좌표에서 평가하였다. 경계조건 부과 방식, 학습 순서 및 에너지 항의 영향을 성분별 오차로 분석하였다.'
    if all(name in results for name in ('mixed_circle_coupled','mixed_circle_sequential')):
        a=results['mixed_circle_coupled']['derived']['ux']['relative_l2']*100
        b=results['mixed_circle_sequential']['derived']['ux']['relative_l2']*100
        abstract+=f' 원형 문제의 수평변위 상대 L2 오차는 동시 학습 {a:.2f}%, 순차 학습 {b:.2f}%로, 학습 방식의 우열이 일률적이지 않음을 확인하였다.'
    abstract+=f' 본 보고서는 검증된 {len(results)}개 신경망 실험을 수록하며, 미완료 실험과 원문의 미기재 조건은 별도로 제시한다.'
    r.p(abstract)
    r.p('핵심용어: 열탄성, 이질 재료, 물리정보 신경망, 혼합 정식화, 유한요소법, 수치 재현성','small')
    r.p(f'<b>작성 시점의 검증 완료 작업은 {len(complete)}/{len(specs)}개이다.</b> '
        +(f'미완료 {len(pending)}개 작업은 결과 비교에 넣지 않았다. 이 문서는 전체 학습 완료나 방법의 우세를 주장하지 않는 진행 보고서이다.' if partial
          else '등록된 작업의 완료 기준을 충족했으나, 원문의 미기재 설정을 포함한 비트 단위 동일 재현을 뜻하지는 않는다.'))
    r.table(['구분','등록','검증 완료'],[[label,sum(j['kind']==kind for j in specs),sum(j['kind']==kind and j['id'] in complete for j in specs)]
            for kind,label in [('mixed','혼합 PINN·구조·추가 예산'),('baseline','표준 PINN·DEM 및 민감도'),('transfer','전이학습'),('timing','독립 시간 측정')]], [.58,.21,.21])
    r.page(); r.h('Abstract',2)
    english='This study examines the numerical reproducibility and limitations of mixed physics-informed neural networks for steady-state thermoelastic problems in heterogeneous materials. FEniCS reference solutions are constructed for circular and rectangular inclusions, and mixed and standard neural networks and the deep energy method are evaluated at identical spatial points. Component-wise errors are used to assess boundary enforcement, training order, and energy terms. '
    if all(name in results for name in ('mixed_circle_coupled','mixed_circle_sequential')):
        english+=f'For the circular inclusion, the relative L2 errors in horizontal displacement are {a:.2f}% for coupled training and {b:.2f}% for sequential training, indicating that neither training strategy is uniformly superior. '
    english+=f'This report includes {len(results)} verified neural-network experiments and separately identifies unfinished experiments and settings that are unspecified in the source paper.'
    r.p(english)
    r.p('Keywords: thermoelasticity, heterogeneous materials, physics-informed neural networks, mixed formulation, finite element method, numerical reproducibility','small')
    r.h('기호 및 용어',2)
    r.table(['기호 / 용어','정의'],[['u, T','변위 벡터 및 온도'],['σ, q','Cauchy 응력 텐서 및 열유속 벡터'],['E, ν, α, k','탄성계수, Poisson 비, 열팽창계수, 열전도도'],['FEM / PINN / DEM','유한요소법 / 물리정보 신경망 / 심층 에너지법'],['동시 / 순차 학습','열·역학 손실의 공동 최적화 / 장별 교대 최적화'],['연성 / 강제 경계조건','손실 페널티를 통한 부과(soft) / 출력 변환에 의한 만족(hard)'],['반복 수 / 상대 L2 오차','실제 최적화기 갱신 횟수 / 기준해의 벡터 노름으로 정규화한 오차']],[.30,.70],caption='주요 기호와 수치해석 용어')
    r.p('본 문서는 학술지 게재 논문이 아닌 재현 연구 보고서이다. 국문 공학 논문의 서술 체계를 참고하되, 전체 수치 자료를 확인할 수 있도록 상세 표와 실행 명세를 부록에 수록하였다.','small')
    r.page(); r.h('보고서 읽기와 목차')
    r.p('본문은 물리·수치 방법, 비교의 공정성, 실험별 관측, 재현 및 한계 순으로 구성한다. 부록에는 모든 작업의 정확한 예산·완료 상태, 8개 성분의 오차, 원문 28개 그림과 4개 표의 대응 및 출처 기록을 둔다. 미완료 항목은 숫자 0이 아니라 대기로 표시한다.')
    r.toc()
    r.page(); r.h('1. 서론')
    r.p('이질 재료의 열탄성 해석에서는 재료 경계에서 온도 기울기와 응력의 변화가 커지므로, 변위와 온도의 근사 정확도만으로 수치해의 신뢰성을 평가하기 어렵다. 특히 신경망 기반 해석은 잔차의 구성, 경계조건의 부과 및 최적화 절차에 따라 서로 다른 오차 양상을 나타낼 수 있다. 이에 따라 해의 공간 분포와 물리량별 오차를 함께 검토할 필요가 있다.')
    r.p('Harandi 등(2024)은 응력과 열유속을 독립 출력으로 도입한 혼합 정식화를 제시하였다. 본 연구의 목적은 이 정식화를 FEniCS 유한요소해와 비교하여 재현하고, 원문에서 보고한 개선 경향과 예외가 구현 조건에 따라 어떻게 나타나는지 검토하는 데 있다. 이를 위해 원형·사각형 개재물 문제, 비교 방법, 경계조건 민감도, 신경망 구조 및 재료비 전이학습을 하나의 실행 체계로 구성하였다.')
    r.h('1.1 연구 범위와 재현의 수준',2)
    r.p('원문의 실험 기록 수와 실제 실행 작업 수는 일치하지 않는다. 깊이·너비 연구의 5×40은 중복되며, 부록 B는 하드 L-BFGS 전체장과 연결된다. Table 4의 25개 조합은 100 업데이트의 시간 측정이지 25개 추가 10k 수렴 연구가 아니다. Remark 7의 1M Adam은 Table 1과 구분한 추가 연속 실행이고, 두 DEM 하드 경계 제어는 원문에 없는 공정성 민감도 연구이다.')
    r.p(f'현재 목록은 정확도·물리 비교용 {sum(j["kind"]!="timing" for j in specs)}개와 시간 측정용 {sum(j["kind"]=="timing" for j in specs)}개 작업으로 구성된다. 각 작업의 분류와 예산은 하드코딩한 완료 문구가 아니라 실행 시점의 catalogue에서 읽는다.')
    r.h('1.2 수치 결과의 검증 기준',2)
    r.p('완료 판정은 설정 일치, 실제 업데이트 수, 최종 상태, 101×101 예측 배열의 유한성, 가중치의 유한성 및 모델 구조 적합성을 확인한다. 체크포인트나 그림이 존재하는 것만으로 완료를 인정하지 않는다. 보고서 생성기는 별도의 프로젝트 Python에서 이 검사를 수행한 뒤, 완료 예측의 직접·미분 출력 오차를 NPZ로 다시 계산한다. 부분 결과·짧은 시험·기존 빠른 실행은 전체 예산의 결과로 대체하지 않는다.')
    r.table(['자료','역할'],[['PAPER_EXPERIMENT_INVENTORY.json','원문 본문·그림·표의 감사 목록'],['FULL_STUDY_PROTOCOL.json','재구성 선택과 전체 학습 계획'],['FULL_STUDY_RESULTS.json','검증된 그림·출처 해시·수치 비교'],['results/full_study/*','실제 설정·모델·예측·손실·완료 상태'],['Full_Paper_Reproduction_KO.ipynb','편집 가능한 실행·독립 재계산 절차']],[.46,.54])
    r.p('그림집 생성 시점의 완료 목록과 현재 검증 목록이 다르면, 보고서는 현재 검증 결과와 새로 계산한 오차를 기준으로 한다. 엄격한 최종 보고서는 그림집도 최신 전체 완료 상태여야 생성된다.','small')

    r.page(); r.h('2. 물리적 문제와 경계조건')
    r.p('영역 Ω=[0,1]×[0,1]에서 정상 열전도와 소변형 등방성 열탄성을 계산한다. 열전도 해가 열팽창을 통해 역학에 영향을 주며, 역학이 열전도에 되먹임하는 열원이나 재료 변화는 포함하지 않는다. 3차원 평면변형률 조건 εzz=0을 적용하되 면외 열변형률과 면외 응력을 에너지에서 무시하지 않는다.')
    r.equation('\\nabla\\!\\cdot\\boldsymbol{\\sigma}=\\boldsymbol{0},\\qquad \\nabla\\!\\cdot\\boldsymbol{q}=0,\\qquad \\boldsymbol{q}=-k\\nabla T')
    r.equation('\\boldsymbol{\\varepsilon}(\\boldsymbol{u})=\\tfrac12\\!\\left[\\nabla\\boldsymbol{u}+(\\nabla\\boldsymbol{u})^{\\mathsf T}\\right],\\qquad \\mu=\\frac{E}{2(1+\\nu)},\\qquad \\lambda=\\frac{E\\nu}{(1+\\nu)(1-2\\nu)}')
    r.equation('\\boldsymbol{\\sigma}=2\\mu\\boldsymbol{\\varepsilon}+\\lambda\\operatorname{tr}(\\boldsymbol{\\varepsilon})\\boldsymbol{I}-\\beta(T-T_0)\\boldsymbol{I},\\qquad \\beta=\\frac{E\\alpha}{1-2\\nu}')
    r.p(r'공통 값은 $\nu=0.3$, $\alpha=1$, $T_0=0$이다. 좌우 경계에서 $u_x=0$, 상하에서 $u_y=0$이며 네 변의 접선력은 0이다. 왼쪽 $T=1$, 오른쪽 $T=0$이고 상하는 단열 $q_y=0$이다. 전단응력은 $\sigma_{xy}=\mu(u_{x,y}+u_{y,x})$로 계산한다. $\alpha=1$은 벤치마크 값이며 실제 재료의 열팽창 계수로 해석하지 않는다.')
    r.table(['기하','기지 E,k','개재물 E,k','좌표/재료 선택'],[['원형','0.3, 0.3','1.0, 1.0','중심 (0.60,0.35), 반지름 0.15'],['사각형','1.0, 1.0','0.5, 0.5','저자 제공 재료 자료의 네 사각형'],['전이학습','1.0, 1.0','1/r, 1/r','matrix/inclusion 비율 r']],[.17,.18,.18,.47])
    r.fig('fig02_geometry','그림 1. 구현된 재료장. 원문 단위·좌표의 불명확성을 구분하며, 도해의 모양을 맞추기 위해 예측값을 조정하지 않는다.',manifest,88*mm)
    r.p('원문 도해의 μm와 본문·축의 mm가 충돌한다. 본 프로젝트는 수치 단위 영역을 사용하고 mm, GPa, K, W/mK는 원문 벤치마크 표기임을 밝힌다. 실제 SI 열유속에는 길이·온도 차에 따른 환산이 필요하다.','small')

    r.page(); r.h('3. FEniCS 기준해와 오차 정의')
    r.p('Q1 사각형 유한요소로 열전도 선형 시스템을 먼저 풀고 온도 해를 열하중으로 대입하여 변위를 구한다. 주 기준해는 요소 중심의 불연속 재료 구분을 사용한다. 신경망에는 50×50 재료 격자의 절점 평균·쌍선형 보간 계수가 쓰인다. 두 재료 표현의 차이는 지워서는 안 되며, 동일 절점 재료장의 별도 기준해로도 확인할 수 있다.')
    r.equation('\\begin{aligned}\\int_\\Omega k\\nabla T_h\\!\\cdot\\nabla w_h\\,\\mathrm d\\Omega&=0,\\\\\\int_\\Omega\\!\\left[2\\mu\\boldsymbol{\\varepsilon}(\\boldsymbol{u}_h):\\boldsymbol{\\varepsilon}(\\boldsymbol{v}_h)+\\lambda(\\nabla\\!\\cdot\\boldsymbol{u}_h)(\\nabla\\!\\cdot\\boldsymbol{v}_h)\\right]\\mathrm d\\Omega&=\\int_\\Omega\\beta(T_h-T_0)\\nabla\\!\\cdot\\boldsymbol{v}_h\\,\\mathrm d\\Omega.\\end{aligned}')
    r.p('응력·열유속의 요소 평균값을 보존하고, 비교용 연속 절점값은 명시적인 주변 요소 복원으로 만든다. 국소 경계에서 복원 방식과 재료 불연속은 최대오차에 영향을 준다. 선형 시스템의 잔차가 작다는 사실은 이산 해법 검증이며 공간 이산화 오차가 0이라는 의미는 아니다.')
    femrows=[]
    for geom in ('circle','rectangles'):
        path=root/'results'/f'fem_{geom}_n200/metadata.json'
        if path.exists():
            meta=read_json(path); d=meta['diagnostics']
            femrows.append([geom,meta['n'],number(d['thermal_relative_residual']),number(d['mechanical_relative_residual']),number(d['heat_balance'])])
    r.table(['기하','요소/변','열 잔차','역학 잔차','열수지'],femrows,[.23,.14,.21,.21,.21])
    r.h('평가 기준과 세 가지 다른 상대화',2)
    r.p('본문·부록의 일반 실행 정량 기준은 FEM200이며, 비율 15만 별도 FEM50을 사용한다. 원문 형식 그림은 FEM50을 사용하는 경우가 있으므로 그림 캡션의 기준을 따른다. 모든 신경망 평가는 101×101 좌표이다. 직접 혼합 출력과 변위·온도 미분으로 얻은 구성식/Fourier 출력은 별도로 계산한다.')
    r.equation('\\begin{aligned}e_{L^2}(f)&=\\frac{\\|\\widehat{\\boldsymbol f}-\\boldsymbol f^{\\mathrm{ref}}\\|_2}{\\|\\boldsymbol f^{\\mathrm{ref}}\\|_2},\\qquad e_{\\mathrm{MAE}}(f)=\\frac1N\\sum_{i=1}^{N}|\\widehat f_i-f_i^{\\mathrm{ref}}|,\\\\ e_{\\mathrm{mean,peak}}(f)&=\\frac{e_{\\mathrm{MAE}}(f)}{\\max_i|f_i^{\\mathrm{ref}}|},\\qquad e_{\\mathrm{max,peak}}(f)=\\frac{\\max_i|\\widehat f_i-f_i^{\\mathrm{ref}}|}{\\max_i|f_i^{\\mathrm{ref}}|}.\\end{aligned}')
    r.p('평균 절대오차, 최대 절대오차와 상대 L2는 서로 다른 기준이다. 기준해가 0인 분모는 정의하지 않고 빈값으로 남긴다. 작은 전단·횡열유속 성분은 상대오차가 커질 수 있으므로 절대오차와 함께 읽는다. 본 정의가 원문의 모호한 평균 상대오차 문구와 완전히 동일하다고 단정하지 않는다.')

    r.page(); r.h('4. 혼합 PINN과 학습 절차')
    r.p('8개 독립 스칼라 신경망이 ux, uy, σxx, σyy, σxy, T, qx, qy를 출력한다. 기본은 각각 5개 은닉층·층당 40개 tanh 뉴런이다. 독립 응력·열유속 출력은 평형·보존식의 1차 미분을 가능하게 하며, 구성식 및 Fourier 연결 손실이 주 변수의 미분 결과와 맞춘다.')
    r.p('원형의 기본 배치는 51×51 직교 격자와 시드가 고정된 내부 균등 난수 2,399점으로 총 5,000점이다. 원문 그림 5의 계면·경계 보강점 좌표는 공개되지 않았으므로 그 분포를 그대로 재현하지 않았다. 같은 점 개수가 같은 배치 또는 같은 정확도를 뜻하지 않는다. 각 점의 동일 가중치도 면적에 정확한 적분 공식과 구분한다.','small')
    r.equation('\\begin{aligned}\\mathcal L&=\\mathcal L_{\\mathrm M}+\\mathcal L_{\\mathrm T},\\\\\\mathcal L_{\\mathrm M}&=\\mathcal L_{\\mathrm{link},\\sigma}+\\mathcal L_{\\mathrm{eq}}+\\mathcal L_{\\mathrm D,u}+\\mathcal L_{\\mathrm N,t}+\\mathcal L_{\\mathrm E,M},\\\\\\mathcal L_{\\mathrm T}&=\\mathcal L_{\\mathrm{link},q}+\\mathcal L_{\\mathrm{div},q}+\\mathcal L_{\\mathrm D,T}+\\mathcal L_{\\mathrm N,q}+\\mathcal L_{\\mathrm E,T}.\\end{aligned}')
    r.p(r'아래 첨자 $\mathrm{link}$는 독립 출력과 구성식의 연결, $\mathrm{eq}$는 역학 평형, $\mathrm D$와 $\mathrm N$은 각각 Dirichlet 및 Neumann 경계조건, $\mathrm E$는 일 기반 보조 항을 뜻한다. 에너지 항은 저자 코드의 내부 일과 경계 외부 일의 차를 사용한다. 이 항은 온도를 고정한 열탄성 퍼텐셜 에너지와 동일하지 않으며, 정확해에서 반드시 0이 되는 잔차로 해석해서는 안 된다.')
    r.h('4.1 원저자 구현과 수식 표기의 대응',2)
    r.p('원저자 공개 저장소 phyml4e/PINNs의 Thermomechanic/Thermo_mechanic.py를 대조하였다. 공개 파일은 SciANN/TensorFlow를 사용하고, 좌표를 입력으로 하는 8개 독립 함수와 강제 경계 출력 변환을 정의한다. 본 프로젝트는 같은 물리량 구분을 PyTorch로 구현하되, 논문의 소프트 경계 주 실험과 공개 코드의 하드 경계 실행을 구분한다.')
    r.table(['저자 코드의 변수','보고서 표기','의미'],[
        ['sig_x_O, sig_y_O, sig_xy_O',r'$\widehat\sigma_{xx},\widehat\sigma_{yy},\widehat\sigma_{xy}$','독립 응력 신경망의 출력'],
        ['sigma_x, sigma_y, sigma_xy',r'$\sigma_{xx}(\boldsymbol u,T),\sigma_{yy}(\boldsymbol u,T),\sigma_{xy}(\boldsymbol u)$','변위 미분과 온도에서 계산한 구성식 응력'],
        ['q_x_O, q_y_O',r'$\widehat q_x,\widehat q_y$','독립 열유속 신경망의 출력'],
        ['q_x, q_y','$-kT_{,x},-kT_{,y}$','Fourier 법칙에서 계산한 열유속'],
        ['EQ1_m-EQ3_m; EQ1_t-EQ2_t',r'$\mathcal L_{\mathrm{link},\sigma},\mathcal L_{\mathrm{link},q}$','독립 출력과 유도 출력의 연결 조건'],
        ['PDE1_m, PDE2_m; PDE1_t',r'$\nabla\cdot\widehat{\boldsymbol\sigma},\nabla\cdot\widehat{\boldsymbol q}$','독립 출력의 1차 미분 잔차']],[.32,.32,.36],caption='원저자 코드와 본 보고서의 표기 대응')
    r.equation(r'\begin{aligned}\boldsymbol r_{\sigma}&=\widehat{\boldsymbol\sigma}-\boldsymbol\sigma(\boldsymbol u,T),&\quad\boldsymbol r_q&=\widehat{\boldsymbol q}+k\nabla T,\\\boldsymbol r_{\mathrm M}&=\nabla\!\cdot\widehat{\boldsymbol\sigma},&r_{\mathrm T}&=\nabla\!\cdot\widehat{\boldsymbol q}.\end{aligned}')
    r.p(r'저자 파일의 strain_xy는 텐서 전단변형률 $\varepsilon_{xy}=(u_{x,y}+u_{y,x})/2$이며, sigma_xy에 곱하는 계수는 통상적인 평면변형률 구성식과 2배 차이를 보인다. 본 구현은 $\sigma_{xy}=2\mu\varepsilon_{xy}=\mu(u_{x,y}+u_{y,x})$로 통일하였다. 이는 원저자 파일의 문자 단위 복제가 아니라, 명시된 등방성 열탄성 모델에 대한 수정이다. 원문·공개 코드·재구성의 차이를 지운 채 동일 재현으로 서술하지 않는다.')
    r.p('출처: <link href="https://github.com/phyml4e/PINNs/blob/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic/Thermo_mechanic.py#L17-L55">저자 코드의 출력·구성식 정의</link>; <link href="https://github.com/phyml4e/PINNs/blob/5b426dc8eb2255b1bb3b8630619228b36cdb464e/Thermomechanic/Thermo_mechanic.py#L175-L278">일·연결·평형 손실</link>. 고정 커밋: 5b426dc8eb2255b1bb3b8630619228b36cdb464e.','small')
    r.fig('fig01_04_training_architecture','그림 2. 동시·순차 학습과 혼합 네트워크의 구현 절차. 독립 신경망과 장 동결을 구분한다.',manifest,110*mm)
    r.p('동시 학습은 두 장의 손실을 함께 최소화한다. 순차 학습은 한 장의 파라미터만 갱신하고 다른 장을 고정한다. 순차의 역학 블록에서 온도는 처방된 입력이므로 기계 에너지의 온도 방향 기울기를 차단한다. 본 구현은 열 블록 이후 새 역학 블록에서 Adam 상태를 재설정한다. 이 정책은 원문에 명시되지 않은 선택이다.')
    r.p('float64와 시드 42를 유지한다. CUDA 그래프 및 tanh 연쇄법칙의 공간 미분 가속은 동일 목적함수·업데이트를 목표로 검증했으나, 부동소수점 연산 순서에 따른 장기 궤적 차이가 가능하다. 정밀도나 예정 예산을 줄인 결과로 치환하지 않는다.','small')

    r.page(); r.h('5. 비교 해법과 경계조건의 영향')
    r.h('표준 PINN',2)
    r.p('주 변수 ux, uy, T만 출력하고 응력·열유속은 미분하여 만든다. 평형·열보존 잔차에는 공간 2차 미분과 재료 계수 미분이 필요하다. 절점 쌍선형 재료장의 보간 결절에서는 고전적 미분이 정의되지 않으므로 표준 PINN의 내부 배치는 재현 가능한 비결절 점을 사용한다. 이것은 방법 비교에 영향을 주는 공개된 정규화·배치 선택이다.')
    r.h('심층 에너지법(DEM)',2)
    r.p('DEM은 주 변수의 1차 미분으로 열전도 에너지와 평면변형률 탄성 에너지를 최소화한다. 열 에너지는 $\\frac12\\int_\\Omega k|\\nabla T|^2\\,\\mathrm d\\Omega$이고, 역학 에너지는 온도를 고정한 상태의 완전한 평면변형률 탄성 에너지다. 기본 자연 경계는 변분 문제에 포함하며 별도 Neumann 페널티를 임의로 추가하지 않는다. 기본 비교는 단위 가중치의 소프트 Dirichlet 페널티를 사용하고 하드 경계 제어를 추가한다.')
    r.p('<b>동시 DEM의 온도 기울기 처리도 구현 선택이다.</b> 역학 에너지를 계산할 때 T를 처방값으로 취급하여 그 항에서 온도망으로 가는 기울기를 차단한다. 반면 동시 표준·혼합 PINN은 역학 손실의 기울기가 온도망으로 전달된다. 이 선택은 열전도에서 역학으로 향하는 단방향 물리를 위한 변분 재구성이며, 원문 식 14의 문구를 그대로 따른 u,T 공동 최소화는 아니다. 따라서 방법 간 차이에는 이 최적화 경로의 차이도 포함된다.','small')
    r.h('소프트 페널티 편향을 방법 자체의 실패와 구분',2)
    r.p('균질 1차원 단위 구간에서 T(x)=ax+b라 두고 단위 경계 페널티를 쓰면 다음 목적함수가 된다. 이는 학습 결과를 꾸며낸 예시가 아니라 해석적으로 계산할 수 있는 비교 진단이다.')
    r.equation('\\begin{aligned}J(a,b)&=\\tfrac12a^2+(b-1)^2+(a+b)^2,\\\\\\frac{\\partial J}{\\partial a}&=3a+2b=0,\\qquad \\frac{\\partial J}{\\partial b}=2a+4b-2=0,\\\\(a^*,b^*)&=\\left(-\\tfrac12,\\tfrac34\\right),\\qquad T^*(x)=\\tfrac34-\\tfrac12x.\\end{aligned}')
    r.p('이 최소해의 경계값은 0.75와 0.25이며, 정확한 강제 경계 해 T=1-x와 다르다. J는 정확한 경계 해의 0.5보다 소프트 최소해의 0.25에서 더 작다. 따라서 최적화가 성공해도 유한한 페널티는 Dirichlet 문제의 정답을 그대로 최소화하지 않을 수 있다. 하드 경계 제어는 이 편향과 DEM의 표현력·최적화 한계를 분리하기 위해 필요하다.')
    r.table(['비교','읽는 원칙'],[['기본 소프트 DEM','원문의 비교 구조를 재구성하되 경계 페널티 편향을 공개'],['추가 하드 DEM 두 방식','동일 방법에서 경계 처리에 따른 민감도를 확인하는 추가 연구'],['방법 간 우세 판단','성분별 실제 오차·경계 오차·예산을 함께 보고한다'],['순차 열 장','역학을 바꾸어 비활성 열손실이 변하도록 인위적 역연성을 만들지 않는다']],[.32,.68])

    r.page(); r.h('6. 수치실험의 구성 및 학습 조건')
    r.table(['실험군','설정·예산','출처/선택'],[
      ['주 혼합 PINN 4개','원형/사각형 × 동시/순차; Adam 100k 또는 200k 총 업데이트; 5×40; soft','Table 1, 4.1-4.2절'],
      ['표준 PINN·DEM 4개','원형 × 두 방법 × 두 방식; 각 장 100k; soft','4.1.2절; 세부 페널티는 재구성'],
      ['하드 DEM 제어 2개','기본 DEM과 같은 예산, hard 경계','원문 외 공정성 민감도'],
      ['경계 보강·에너지 제외','각 100k 동시 Adam; 왼쪽 추가 1,000점 / energy=none','그림 15 / 정성적 절제 연구'],
      ['하드 Adam','사각형 동시 10k, 5×40','그림 17-18'],
      ['하드 L-BFGS 구조 9개','각 10k; 깊이 1,2,4,5,6×40 및 5층×5,10,20,40,60','그림 19-21; 5×40 중복 제외'],
      ['전이학습 4개','비율 1-10 각 Adam 10k; 가중치 0,0.1,0.5,1; 테스트 15','4.4절; 0.5 추가 선택'],
      ['시간 연구 25개','5깊이×5너비 각 L-BFGS 100회; 단일 작업','Table 4'],
      ['Remark 7 추가 실행','사각형 Adam 1M; 100k 궤적에서 900k 추가','본문 문구 충돌; soft는 재구성']],[.25,.49,.26])
    r.p('부록 A는 이 요약을 실제 작업 ID와 연결한다. 순차 학습의 총 업데이트 수는 두 장을 합친 수이며 각 장의 배분도 보존한다. L-BFGS의 한 업데이트에는 여러 closure 평가가 들어갈 수 있어 Adam과 반복 수만으로 비용을 비교하지 않는다.')
    r.h('중단·재시작과 선택 편향',2)
    r.p('체크포인트는 모델과 최적화기 상태, 진행 위치, 누적 시간을 함께 저장한다. 전이학습은 비율 내부의 Adam 상태를 보존하고 비율 경계에서 모델 가중치만 전달한다. 최종 비교는 마지막 예정 업데이트를 사용한다. 비율 15의 오차로 조기 종료하거나 좋은 시드·체크포인트만 선택하지 않는다. 결과가 나쁘더라도 완료 기준을 충족한 실행은 보고한다.')

    def focus(ids,metric='mean_peak_percent'):
        rows=[]
        for name in ids:
            if name not in results: continue
            values=results[name]['derived']
            rows.append([codes[name]]+[number(values[f][metric]) for f in ('ux','sxx','T','qx')])
        r.table(['실행','ux','sxx','T','qx'],rows,[.20,.20,.20,.20,.20])
        r.p('표의 값은 '+('평균 절대오차/기준해 최대 절댓값(%)' if metric=='mean_peak_percent' else metric)+
            '이다. 일반 실행 FEM200, 비율 15는 별도 FEM50을 기준으로 하며, 구성식/Fourier 출력을 사용한다. 전체 8성분·직접 출력은 부록 B에 있다.','small')
    def delta(left,right):
        if left not in results or right not in results:
            r.p('비교 대기: '+codes.get(left,left)+' 대 '+codes.get(right,right)+'. 두 실행이 모두 완료·평가되기 전에는 개선 여부를 결론내리지 않는다.','small'); return
        rows=[]; means=maximums=0
        for f in FIELDS:
            a=results[left]['derived'][f]; b=results[right]['derived'][f]
            dmean=None if a['mean_peak_percent'] is None or b['mean_peak_percent'] is None else a['mean_peak_percent']-b['mean_peak_percent']
            dmax=None if a['max_peak_percent'] is None or b['max_peak_percent'] is None else a['max_peak_percent']-b['max_peak_percent']
            means+=int(dmean is not None and dmean<0); maximums+=int(dmax is not None and dmax<0)
            rows.append([f,number(dmean),number(dmax)])
        r.table(['성분',codes[left]+' - '+codes[right]+' 평균(pp)','최대(pp)'],rows,[.18,.44,.38])
        r.p(f'왼쪽 실행의 평균오차가 작은 성분은 {means}/8개, 최대오차가 작은 성분은 {maximums}/8개이다. 양의 변화는 악화, 음의 변화는 개선이다. 단일 시드 결과이며 통계적 우세나 모든 상황의 순위로 확대하지 않는다.','small')

    r.page(); r.h('7. 수치 결과 및 고찰: 기본 벤치마크')
    circle=['mixed_circle_coupled','mixed_circle_sequential','baseline_circle_standard_coupled','baseline_circle_standard_sequential','baseline_circle_dem_coupled','baseline_circle_dem_sequential']
    rectangles=['mixed_rectangles_coupled','mixed_rectangles_sequential']
    r.p('원형 문제에서는 혼합 PINN, 표준 PINN 및 DEM에 동시·순차 학습을 적용한 여섯 조건을 비교한다. 사각형 문제에서는 혼합 PINN의 두 학습 방식을 평가한다. 비교 지표는 성분별 평균 및 최대 정규화오차이며, 각 방법의 경계조건과 학습 횟수를 함께 제시한다. 원문에서 보고한 방법 간 경향의 재현 여부는 이 수치 비교를 기준으로 판단한다.')
    focus(circle+rectangles)
    r.fig('fig11_circle_methods','그림 3. 본 프로젝트에서 학습한 6개 모델과 직접 계산한 FEniCS50의 x=0.5 단면. 비교 구성: Harandi et al. (2024), Fig. 11, DOI 10.1002/nme.7388. 모든 곡선은 자체 계산 결과이며 원문 그림의 전재·디지타이징이 아니다. 그림의 FEM50과 본문 표의 FEM200은 구분한다.',manifest,137*mm)
    r.p('곡선별 원본·학습 횟수·단면 CSV: <link href="https://github.com/JIN9811/AM_Project_Reproduce/blob/main/docs/FIGURE_11_PROVENANCE_KO.md">데이터 출처 명세</link>. 응력·열유속은 구성식/Fourier 출력이다.','small')
    r.h('동시·순차의 성분별 비교',2)
    delta('mixed_circle_sequential','mixed_circle_coupled')
    delta('mixed_rectangles_sequential','mixed_rectangles_coupled')
    sensitivity=['sensitivity_circle_dem_hard_coupled','sensitivity_circle_dem_hard_sequential']
    r.h('추가 DEM 경계 민감도',2)
    focus(sensitivity)
    delta('sensitivity_circle_dem_hard_coupled','baseline_circle_dem_coupled')
    delta('sensitivity_circle_dem_hard_sequential','baseline_circle_dem_sequential')
    r.p('이 제어 실험이 경계오차를 줄이더라도 DEM의 모든 오차가 작아진다고 가정하지 않는다. 소프트 페널티 문제와 하드 제약 문제는 최적화 문제가 다르며, 기본 재현과 추가 진단을 분리하여 해석한다.')

    for geometry,label,mfig,tfig in [('rectangles','사각형','fig13','fig14'),('circle','원형','fig08','fig09')]:
        name=f'mixed_{geometry}_coupled'
        if name not in results: continue
        for prefix,field,title in [(mfig,'mechanical','역학 전체장'),(tfig,'thermal','열 전체장')]:
            identifier=f'{prefix}_{geometry}_coupled_{field}'
            if not any(item['id']==identifier for item in manifest.get('figures',[])): continue
            r.page(); r.h(f'7. {label} 동시 학습: {title}',2)
            r.p('완료된 100,000회 실행의 예측, FEniCS50 기준해 및 절대 차이를 원문 색 범위로 표시한다. 응력과 열유속은 구성식/Fourier 미분 출력이다. 색 범위 밖 값은 색 막대의 삼각형으로 표시하며, 수치 배열을 범위에 맞추어 수정하지 않는다. FEM200 기준의 정량 오차는 본문과 부록 B에 별도로 제시한다.','small')
            r.fig(identifier,f'{label} {title}. 원문 그림 {prefix[3:]}에 대응하는 실제 계산 결과.',manifest,210*mm)
        identifier=f'fig06_{geometry}_coupled_losses'
        if any(item['id']==identifier for item in manifest.get('figures',[])):
            r.page(); r.h(f'7. {label} 동시 학습: 계산 과정',2)
            r.p('원래 CSV에 기록된 실제 손실 항을 표시한다. 이력의 진동을 평활화하지 않았으며, 손실 감소가 모든 성분의 오차 감소를 보장하지 않는다.')
            r.fig(identifier,f'{label} 100,000회 학습의 열·역학 손실 이력.',manifest,180*mm)

    r.page(); r.h('8. 왼쪽 경계 보강과 에너지 항 제외')
    r.p('두 학습 방식의 소프트 경계에서 남는 오차를 점검하고, 사각형 왼쪽에 약 1,000점을 추가한 실행과 기본 실행을 비교한다. 추가점은 점 평균 손실의 상대적인 가중도 바꿀 수 있으므로 단순히 공간 해상도만 높인 것으로 해석하지 않는다.')
    boundary_ids=['mixed_rectangles_coupled','mixed_rectangles_sequential','refine_rectangles_left1000','hard_rectangles_adam10k','arch_l5_w40']
    boundary_rows=[]
    for name in boundary_ids:
        if name in results:
            b=results[name]['boundary']; boundary_rows.append([codes[name]]+[number(b[k]) for k in ('ux_left','T_left','normal_u','temperature')])
    r.table(['실행','좌측 max|ux|','좌측 max|T-1|','전체 법선 u','전체 온도 DBC'],boundary_rows,[.16,.21,.21,.21,.21])
    r.fig('fig15_left_refinement','그림 4. 왼쪽 점 보강 배치와 오차. 보강 전후를 실제 완료 결과로 비교한다.',manifest,135*mm)
    delta('refine_rectangles_left1000','mixed_rectangles_coupled')
    r.h('원형의 에너지 항 제외',2)
    r.p('원문은 학습 중 변화가 작은 에너지 항을 제거하였을 때 정확도가 저하되는 사례를 보고하였다. 이를 검토하기 위해 나머지 설정을 유지하고 에너지 항만 제외한 절제 실험을 구성하였다. 효과의 방향과 크기는 기본 실행 대비 성분별 오차 변화로 평가한다. 이는 손실 항의 변화량과 최종 예측 정확도의 관계를 확인하기 위한 실험이다.')
    delta('ablation_circle_noenergy','mixed_circle_coupled')

    r.page(); r.h('9. 하드 경계·최적화기·1M 연속 실행')
    r.p('강제 경계조건은 다음 출력 변환으로 부과한다. 원문의 $x(x-1)$ 표현과 부호 차이는 기저 신경망에 흡수될 수 있지만 초기화 궤적까지 같음을 보장하지 않는다. 정확히 0인 경계 오차를 로그 축에서 임의의 양수로 바꾸지 않는다.')
    r.equation(r'u_x=x(1-x)N_{u_x},\qquad u_y=y(1-y)N_{u_y},\qquad T=1-x+x(1-x)N_T')
    focus(['hard_rectangles_adam10k','arch_l5_w40'])
    r.fig('fig18_hard_objective','그림 5. 하드 Adam과 L-BFGS의 실제 목적함수 이력. 한 L-BFGS 업데이트의 내부 closure 비용은 별도이다.',manifest,108*mm)
    r.p('하드 두 실행은 각각 10k이지만 소프트 기본 Adam은 100k이다. 따라서 소프트/하드 결과 차이를 경계조건 하나의 인과 효과로 부르지 않는다. Adam과 L-BFGS는 목적함수 평가 횟수와 벽시계 시간까지 함께 확인한다.')
    r.h('Remark 7의 1M 문구',2)
    r.p('Table 1의 100k와 Remark 7의 1M이 충돌하므로 사각형 100k 체크포인트를 보존하고 같은 궤적에 900k를 추가한 별도 실행을 둔다. 원문의 원래 시드·하드/소프트·실행 동일성은 확인되지 않으므로 이 추가 실행은 명시적인 재구성이다. 1M 실행으로 100k 결과를 덮어쓰지 않는다.')
    focus(['mixed_rectangles_coupled','remark_rectangles_adam1m'])
    delta('remark_rectangles_adam1m','mixed_rectangles_coupled')
    r.fig('remark07_runtime','그림 6. 100k와 1M의 기록된 누적 시간·업데이트 비교. 연속 실행의 재사용 예산을 이중 계산하지 않는다.',manifest,90*mm)

    r.page(); r.h('10. 신경망 구조의 정확도')
    r.p('깊이 연구는 너비 40을 고정한 1,2,4,5,6층이며, 너비 연구는 5층에서 5,10,20,40,60 뉴런이다. 모두 하드 경계의 동시 L-BFGS 10k이다. 겹치는 5×40은 동일 실행을 재사용한다. 3층은 원문의 실험값이 아니므로 임의로 추가하지 않는다.')
    arch=[j['id'] for j in specs if j['id'].startswith('arch_')]
    focus(arch)
    r.fig('fig20_depth_derived','그림 7. 깊이에 따른 구성식 출력의 평균·최대 절대오차. FEM50 기준이며 모든 8개 장을 유지한다.',manifest,178*mm)
    r.fig('fig21_width_derived','그림 8. 너비에 따른 구성식 출력의 평균·최대 절대오차. 큰 망의 단조 개선을 가정하지 않는다.',manifest,178*mm)
    r.p('원문에서는 5개 은닉층을 갖는 구조가 전반적으로 양호한 성능을 보이나, σxx의 최대오차에는 예외가 제시되어 있다. 본 연구에서는 평균오차와 최대오차를 함께 비교하여 이러한 경향을 검토한다. 최대오차는 재료 경계의 위치, 응력 복원 방식 및 평가 격자에 민감하므로 단일 지표만으로 구조의 적합성을 판단하기 어렵다.')
    delta('arch_l5_w40','arch_l4_w40')

    r.page(); r.h('11. Table 4 시간 측정과 실행 환경')
    r.p('25개 조합은 각 100 L-BFGS 업데이트를 다른 연구 작업과 겹치지 않게 실행한다. 지표는 원본 이력의 정확한 update=100 누적 시간/100이다. 초기 설정·closure·로그·체크포인트 오버헤드를 포함한 벽시계 시간이며 순수 GPU 커널 시간은 아니다. 여러 작업 동시 실행의 시간으로 이를 대체하지 않는다.')
    time_rows=[]
    for layer in (1,2,4,5,6):
        row=[layer]
        for width in (5,10,20,40,60):
            name=f'timing_l{layer}_w{width}'
            timing=manifest.get('jobs',{}).get(name,{}).get('timing',{}) if name in complete else {}
            row.append(number(timing.get('seconds_per_iteration_first_100')))
        time_rows.append(row)
    r.table(['층 / 뉴런','5','10','20','40','60'],time_rows,[.20,.16,.16,.16,.16,.16])
    r.p('단위는 s/update이다. 빈칸(-)은 검증 완료의 정확한 100회 측정값이 아직 없다는 뜻이며 0초가 아니다.','small')
    environments=[]
    for job in specs:
        path=root/'results/full_study'/job['id']/'environment.json'
        if path.exists():
            e=read_json(path)
            key=(e.get('python'),e.get('torch'),e.get('GPU'),e.get('device'))
            if key not in [x[0] for x in environments]: environments.append((key,str(path.relative_to(root))))
    r.table(['Python / torch','장치','기록 파일'],[[str(k[0])+' / '+str(k[1]),str(k[2] or k[3]),path] for k,path in environments],[.25,.28,.47])
    r.p('환경 파일은 진행 중 실행에도 존재할 수 있으므로 이 표 자체는 완료의 증거가 아니다. 원문 시간은 Apple M2 Pro이며 로컬 GPU·운영체제·라이브러리가 다르다. Table 4의 5×40 0.3409초를 10k에 곱하면 약 56.8분으로, 본문의 약 40분과도 차이가 있다. 논문 시간을 로컬 종료 보장값으로 사용하지 않는다.')
    r.h('효율에 관한 결론의 제한',2)
    r.p('최적화기 비교에는 실제 정확도, 업데이트 및 closure 수, 전처리/평가 포함 범위를 함께 명시해야 한다. 시간만 줄었다고 방법이 우수하다고 결론내리지 않는다. 재시작으로 누적된 시간과 연속 실행의 재사용 예산도 구분한다.')

    r.page(); r.h('12. 재료비 전이학습의 설계')
    r.p('그림 22의 명시적 정의에 따라 E기지/E개재물=k기지/k개재물=r이다. 기지 E=k=1, 개재물 E=k=1/r로 둔다. r=1,...,10을 순서대로 학습하고 각 비율에 10k Adam을 배분한다. 총 100k 업데이트 후 최종 가중치로 r=15를 평가한다. 두 재료비는 이 연구에서 항상 같으므로 독립 E비율·k비율 일반화를 검증한 것으로 해석하지 않는다.')
    r.p('입력은 x,y와 전역 log10 재료비 두 개이다. 같은 기지 좌표에서 국소 E,k가 같아도 전체 BVP는 달라지므로 전역 비율이 문제를 식별한다. 국소 E(x,y),k(x,y)를 말하는 본문과 이 인코딩 선택의 차이를 공개한다.')
    r.equation('\\mathcal L_{\\mathrm{total}}=\\mathcal L_{\\mathrm{data}}+w\\!\\left(\\mathcal L_{\\mathrm T}+\\mathcal L_{\\mathrm M}\\right)')
    transfer=[j for j in specs if j['kind']=='transfer']
    r.table(['실행','w','경계','데이터 정규화','총 업데이트'],[[codes[j['id']],j['config']['weight'],'hard' if j['config']['hard'] else 'soft',j['config'].get('normalization'),budget(j)] for j in transfer],[.15,.12,.16,.32,.25])
    r.fig('fig22_transfer_protocol','그림 9. 비율 1-10 전이와 미학습 비율 15 평가. 가중치 0.5는 원문에 미기재인 보완 선택이다.',manifest,108*mm)
    r.p('각 비율의 Adam 상태는 새로 시작하고 모델 가중치를 전달한다. 비율 내부 재시작은 Adam 상태를 보존한다. 원시 8성분 MSE의 단위별 규모와 physics 손실의 상대 크기는 가중치 효과에 영향을 준다. 다른 정규화로 얻은 w=0.1과 같은 의미의 가중치라고 가정하지 않는다.')
    r.p('테스트 비율 15는 마지막 학습까지 라벨을 읽지 않고, 조기 종료·최적 체크포인트·가중치 선택에 쓰지 않는다. 네 사전 고정 가중치의 테스트 비교를 공개하되 이 결과로 다시 선택한 가중치를 독립 검증 최적값이라고 부르지 않는다.')

    r.page(); r.h('13. 미학습 비율 15의 실제 결과')
    transfer_ids=[j['id'] for j in transfer]
    focus(transfer_ids)
    r.fig('fig25_transfer_sections','그림 10. 비율 15의 x=0.35 단면. 패널 제목·본문을 따랐으며 캡션 x=0.5와의 충돌을 남긴다. FEM15/50 기준이다.',manifest,145*mm)
    r.h('물리 항의 개선과 반례를 함께 검사',2)
    delta('transfer_w0.1','transfer_w0')
    r.p('원문의 전이학습 결과에서는 물리 손실을 추가하였을 때 여러 성분의 오차가 감소하는 반면, qx의 최대오차는 일부 위치에서 증가하는 예외가 나타난다. 이에 따라 본 연구는 평균오차와 최대오차의 변화량을 각각 산정한다. 음의 변화량은 개선, 양의 변화량은 악화를 의미하며, 그 부호를 통해 물리 손실의 이점과 국소적인 한계를 함께 평가한다.')
    if 'transfer_w0' in results and 'transfer_w0.1' in results:
        t0=results['transfer_w0']['status'].get('training_elapsed_seconds')
        t1=results['transfer_w0.1']['status'].get('training_elapsed_seconds')
        if t0 is not None and t1:
            r.table(['측정','값'],[['w=0 학습초',number(t0)],['w=0.1 학습초',number(t1)],['1-t(w=0)/t(w=0.1)',number((1-t0/t1)*100)+'%']],[.64,.36])
    else: r.p('데이터만/물리+데이터의 시간 비교는 두 실행 완료까지 대기한다. 원문의 약 82% 비용 감소를 본 계산값인 것처럼 복사하지 않는다.','small')

    r.page(); r.h('14. FEM 격자 수렴과 해석 한계')
    convergence=[]
    for geometry in ('circle','rectangles'):
        fine=root/'results'/f'fem_{geometry}_n200/solution.npz'
        if not fine.exists(): continue
        with np.load(fine) as d: xy,ref=d['xy'].copy(),d['fields'].copy()
        for n in (50,100):
            coarse=root/'results'/f'fem_{geometry}_n{n}/solution.npz'
            if not coarse.exists(): continue
            with np.load(coarse) as d:
                if not np.allclose(xy,d['xy'],atol=1e-12,rtol=0): raise ValueError('FEM convergence grids differ')
                m=metrics(d['fields'],ref)
            convergence.append([geometry+' '+str(n)]+[number(m[f]['relative_l2'],True) for f in FIELDS])
    r.table(['기하/요소']+list(FIELDS),convergence,[.24]+[.095]*8,numeric=True)
    r.p('표는 200×200 대비 상대 L2(%)를 NPZ에서 새로 계산했다. 기준이 0인 분모는 빈값이다. 해법 잔차와 격자 수렴은 서로 다른 검증이다.','small')
    r.fig('figA1_mesh_convergence','그림 11. 사각형의 FEM 50/100/200 단면. 200은 비교 기준이며 해석적 정답은 아니다.',manifest,146*mm)
    r.p('응력·열유속은 재료 경계에서 불연속일 수 있고 연속 절점 복원에 의존한다. 따라서 작은 평균오차와 큰 국소 최대오차가 동시에 나타날 수 있다. 원문의 “100에서 거의 수렴”에는 정량 허용오차가 없으므로 본 표의 변화를 보고 판단한다.')

    r.page(); r.h('15. 결론')
    if not results:
        r.p('<b>현재 검증 완료 신경망 결과가 없어 방법 순위·구조 최적값·물리 손실의 이점에 관한 수치 결론을 제시하지 않는다.</b> 이 단계에서 확인 가능한 것은 구현된 실험 계획, 기존 FEM 기준해의 실제 검증·수렴, 데이터와 완료 판정 경로이다. 전체 완료 전의 결과 표는 대기로 남는다.')
    else:
        r.p(f'현재 {len(results)}개 신경망 실행의 예측 배열과 기준해를 정렬하여 오차를 다시 계산했다. 각 절의 표와 변화량은 실제 완료 결과에 한정된다. 개선 성분과 악화 성분을 모두 남기며, 아래 제한 없이 보편적인 방법 순위로 확대하지 않는다.')
        if all(name in results for name in ('mixed_circle_coupled','mixed_circle_sequential','mixed_rectangles_coupled','mixed_rectangles_sequential')):
            rows=[]
            for geom,label in [('circle','원형'),('rectangles','사각형')]:
                for mode_name,mode_label in [('coupled','동시'),('sequential','순차')]:
                    values=results[f'mixed_{geom}_{mode_name}']['derived']
                    rows.append([label,mode_label,f"{values['ux']['relative_l2']*100:.2f}",f"{values['T']['relative_l2']*100:.2f}"])
            r.table(['기하','학습 방식','수평변위 오차 (%)','온도 오차 (%)'],rows,[.20,.20,.30,.30],caption='주 혼합 PINN의 상대 L2 오차 요약(FEM200 기준)')
            r.p('위 결과는 순차 학습의 효과가 기하와 평가 성분에 의존함을 보여준다. 따라서 학습 방법의 선택에는 전체 목적함수의 감소뿐 아니라 변위·응력·온도·열유속의 오차와 경계조건의 만족도를 함께 고려해야 한다. 연성 경계조건의 에너지 최소화에서는 페널티 자체가 유발하는 편향도 별도로 평가할 필요가 있다.')
    r.h('15.1 적용 범위와 후속 검증',2)
    r.p('첫째, 단일 시드와 지정 기하·하중·재료비에 대한 결과이므로 표본 변동이나 다른 문제에 대한 일반화를 확정하지 않는다. 둘째, 소프트 페널티·에너지·재료 미분·학습 배치·옵티마이저 상태 정책은 비교의 의미를 바꾼다. 셋째, 원문의 미기재 설정과 도해/본문 충돌 때문에 완전한 동일 실험의 유일한 해석은 존재하지 않는다.')
    r.p('넷째, 비율 15의 성공은 새로운 미세구조·다른 경계조건·독립 두 재료비의 일반화를 입증하지 않는다. 다섯째, 원문 시각 형식과 색 범위를 맞춘 것은 수치해를 맞춘 것과 다르다. 포화 색 영역은 감사표로 드러내고 예측 배열은 그대로 둔다. 여섯째, 장기 궤적과 실제 장비 시간이 원문과 다른 것은 소프트웨어·장비·재구성 차이를 함께 검토해야 한다.')
    r.h('15.2 계산 자료의 재사용',2)
    r.p('Full_Paper_Reproduction_KO.ipynb는 완료·대기 상태, 독립 FEM/PINN 검증, NPZ 오차 재계산 및 편집 가능한 그림을 제공한다. 기본값은 전체 학습을 시작하지 않고 완료된 자료만 읽는다. RUN_FULL_STUDY=True는 누락 FEM과 신경망 전체를 실행한 뒤 시간 연구를 단독 실행한다. 동일 설정의 체크포인트는 이어서 계산한다.')
    r.p('명령행: python -m harandi.full_study --groups mixed baseline transfer --workers 1<br/>이후: python -m harandi.full_study --groups timing --workers 1<br/>그림: python -m harandi.full_figures<br/>보고서: python scripts/build_full_korean_report.py --pinn-python PROJECT_PYTHON','small')
    r.p('중간 상태만 공유할 때는 그림과 보고서에 --partial을 명시한다. 새로운 조건은 별도 폴더에 저장하고 기존 결과를 바꾸어 재현 성공처럼 보이게 하지 않는다. 원본 로그를 완만하게 보간하거나 기록되지 않은 비활성 손실을 0으로 채우지 않는다.')

    r.page(); r.h('참고문헌')
    r.p('Harandi, A., Moeineddin, A., Kaliske, M., Reese, S., Rezaei, S. (2024). Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. International Journal for Numerical Methods in Engineering, 125(4), e7388. <link href="https://doi.org/10.1002/nme.7388" color="#245D96">https://doi.org/10.1002/nme.7388</link>. 온라인 선게재 연도는 2023년이다.','small')
    r.p('phyml4e. PINNs: 저자 공개 구현 및 벤치마크 자료. <link href="https://github.com/phyml4e/PINNs" color="#245D96">https://github.com/phyml4e/PINNs</link>. 사용 커밋과 파일 해시는 SOURCE_PROVENANCE.json에 기록하였다.','small')
    r.h('국문 서술 및 편집 형식 참고',2)
    r.p('아래 자료는 국·영문 제목과 초록, 용어의 최초 정의, 번호가 부여된 수식·표·그림, 결과와 해석을 연결하는 문단 구성에 참고하였다. 본 연구의 수치 결과는 자체 계산에서 얻었으며, 아래 논문의 계산 결과를 검증 자료로 사용하지 않았다. 학술지의 저자·소속·게재 정보나 편집 서식을 그대로 재현하지 않았다.','small')
    r.p('한국전산구조공학회. 논문 작성 및 투고 지침, 3절 원고 작성 요령. <link href="https://www.jcoseik.or.kr/journal-information/instructions-for-authors/" color="#245D96">학회 공식 작성 지침</link>. 확인일: 2026-11-04.','small')
    r.p('신호철, 이재훈 (2026). 선형 동적 시스템의 주파수 응답 해석을 위한 심층 연산자 네트워크 기반 예측 알고리즘 응용. 한국전산구조공학회논문집, 39(4), 265-272. <link href="https://doi.org/10.7734/COSEIK.2026.39.4.265" color="#245D96">https://doi.org/10.7734/COSEIK.2026.39.4.265</link>.','small')
    r.p('김준식 (2026). 변분 사영 관점에서 본 공회전 및 내재적 보 이론의 비선형 구조. 한국전산구조공학회논문집, 39(4), 219-228. <link href="https://doi.org/10.7734/COSEIK.2026.39.4.219" color="#245D96">https://doi.org/10.7734/COSEIK.2026.39.4.219</link>.','small')

    r.page(); r.h('부록 A. 모든 작업의 예산과 완료·대기 상태')
    r.p('J 코드는 본 보고서의 표에서 사용하는 실행 식별자이다. 현재 catalogue 순서에 대응하며 원문의 독립 실험 번호는 아니다. 요구 업데이트는 열·역학의 실제 갱신을 합친 수이다. 시간 연구는 정확도 연구와 구분한다.')
    inventory_rows=[]
    for job in specs:
        c=job['config']; name=job['id']; folder=root/'results/full_study'/name
        if name in complete: actual=str(budget(job)); status='검증 완료'
        else:
            actual='-'; status='대기'
            meta_path=folder/'experiment.json' if job['kind'] in ('baseline','transfer') else folder/'run_status.json'
            if meta_path.exists():
                try:
                    info=read_json(meta_path)
                    count=info.get('completed',info.get('completed_updates',info.get('actual_optimizer_updates')))
                    if isinstance(count,dict): count=sum(count.values())
                    if count is not None: actual=str(count)
                except json.JSONDecodeError: pass
            if (folder/'checkpoint.pt').exists(): status='진행/재시작'
        optimizer='Adam' if job['kind']=='transfer' or c.get('optimizer')=='adam' or c.get('adam_steps',0)>0 else 'L-BFGS'
        form=f"{c['layers']}×{c['width']} / "+('hard' if c['hard'] else 'soft')+' / '+optimizer
        inventory_rows.append([codes[name],name,form,f'{actual} / {budget(job):,}',status])
    r.table(['코드','실행 ID','구조·경계·최적화','실제 / 요구','상태'],inventory_rows,[.07,.39,.26,.16,.12])
    current_missing_refs=sorted(set(pending_refs+[name for name in manifest.get('pending_references',[])
                                               if not (root/name).is_file()]))
    if current_missing_refs:
        r.h('기준해 대기',2)
        r.table(['파일','의미'],[[v,'보고서 생성 시점에 파일이 없는 기준해; 해당 비교는 완료로 처리하지 않음'] for v in current_missing_refs],[.58,.42])

    r.page(); r.h('부록 B. 전체 8성분 오차')
    r.p('다음 값은 보고서 생성 때 NPZ 배열에서 다시 계산한다. 일반 실행은 FEM200, 전이학습은 재료비 15의 FEM50을 기준으로 한다. σxx,σyy,σxy는 sxx,syy,sxy로 표기한다. 변위·응력·온도·열유속은 각각 원문 벤치마크의 mm,GPa,K,W/mK 표기이며 혼합된 단위를 합산하여 하나의 정확도 점수로 만들지 않는다.')
    if not results:
        r.p('현재 완료 예측이 없어 수치 표를 비워 둔다. 모든 8개 성분과 직접/미분 표현의 계산 경로는 준비되어 있으며, 완료 실행이 확보되면 아래 여섯 표가 실제 값으로 자동 생성된다. 미완료 예측이나 과거의 짧은 실행 값으로 표를 채우지 않는다.')
        r.table(['표현','자동 생성할 지표'],[['직접 혼합 출력','8성분 상대 L2(%), 평균 절대오차, 최대 절대오차'],['구성식/Fourier 출력','8성분 상대 L2(%), 평균 절대오차, 최대 절대오차']],[.38,.62])
    else:
        names=[j['id'] for j in specs if j['id'] in results]
        first=True
        for representation,label in [('direct','직접 출력'),('derived','구성식/Fourier 출력')]:
            for key,caption in [('relative_l2','상대 L2 (%)'),('mean_absolute_error','평균 절대오차'),('max_absolute_error','최대 절대오차')]:
                if not first and (len(names)>14 or key=='relative_l2'): r.page()
                first=False
                r.h(label+' - '+caption,2)
                rows=[[codes[name]]+[number(results[name][representation][f][key],key=='relative_l2') for f in FIELDS] for name in names]
                r.table(['실행']+list(FIELDS),rows,[.16]+[.105]*8,numeric=True)
                r.p('빈값(-)은 정의할 수 없는 분모를 뜻한다. 작업 ID 및 경계·구조·예산은 부록 A, 평균/최대 정규화의 원시 자료는 FULL_STUDY_RESULTS.json에 있다.','small')
    r.h('재계산한 참조의 감사 기록',2)
    r.table(['실행','기준해','예측 SHA-256 앞 12자리'],[[codes[name],item['reference'],item['prediction_sha256'][:12]] for name,item in results.items()],[.12,.66,.22])

    r.page(); r.h('부록 C. 원문 28개 그림·4개 표의 대응')
    r.p('원문 1-25와 A1,B1,B2의 28개 그림, Table 1-4를 모두 목록화한다. 아래 “생성”은 해당 대응 그림 파일이 있다는 뜻이며 원문의 숨은 설정까지 동일하다는 뜻은 아니다. 도식·격자·시간 표와 신경망 결과를 구분하고, 원래 그림 전체는 FULL_STUDY_FIGURES.pdf와 HTML 갤러리에서 확대해 확인한다.')
    coverage=[]
    for entry in inventory['figures']:
        identifier=entry['figure']; token=str(identifier)
        found=[]
        for item in manifest.get('figures',[]):
            if item['id'].startswith('table'): continue
            sources=re.split(r'[,;/ ]+',str(item.get('source_figure','')))
            if token in sources: found.append(item['id'])
        coverage.append([str(identifier),FIGURE_NAMES.get(identifier,entry['content']),
                         '생성: '+', '.join(found) if found else '대응 결과/자료 대기'])
    r.table(['원문 그림','내용','현재 대응'],coverage,[.10,.34,.56],numeric=True)
    r.table(['원문 표','내용','본 보고서 대응'],[['1','기본 구조·Adam·예산·전환','4-6절 및 부록 A'],['2','원형 재료 상수','2절; 기하·재료 표'],['3','사각형 재료 상수','2절; 기하·재료 표'],['4','25개 구조의 계산 시간','11절; 완료 측정값만 표시']],[.11,.39,.50],numeric=True)

    r.page(); r.h('부록 D. 원문의 불명확성과 재구성 선택')
    r.p('아래 항목은 결과를 유리하게 해석하기 위한 사후 설명이 아니라 원문 감사에서 구분한 출처 문제이다. 수치 결과와 어떤 판단이 직접 관측이고 어떤 부분이 구현 선택인지 함께 확인한다.')
    issues=[]
    for item in inventory['source_issues']:
        topic,implication=ISSUES.get(item['id'],(item['topic'],item['reproduction_implication']))
        issues.append([item['id'],','.join(map(str,item['pages'])),topic,implication])
    r.table(['ID','쪽','주제','공개한 처리'],issues,[.08,.10,.21,.61])
    r.p('전이학습의 미기재 네 번째 가중치, 경계 처리, 데이터 정규화, 전역 인코딩과 최적화기 상태는 프로토콜에 기록된다. 동일 이름의 방법이라도 서로 다른 페널티·재료 정규화·에너지·배치를 쓰면 비교 대상 자체가 달라질 수 있다. 보고서에서 원문의 결과를 그대로 재현했다고 과장하지 않는다.')

    r.page(); r.h('부록 E. 파일·출처·보고서 재생성')
    r.table(['자료','위치와 용도'],[['기준 논문','Harandi et al. (2024), Int. J. Numer. Methods Eng., 125(4), e7388; DOI 10.1002/nme.7388'],
        ['원문 감사','PAPER_EXPERIMENT_INVENTORY.json; 본문·도해·표·부록을 분리 기록'],
        ['구현 선택','FULL_STUDY_PROTOCOL.json; 목적함수·경계·예산·입력·추가 제어'],
        ['수치 결과','results/full_study/각 ID의 prediction.npz, model.pt, checkpoint.pt, history.csv'],
        ['원시 FEM','results/fem_*/solution.npz 및 results/full_study/parametric_fem/ratio_*/solution.npz'],
        ['편집 가능한 실험','Full_Paper_Reproduction_KO.ipynb, README_FULL_STUDY_KO.md'],
        ['그림집','FULL_STUDY_FIGURES.pdf, FULL_STUDY_GALLERY_KO.html, figures/full_study/*.svg'],
        ['본 보고서 원본','scripts/build_full_korean_report.py, FULL_PROJECT_REPORT_KO.tex, FULL_PROJECT_REPORT_KO.md'],['별도 FEM 보고서','FEM_REPORT_KO.pdf 및 FEM_REPORT_KO.tex; docs/LATEX_REPORTS_KO.md의 재생성 안내']],[.24,.76])
    r.p('출처 식별용 SHA-256은 아래와 같다. 예측·기준해별 해시는 수치 JSON과 작업 폴더에 연결된다. 이 보고서의 그림은 현재 그림집에서 가져오며, 수치 표는 별도로 다시 계산한다.','small')
    for filename in ('PAPER_EXPERIMENT_INVENTORY.json','FULL_STUDY_PROTOCOL.json','FULL_STUDY_RESULTS.json'):
        r.p(html.escape(filename)+'<br/>'+sha(root/filename),'small')
    r.p('재생성 명령은 문서용 Python으로 scripts/build_full_korean_report.py를 실행하고 --pinn-python에 torch가 있는 프로젝트 Python을 지정하는 것이다. 엄격한 기본값은 전체 작업 및 최신 그림집의 완료를 요구한다. --partial만 진행 보고서를 허용하며 표지와 모든 페이지의 꼬리말에 이를 표시한다. 보고서 생성 과정은 신경망 학습을 수행하지 않는다. LaTeX 원본과 Markdown 및 PDF를 함께 갱신한다. XeLaTeX 설치와 재생성 명령은 docs/LATEX_REPORTS_KO.md에 정리하였다.')
    r.p('서지: Harandi A., Moeineddin A., Kaliske M., Reese S., Rezaei S. Mixed formulation of physics-informed neural networks for thermo-mechanically coupled systems and heterogeneous domains. DOI <link href="https://doi.org/10.1002/nme.7388" color="#245D96">10.1002/nme.7388</link>. 저자 공개 코드 및 원문 자료의 상세 출처는 SOURCE_PROVENANCE.json과 원문 감사 목록을 따른다.','small')
    target=r.build()
    print(f'{target}: mode={"partial" if partial else "complete"}; completed={len(complete)}/{len(specs)}; evaluated={len(results)}; figures={len(r.figures)}')
    return {'pdf':str(target),'completed':len(complete),'catalogue':len(specs),'evaluated':len(results),'figures':r.figures,'partial':partial}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--partial',action='store_true',help='Create an explicitly labeled progress report')
    parser.add_argument('--pinn-python','--torch-python',dest='pinn_python',type=Path)
    args=parser.parse_args(); root=args.root.resolve()
    create_report(root,args.partial,find_pinn_python(root,args.pinn_python))


if __name__=='__main__': main()
