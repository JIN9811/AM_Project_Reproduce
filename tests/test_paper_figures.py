import numpy as np
from harandi.paper_figures import scale_audit,history_rows


def test_color_scale_audit_records_saturation_without_changing_values():
    data=np.array([-2.,0.,1.,3.]);original=data.copy()
    info=scale_audit(data,(0.,2.))
    assert info['below_count']==1 and info['above_count']==1
    assert info['extend']=='both' and info['data_min']==-2 and info['data_max']==3
    np.testing.assert_array_equal(data,original)


def test_history_keeps_optimizer_offsets_and_missing_frozen_losses(tmp_path):
    f=tmp_path/'history.csv'
    f.write_text('optimizer,step,active,T_connection,M_connection\nadam,1,thermal,1,\nadam,4,mechanical,,2\nlbfgs,1,thermal,.5,\nlbfgs,4,mechanical,,.8\n')
    rows=history_rows(f)
    assert [r['update'] for r in rows]==[1,4,5,8]
    assert np.isnan(rows[0]['M_connection']) and np.isnan(rows[1]['T_connection'])
    assert rows[2]['optimizer']=='lbfgs'


def test_evolution_audits_saturation_and_uses_only_current_metadata(tmp_path):
    import json,shutil
    from pathlib import Path
    from harandi.paper_figures import FigureWriter,evolution_figure
    root=Path(__file__).parents[1]
    (tmp_path/'data').mkdir()
    shutil.copy(root/'data/paper_palette.json',tmp_path/'data/paper_palette.json')
    run=tmp_path/'results/process_circle_coupled';folder=run/'snapshots';folder.mkdir(parents=True)
    (run/'process_metadata.json').write_text(json.dumps({'status':'complete','snapshot_updates':[0,2],
          'total_optimizer_updates':2,'adam_steps':1,'lbfgs_steps':1,'mode':'coupled'}))
    (run/'config.json').write_text(json.dumps({'adam_steps':1,'lbfgs_steps':1,'mode':'coupled'}))
    from harandi.physics import grid
    for step in (0,2,999):
        f=np.zeros((2601,8));f[:,0]=-2 if step==0 else .1;f[:,2]=-.5
        np.savez(folder/f'step_{step:06d}.npz',xy=grid(51),fields=f,step=step,
                 constitutive_stress=f[:,2:5],fourier_flux=f[:,6:8])
    writer=FigureWriter(tmp_path)
    try:evolution_figure(writer)
    finally:writer.close()
    assert len(writer.audit)==6
    assert {r['panel'] for r in writer.audit}=={'update_0','update_2'}
    initial=next(r for r in writer.audit if r['field']=='ux' and r['panel']=='update_0')
    assert initial['below_count']==2601 and initial['above_count']==0 and initial['extend']=='min'
    assert '2' in writer.items[0]['note']


def test_running_snapshot_experiment_is_not_presented_as_complete(tmp_path):
    import json
    from harandi.paper_figures import FigureWriter,evolution_figure
    run=tmp_path/'results/process_circle_coupled';run.mkdir(parents=True)
    (run/'process_metadata.json').write_text(json.dumps({'status':'running','snapshot_updates':[0,2]}))
    writer=FigureWriter(tmp_path)
    try:evolution_figure(writer)
    finally:writer.close()
    assert not writer.items and not writer.audit
