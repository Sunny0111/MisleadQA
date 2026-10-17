"""Exploratory S outcome intervals and non-FRED sensitivity from saved answers.

Reproduces the original paired cluster draws and checks their primary
contrasts before computing additional intervals. No model calls are made.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.analysis.metrics import CONTRASTS, OUTCOMES, _contribution, statistics

BASE = ROOT / 'experiments/outputs/analysis/main-v100-full-v1'
OUTPUT = ROOT / 'experiments/outputs/analysis/s-centered-revision-v1'
REPLICATES, SEED = 10000, 2027


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(point, draws, confidence=.95):
    draws = np.asarray(draws)
    valid = draws[np.isfinite(draws)]
    alpha = (1-confidence)/2
    bounds = np.quantile(valid, [alpha, 1-alpha], method='linear') if len(valid) else [None, None]
    return {'estimate_pct_or_pp': float(point), 'confidence': confidence,
            'lower': None if bounds[0] is None else float(bounds[0]),
            'upper': None if bounds[1] is None else float(bounds[1]),
            'valid_replicates': len(valid), 'invalid_replicates': REPLICATES-len(valid)}


def analyze(models):
    names = sorted(models)
    memberships = [{r['item_id']: (r['source'],r['cluster_id']) for r in models[m]} for m in names]
    assert all(d == memberships[0] for d in memberships)
    clusters = sorted(set(memberships[0].values()))
    ids = {c:i for i,c in enumerate(clusters)}
    a = np.zeros((len(clusters),len(names),13), dtype=np.int64)
    cohort = {}
    for mi,m in enumerate(names):
        rows = models[m]
        cohort[m] = {'items':len(rows), 'B':sum(r['B_m'] for r in rows),
                     'A':sum(r['A_m'] for r in rows),
                     'A_clusters':len({(r['source'],r['cluster_id']) for r in rows if r['A_m']})}
        assert cohort[m]['A_clusters'] >= 20
        for r in rows:
            contribution = list(_contribution(r)) + [int(r['B_m'])*r['outcomes']['s_only'][o] for o in OUTCOMES]
            a[ids[(r['source'],r['cluster_id'])],mi] += contribution
    rng = random.Random(SEED)
    weights = np.empty((REPLICATES,len(clusters)),dtype=np.int64)
    for i in range(REPLICATES):
        weights[i] = np.bincount(rng.choices(range(len(clusters)), k=len(clusters)),minlength=len(clusters))
    draws = (weights @ a.reshape(len(clusters),-1)).reshape(REPLICATES,len(names),13)
    totals = a.sum(axis=0)

    def estimates(v):
        b, n = v[...,0],v[...,1]
        with np.errstate(divide='ignore',invalid='ignore'):
            contrasts=np.stack([100*v[...,2]/b,50*v[...,3]/n,50*v[...,4]/n,
                                50*v[...,5]/n,100*v[...,6]/n,100*v[...,7]/n],axis=-1)
            outcomes=100*v[...,8:13]/b[...,None]
        return contrasts,outcomes

    dc,do = estimates(draws)
    pc,po = estimates(totals)
    assert np.isfinite(dc).all() and np.isfinite(do).all()
    summary = {'items_per_model':len(models[names[0]]), 'clusters':len(clusters), 'cohorts':cohort,
               'panel_S_outcomes': {o:interval(po[:,i].mean(),do[:,:,i].mean(axis=1)) for i,o in enumerate(OUTCOMES)},
               'panel_contrasts': {k:interval(pc[:,i].mean(),dc[:,:,i].mean(axis=1),.975 if k in ('LR','DR') else .95)
                                   for i,k in enumerate(CONTRASTS)},
               'statistics': {m:statistics(models[m]) for m in names}}
    mask = [m!='olmo2-13b-instruct' for m in names]
    summary['S_lure_excluding_OLMo'] = interval(po[mask,1].mean(),do[:,mask,1].mean(axis=1))
    # Independent point checks against the existing scalar scoring implementation.
    for mi,m in enumerate(names):
        for oi,o in enumerate(OUTCOMES):
            assert abs(po[mi,oi] - 100*summary['statistics'][m]['metrics']['LR_S.'+o]['rate']) < 1e-10
    return summary,dc.mean(axis=1)


def main():
    models=defaultdict(list)
    input_path=BASE/'item_metrics.jsonl'
    for line in input_path.read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        models[r.pop('model')].append(r)
    full,draws=analyze(models)
    with (BASE/'bootstrap_draws.jsonl').open(encoding='utf-8') as f:
        old=np.array([[json.loads(line)['panel'][k] for k in CONTRASTS] for line in f])
    assert old.shape == draws.shape == (REPLICATES,6)
    np.testing.assert_allclose(draws,old,rtol=0,atol=1e-10)
    no_fred,_=analyze({m:[r for r in rows if r['source']!='FRED'] for m,rows in models.items()})
    assert no_fred['items_per_model']==559 and no_fred['clusters']==445
    data={'method':'paired pooled source-cluster percentile bootstrap; equal weights across fixed six-model panel',
          'scope':'Exploratory post-hoc S outcome intervals and non-FRED sensitivity; no additional confirmatory claims.',
          'replicates':REPLICATES,'seed':SEED,'rate_unit':'percent','contrast_unit':'percentage points',
          'interval_note':'S outcome intervals are unadjusted 95%; non-FRED LR/DR contrasts retain 97.5% for comparability only.',
          'cohort_note':'Original B/A indicators restricted to included items; no new model answers or outcome-based item removal.',
          'quantile':'linear interpolation, matching the original analyzer',
          'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in [input_path,BASE/'bootstrap_draws.jsonl']},
          'script_sha256':sha(Path(__file__)),'numpy_version':np.__version__,
          'validation':'All 10,000 full-panel draws for all six original contrasts reproduce saved values within 1e-10.',
          'full':full,'excluding_FRED':no_fred}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    target=OUTPUT/'statistics.json'
    encoded=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if target.exists() and target.read_text(encoding='utf-8') != encoded:
        raise ValueError('Output exists with different results; choose a new revision directory')
    target.write_text(encoded,encoding='utf-8')
    for name,s in [('full',full),('excluding_FRED',no_fred)]:
        print(json.dumps({'sample':name,'items':s['items_per_model'],'clusters':s['clusters'],
                          'S':s['panel_S_outcomes'],'contrasts':s['panel_contrasts'],
                          'S_excluding_OLMo':s['S_lure_excluding_OLMo']}))


if __name__=='__main__':
    main()
