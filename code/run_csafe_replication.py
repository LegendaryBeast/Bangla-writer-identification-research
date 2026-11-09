#!/usr/bin/env python3
"""CSAFE cross-session replication with outer-writer-nested attribute fusion."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import csafe_calibration as calibration
import revised_protocol as rp
import run_wid_handedness as wh
from run_nested_fusion import choose_lambda

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/csafe_replication_2026-11-08'
CONDITIONS=('S2_LND_r01','S2_WOZ_r01','S3_WOZ_r01')
REPS=('resnet18','vit_b_16','hc183','stroke')
HAND_MODELS=('hc183','stroke','hc183_adj')
SEED=20261108


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_manifest(train,groups,labels):
    tr=np.flatnonzero(train);rows=[]
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=2031).split(tr,labels[tr],groups[tr]):
        left,right=sorted(set(groups[tr[a]])),sorted(set(groups[tr[b]]))
        assert not set(left)&set(right)
        rows.append(dict(train_writers=left,calibration_writers=right))
    return rows


def outer_attribute(outer,writers,X,y,cov,groups,auxiliary,query,conditions,adjust):
    train=auxiliary & np.isin(groups,writers) & (groups!=outer)
    test=np.flatnonzero(query & (groups==outer))
    assert len(test)==3 and outer not in set(groups[train])
    assert train.sum()==6*(len(writers)-1)
    _,raw,prediction,calibration_fit=calibration.lowo_writer(X,y,cov,groups,train,test,adjust,2031)
    ci={c:i for i,c in enumerate(CONDITIONS)};wi={w:i for i,w in enumerate(writers)}
    raw_out=np.full(3,np.nan);outer_out=np.full(3,np.nan)
    for idx,score,logit in zip(test,raw,prediction):
        raw_out[ci[conditions[idx]]]=score;outer_out[ci[conditions[idx]]]=logit
    dev=np.array([w for w in writers if w!=outer])
    labels=np.array([y[np.flatnonzero(groups==w)[0]] for w in dev])
    logits=np.full((3,len(writers)),np.nan)
    manifest=dict(outer_writer=outer,final_attribute_train_writers=sorted(set(groups[train])),
        final_training_pages=int(train.sum()),final_platt_fit=calibration_fit,final_calibration_folds=calibration_manifest(train,groups,y),development=[])
    for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=2031).split(dev,labels)):
        trainw,heldw=dev[a],dev[b]
        assert outer not in set(trainw)|set(heldw) and not set(trainw)&set(heldw)
        mask=auxiliary & np.isin(groups,trainw)
        idx=np.flatnonzero(query & np.isin(groups,heldw))
        _,_,ell,development_calibration=calibration.lowo_writer(X,y,cov,groups,mask,idx,adjust,2031)
        for j,value in zip(idx,ell):logits[ci[conditions[j]],wi[groups[j]]]=value
        inner=calibration_manifest(mask,groups,y)
        for row in inner:
            assert not (set(row['train_writers'])|set(row['calibration_writers']))&(set(heldw)|{outer})
        manifest['development'].append(dict(fold=fold,train_writers=list(trainw),held_writers=list(heldw),
            training_pages=int(mask.sum()),platt_fit=development_calibration,calibration_folds=inner))
    assert np.isfinite(logits[:,np.array(writers)!=outer]).all()
    assert np.isnan(logits[:,wi[outer]]).all() and np.isfinite(outer_out).all()
    return raw_out,outer_out,logits,manifest


def attribute_predictions(U,F,writers,hand_model,cache,jobs,signature):
    key=hashlib.sha256(json.dumps(dict(writers=writers,hand_model=hand_model,signature=signature),sort_keys=True).encode()).hexdigest()
    arrays,meta=cache/(key+'.npz'),cache/(key+'.json')
    if arrays.exists() and meta.exists():
        with np.load(arrays) as z:raw,outer,development=z['raw'],z['outer'],z['development']
    else:
        rep='stroke' if hand_model=='stroke' else 'hc183'
        cov=np.column_stack([U.ink_density.to_numpy(),F['global_intensity']])
        with parallel_config(backend='loky',n_jobs=jobs,inner_max_num_threads=1):
            fitted=Parallel()(delayed(outer_attribute)(w,writers,F[rep],U.label.to_numpy(),cov,
                U.writer_id.to_numpy(),U.attribute_train.to_numpy(),U['query'].to_numpy(),
                U.condition.to_numpy(),hand_model=='hc183_adj') for w in writers)
        raw=np.stack([r[0] for r in fitted],axis=1)
        outer=np.stack([r[1] for r in fitted],axis=1)
        development=np.stack([r[2] for r in fitted])
        np.savez_compressed(arrays,raw=raw,outer=outer,development=development)
        meta.write_text(json.dumps([r[3] for r in fitted],indent=2)+'\n')
    mask=np.eye(len(writers),dtype=bool)
    assert all(np.isnan(development[:,c,:][mask]).all() for c in range(3))
    assert all(np.isfinite(development[:,c,:][~mask]).all() for c in range(3))
    return raw,outer,development,key


def inference(delta,labels,rng):
    delta=np.asarray(delta,float);draws=np.zeros(10000)
    for hand in (0,1):
        values=delta[labels==hand]
        draws+=.5*values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    lo,hi=np.quantile(draws,[.025,.975]);observed=delta.mean()
    flips=rng.choice([-1,1],size=(9999,len(delta)))
    p=(1+np.sum(np.abs(flips@delta/len(delta))>=abs(observed)-1e-12))/10000
    return dict(delta_top1=observed,ci_lo=lo,ci_hi=hi,p_signflip=p)


def cohorts(U):
    writers=sorted(U.writer_id.unique());labels=U.groupby('writer_id').label.first()
    rng=np.random.default_rng(SEED)
    result=[('full_balanced',writers)]
    for i in range(20):
        group=sorted(w for hand in (0,1) for w in rng.choice(labels.index[labels==hand],21,replace=False))
        result.append((f'balanced42_{i:02d}',group))
    (OUT/'gallery_manifest.json').write_text(json.dumps(dict(seed=SEED,
        galleries=[dict(name=n,writers=w) for n,w in result]),indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args();start=time.time()
    U=pd.read_csv(OUT/'units.csv');A=pd.read_csv(OUT/'images.csv')
    assert np.array_equal(U.image_id,A.image_id)
    if not U.valid.all():raise ValueError('Invalid source page: stop and record eligibility before evaluation')
    assert len(U)==864 and U.writer_id.nunique()==96 and U.image_sha256.is_unique
    assert (U.groupby('writer_id').attribute_train.sum()==6).all()
    with np.load(OUT/'features.npz') as z:
        assert np.array_equal(z['image_id'],U.image_id)
        F={rep:z[rep].astype(float) for rep in (*REPS,'global_intensity')}
    for rep in ('hc183','stroke','global_intensity'):assert np.isfinite(F[rep]).all()
    for rep in ('resnet18','vit_b_16'):assert np.isfinite(F[rep][U.deep_required]).all()
    sources=[Path(__file__),Path(calibration.__file__),Path(wh.__file__),Path(rp.__file__),ROOT/'code/run_nested_fusion.py',
        OUT/'protocol_v1_1.json',OUT/'units.csv',OUT/'features.npz']
    signature={str(p.relative_to(ROOT)):sha(p) for p in sources}
    cache=OUT/'attribute_cache';cache.mkdir(exist_ok=True)
    summaries,outcomes,params,tests,diagnostics,fit_records,hand_rows,loss_rows=[],[],[],[],[],[],[],[]
    lookup={(r.writer_id,r.condition):i for i,r in enumerate(U.itertuples())}
    rng=np.random.default_rng(SEED+1)
    groups=cohorts(U)
    for cohort,writers in groups:
        size=len(writers);labels=U.groupby('writer_id').label.first().reindex(writers).to_numpy()
        assert np.sum(labels==0)==np.sum(labels==1)
        models=HAND_MODELS if cohort=='full_balanced' else ('hc183',)
        active_conditions=CONDITIONS if cohort=='full_balanced' else CONDITIONS[:2]
        predictions={}
        for hand_model in models:
            predictions[hand_model]=attribute_predictions(U,F,writers,hand_model,cache,args.jobs,signature)
            print(f'{cohort} {hand_model}: nested attribute fits ready; {time.time()-start:.1f}s',flush=True)
        ri=np.array([lookup[(w,'S1_LND_r01')] for w in writers])
        for condition in active_conditions:
            cindex=CONDITIONS.index(condition)
            qi=np.array([lookup[(w,condition)] for w in writers])
            assert not set(ri)&set(qi) and (U.session.to_numpy()[ri]==1).all()
            assert (U.session.to_numpy()[qi]>1).all()
            assert not set(U.parent_form_id.to_numpy()[ri])&set(U.parent_form_id.to_numpy()[qi])
            truth=np.arange(size);qw=np.array(writers);weights=np.full(size,1/size)
            for hand_model in models:
                raw,outer,development,cache_key=predictions[hand_model]
                ell=outer[cindex];probability=expit(ell);hand_ok=(ell>0)==labels
                hand_rows.append(dict(cohort=cohort,condition=condition,hand_model=hand_model,n_writers=size,
                    accuracy=float(hand_ok.mean()),left_recall=float(hand_ok[labels==1].mean()),
                    right_recall=float(hand_ok[labels==0].mean()),nll=float(np.mean(np.logaddexp(0,ell)-labels*ell)),
                    brier=float(np.mean((probability-labels)**2)),raw_auc=float(roc_auc_score(labels,raw[cindex])),
                    calibrated_auc=float(roc_auc_score(labels,ell)),predicted_left=int((ell>0).sum())))
                fit_records.append(dict(cohort=cohort,condition=condition,hand_model=hand_model,cache_key=cache_key,
                    writers=writers,reference_image_ids=U.image_id.to_numpy()[ri].tolist(),
                    query_image_ids=U.image_id.to_numpy()[qi].tolist(),attribute_training_image_ids=U.loc[
                        U.attribute_train&U.writer_id.isin(writers),'image_id'].tolist()))
            for rep in REPS:
                X=F[rep];scaler=StandardScaler().fit(X[ri])
                zr,zq=scaler.transform(X[ri]),scaler.transform(X[qi])
                zr/=np.maximum(np.linalg.norm(zr,axis=1,keepdims=True),1e-12)
                zq/=np.maximum(np.linalg.norm(zq,axis=1,keepdims=True),1e-12)
                S=zq@zr.T;T=rp.fit_temperatures(S,truth,qw,weights)
                for hand_model in models:
                    raw,outer,development,cache_key=predictions[hand_model]
                    ell=outer[cindex];hand_ok=(ell>0)==labels;M=rp.match_logprob(ell,labels)
                    lambdas=np.zeros(size)
                    for j,w in enumerate(writers):
                        lambdas[j],losses=choose_lambda(S,truth,qw,weights,development[j,cindex],w,T[j],labels)
                        params.append(dict(cohort=cohort,condition=condition,rep=rep,hand_model=hand_model,
                            writer_id=w,temperature=T[j],fusion_weight=lambdas[j],raw_hand_score=raw[cindex,j],hand_logit=ell[j]))
                        if cohort in ('full_balanced','balanced42_00'):
                            for lam,loss in zip(wh.LAM_GRID,losses):
                                loss_rows.append(dict(cohort=cohort,condition=condition,rep=rep,hand_model=hand_model,
                                    writer_id=w,fusion_weight=lam,development_nll=loss))
                    scores=dict(identity_only=S,oracle_known_hand=S+10*(labels[:,None]==labels[None,:]),
                        predicted_hard_filter=S+10*((ell>0)[:,None]==labels[None,:]),
                        fixed_fusion_lambda1=S/T[:,None]+M,nested_fusion=S/T[:,None]+lambdas[:,None]*M)
                    ranks={method:rp.ranks(score,truth) for method,score in scores.items()}
                    base=ranks['identity_only']==1;oracle=ranks['oracle_known_hand']==1
                    filtered=ranks['predicted_hard_filter']==1
                    assert np.array_equal(filtered,hand_ok&oracle)
                    same=labels[:,None]==labels[None,:]
                    own=np.argmax(np.where(same,S,-np.inf),axis=1)
                    opposite=np.argmax(np.where(~same,S,-np.inf),axis=1)
                    margin=(S[np.arange(size),truth]-S[np.arange(size),opposite])/T
                    hand_logodds=(2*labels-1)*ell
                    for method,lam in [('fixed_fusion_lambda1',np.ones(size)),('nested_fusion',lambdas)]:
                        boundary=margin+lam*hand_logodds
                        predicted=(own==truth)&((boundary>0)|((boundary==0)&(truth<opposite)))
                        assert np.array_equal(predicted,ranks[method]==1)
                    for method,rank in ranks.items():
                        correct=rank==1
                        tag=dict(cohort=cohort,condition=condition,rep=rep,hand_model=hand_model,method=method)
                        summaries.append(dict(**tag,n_writers=size,top1=float(correct.mean()),
                            top5=float((rank<=5).mean()),mrr=float((1/rank).mean()),left_top1=float(correct[labels==1].mean()),
                            right_top1=float(correct[labels==0].mean()),hand_balanced_accuracy=float(hand_ok.mean()),
                            rescued=float((correct&~base).mean()),damaged=float((~correct&base).mean())))
                        for j,w in enumerate(writers):
                            outcomes.append(dict(**tag,writer_id=w,label=int(labels[j]),rank=int(rank[j]),
                                correct=int(correct[j]),delta_vs_identity=int(correct[j])-int(base[j]),
                                hand_correct=int(hand_ok[j]),identity_margin=margin[j],true_hand_logodds=hand_logodds[j]))
                        if method!='identity_only' and cohort in ('full_balanced','balanced42_00'):
                            primary=(cohort=='full_balanced' and condition in CONDITIONS[:2] and hand_model=='hc183'
                                and rep in ('resnet18','vit_b_16') and method in ('fixed_fusion_lambda1','nested_fusion'))
                            tests.append(dict(**tag,primary=primary,**inference(correct.astype(float)-base,labels,rng)))
                    partition=np.array([math.comb(size-size//2,int(r)-1)/math.comb(size-1,int(r)-1)
                        if int(r)-1<=size-size//2 else 0. for r in ranks['identity_only']])
                    diagnostics.append(dict(cohort=cohort,condition=condition,rep=rep,hand_model=hand_model,n_writers=size,
                        random_balanced_partition_expected_top1=float(partition.mean()),
                        random_partition_expected_gain=float(partition.mean()-base.mean()),oracle_gain=float(oracle.mean()-base.mean()),
                        filtering_exact_joint=float((hand_ok&oracle).mean()),filtering_observed=float(filtered.mean()),
                        filtering_independence_approximation=float(hand_ok.mean()*oracle.mean()),
                        filtering_independence_residual=float(filtered.mean()-hand_ok.mean()*oracle.mean()),
                        top_score_ties=int(np.sum((S==S.max(1,keepdims=True)).sum(1)>1)),
                        fixed_boundary_ties=int(np.sum(margin+hand_logodds==0)),
                        nested_boundary_ties=int(np.sum(margin+lambdas*hand_logodds==0))))
        pd.DataFrame(summaries).to_csv(OUT/'id_summary.partial.csv',index=False)
        print(f'{cohort}: completed comparisons; {time.time()-start:.1f}s',flush=True)
    test=pd.DataFrame(tests);test['p_holm_primary']=np.nan
    selected=test[test.primary].sort_values('p_signflip');assert len(selected)==8
    test.loc[selected.index,'p_holm_primary']=np.maximum.accumulate([
        min(1.,(len(selected)-i)*p) for i,p in enumerate(selected.p_signflip)])
    pd.DataFrame(summaries).to_csv(OUT/'id_summary.csv',index=False)
    pd.DataFrame(outcomes).to_csv(OUT/'writer_outcomes.csv.gz',index=False)
    pd.DataFrame(params).to_csv(OUT/'selected_parameters.csv',index=False)
    pd.DataFrame(loss_rows).to_csv(OUT/'lambda_loss_profiles.csv.gz',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    pd.DataFrame(hand_rows).to_csv(OUT/'handedness_summary.csv',index=False)
    test.to_csv(OUT/'paired_tests.csv',index=False)
    (OUT/'fitting_manifest.json').write_text(json.dumps(fit_records,indent=2)+'\n')
    report=dict(completed=True,protocol_sha256=sha(OUT/'protocol_v1_1.json'),input_code_sha256=signature,
        python=platform.python_version(),sklearn=sklearn.__version__,seconds=time.time()-start,
        galleries=21,sensitivity_galleries=20,full_gallery_writers=96,settings=len(diagnostics),
        primary_tests=8,representations=list(REPS),hand_models=list(HAND_MODELS),
        query_reference_image_overlap=False,query_reference_session_overlap=False,
        attribute_fit_sessions=[1],query_sessions=[2,3],
        inference='Conditional on fitted models and observed gallery; eight Holm-adjusted primary comparisons; remaining comparisons descriptive.',
        interpretation='Independent-dataset, separate-page and cross-session evaluation with within-dataset refitting; not cross-dataset transfer of a Bangla-trained model.')
    (OUT/'run_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    (OUT/'id_summary.partial.csv').unlink(missing_ok=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
