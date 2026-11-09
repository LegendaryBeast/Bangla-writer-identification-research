#!/usr/bin/env python3
"""Audit CSAFE session separation, writer nesting, hashes and reported values."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/csafe_replication_2026-11-08'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--saved-only',action='store_true')
    args=ap.parse_args()
    run=json.loads((OUT/'run_provenance.json').read_text())
    assert run['completed'] and run['galleries']==21 and run['settings']==196 and run['primary_tests']==8
    for name,digest in run['input_code_sha256'].items():assert sha(ROOT/name)==digest,name
    feature=json.loads((OUT/'feature_provenance.json').read_text())
    assert feature['n_images']==864 and not feature['invalid_images']
    equivalence=json.loads((OUT/'feature_equivalence.json').read_text())
    assert equivalence['passed']
    warm=json.loads((OUT/'handcrafted_cache_provenance.json').read_text())
    assert sha(ROOT/'code/prefetch_csafe_handcrafted.py')==warm['code_sha256']
    for key in ('code','checkpoints','manifest_sha256','neural_device'):
        assert warm['main_extractor_signature'][key]==feature[key]
    for name,digest in feature['code'].items():assert sha(ROOT/'code'/name)==digest,name
    if not args.saved_only:
        for info in feature['checkpoints'].values():
            assert sha(ROOT/'dataset/model_cache/checkpoints'/info['file'])==info['sha256']
    U=pd.read_csv(OUT/'units.csv');A=pd.read_csv(OUT/'images.csv')
    assert len(U)==864 and U.writer_id.nunique()==96
    assert np.array_equal(U.image_id,A.image_id) and U.image_sha256.is_unique
    assert (U.groupby('writer_id').size()==9).all()
    assert (U.groupby('writer_id').attribute_train.sum()==6).all()
    assert (U.groupby('writer_id').label.nunique()==1).all()
    assert U.groupby('writer_id').label.first().value_counts().to_dict()=={0:48,1:48}
    assert (U.loc[U.attribute_train,'session']==1).all()
    assert (U.loc[U.gallery,'session']==1).all() and (U.loc[U['query'],'session']>1).all()
    acquired=json.loads((OUT/'acquisition.json').read_text())
    for row in A.itertuples():
        expected=U.loc[U.image_id==row.image_id,'image_sha256'].iloc[0]
        assert acquired['image_sha256'][row.relative_path]==expected
        if not args.saved_only:assert sha(ROOT/'dataset/csafe'/row.relative_path)==expected
    with np.load(OUT/'features.npz') as F:
        assert np.array_equal(F['image_id'],U.image_id)
        for rep,dim in [('hc183',183),('stroke',208),('resnet18',512),('vit_b_16',768),('global_intensity',7)]:
            assert F[rep].shape==(864,dim)
            if rep in ('resnet18','vit_b_16'):
                assert np.isfinite(F[rep][U.deep_required]).all()
                assert np.isnan(F[rep][~U.deep_required]).all()
            else:assert np.isfinite(F[rep]).all()
    manifest=json.loads((OUT/'fitting_manifest.json').read_text())
    assert len(manifest)==49
    n_outer,n_models,n_inner=0,0,0
    max_calibration_gradient=0.
    non_success_status_with_certified_gradient=0
    seen=set()
    for cell in manifest:
        writers=set(cell['writers']);refs=U[U.image_id.isin(cell['reference_image_ids'])]
        query=U[U.image_id.isin(cell['query_image_ids'])]
        aux=U[U.image_id.isin(cell['attribute_training_image_ids'])]
        assert set(refs.writer_id)==set(query.writer_id)==set(aux.writer_id)==writers
        assert refs.condition.eq('S1_LND_r01').all() and refs.session.eq(1).all()
        assert query.condition.eq(cell['condition']).all() and query.session.gt(1).all()
        assert not set(refs.image_id)&set(query.image_id)
        assert not set(refs.parent_form_id)&set(query.parent_form_id)
        assert aux.session.eq(1).all() and aux.attribute_train.all()
        assert (aux.groupby('writer_id').size()==6).all()
        key=cell['cache_key']
        if key in seen:continue
        seen.add(key)
        cached=json.loads((OUT/'attribute_cache'/(key+'.json')).read_text())
        assert {r['outer_writer'] for r in cached}==writers
        with np.load(OUT/'attribute_cache'/(key+'.npz')) as z:
            assert z['outer'].shape==(3,len(writers))
            assert z['development'].shape==(len(writers),3,len(writers))
            mask=np.eye(len(writers),dtype=bool)
            for c in range(3):
                assert np.isnan(z['development'][:,c,:][mask]).all()
                assert np.isfinite(z['development'][:,c,:][~mask]).all()
        for row in cached:
            n_outer+=1;outer=row['outer_writer'];expected=writers-{outer}
            assert set(row['final_attribute_train_writers'])==expected
            assert row['final_training_pages']==6*len(expected)
            stages=[(expected,row['final_calibration_folds'],set())]
            calibration_fits=[row['final_platt_fit']]
            held=[]
            for dev in row['development']:
                a,b=set(dev['train_writers']),set(dev['held_writers'])
                assert not a&b and a|b==expected
                assert dev['training_pages']==6*len(a)
                held.extend(b);stages.append((a,dev['calibration_folds'],b))
                calibration_fits.append(dev['platt_fit'])
            assert len(held)==len(expected) and set(held)==expected
            for fit in calibration_fits:
                assert fit['C']==10000 and fit['scale']>0
                assert np.isfinite(fit['mean_penalized_logistic_objective'])
                assert fit['gradient_infinity_norm']<=1e-7
                max_calibration_gradient=max(max_calibration_gradient,fit['gradient_infinity_norm'])
                non_success_status_with_certified_gradient+=not fit['optimizer_success']
            for training,folds,forbidden in stages:
                n_models+=1;held=[]
                for fold in folds:
                    n_inner+=1;a,b=set(fold['train_writers']),set(fold['calibration_writers'])
                    assert not a&b and a|b==training
                    assert not (a|b)&(forbidden|{outer})
                    held.extend(b)
                assert len(held)==len(training) and set(held)==training
    S=pd.read_csv(OUT/'id_summary.csv');W=pd.read_csv(OUT/'writer_outcomes.csv.gz')
    D=pd.read_csv(OUT/'diagnostics.csv');T=pd.read_csv(OUT/'paired_tests.csv')
    H=pd.read_csv(OUT/'handedness_summary.csv');P=pd.read_csv(OUT/'selected_parameters.csv')
    assert len(S)==980 and len(W)==50880 and len(D)==196 and len(H)==49 and len(P)==10176
    keys=['cohort','condition','rep','hand_model','method']
    recalculated=W.groupby(keys).agg(top1=('correct','mean'),n_writers=('writer_id','nunique'))
    paired=S.set_index(keys).join(recalculated,rsuffix='_checked')
    assert np.allclose(paired.top1,paired.top1_checked,atol=1e-14)
    assert np.array_equal(paired.n_writers,paired.n_writers_checked)
    basekey=['cohort','condition','rep','hand_model','writer_id']
    base=W[W.method=='identity_only'].set_index(basekey).correct
    for method in W.method.unique():
        rows=W[W.method==method].set_index(basekey)
        assert np.array_equal(rows.delta_vs_identity,rows.correct-base.reindex(rows.index))
    # Shared identity/oracle comparators must not depend on the auxiliary model.
    for method in ('identity_only','oracle_known_hand'):
        assert (S[(S.cohort=='full_balanced')&(S.method==method)].groupby(['condition','rep']).top1.nunique()==1).all()
    assert np.allclose(D.filtering_exact_joint,D.filtering_observed,atol=1e-14)
    primary=T[T.primary].sort_values('p_signflip')
    assert len(primary)==8 and primary.cohort.eq('full_balanced').all() and primary.hand_model.eq('hc183').all()
    assert set(primary.condition)=={'S2_LND_r01','S2_WOZ_r01'}
    adjusted=np.maximum.accumulate([min(1.,(8-i)*p) for i,p in enumerate(primary.p_signflip)])
    assert np.allclose(primary.p_holm_primary,adjusted)
    assert ((T.ci_lo<=T.delta_top1+1e-14)&(T.ci_hi>=T.delta_top1-1e-14)).all()
    expected_t=np.logspace(-3,0,31)
    assert np.isclose(P.temperature.to_numpy()[:,None],expected_t).any(1).all()
    assert P.fusion_weight.isin([0,.1,.2,.3,.5,.7,1,1.5,2,3,5]).all()
    for row in H.itertuples():
        rows=W[(W.cohort==row.cohort)&(W.condition==row.condition)&(W.hand_model==row.hand_model)
            &(W.rep=='resnet18')&(W.method=='identity_only')]
        assert np.isclose(rows.hand_correct.mean(),row.accuracy)
    result=dict(passed=True,settings=196,summary_rows=len(S),writer_method_outcomes=len(W),primary_tests=8,
        outer_writer_fits_audited=n_outer,attribute_models_audited=n_models,calibration_folds_audited=n_inner,
        unique_attribute_caches=len(seen),input_code_hashes_verified=True,
        raw_images_checkpoints_verified=not args.saved_only,separate_query_gallery_pages_and_sessions=True,
        auxiliary_training_session1_only=True,zero_outer_writer_in_attribute_fitting=True,
        zero_development_writer_in_its_attribute_fit=True,exact_filter_identity_verified=True,
        margin_reconstruction_checked_during_run=True,shared_base_oracle_across_attribute_models=True,
        full_source_zip_md5_independently_verified=False)
    result.update(calibration_gradient_certificate=True,
        maximum_calibration_gradient=max_calibration_gradient,
        optimizer_status_non_success_but_gradient_certified=non_success_status_with_certified_gradient)
    dest=OUT/('validation_saved_only.json' if args.saved_only else 'validation.json')
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
