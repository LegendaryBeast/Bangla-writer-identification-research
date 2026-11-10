#!/usr/bin/env python3
"""Numerically scaled Platt fit retaining the original C=10000 objective.

Only training OOF scores determine the shift/scale and calibration parameters.
The raw-slope L2 penalty is transformed algebraically, not retuned.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold

import run_wid_handedness as wh


def fit_platt(scores,labels,C=1e4):
    scores=np.asarray(scores,float);labels=np.asarray(labels,float)
    assert np.isfinite(scores).all() and set(np.unique(labels))=={0.,1.}
    center=float(scores.mean());scale=float(scores.std())
    if scale==0:scale=1.
    z=(scores-center)/scale
    penalty=1./(C*len(scores)*scale**2)
    assert np.isfinite(penalty) and penalty>0
    def objective(theta):
        a,b=theta;linear=a*z+b
        loss=np.mean(np.logaddexp(0.,linear)-labels*linear)+.5*penalty*a*a
        error=expit(linear)-labels
        gradient=np.array([np.mean(error*z)+penalty*a,error.mean()])
        return loss,gradient
    prior=float(labels.mean())
    result=minimize(objective,np.array([0.,np.log(prior/(1-prior))]),jac=True,method='L-BFGS-B',
        options=dict(gtol=1e-10,ftol=1e-15,maxiter=2000,maxls=100))
    value,gradient=objective(result.x)
    norm=float(np.max(np.abs(gradient)))
    if not np.isfinite(result.x).all() or not np.isfinite(value) or norm>1e-7:
        raise RuntimeError(f'Calibration not numerically certified: {result.message}; gradient={norm}')
    a,b=map(float,result.x)
    diagnostic=dict(center=center,scale=scale,scaled_slope=a,scaled_intercept=b,
        raw_slope=a/scale,raw_intercept=b-a*center/scale,C=C,n_training_scores=len(scores),
        mean_penalized_logistic_objective=float(value),gradient_infinity_norm=norm,
        optimizer_success=bool(result.success),optimizer_status=int(result.status),iterations=int(result.nit),
        accepted_by='Finite objective/parameters and analytic gradient infinity norm <= 1e-7')
    return diagnostic


def lowo_writer(X,y,cov,groups,train_mask,test_idx,adjust,seed):
    tr=np.flatnonzero(train_mask)
    raw=wh.fit_score(X[tr],y[tr],cov[tr],X[test_idx],cov[test_idx],adjust)
    oof=np.empty(len(tr))
    for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(tr,y[tr],groups[tr]):
        oof[b]=wh.fit_score(X[tr[a]],y[tr[a]],cov[tr[a]],X[tr[b]],cov[tr[b]],adjust)
    diagnostic=fit_platt(oof,y[tr])
    logit=diagnostic['scaled_slope']*((raw-diagnostic['center'])/diagnostic['scale'])+diagnostic['scaled_intercept']
    assert np.isfinite(logit).all()
    return test_idx,raw,logit,diagnostic
