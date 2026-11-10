#!/usr/bin/env python3
"""Independent numerical checks of the calibration reparameterization."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from csafe_calibration import fit_platt


def main():
    rng=np.random.default_rng(411);scores=rng.normal(size=256)
    labels=rng.binomial(1,expit(.4+1.2*scores))
    fit=fit_platt(scores,labels)
    p=expit(fit['scaled_slope']*((scores-fit['center'])/fit['scale'])+fit['scaled_intercept'])
    reference=LogisticRegression(C=1e4,tol=1e-10,max_iter=2000).fit(scores[:,None],labels)
    difference=float(np.max(np.abs(p-reference.predict_proba(scores[:,None])[:,1])))
    assert difference<1e-6
    a,b=.7,-.3;scale=1e6;center=7e8;raw=scale*scores+center
    beta=a/scale;intercept=b-a*center/scale
    transformed=np.mean(np.logaddexp(0,a*scores+b)-labels*(a*scores+b))+.5*a*a/(1e4*len(scores)*scale**2)
    original=np.mean(np.logaddexp(0,beta*raw+intercept)-labels*(beta*raw+intercept))+.5*beta*beta/(1e4*len(scores))
    objective_error=float(abs(transformed-original));assert objective_error<1e-12
    large=fit_platt(raw,labels);constant=fit_platt(np.ones(256)*1e8,labels)
    assert large['gradient_infinity_norm']<1e-7 and constant['gradient_infinity_norm']<1e-7
    root=Path(__file__).resolve().parents[1];dest=root/'results/csafe_replication_2026-11-08/calibration_objective_validation.json'
    result=dict(passed=True,reference_probability_max_difference=difference,
        raw_vs_transformed_objective_difference=objective_error,
        large_offset_gradient=large['gradient_infinity_norm'],constant_score_gradient=constant['gradient_infinity_norm'],
        test_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
