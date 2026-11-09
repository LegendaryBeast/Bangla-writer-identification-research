#!/usr/bin/env python3
"""Verify, regenerate artwork, or recompute the released analysis from features."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(script,*args):
    print('+',sys.executable,'code/'+script,*map(str,args),flush=True)
    subprocess.run([sys.executable,str(ROOT/'code'/script),*map(str,args)],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--verify',action='store_true',help='Check supplied numerical results; no model fitting')
    modes.add_argument('--figures',action='store_true',help='Regenerate every table and figure from supplied outputs')
    modes.add_argument('--full',action='store_true',help='Recompute the revised analysis from frozen features')
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--refit-development',action='store_true',help='Ignore the supplied development/calibration caches')
    a=p.parse_args()
    if a.full:
        if not a.refit_development:run('prepare_cached_inputs.py')
        else:
            for name in ['nested_development_provenance.json','calibration_seed_provenance.json']:
                path=ROOT/'results/wid_fusion_robustness_2026-11-08'/name
                if path.exists():path.rename(path.with_suffix('.json.bak'))
        run('run_revised_identification.py')
        run('run_revision_checks.py','--both-controls','--jobs',a.jobs)
        run('summarize_revision_checks.py')
        run('summarize_revision_checks.py','--own-only')
        run('run_revision_simulations.py')
        run('run_nested_fusion.py','--jobs',a.jobs)
        run('run_robustness_diagnostics.py','--jobs',a.jobs,'--nested',
            'results/wid_fusion_robustness_2026-11-08/nested_writer_outcomes.csv')
    if a.figures or a.full:
        for script in ['make_revision_paper.py','make_fusion_robustness_paper.py',
                       'make_representation_utility.py','run_margin_diagnostic.py']:
            run(script)
    run('verify_results.py')
    run('validate_fusion_robustness.py')


if __name__=='__main__':main()
