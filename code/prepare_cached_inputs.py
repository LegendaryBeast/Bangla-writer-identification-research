#!/usr/bin/env python3
"""Bind verified cache input paths to this installation without changing hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/wid_fusion_robustness_2026-11-08'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for filename in ['nested_development_provenance.json','calibration_seed_provenance.json']:
        path=RUN/filename;meta=json.loads(path.read_text());out={}
        for name,expected in meta['input_sha256'].items():
            # Published records use repository-relative paths. Local code-path
            # strings are absolute in the original cache loader; data stay relative.
            parts=Path(name).parts
            if Path(name).is_absolute():
                positions=[i for i,x in enumerate(parts) if x in {'code','results'}]
                if not positions:raise ValueError('Unrecognized cache input path')
                relative=Path(*parts[positions[-1]:])
            else:relative=Path(name)
            current=ROOT/relative
            if not current.is_file() or digest(current)!=expected:
                raise ValueError(f'Cache input changed: {relative}; use --refit-development to refit.')
            key=str(current) if relative.parts[0]=='code' else str(relative)
            out[key]=expected
        meta['input_sha256']=out
        path.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
        print('Verified and bound cache:',filename)


if __name__=='__main__':main()
