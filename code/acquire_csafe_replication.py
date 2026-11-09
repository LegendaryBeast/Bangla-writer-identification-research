#!/usr/bin/env python3
"""Acquire a fixed CSAFE v2 balanced subset, verifying selected ZIP members."""
import concurrent.futures
import hashlib
import io
import json
import re
import struct
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dataset/csafe'
OUT=ROOT/'results/csafe_replication_2026-11-08'
PINNED={
 'surveydata_v2.csv':'1bb9e53728ba71cf19af23d989f6bf5b',
 'Session 1.zip':'1b8fb1c877156905b247e8f5ef13bdc0',
 'Session 2.zip':'acfbd47a665acc96545f6e8cef0d25a3',
 'Session 3.zip':'8a8efdfc38d0ee8e4e7daab8ba21ef3c'}


class RemoteArchive(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.position=url,size,0
        self.session=requests.Session()
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        return self.position
    def tell(self): return self.position
    def seekable(self): return True
    def read(self,size=-1):
        if size<0: size=self.size-self.position
        if size==0: return b''
        end=min(self.position+size,self.size)-1
        response=self.session.get(self.url,headers={'Range':f'bytes={self.position}-{end}'},timeout=60)
        response.raise_for_status()
        assert response.status_code==206 and len(response.content)==end-self.position+1
        assert response.headers.get('Content-Range','').startswith(f'bytes {self.position}-')
        self.position=end+1
        return response.content


def initialize():
    DATA.mkdir(exist_ok=True,parents=True)
    record_file=DATA/'record.json'
    if not record_file.exists():
        r=requests.get('https://api.figshare.com/v2/articles/10062203/versions/2',timeout=30)
        r.raise_for_status();record_file.write_text(json.dumps(r.json(),indent=2)+'\n')
    record=json.loads(record_file.read_text())
    assert record['version']==2
    for f in record['files']:
        if f['name'] in PINNED: assert f['computed_md5']==PINNED[f['name']]
        if f['name']=='surveydata_v2.csv':
            dest=DATA/f['name']
            if not dest.exists():
                r=requests.get(f['download_url'],timeout=30);r.raise_for_status();dest.write_bytes(r.content)
            assert hashlib.md5(dest.read_bytes()).hexdigest()==PINNED[f['name']]
        if f['name'].endswith('.zip'):
            dest=DATA/f['name'].replace('.zip','_inventory.json')
            if not dest.exists():
                with zipfile.ZipFile(RemoteArchive(f['download_url'],f['size'])) as z:
                    rows=[dict(name=x.filename,size=x.file_size,compressed_size=x.compress_size,
                        crc=x.CRC,offset=x.header_offset) for x in z.infolist()]
                dest.write_text(json.dumps(rows,indent=2)+'\n')
    return record


def download_member(task):
    info,url=task
    name=info['name'];relative=Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    dest=DATA/'selected'/relative
    if dest.exists():
        body=dest.read_bytes()
        if len(body)==info['size'] and zlib.crc32(body)==info['crc']:
            return str(Path('selected')/relative),hashlib.sha256(body).hexdigest()
    start=info['offset']
    end=start+30+len(name.encode())+4096+info['compressed_size']-1
    for attempt in range(6):
        try:
            r=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=60)
            r.raise_for_status()
            assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-')
            fields=struct.unpack('<I5H3I2H',r.content[:30])
            assert fields[0]==0x04034B50 and not fields[2]&1
            method,nname,nextra=fields[3],fields[-2],fields[-1]
            assert r.content[30:30+nname].decode()==name
            a=30+nname+nextra;b=a+info['compressed_size']
            if b>len(r.content):end+=65536;continue
            body=zlib.decompress(r.content[a:b],-15) if method==8 else r.content[a:b] if method==0 else None
            assert body is not None and len(body)==info['size'] and zlib.crc32(body)==info['crc']
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(body)
            return str(Path('selected')/relative),hashlib.sha256(body).hexdigest()
        except (requests.RequestException,AssertionError):
            if attempt==5:raise
            time.sleep(min(2**attempt,20))
    raise RuntimeError(name)


def main():
    record=initialize();meta=pd.read_csv(DATA/'surveydata_v2.csv')
    assert meta.wid.is_unique
    left=np.sort(meta.loc[meta.hand=='left','wid'].to_numpy())
    right=np.sort(meta.loc[meta.hand=='right','wid'].to_numpy())
    assert len(left)==48 and len(right)==423
    rng=np.random.default_rng(20261108)
    right=np.sort(rng.choice(right,len(left),replace=False))
    selected=meta[meta.wid.isin(np.r_[left,right])].sort_values('wid')
    OUT.mkdir(exist_ok=True,parents=True)
    selected[['wid','hand']].to_csv(OUT/'selected_writers.csv',index=False)
    timing=dict(selected_writers=len(selected),reported_days_after_enrolment={})
    for target,base in [('s2_dae','s1_dae'),('s3_dae','s1_dae'),('s3_dae','s2_dae')]:
        delta=pd.to_numeric(selected[target],errors='coerce')-pd.to_numeric(selected[base],errors='coerce')
        timing['reported_days_after_enrolment'][target+' minus '+base]=dict(nonmissing=int(delta.notna().sum()),
            minimum=float(delta.min()),median=float(delta.median()),maximum=float(delta.max()),nonpositive=int((delta<=0).sum()))
    (OUT/'session_timing_audit.json').write_text(json.dumps(timing,indent=2)+'\n')
    files={f['name']:f for f in record['files']}
    indexes={session:{r['name']:r for r in json.loads((DATA/f'Session {session}_inventory.json').read_text())}
        for session in (1,2,3)}
    configurations=[(1,p,r) for p in ('LND','WOZ') for r in (1,2,3)]+[(2,'LND',1),(2,'WOZ',1),(3,'WOZ',1)]
    rows,tasks=[],[]
    for writer in selected.itertuples():
        for session,prompt,repetition in configurations:
            name=f'Session {session}/w{writer.wid:04d}_s{session:02d}_p{prompt}_r{repetition:02d}.png'
            assert name in indexes[session],name
            gallery=session==1 and prompt=='LND' and repetition==1
            query=session>1
            rows.append(dict(image_id=Path(name).stem,writer_id=f'CSAFE_{writer.wid:04d}',
                original_wid=writer.wid,label=int(writer.hand=='left'),session=session,prompt=prompt,
                repetition=repetition,attribute_train=session==1,gallery=gallery,query=query,
                condition=f'S{session}_{prompt}_r{repetition:02d}',deep_required=gallery or query,
                parent_form_id=Path(name).stem,relative_path=str(Path('selected')/name)))
            tasks.append((indexes[session][name],files[f'Session {session}.zip']['download_url']))
    pd.DataFrame(rows).to_csv(OUT/'images.csv',index=False)
    hashes={};start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for i,(name,digest) in enumerate(executor.map(download_member,tasks),1):
            hashes[name]=digest
            if i%48==0:print(f'Acquired and verified {i}/{len(tasks)} pages; {time.time()-start:.1f}s',flush=True)
    report=dict(dataset='CSAFE Handwriting Database v2',doi=record['doi'],license=record['license'],
        seed=20261108,selected_left=48,selected_right=48,selected_writers=96,images=len(rows),
        source_survey_md5=PINNED['surveydata_v2.csv'],source_archive_md5=PINNED,
        retrieval='Selected original ZIP members checked against CRC/size; SHA256 recorded. Whole archive MD5 not independently checked because archives were not fully downloaded.',
        image_sha256=hashes,archive_file_ids={f['name']:f['id'] for f in record['files']},
        distinct_sessions=[1,2,3],attribute_training_pages=576,neural_feature_pages=384,
        ambiguous_or_missing_hand_excluded_before_sampling=4)
    (OUT/'acquisition.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Acquisition complete:',len(rows),'pages;',96,'writers',flush=True)


if __name__=='__main__':main()
