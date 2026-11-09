#!/usr/bin/env python3
"""Fetch a deterministic balanced MPHD paragraph subset, with ZIP CRC checks.

Uses the published ZIP directory saved during the access audit. No dataset
image is added to the code or submission archives. No remote code is executed.
"""
import concurrent.futures
import hashlib
import io
import json
import struct
import time
import zlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset/mphd"
OUT = ROOT / "results/external_replication_2026-11-08"
URL = "https://zenodo.org/api/records/23557640/files/MPHD_v2.zip/content"


class RemoteArchive(io.RawIOBase):
    """Seekable range reader; never downloads the complete ZIP implicitly."""
    def __init__(self, size):
        self.position = 0
        self.size = size
        self.session = requests.Session()

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        if size == 0:
            return b""
        response = self.session.get(URL, headers={"Range": f"bytes={self.position}-{self.position+size-1}"}, timeout=60)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != size:
            raise ValueError("Partial archive read failed")
        if not response.headers.get("Content-Range", "").startswith(f"bytes {self.position}-"):
            raise ValueError("Partial archive position mismatch")
        self.position += size
        return response.content


def bootstrap_metadata():
    DATA.mkdir(exist_ok=True, parents=True)
    if (DATA / "metadata.csv").exists() and (DATA / "zip_inventory.json").exists():
        return
    response = requests.get("https://zenodo.org/api/records/23557640", timeout=30)
    response.raise_for_status()
    record = response.json()
    member = next(f for f in record["files"] if f["key"] == "MPHD_v2.zip")
    if member["checksum"] != "md5:536a34e05a2d2165c94ae1210fdd107b":
        raise ValueError("The published dataset version has changed; audit before proceeding")
    (DATA / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    with zipfile.ZipFile(RemoteArchive(member["size"])) as archive:
        rows = [dict(name=x.filename, size=x.file_size, compressed_size=x.compress_size,
            crc=x.CRC, offset=x.header_offset) for x in archive.infolist()]
        (DATA / "zip_inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
        (DATA / "metadata.csv").write_bytes(archive.read("MPHD_v2/metadata.csv"))


def download_member(info):
    name = info["name"]
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe archive path")
    dest = DATA / relative
    if dest.exists():
        content = dest.read_bytes()
        if len(content) == info["size"] and zlib.crc32(content) == info["crc"]:
            return name, hashlib.sha256(content).hexdigest()
    start = info["offset"]
    end = start + 30 + len(name.encode()) + 4096 + info["compressed_size"] - 1
    for attempt in range(5):
        try:
            response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError("Server did not honor the partial download")
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise ValueError("Unexpected Content-Range")
            header = response.content[:30]
            fields = struct.unpack("<I5H3I2H", header)
            if fields[0] != 0x04034B50 or fields[2] & 1:
                raise ValueError("Invalid/encrypted ZIP member")
            method, nname, nextra = fields[3], fields[-2], fields[-1]
            if response.content[30:30+nname].decode() != name:
                raise ValueError("ZIP member name mismatch")
            body_start = 30 + nname + nextra
            if body_start + info["compressed_size"] > len(response.content):
                end += 65536
                continue
            compressed = response.content[body_start:body_start+info["compressed_size"]]
            content = zlib.decompress(compressed, -15) if method == 8 else compressed if method == 0 else None
            if content is None or len(content) != info["size"] or zlib.crc32(content) != info["crc"]:
                raise ValueError("ZIP size/CRC verification failed")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return name, hashlib.sha256(content).hexdigest()
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError("ZIP member could not be retrieved")


def main():
    bootstrap_metadata()
    metadata = pd.read_csv(DATA / "metadata.csv")
    assert metadata.subject_id.is_unique
    assert set(metadata.handedness_code.unique()) == {0, 1}
    assert (metadata.handedness_name == metadata.handedness_code.map({0: "Right-handed", 1: "Left-handed"})).all()
    counts = metadata.groupby("handedness_code").size()
    size = int(counts.min())
    rng = np.random.default_rng(20261108)
    ids = []
    for hand in (0, 1):
        available = np.sort(metadata.loc[metadata.handedness_code == hand, "subject_id"].to_numpy())
        ids.extend(available if len(available) == size else np.sort(rng.choice(available, size, replace=False)))
    selected = metadata[metadata.subject_id.isin(ids)].sort_values("subject_id").copy()
    inventory = {row["name"]: row for row in json.loads((DATA / "zip_inventory.json").read_text())}
    names = ["MPHD_v2/LICENSE"]
    image_rows = []
    for row in selected.itertuples():
        for field in ("path_json", "path_text1_image", "path_text2_image"):
            original = getattr(row, field).replace("\\", "/")
            name = "MPHD_v2/" + original.split("/", 1)[1]
            if name not in inventory:
                raise ValueError(f"Missing published member: {name}")
            names.append(name)
            if field != "path_json":
                text = 1 if field == "path_text1_image" else 2
                image_rows.append(dict(image_id=f"MPHD_{row.subject_id:03d}_T{text}",
                    writer_id=f"MPHD_{row.subject_id:03d}", original_subject_id=row.subject_id,
                    label=int(row.handedness_code), text=text, parent_form_id=f"MPHD_{row.subject_id:03d}_form",
                    relative_path=name, source_page_separation=False))
    OUT.mkdir(exist_ok=True, parents=True)
    selected.to_csv(OUT / "cohort_metadata.csv", index=False)
    pd.DataFrame(image_rows).to_csv(OUT / "images.csv", index=False)
    hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for i, (name, digest) in enumerate(executor.map(download_member, [inventory[n] for n in names]), 1):
            hashes[name] = digest
            if i % 30 == 0:
                print(f"Verified {i}/{len(names)} files", flush=True)
    for row in selected.itertuples():
        name = "MPHD_v2/" + row.path_json.replace("\\", "/").split("/", 1)[1]
        annotation = json.loads((DATA / name).read_text(encoding="utf-8-sig"))
        meta = annotation["meta"]
        assert int(meta["writer_id"]) == row.subject_id
        assert meta["handedness"] == row.handedness_name
    report = dict(dataset="MPHD v2", doi="10.5281/zenodo.23557640",
        source_url=URL, source_archive_md5="536a34e05a2d2165c94ae1210fdd107b",
        retrieval="Selected ZIP members via HTTP Range; each member checked against published ZIP CRC and size; SHA256 recorded. Full archive MD5 not independently checked because full archive was not downloaded.",
        seed=20261108, total_writers=len(metadata), release_counts={str(k): int(v) for k,v in counts.items()},
        selected_writers=len(selected), selected_per_hand=size, paragraph_images=len(image_rows),
        annotation_handedness_agreement=True, image_sha256=hashes,
        interpretation="Independent writers and dataset; two paragraphs from same scanned form; no new-session or separate-page claim.")
    (OUT / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Acquired {len(selected)} writers / {len(image_rows)} paragraph images; labels verified.", flush=True)


if __name__ == "__main__":
    main()
