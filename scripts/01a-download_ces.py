"""
Purpose : Download the official 2024 and 2025 Cooperative Election Study (CES)
          files from Harvard Dataverse so the cleaning pipeline can use them.
Author  : Vijval Rallan (Member 1)
Date    : 2026-10-24
Inputs  : Harvard Dataverse data files configured in FILES:
          - CES 2024: CCES24_Common_OUTPUT_vv_topost_final.csv (file ID 12050325)
          - CES 2025: CES25_Common.dta (file ID 13600926)
          No local input data files are required; internet access and pyDataverse are required.
Outputs : data/01-raw_data/ces/CCES24_Common_OUTPUT_vv_topost_final.csv  (gitignored)
          data/01-raw_data/ces/CES25_Common.dta                            (gitignored)
          data/01-raw_data/ces/CES24_sample.csv  (committed)
          data/01-raw_data/ces/CES25_sample.dta  (committed)
          data/01-raw_data/ces/ces_download_manifest.json, including file IDs, source URLs,
          file sizes, SHA-256 checksums, download/existing-file status, and the samples.
Notes   : Creates the CES directory if needed. Checks that downloads are non-empty and
          checks the 2025 response for a Stata-file signature before saving. Writes through
          a temporary file before replacing the destination. Records SHA-256 checksums.
          An existing non-empty 2024 file is reused and checksummed. An existing non-empty
          2025 file currently raises FileExistsError rather than silently reusing it.
          Samples: the full files are too large for GitHub, so after the download the
          script saves a 1,000-row sample of each year next to them. Rows are drawn
          evenly from every state (seed 2453), so every 2026 race state appears in both
          years, which 02-clean_data.py requires. Every column is kept. The 2024 sample
          keeps each value exactly as written in the CSV; the 2025 sample keeps the raw
          codes with their Stata value labels, so pd.read_stata reads the same labels
          from it as from the full file. Stata's extended missing codes (.a to .z)
          become plain missing values in the sample.
          With --sample-only the script skips the download and only rebuilds the
          samples from full files already on disk.
Run     : uv run python scripts/01a-download_ces.py                 (download, then sample)
          uv run python scripts/01a-download_ces.py --sample-only   (sample only)
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd
from pandas.io.stata import StataReader

from common import CES_DIR, SEED

BASE_URL = "https://dataverse.harvard.edu"

FILES = {
    "ces_2024": {
        "file_id": 12050325,
        "filename": "CCES24_Common_OUTPUT_vv_topost_final.csv",
    },
    "ces_2025": {
        "file_id": 13600926,
        "filename": "CES25_Common.dta",
    },
}

MANIFEST_FILE = CES_DIR / "ces_download_manifest.json"

# Size of each committed sample. Rows are spread evenly across the states in
# the file, then topped up at random to exactly this many.
SAMPLE_ROWS = 1000
SAMPLE_FILES = {
    "ces_2024": "CES24_sample.csv",
    "ces_2025": "CES25_sample.dta",
}
STATE_COLUMN = "inputstate"


def sha256_file(path):
    """Calculate a SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def pick_sample_rows(states, n_rows, seed):
    """Choose n_rows row positions, spread evenly across states.

    `states` holds one state value per row. Each state gives the same number of
    rows (or all of its rows, if it has fewer); the rest are drawn at random
    from the rows not yet chosen. Returns the chosen positions in file order.
    """
    rows = pd.DataFrame({"state": states.to_numpy()})
    rows = rows[rows["state"].notna() & (rows["state"].astype(str) != "")]
    n_states = rows["state"].nunique()
    per_state = max(1, n_rows // n_states)

    chosen = rows.groupby("state").sample(
        frac=1, random_state=seed
    ).groupby("state").head(per_state)

    remaining = rows.drop(chosen.index)
    top_up = remaining.sample(
        n=min(n_rows - len(chosen), len(remaining)), random_state=seed
    )
    return sorted(chosen.index.tolist() + top_up.index.tolist())


def sample_ces_2024(full_path, sample_path, n_rows, seed):
    """Save n_rows of the 2024 CSV with every column, values unchanged.

    Every column is read as text and nothing is treated as missing, so each
    value is written back exactly as it appears in the full file.
    """
    ces = pd.read_csv(full_path, dtype=str, keep_default_na=False)
    keep = pick_sample_rows(ces[STATE_COLUMN], n_rows, seed)
    ces.iloc[keep].to_csv(sample_path, index=False)
    return len(keep), ces.shape[1], ces.iloc[keep][STATE_COLUMN].nunique()


def sample_ces_2025(full_path, sample_path, n_rows, seed):
    """Save n_rows of the 2025 Stata file with every column and its labels.

    The raw numeric codes are kept and the file's value labels and variable
    labels are written back, so pd.read_stata (as in 02-clean_data.py) turns
    the sample's codes into the same labels as the full file's.
    """
    with StataReader(full_path) as reader:
        codes = reader.read(convert_categoricals=False)
        label_sets = reader.value_labels()
        variable_labels = reader.variable_labels()
        # The name of the label set attached to each column, in column order.
        # pandas has no public accessor for this, hence the private attribute.
        label_set_names = reader._lbllist

    value_labels = {
        column: label_sets[set_name]
        for column, set_name in zip(codes.columns, label_set_names)
        if set_name in label_sets
    }

    keep = pick_sample_rows(codes[STATE_COLUMN], n_rows, seed)
    sample = codes.iloc[keep].reset_index(drop=True)

    # Text longer than Stata's 2,045-character limit has to be stored as strL.
    long_text = [
        column for column in sample.columns
        if sample[column].dtype == object
        and sample[column].astype(str).str.len().max() > 2045
    ]

    sample.to_stata(
        sample_path,
        write_index=False,
        version=118,
        value_labels=value_labels,
        variable_labels=variable_labels,
        convert_strl=long_text,
    )
    return len(keep), sample.shape[1], sample[STATE_COLUMN].nunique()


def build_samples(ces_dir, n_rows=SAMPLE_ROWS, seed=SEED):
    """Write the committed samples from the full files in ces_dir.

    Returns manifest entries for the two samples.
    """
    samplers = {"ces_2024": sample_ces_2024, "ces_2025": sample_ces_2025}
    entries = {}

    for name, sampler in samplers.items():
        full_path = ces_dir / FILES[name]["filename"]
        sample_path = ces_dir / SAMPLE_FILES[name]

        if not full_path.exists():
            raise FileNotFoundError(
                f"{full_path} is missing. Run this script without "
                "--sample-only to download it first."
            )

        print(f"Sampling {full_path.name}...")
        rows, columns, states = sampler(full_path, sample_path, n_rows, seed)

        entries[name] = {
            "filename": sample_path.name,
            "sampled_from": full_path.name,
            "rows": rows,
            "columns": columns,
            "states": states,
            "seed": seed,
            "size_bytes": sample_path.stat().st_size,
            "sha256": sha256_file(sample_path),
        }
        print(f"  Saved: {sample_path.name}")
        print(f"  {rows:,} rows, {columns:,} columns, {states} states")

    return entries


def download_full_files(manifest):
    """Download the full CES files (or reuse 2024), adding each to the manifest."""
    # Imported here so that --sample-only works without pyDataverse.
    from pyDataverse.api import DataAccessApi

    api = DataAccessApi(base_url=BASE_URL)

    for name, details in FILES.items():
        file_id = details["file_id"]
        destination = CES_DIR / details["filename"]

        url = f"{BASE_URL}/api/access/datafile/{file_id}"
        if name == "ces_2025":
            url += "?format=original"

        if destination.exists() and destination.stat().st_size > 0:
            if name == "ces_2025":
                print(
                    f"CES 2025 destination already exists: {destination.name}"
                )
                print(
                    "Skipping it to avoid assuming the existing file "
                    "is a valid Stata file."
                )
                raise FileExistsError(
                    f"Please rename or move {destination} before downloading "
                    "a fresh copy."
                )

            print(f"Already exists; skipping download: {destination.name}")
            status = "existing_file_checksum_recorded"

        else:
            print(f"Downloading {name} (file ID: {file_id})...")

            if name == "ces_2025":
                response = api.get_datafile(
                    file_id,
                    data_format="original",
                    is_pid=False,
                )
            else:
                response = api.get_datafile(
                    file_id,
                    is_pid=False,
                )

            response.raise_for_status()

            if not response.content:
                raise RuntimeError(f"Downloaded file is empty: {name}")

            # Check the Stata file signature before saving CES 2025.
            # Stata .dta files begin with a format-specific header.
            if name == "ces_2025":
                valid_stata_header = (
                    response.content.startswith(b"<stata_dta>")
                    or response.content[:1] in (b"\x6c", b"\x72", b"\x73")
                )

                if not valid_stata_header:
                    raise RuntimeError(
                        "CES 2025 response does not appear to be a Stata "
                        ".dta file. The file was NOT saved. "
                        f"First 32 bytes: {response.content[:32]!r}. "
                        f"Content-Type: "
                        f"{response.headers.get('Content-Type')!r}"
                    )

            # Write to a temporary file before replacing the destination.
            temporary_file = destination.with_suffix(
                destination.suffix + ".part"
            )

            try:
                temporary_file.write_bytes(response.content)
                temporary_file.replace(destination)
            finally:
                if temporary_file.exists():
                    temporary_file.unlink()

            status = "downloaded"

        file_info = {
            "file_id": file_id,
            "url": url,
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "status": status,
        }

        manifest["files"][name] = file_info

        print(f"  Saved: {destination.name}")
        print(f"  Size: {file_info['size_bytes']:,} bytes")
        print(f"  SHA-256: {file_info['sha256']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download the CES 2024 and 2025 files and save a sample of each."
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="skip the download and only rebuild the samples from full files on disk",
    )
    args = parser.parse_args()

    CES_DIR.mkdir(parents=True, exist_ok=True)

    if args.sample_only and MANIFEST_FILE.exists():
        # Keep the record of the full files from the last download.
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    else:
        manifest = {
            "source": BASE_URL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {},
        }

    if not args.sample_only:
        download_full_files(manifest)

    manifest["samples"] = build_samples(CES_DIR)
    manifest["samples_created_at_utc"] = datetime.now(timezone.utc).isoformat()

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nManifest saved: {MANIFEST_FILE}")
    print("All CES files and samples are ready.")


if __name__ == "__main__":
    main()
