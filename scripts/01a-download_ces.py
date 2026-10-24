

from datetime import datetime, timezone
import hashlib
import json

from pyDataverse.api import DataAccessApi

from common import CES_DIR

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


def sha256_file(path):
    """Calculate a SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    CES_DIR.mkdir(parents=True, exist_ok=True)
    api = DataAccessApi(base_url=BASE_URL)

    manifest = {
        "source": BASE_URL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {},
    }

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

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"\nManifest saved: {MANIFEST_FILE}")
    print("All CES files are ready.")


if __name__ == "__main__":
    main()
