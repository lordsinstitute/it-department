import json
import zipfile
from pathlib import Path

def build_artifact_zip(
    zip_path: Path,
    input_path: Path,
    output_path: Path,
    findings: dict,
    ledger_hash: str
) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    findings_json = json.dumps(findings, indent=2, ensure_ascii=False).encode("utf-8")
    checksums_txt = "\n".join([
        f"input_file={input_path.name}",
        f"output_file={output_path.name}",
        f"input_sha256={findings.get('checksums', {}).get('input_sha256','')}",
        f"output_sha256={findings.get('checksums', {}).get('output_sha256','')}",
        f"ledger_hash={ledger_hash}",
    ]).encode("utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if input_path.exists():
            z.write(str(input_path), arcname=f"input_{input_path.name}")
        if output_path.exists():
            z.write(str(output_path), arcname=f"output_{output_path.name}")
        z.writestr("findings.json", findings_json)
        z.writestr("checksums.txt", checksums_txt)
        z.writestr("ledger_hash.txt", (ledger_hash + "\n").encode("utf-8"))

    return zip_path