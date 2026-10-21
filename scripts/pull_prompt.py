#!/usr/bin/env python3
"""Pull a server-side prompt version from Vertex AI Prompt Management into prompts/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vertexai

from config import settings
from prompt_registry import _extract_instruction_text

RELEASE_FILE = ROOT / "release" / "sig-normalizer.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-id", required=True, help="Server-side version_id (e.g. '3')")
    parser.add_argument("--out", required=True, help="Output file path (e.g. prompts/v3_pulled.txt)")
    args = parser.parse_args()

    doc = json.loads(RELEASE_FILE.read_text(encoding="utf-8"))
    raw_vid = str(args.version_id).strip()
    if raw_vid in doc.get("versions", {}):
        vid = str(doc["versions"][raw_vid]["version_id"])
    else:
        vid = raw_vid[1:] if raw_vid.startswith("v") and raw_vid[1:].isdigit() else raw_vid

    client = vertexai.Client(
        project=settings.project_id,
        location=settings.resource_location,
    )
    prompt_v = client.prompts.get_version(
        prompt_id=str(doc["prompt_id"]),
        version_id=vid,
    )
    text = _extract_instruction_text(prompt_v)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"Pulled prompt_id={doc['prompt_id']} version_id={vid} -> {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
