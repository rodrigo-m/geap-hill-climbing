#!/usr/bin/env python3
"""Retrieve the latest Reasoning Engine ID for the deployed ADK agent."""

import os
import sys
from dotenv import load_dotenv
import google.auth
from google.auth.transport.requests import Request
import requests

load_dotenv()

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("AGENT_LOCATION") or os.getenv("RESOURCE_LOCATION") or "us-central1"
if LOCATION == "global":
    LOCATION = "us-central1"

if not PROJECT or PROJECT == "your-project-id":
    print("Error: GOOGLE_CLOUD_PROJECT not set in .env", file=sys.stderr)
    sys.exit(1)

DISPLAY_NAME_FILTER = sys.argv[1] if len(sys.argv) > 1 else None

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(Request())

url = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/locations/{LOCATION}/reasoningEngines"
headers = {
    "Authorization": f"Bearer {credentials.token}",
    "Content-Type": "application/json",
    "x-goog-user-project": PROJECT,
}

resp = requests.get(url, headers=headers, timeout=30)
if not resp.ok:
    print(f"API Error ({resp.status_code}): {resp.text}", file=sys.stderr)
    sys.exit(1)

engines = resp.json().get("reasoningEngines", [])
if DISPLAY_NAME_FILTER:
    filtered = [
        e
        for e in engines
        if e.get("displayName")
        in (DISPLAY_NAME_FILTER, DISPLAY_NAME_FILTER.replace("_", "-"))
    ]
    if filtered:
        engines = filtered

if not engines:
    print("No matching reasoning engines found.", file=sys.stderr)
    sys.exit(1)

engines.sort(key=lambda x: x.get("createTime", ""), reverse=True)
latest = engines[0]
engine_id = latest["name"].split("/")[-1]
print(engine_id)
