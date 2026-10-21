#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <agent_folder_name>"
    echo "Example: $0 sig_normalizer"
    exit 1
fi

AGENT_FOLDER="$1"

if [ ! -d "$AGENT_FOLDER" ]; then
    echo "Error: Agent directory '$AGENT_FOLDER' does not exist."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "Error: .env file not found in project root."
    exit 1
fi

set -a
source .env
set +a

if [ -z "${GOOGLE_CLOUD_PROJECT:-}" ] || [ "${GOOGLE_CLOUD_PROJECT}" = "your-project-id" ]; then
    echo "Error: GOOGLE_CLOUD_PROJECT is not set in .env."
    exit 1
fi

DEPLOY_REGION="${AGENT_LOCATION:-${RESOURCE_LOCATION:-us-central1}}"
if [ "$DEPLOY_REGION" = "global" ]; then
    DEPLOY_REGION="us-central1"
fi

# Ensure symlinked .env and release pointer exist inside the agent package
ln -sf ../.env "$AGENT_FOLDER/.env"
if [ -f "release/sig-normalizer.json" ]; then
    mkdir -p "$AGENT_FOLDER/release"
    cp "release/sig-normalizer.json" "$AGENT_FOLDER/release/sig-normalizer.json"
fi

echo "Deploying '$AGENT_FOLDER' to Vertex AI Agent Runtime..."
echo "  Project: $GOOGLE_CLOUD_PROJECT"
echo "  Region:  $DEPLOY_REGION"

if [ -x ".venv/bin/adk" ]; then
    ADK_CMD=".venv/bin/adk"
    PYTHON_CMD=".venv/bin/python"
else
    ADK_CMD="uv run adk"
    PYTHON_CMD="uv run python"
fi

EXISTING_ENGINE_ID=$($PYTHON_CMD ./get_agent_id.py "$AGENT_FOLDER" 2>/dev/null || true)

echo "  Telemetry: --otel_to_cloud (Cloud Trace, Cloud Monitoring, Cloud Logging)"
echo "  Prompt-Response Logging: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=${OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:-true}"

if [ -n "$EXISTING_ENGINE_ID" ]; then
    echo "  Updating existing Reasoning Engine in-place: $EXISTING_ENGINE_ID"
    PYTHONUNBUFFERED=1 $ADK_CMD deploy agent_engine \
        --project="$GOOGLE_CLOUD_PROJECT" \
        --region="$DEPLOY_REGION" \
        --display_name="$AGENT_FOLDER" \
        --agent_engine_id="$EXISTING_ENGINE_ID" \
        --otel_to_cloud \
        "$AGENT_FOLDER"
else
    echo "  Creating new Reasoning Engine instance..."
    PYTHONUNBUFFERED=1 $ADK_CMD deploy agent_engine \
        --project="$GOOGLE_CLOUD_PROJECT" \
        --region="$DEPLOY_REGION" \
        --display_name="$AGENT_FOLDER" \
        --otel_to_cloud \
        "$AGENT_FOLDER"
fi
