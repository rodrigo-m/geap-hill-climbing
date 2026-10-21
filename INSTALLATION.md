# Installation & Setup Guide

This guide walks through setting up the **Prescription Sig Normalizer** ADK agent, seeding prompt versions into **Vertex AI Prompt Management**, running the **GEAP Evaluation** hill-climbing suite, and deploying to **Vertex AI Agent Runtime (Reasoning Engine)**.

---

## 1. Prerequisites

- **Google Cloud SDK (`gcloud`)** installed and authenticated
- **Python 3.11+** and **[`uv`](https://docs.astral.sh/uv/)** installed
- A **Google Cloud project** with billing enabled

---

## 2. Clone & Install Dependencies

```bash
# Create a virtual environment and install dependencies via uv
uv venv
source .venv/bin/activate
uv pip install -e .
```

---

## 3. Authenticate & Enable Google Cloud APIs

Set your project ID (replace `your-project-id` with your own GCP project ID) and enable the required Vertex AI, Storage, and Observability APIs:

```bash
export PROJECT_ID="your-project-id"

gcloud auth login
gcloud auth application-default login
gcloud config set project "${PROJECT_ID}"
gcloud auth application-default set-quota-project "${PROJECT_ID}"

gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  cloudtrace.googleapis.com \
  telemetry.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com \
  iamcredentials.googleapis.com

# Grant the Vertex AI Reasoning Engine Service Agent access to Prompt Management, GCS release pointers, and OpenTelemetry exporters
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
gcloud beta services identity create --service=aiplatform.googleapis.com --project="${PROJECT_ID}" || true
for ROLE in \
  roles/aiplatform.user \
  roles/storage.objectViewer \
  roles/telemetry.writer \
  roles/cloudtrace.agent \
  roles/logging.logWriter \
  roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform-re.iam.gserviceaccount.com" \
    --role="${ROLE}"
done
```

---

## 4. Configure Environment Variables (`.env`)

Copy `.env.example` to `.env` and set your project ID and release bucket name:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
LLM_MODEL=gemini-3.8-flash
LLM_LOCATION=global
RESOURCE_LOCATION=us-central1
AGENT_LOCATION=us-central1
PROMPT_ENV=staging
PROMPT_CACHE_TTL_SECONDS=60
PROMPT_RELEASE_BUCKET=gs://your-project-id-prompt-release
GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=true
ENABLE_PROMPT_RESPONSE_LOGGING=true
```

> **Why two locations?**
> - `LLM_LOCATION=global` (`GOOGLE_CLOUD_LOCATION=global`): `gemini-3.8-flash` inference is served from the Vertex AI `global` multiregion endpoint.
> - `RESOURCE_LOCATION=us-central1`: Vertex AI Prompt Management (`client.prompts`), Vertex AI GenAI Evaluation (`client.evals`), and Vertex AI Agent Runtime (`ReasoningEngine`) are regional resources hosted in `us-central1`.

Ensure the `.env` symlink inside the `sig_normalizer/` package is present:

```bash
ln -sf ../.env sig_normalizer/.env
```

---

## 5. Seed Prompt Versions in Vertex AI Prompt Management

Register all three immutable prompt versions (`v1_naive`, `v2_schema`, `v3_policy_fewshot`) in Vertex AI Prompt Management and initialize the release pointer (`release/sig-normalizer.json` + GCS replica):

```bash
uv run python scripts/seed_prompts.py
```

Verify the registered server-side versions and active environment pointers (`prod -> v1`, `staging -> v3`):

```bash
uv run python scripts/show_versions.py
```

---

## 6. Deploy to Vertex AI Agent Runtime (Reasoning Engine)

Deploy (or update in-place) the `sig_normalizer` ADK agent package to Vertex AI Agent Runtime with OpenTelemetry (`--otel_to_cloud`) and prompt-response logging enabled:

```bash
./deploy_agent.sh sig_normalizer
```

Verify the deployed agent via REST API (`async_create_session` + `async_stream_query`):

```bash
# Query using default environment (staging -> v3)
./verify_agent_api.py --prompt "metformin 500mg tab: 1 tab po bid c meals"

# Query a high-risk ISMP sig against prod (v1) vs. staging (v3)
./verify_agent_api.py --prompt "insulin glargine: 10 u subq qhs" --prompt-env prod
./verify_agent_api.py --prompt "insulin glargine: 10 u subq qhs" --prompt-env staging
```

Inspect live Reasoning Engine logs and prompt-response telemetry in Cloud Logging:

```bash
# View recent Reasoning Engine container logs
./read_cloud_logs.py

# Filter specifically for OpenTelemetry GenAI events and structured prompt-response logs
./read_cloud_logs.py --prompt-response-only
```

---

## 7. Run the GEAP Hill-Climbing Evaluation Suite (Cloud-Only)

Both inference (`client.evals.run_inference` against the deployed Reasoning Engine) and metric evaluation (`client.evals.evaluate` + Console `EvaluationExperiment` / `EvaluationRun`) execute remotely in Google Cloud:

```bash
# Run the complete v1 -> v2 -> v3 hill-climb comparison and publish to Vertex AI Console
uv run python evals/run_eval.py --compare v1 v2 v3

# Or evaluate a specific version or environment individually
uv run python evals/run_eval.py --version v1 --allow-gate-failure
uv run python evals/run_eval.py --version v2 --allow-gate-failure
uv run python evals/run_eval.py --version v3
uv run python evals/run_eval.py --env staging
```

Scorecards are written locally and published to **Vertex AI -> Evaluation** in the Google Cloud Console:
- `results/scorecard_v1.json`, `results/scorecard_v2.json`, `results/scorecard_v3.json`
- `results/hill_climb_summary.json`
- `results/hill_climb_scorecard.md`

---

## 8. Promote `v3` to `prod` Without Redeploying the Agent

Attempt to promote `v2` (blocked because `v2` fails the `safety_escalation` gate):

```bash
uv run python scripts/promote.py --env prod --to-version v2
# -> REFUSED: Version 2 (v2_schema) failed one or more quality gates
```

Promote `v3` (passes all three quality gates in `results/scorecard_v3.json`):

```bash
uv run python scripts/promote.py --env prod --to-version v3
```

Because `sig_normalizer` resolves its instruction dynamically via `sig_instruction(ctx: ReadonlyContext) -> str` and caches the release pointer from GCS (`PROMPT_RELEASE_BUCKET`) with a 60-second TTL, the live Vertex AI Agent Runtime picks up `v3` in `prod` automatically **without redeploying the agent container**:

```bash
./verify_agent_api.py --prompt "insulin glargine: 10 u subq qhs" --prompt-env prod
```
