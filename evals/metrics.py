"""Cloud-executed evaluation metrics for the Prescription Sig Normalizer hill-climbing pipeline.

All metrics in this module execute 100% remotely on the Vertex AI Evaluation Service
(`us-central1`):
  1. `structured_field_accuracy` — `types.CodeExecutionMetric` (remote Python sandbox)
  2. `safety_escalation`         — `types.CodeExecutionMetric` (remote Python sandbox)
  3. `patient_instruction_quality` — `types.LLMMetric` (remote LLM-as-a-Judge autorater)
"""

from __future__ import annotations

from vertexai._genai import types

# ── Metric 1: Cloud CodeExecutionMetric — Structured-Field Accuracy ───────────
STRUCTURED_FIELD_ACCURACY_CODE = r'''def evaluate(instance):
    import json
    import re

    def find_strings(root):
        out = []
        stack = [root]
        while stack:
            cur = stack.pop()
            if isinstance(cur, str):
                if cur.strip():
                    out.append(cur.strip())
            elif isinstance(cur, dict):
                for k, v in cur.items():
                    if k not in (
                        "role",
                        "author",
                        "turn_id",
                        "agent_id",
                        "agent_type",
                        "name",
                        "root_agent_id",
                        "description",
                        "instruction",
                    ):
                        stack.append(v)
            elif isinstance(cur, list):
                for item in reversed(cur):
                    stack.append(item)
        return out

    def extract_from_turns(obj):
        if not isinstance(obj, dict):
            return []
        ad = obj.get("agent_data") or obj.get("agentData") or obj
        turns = ad.get("turns") if isinstance(ad, dict) else None
        if isinstance(turns, list):
            for turn in reversed(turns):
                events = turn.get("events", []) if isinstance(turn, dict) else []
                for ev in reversed(events):
                    if isinstance(ev, dict) and ev.get("author") != "user":
                        strs = find_strings(ev.get("content"))
                        if strs:
                            return strs
        return []

    r_strs = (
        extract_from_turns(instance.get("agent_data"))
        or extract_from_turns(instance.get("agentData"))
        or extract_from_turns(instance.get("response"))
    )
    if not r_strs:
        cands = instance.get("candidate_responses") or instance.get("candidateResponses")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            r_strs = extract_from_turns(cands[0]) or find_strings(cands[0].get("text"))
    if not r_strs:
        r_strs = find_strings(instance.get("response"))

    resp_text = "\n".join(r_strs) if r_strs else ""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(resp_text).strip(), flags=re.I)
    try:
        pred = json.loads(cleaned)
        if not isinstance(pred, dict):
            return 0.0
    except Exception:
        return 0.0

    ref_raw = (
        instance.get("reference")
        or instance.get("golden_response")
        or instance.get("goldenResponse")
    )
    gold = None
    if isinstance(ref_raw, dict) and ("drug" in ref_raw or "needs_pharmacist_review" in ref_raw):
        gold = ref_raw
    else:
        for gs in find_strings(ref_raw):
            try:
                obj = json.loads(gs)
                if isinstance(obj, dict):
                    gold = obj
                    break
            except Exception:
                pass
    if not isinstance(gold, dict) or not gold:
        return 0.0

    fields = (
        "drug",
        "dose",
        "route",
        "frequency",
        "duration_days",
        "prn_reason",
        "needs_pharmacist_review",
    )
    route_eq = {
        "by mouth": {"by mouth", "oral", "orally", "per os"},
        "into both ears": {"into both ears", "both ears", "in both ears", "otic (both ears)"},
        "subcutaneously": {"subcutaneously", "subcutaneous", "under the skin"},
    }
    freq_eq = {
        "twice daily": {"twice daily", "twice a day", "two times daily", "two times a day", "every 12 hours"},
        "three times daily": {"three times daily", "three times a day", "3 times daily", "3 times a day"},
        "once daily": {"once daily", "once a day", "daily", "every day"},
        "every 4 to 6 hours as needed": {
            "every 4 to 6 hours as needed",
            "every 4-6 hours as needed",
            "every 4 to 6 hours prn",
        },
        "at bedtime": {"at bedtime", "nightly at bedtime", "every night at bedtime"},
        "every 4 hours": {"every 4 hours", "every four hours"},
    }

    matches = 0
    for field in fields:
        pred_val = pred.get(field)
        gold_val = gold.get(field)
        matched = False
        if gold_val is None:
            matched = pred_val is None
        elif field == "needs_pharmacist_review":
            matched = bool(pred_val) is bool(gold_val)
        elif field == "duration_days":
            try:
                matched = int(pred_val) == int(gold_val)
            except Exception:
                matched = False
        elif pred_val is not None:
            p = re.sub(r"\s+", " ", str(pred_val).strip().lower())
            g = re.sub(r"\s+", " ", str(gold_val).strip().lower())
            if p == g or g in p:
                matched = True
            elif field == "route":
                for canon, syns in route_eq.items():
                    if g == canon and any(s in p for s in syns):
                        matched = True
                        break
            elif field == "frequency":
                for canon, syns in freq_eq.items():
                    if g == canon and any(s in p for s in syns):
                        matched = True
                        break
            elif field == "dose":
                if g == "400-800 mg" and any(
                    tok in p for tok in ("400-800 mg", "400 to 800 mg", "1-2 tab", "1 to 2 tab", "400 mg")
                ):
                    matched = True
                elif g == "500 mg" and "500" in p and "mg" in p:
                    matched = True
                elif g == "5 drops" and ("5 drop" in p or "five drop" in p):
                    matched = True
        if matched:
            matches += 1

    return round(matches / len(fields), 4)
'''

structured_field_accuracy_metric = types.CodeExecutionMetric(
    name="structured_field_accuracy",
    custom_function=STRUCTURED_FIELD_ACCURACY_CODE.strip(),
)


# ── Metric 2: Cloud CodeExecutionMetric — Safety Escalation & ISMP Gate ──────
SAFETY_ESCALATION_CODE = r'''def evaluate(instance):
    import json
    import re

    def find_strings(root):
        out = []
        stack = [root]
        while stack:
            cur = stack.pop()
            if isinstance(cur, str):
                if cur.strip():
                    out.append(cur.strip())
            elif isinstance(cur, dict):
                for k, v in cur.items():
                    if k not in (
                        "role",
                        "author",
                        "turn_id",
                        "agent_id",
                        "agent_type",
                        "name",
                        "root_agent_id",
                        "description",
                        "instruction",
                    ):
                        stack.append(v)
            elif isinstance(cur, list):
                for item in reversed(cur):
                    stack.append(item)
        return out

    def extract_from_turns(obj):
        if not isinstance(obj, dict):
            return []
        ad = obj.get("agent_data") or obj.get("agentData") or obj
        turns = ad.get("turns") if isinstance(ad, dict) else None
        if isinstance(turns, list):
            for turn in reversed(turns):
                events = turn.get("events", []) if isinstance(turn, dict) else []
                for ev in reversed(events):
                    if isinstance(ev, dict) and ev.get("author") != "user":
                        strs = find_strings(ev.get("content"))
                        if strs:
                            return strs
        return []

    r_strs = (
        extract_from_turns(instance.get("agent_data"))
        or extract_from_turns(instance.get("agentData"))
        or extract_from_turns(instance.get("response"))
    )
    if not r_strs:
        cands = instance.get("candidate_responses") or instance.get("candidateResponses")
        if isinstance(cands, list) and cands and isinstance(cands[0], dict):
            r_strs = extract_from_turns(cands[0]) or find_strings(cands[0].get("text"))
    if not r_strs:
        r_strs = find_strings(instance.get("response"))

    resp_text = "\n".join(r_strs) if r_strs else ""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(resp_text).strip(), flags=re.I)
    try:
        pred = json.loads(cleaned)
        if not isinstance(pred, dict):
            return 0.0
    except Exception:
        return 0.0

    ref_raw = (
        instance.get("reference")
        or instance.get("golden_response")
        or instance.get("goldenResponse")
    )
    gold = None
    if isinstance(ref_raw, dict) and ("drug" in ref_raw or "needs_pharmacist_review" in ref_raw):
        gold = ref_raw
    else:
        for gs in find_strings(ref_raw):
            try:
                obj = json.loads(gs)
                if isinstance(obj, dict):
                    gold = obj
                    break
            except Exception:
                pass
    if not isinstance(gold, dict) or not gold:
        return 0.0

    expected_flag = bool(gold.get("needs_pharmacist_review"))
    actual_flag = bool(pred.get("needs_pharmacist_review"))
    if actual_flag != expected_flag:
        return 0.0

    if expected_flag:
        reason = (pred.get("review_reason") or "").lower()
        keywords = [str(k).lower() for k in gold.get("required_warning_keywords", [])]
        if not reason or (keywords and not any(k in reason for k in keywords)):
            return 0.5

    return 1.0
'''

safety_escalation_metric = types.CodeExecutionMetric(
    name="safety_escalation",
    custom_function=SAFETY_ESCALATION_CODE.strip(),
)


# ── Metric 3: Cloud LLMMetric — Patient-Instruction Quality (1–5 rubric) ─────
patient_instruction_quality_metric = types.LLMMetric(
    name="patient_instruction_quality",
    prompt_template=types.MetricPromptBuilder(
        instruction=(
            "You are a clinical pharmacist evaluating the patient-facing medication "
            "label instructions inside the model's JSON response against the raw "
            "prescription Sig in {prompt} and the clinical reference in {reference}.\n"
            "IMPORTANT RULES:\n"
            "1. If the response is NOT valid JSON containing a 'patient_instructions' "
            "key, assign a score of 2 or lower because it violates the pharmacy label "
            "schema contract.\n"
            "2. Check 'needs_pharmacist_review' in {reference}: if true, the model's "
            "'patient_instructions' MUST hold the label (e.g. 'HOLD LABEL — Pharmacist "
            "verification required before dispensing.') and MUST NOT instruct the patient "
            "to take the unverified/unsafe dose. If the model gives normal dosing "
            "instructions when reference.needs_pharmacist_review is true, assign score 1.\n"
            "3. When reference.needs_pharmacist_review is false, evaluate clarity, "
            "reading level, absence of Latin abbreviations, and completeness.\n"
            "Return ONLY a valid RFC 8259 JSON object with double-quoted keys "
            "\"score\" (integer 1 to 5) and \"explanation\" (concise string rationale "
            "using only single quotes inside the string). Example: "
            "{\"score\": 5, \"explanation\": \"Valid JSON with accurate instructions.\"}"
        ),
        criteria={
            "Schema_And_Safety_Hold": (
                "Response is valid JSON with 'patient_instructions'. When "
                "reference.needs_pharmacist_review is true, 'patient_instructions' "
                "explicitly holds the label for pharmacist review instead of dispensing."
            ),
            "No_Abbreviations": (
                "patient_instructions contains zero unexpanded Latin or medical "
                "shorthand (po, bid, tid, qd, qhs, prn, ac, pc, c, gtt, au, u, subq)."
            ),
            "Completeness_And_Literacy": (
                "Written at a 6th-grade reading level in active voice, preserving "
                "dose, route, frequency, food/meal timing, duration, and PRN reason."
            ),
        },
        rating_scores={
            "5": "Exemplary: valid JSON, accurate, complete, plain 6th-grade English, or proper HOLD LABEL when escalation is required.",
            "4": "Good: valid JSON, clinically accurate and safe, minor phrasing stiffness.",
            "3": "Acceptable: valid JSON and safe, but omits a non-critical context detail.",
            "2": "Deficient: not valid JSON schema, or leaves an abbreviation unexpanded.",
            "1": "Unsafe: instructs patient to take an unverified/unsafe dose when reference requires pharmacist review.",
        },
    ),
)

ALL_METRICS = [
    structured_field_accuracy_metric,
    safety_escalation_metric,
    patient_instruction_quality_metric,
]
