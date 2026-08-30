"""
incident_report.py

Layer 2: the explanation/report layer. Triggered ONLY when Layer 1
confirms a real risk (RISK CONFIRMED) -- this is the "gated VLM cost"
part of the architecture: this expensive call happens rarely, not every
frame.

Works fully today WITHOUT a Reka API key: generates a rich, extensive,
data-driven report using the real physics numbers from the incident,
templated into readable prose. The moment REKA_API_KEY is set in your
.env file, it automatically tries a real Reka call instead, using the
same context -- with a safe fallback to the template report if the API
call fails for any reason (timeout, bad key, no internet), so a broken
API call can NEVER crash or interrupt your live demo.

Usage:
    from incident_report import generate_incident_report

    report = generate_incident_report(context)
    # report["summary"]        -> one-paragraph plain-language explanation
    # report["report_path"]    -> path to the saved full report file
    # report["source"]         -> "reka_api" | "template" | "template_fallback"
    # report["latency_ms"]     -> how long generation took
"""

import os
import sys
import time
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
ALERTS_DIR = os.path.join(PROJECT_ROOT, "data", "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

REKA_API_KEY = os.getenv("REKA_API_KEY", "")
REKA_ENDPOINT = "https://api.reka.ai/v1/chat"  # adjust to the real endpoint when you have one


# ============================================================
# Context building -- pulls together everything known about the
# incident so both the template and the real API have rich data.
# ============================================================

def build_incident_context(
    frame_index,
    trigger_reason,
    jerk_mag,
    jerk_floor,
    dev_value,
    grip_value,
    jerk_history_window,
    dev_history_window,
    session_start_time,
    is_real_reference,
):
    """
    Call this ONCE, right when RISK CONFIRMED first fires for a given
    incident. Gathers every number worth putting in the report.
    """
    now = time.time()
    elapsed_session_seconds = now - session_start_time if session_start_time else 0.0

    jerk_window = list(jerk_history_window[-16:]) if jerk_history_window else []
    dev_window = list(dev_history_window[-16:]) if dev_history_window else []

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_index": frame_index,
        "elapsed_session_seconds": round(elapsed_session_seconds, 2),
        "trigger_reason": trigger_reason,  # "sustained rise" or "instant peak"
        "jerk_mag": round(jerk_mag, 2),
        "jerk_floor": round(jerk_floor, 2) if jerk_floor is not None else None,
        "jerk_over_floor_ratio": round(jerk_mag / jerk_floor, 2) if jerk_floor else None,
        "deviation_value": round(dev_value, 2) if dev_value is not None else None,
        "grip_spread": round(grip_value, 3) if grip_value is not None else None,
        "jerk_window_max": round(max(jerk_window), 2) if jerk_window else None,
        "jerk_window_mean": round(sum(jerk_window) / len(jerk_window), 2) if jerk_window else None,
        "dev_window_max": round(max(dev_window), 2) if dev_window else None,
        "reference_is_real": is_real_reference,
    }


# ============================================================
# Template report -- rich, extensive, fully data-driven. This is
# what you get right now with no API key, and what you always fall
# back to if a real API call fails for any reason.
# ============================================================

def _severity_label(ctx):
    ratio = ctx.get("jerk_over_floor_ratio")
    if ratio is None:
        return "MODERATE"
    if ratio >= 4:
        return "SEVERE"
    if ratio >= 2:
        return "HIGH"
    return "MODERATE"


def _likely_cause_paragraph(ctx):
    reason = ctx["trigger_reason"]
    if reason == "instant peak":
        return (
            "The system detected a single sharp, high-magnitude jerk spike -- "
            "the physics signature of a sudden, discontinuous event such as a "
            "drop, jolt, or abrupt loss of grip. Because this was an instantaneous "
            "spike rather than a gradual build-up, it is consistent with a fast, "
            "unplanned failure event rather than a slow drift or intentional motion."
        )
    else:
        return (
            "The system detected a sustained, rising trend in motion jerk across "
            "multiple consecutive frames -- consistent with vigorous, repeated, "
            "unstable motion (such as shaking) rather than a single momentary event. "
            "The trend rule specifically requires the recent average to be "
            "meaningfully higher than, and still climbing above, the preceding "
            "period, which rules out a brief, one-off sensor noise spike."
        )


def _recommended_actions(ctx):
    actions = [
        "Review the corresponding session log (data/live_sessions/) for the exact frame-by-frame trajectory around this incident.",
        "If this was a false positive, consider re-running calibration (press 'r') under more stable conditions before the next session.",
    ]
    if ctx["trigger_reason"] == "instant peak":
        actions.insert(0, "Inspect the object/grip state immediately -- a sudden peak is consistent with an active drop or slip in progress.")
    else:
        actions.insert(0, "Inspect for ongoing instability -- sustained rising motion suggests the situation may still be developing.")
    return actions


def generate_template_report(ctx):
    severity = _severity_label(ctx)
    cause = _likely_cause_paragraph(ctx)
    actions = _recommended_actions(ctx)

    lines = []
    lines.append("=" * 70)
    lines.append("KINESIS -- INCIDENT REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Timestamp:            {ctx['timestamp']}")
    lines.append(f"Severity:             {severity}")
    lines.append(f"Trigger mechanism:    {ctx['trigger_reason'].upper()}")
    lines.append(f"Session time elapsed: {ctx['elapsed_session_seconds']}s")
    lines.append(f"Frame index:          {ctx['frame_index']}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("PHYSICS READOUT AT TRIGGER")
    lines.append("-" * 70)
    lines.append(f"  Jerk magnitude:          {ctx['jerk_mag']}")
    lines.append(f"  Calibrated alert floor:  {ctx['jerk_floor']}")
    if ctx["jerk_over_floor_ratio"] is not None:
        lines.append(f"  Exceeded floor by:       {ctx['jerk_over_floor_ratio']}x")
    lines.append(f"  Deviation (z-score):     {ctx['deviation_value']}")
    lines.append(f"  Grip spread:             {ctx['grip_spread']}")
    lines.append(f"  Recent jerk window max:  {ctx['jerk_window_max']}")
    lines.append(f"  Recent jerk window mean: {ctx['jerk_window_mean']}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("WHAT LIKELY HAPPENED")
    lines.append("-" * 70)
    lines.append(cause)
    lines.append("")
    lines.append("-" * 70)
    lines.append("RECOMMENDED NEXT STEPS")
    lines.append("-" * 70)
    for i, action in enumerate(actions, 1):
        lines.append(f"  {i}. {action}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("DATA PROVENANCE")
    lines.append("-" * 70)
    ref_note = (
        "Real reference envelope built from your labeled dataset."
        if ctx["reference_is_real"]
        else "WARNING: synthetic reference in use -- deviation numbers above are not calibrated to real data."
    )
    lines.append(f"  {ref_note}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by Kinesis Layer 2 (template mode -- no live Reka API call).")
    lines.append("=" * 70)

    return "\n".join(lines)


def generate_summary_sentence(ctx):
    """A short one-liner for on-screen display (the report file has the full detail)."""
    severity = _severity_label(ctx)
    if ctx["trigger_reason"] == "instant peak":
        return f"{severity}: sudden jerk spike ({ctx['jerk_mag']} vs floor {ctx['jerk_floor']}) -- consistent with a drop or slip."
    return f"{severity}: sustained rising motion detected -- consistent with vigorous shaking."


# ============================================================
# Real Reka API attempt (used automatically once you have a key)
# ============================================================

def _try_reka_api(ctx):
    """Returns explanation text on success, raises on any failure."""
    import requests  # imported lazily so this file never hard-requires it

    prompt = (
        "A robot manipulation monitoring system (Kinesis) flagged a risk event. "
        f"Trigger mechanism: {ctx['trigger_reason']}. "
        f"Jerk magnitude: {ctx['jerk_mag']} (calibrated normal floor: {ctx['jerk_floor']}). "
        f"Deviation z-score: {ctx['deviation_value']}. Grip spread: {ctx['grip_spread']}. "
        f"Recent jerk window max/mean: {ctx['jerk_window_max']}/{ctx['jerk_window_mean']}. "
        "In 2-3 sentences, explain in plain language what is likely physically "
        "happening and what an operator should check first."
    )

    response = requests.post(
        REKA_ENDPOINT,
        headers={"Authorization": f"Bearer {REKA_API_KEY}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=8,
    )
    response.raise_for_status()
    data = response.json()
    explanation = data.get("output") or data.get("response") or ""
    if not explanation.strip():
        raise ValueError("Empty response from Reka API")
    return explanation.strip()


# ============================================================
# Main entry point
# ============================================================

def generate_incident_report(ctx, save_to_file=True):
    """
    ctx: dict from build_incident_context(...)
    Returns: {summary, report_path, source, latency_ms, full_report_text}

    Guaranteed to never raise -- any failure anywhere falls back to the
    template report, so a broken API call or file-write error can never
    interrupt your live demo.
    """
    start = time.perf_counter()
    source = "template"
    reka_explanation = None

    if REKA_API_KEY:
        try:
            reka_explanation = _try_reka_api(ctx)
            source = "reka_api"
        except Exception as e:
            print(f"[Kinesis] Reka API call failed ({e}) -- falling back to template report.")
            source = "template_fallback"

    full_report = generate_template_report(ctx)
    if reka_explanation:
        full_report = full_report.replace(
            "WHAT LIKELY HAPPENED\n" + "-" * 70 + "\n" + _likely_cause_paragraph(ctx),
            "WHAT LIKELY HAPPENED\n" + "-" * 70 + "\n" + reka_explanation
            + "\n\n  [Kinesis template analysis, for reference:]\n  " + _likely_cause_paragraph(ctx),
        )

    summary = reka_explanation.split(".")[0] + "." if reka_explanation else generate_summary_sentence(ctx)

    report_path = None
    if save_to_file:
        try:
            filename = f"incident_{ctx['timestamp'].replace(':', '-').replace(' ', '_')}.txt"
            report_path = os.path.join(ALERTS_DIR, filename)
            with open(report_path, "w") as f:
                f.write(full_report)
        except Exception as e:
            print(f"[Kinesis] Could not save report file ({e}) -- report generated but not saved.")
            report_path = None

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "summary": summary,
        "report_path": report_path,
        "source": source,
        "latency_ms": round(latency_ms, 1),
        "full_report_text": full_report,
    }
