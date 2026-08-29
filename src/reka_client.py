"""
reka_client.py

explain_risk() is the Layer-2 semantic explainer. If a REKA_API_KEY is
set in .env, it attempts a real API call. Otherwise it returns a
realistic mock explanation with simulated latency, so the app is fully
demoable even without API access configured.
"""

import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

REKA_API_KEY = os.getenv("REKA_API_KEY", "")
REKA_ENDPOINT = "https://api.reka.ai/v1/chat"  # placeholder, adjust to actual Reka endpoint

MOCK_EXPLANATIONS = [
    "Grip aperture is widening while the object's trajectory continues downward — consistent with an early slip.",
    "Deviation from the reference path is rising steadily over the last several frames, suggesting a loss of control mid-lift.",
    "The object's tilt angle has exceeded the stable range seen in reference runs, indicating imminent instability.",
    "Trajectory deviation and grip aperture are both trending upward together — a pattern strongly associated with drops in the reference data.",
]


def explain_risk(context: dict) -> dict:
    """
    context: dict with at least {D_t, dD_dt, P_t, R_t, frame} describing
             the moment that triggered RISK CONFIRMED.

    Returns: {"explanation": str, "latency_ms": float, "source": "mock" | "api"}
    """
    start = time.perf_counter()

    if not REKA_API_KEY:
        # Mock path — simulate realistic API latency so the latency readout
        # is meaningful even without a live key configured
        time.sleep(random.uniform(0.4, 0.9))
        explanation = random.choice(MOCK_EXPLANATIONS)
        latency_ms = (time.perf_counter() - start) * 1000
        return {"explanation": explanation, "latency_ms": latency_ms, "source": "mock"}

    # Real API path
    try:
        import requests
        prompt = (
            f"A robot manipulation monitoring system flagged risk at frame {context.get('frame')}. "
            f"Deviation D_t={context.get('D_t'):.3f}, rate of change dD/dt={context.get('dD_dt'):.3f}, "
            f"grip aperture P_t={context.get('P_t'):.3f}, combined risk score R_t={context.get('R_t'):.3f}. "
            f"In one plain-language sentence, explain what is likely going wrong physically."
        )
        response = requests.post(
            REKA_ENDPOINT,
            headers={"Authorization": f"Bearer {REKA_API_KEY}"},
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        explanation = data.get("output", "").strip() or random.choice(MOCK_EXPLANATIONS)
        latency_ms = (time.perf_counter() - start) * 1000
        return {"explanation": explanation, "latency_ms": latency_ms, "source": "api"}

    except Exception as e:
        # Never let an API failure crash the demo — fall back to mock
        explanation = random.choice(MOCK_EXPLANATIONS)
        latency_ms = (time.perf_counter() - start) * 1000
        return {"explanation": f"{explanation} (fallback: {e})", "latency_ms": latency_ms, "source": "mock-fallback"}