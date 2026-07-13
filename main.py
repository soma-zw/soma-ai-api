import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging — never log context or question content, only shape/metadata.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("soma_ai")

app = FastAPI(title="Soma Health AI API")

# ---------------------------------------------------------------------------
# CORS — restrict to real frontend origins via env var before production use.
# ALLOWED_ORIGINS is a comma-separated list, e.g.
#   ALLOWED_ORIGINS=https://doctor-app-domain.com,https://citizen-app-domain.com
# Falls back to "*" only when nothing is configured, so local/demo use keeps
# working — set ALLOWED_ORIGINS on Render before going further than a demo.
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # never combine allow_credentials=True with a wildcard origin
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
Platform = Literal["doctor", "citizen", "government"]


class AIRequest(BaseModel):
    platform: Platform
    context: str = Field(min_length=1, max_length=20000)
    question: str = Field(default="", max_length=2000)


DISCLAIMERS: Dict[str, str] = {
    "doctor": "AI-generated draft. Clinical review required.",
    "citizen": "This explanation is informational and does not replace advice from a qualified healthcare professional.",
    "government": "This brief is generated from the supplied aggregate data only and may not reflect real-time conditions.",
}

PROMPTS: Dict[str, str] = {
    "doctor": """
You are Soma AI, a clinical documentation assistant for qualified healthcare professionals
using the Soma Health platform.

You will be given a JSON "context" object. It always includes:
- current_page: which page of the app the doctor is looking at (dashboard, schedule,
  pharmacy, or patients)
- signed_in_doctor: the doctor's name, specialty and hospital
- page_data: the data relevant to that page — e.g. today's queue and performance stats on
  the dashboard; internal and partner-pharmacy drug stock and locations on the pharmacy
  page; department rosters and appointments on the schedule page; or the single currently
  unlocked patient record (vitals, allergies, medications, fracture/imaging history,
  medical history entries, risk factors, brain findings) on the patients page.

Answer the doctor's question using ONLY the supplied context. Adapt to whichever page_data
is present:
- On the dashboard: answer questions about today's queue, patient counts, consultation
  times, and forecasts.
- On pharmacy: answer questions about current stock levels (internal and partner-network),
  which partner pharmacy currently holds a given drug, and which pharmacies are nearby,
  using the addresses/coordinates supplied.
- On schedule: summarize rosters, appointments, and workload across doctors/departments.
- On patients: summarize the unlocked patient's medical history, organize diagnoses,
  medications, allergies, investigations and procedures, draft referral letters, discharge
  summaries or consultation notes, and highlight missing information.

You may:
- summarize, organize, and cross-reference the supplied data;
- draft referral letters, discharge summaries and consultation notes from patient data;
- identify missing information that requires professional review;
- answer general operational questions about the current page (stock, schedule, queue).

You must not:
- make a diagnosis;
- recommend, prescribe, stop or change medication;
- suggest treatment or lifestyle interventions;
- invent information not present in the supplied context;
- replace the judgment of a qualified healthcare professional.

If the context does not contain what is needed to answer, say so plainly rather than
guessing. When information is missing, state "Not recorded" rather than inventing it.
Keep the response concise, structured and factual.
""",
    "citizen": """
You are Soma AI, a patient-information assistant.

Explain only the supplied health information in clear and simple language.

You may:
- explain medical terminology;
- explain laboratory results;
- explain diagnoses already recorded by a healthcare professional;
- explain prescriptions and instructions already present in the record;
- summarize appointment, referral or discharge information.

You must not:
- diagnose;
- prescribe;
- recommend changing or stopping medication;
- invent information;
- create treatment plans;
- replace advice from a qualified healthcare professional.

When information is incomplete, say that it is incomplete. Use calm, accessible language
and avoid unnecessary medical jargon.
""",
    "government": """
You are Soma AI, a public-health intelligence assistant.

Analyze only anonymized and aggregated health data supplied in the request.

You may:
- summarize disease trends;
- compare regions;
- identify facility pressure;
- summarize medicine demand;
- explain vaccination coverage;
- identify unusual changes in aggregate indicators;
- produce concise policy briefs.

You must not:
- infer identities;
- request personally identifiable patient data;
- invent statistics;
- claim causation unless supported by the supplied data;
- hide uncertainty or data limitations;
- call something an outbreak unless the supplied data explicitly confirms it.

Structure the response under exactly these headings:
1. Observed Data
2. Interpretation
3. Limitations
4. Recommended Review
""",
}

UNSAFE_PATTERNS = [
    "stop taking",
    "increase your dose",
    "decrease your dose",
    "you have been diagnosed",
    "i recommend this treatment",
    "you definitely have",
    "you should take",
    "start taking",
    "change your medication",
    "switch to",
]

WITHHELD_MESSAGE: Dict[str, str] = {
    "doctor": "The AI response was withheld because it may contain unsupported clinical advice. Please review the source record manually.",
    "citizen": "Soma AI could not safely explain this record. Please contact a qualified healthcare professional.",
    "government": "The AI response was withheld because the supplied data was insufficient for a reliable interpretation.",
}


def contains_unsafe_content(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in UNSAFE_PATTERNS)


# ---------------------------------------------------------------------------
# Very small in-memory rate limiter — good enough for a single-instance demo.
# Keyed by client IP: max N requests per rolling window.
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_request_log: Dict[str, Deque[float]] = defaultdict(deque)


def check_rate_limit(client_id: str) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    log = _request_log[client_id]
    while log and log[0] < window_start:
        log.popleft()
    if len(log) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
        )
    log.append(now)


# ---------------------------------------------------------------------------
# Groq client — created once, with a request timeout so a hung upstream call
# can't hang the API worker indefinitely.
# ---------------------------------------------------------------------------
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

_client: Optional[Groq] = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured.")
        _client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_SECONDS)
    return _client


@app.get("/")
def home():
    return {"service": "Soma Health AI API", "status": "online"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/ai")
def ask_ai(request: AIRequest, http_request: Request):
    client_id = http_request.client.host if http_request.client else "unknown"
    check_rate_limit(client_id)

    started_at = time.monotonic()
    logger.info(
        "ai_request platform=%s context_len=%d question_len=%d",
        request.platform,
        len(request.context),
        len(request.question),
    )

    client = get_client()
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": PROMPTS[request.platform]},
                {
                    "role": "user",
                    "content": (
                        f"Context (JSON):\n{request.context}\n\n"
                        f"Request:\n{request.question or 'Analyze the supplied information.'}"
                    ),
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        answer = (completion.choices[0].message.content or "").strip()
    except Exception as error:  # noqa: BLE001 — deliberately broad, converted to a safe 502
        logger.error("ai_upstream_error platform=%s error=%s", request.platform, type(error).__name__)
        raise HTTPException(status_code=502, detail="Soma AI is temporarily unavailable.") from error

    if contains_unsafe_content(answer):
        logger.warning("ai_response_withheld platform=%s reason=unsafe_pattern", request.platform)
        answer = WITHHELD_MESSAGE[request.platform]
    else:
        disclaimer = DISCLAIMERS.get(request.platform)
        if disclaimer and disclaimer not in answer:
            answer = f"{answer}\n\n{disclaimer}"

    duration_ms = round((time.monotonic() - started_at) * 1000)
    logger.info("ai_response platform=%s duration_ms=%d", request.platform, duration_ms)

    return {
        "platform": request.platform,
        "answer": answer,
        "model": MODEL_NAME,
    }
