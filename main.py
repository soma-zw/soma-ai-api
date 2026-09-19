import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq
from pydantic import BaseModel, Field
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("soma_ai")

app = FastAPI(title="Soma AI API")

_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,   
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


# A truly unhandled exception (anything not raised as an HTTPException) would
# otherwise escape FastAPI's normal response path entirely and come back with
# NO CORS headers attached — the browser then reports it as a CORS/connection
# failure instead of showing the real error, which is exactly what happened
# here. This catch-all guarantees every response — including bugs we haven't
# hit yet — is valid, CORS-safe JSON, and the real traceback still goes to
# the Render logs either way.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_error path=%s error=%s", request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "Soma AI hit an unexpected error. Please try again."},
    )
 
Platform = Literal["page", "assistant"]


class AIRequest(BaseModel):
    platform: Platform
    context: str = Field(min_length=1, max_length=20000)
    question: str = Field(default="", max_length=2000)


DISCLAIMERS: Dict[str, str] = {
    "assistant": "This is a draft to help you — review and rework it before you submit anything as your own work.",
}

PROMPTS: Dict[str, str] = {
    "page": """
You are Soma AI, the sidebar assistant in the Soma student browser.

You are given a JSON "context" object describing the page currently open in the browser —
typically its title, url, and the page's visible text content (and possibly a user-selected
excerpt the student highlighted).

Answer the student's question using ONLY the supplied page context. You may:
- explain, summarize, or simplify what's on the page;
- define terms or concepts that appear on the page;
- answer direct questions about the page's content;
- help the student understand something confusing on the page.

You must not:
- answer questions unrelated to the page by inventing information not present in the
  context;
- pretend to know things the page context doesn't actually contain.

If the question can't be answered from the supplied page content, say so plainly rather
than guessing, and suggest what the student could look up instead. Keep answers short and
conversational — this is a sidebar, not an essay.
""",
    "assistant": """
You are Soma AI, the home assistant in the Soma student browser — a general, ChatGPT-style
companion that knows the student across the conversation.

You are given a JSON "context" object that may include the student's profile (name,
grade/school), recent conversation history, and other Soma account data. Use it to stay
consistent and personable — don't reset context every message, and don't ask the student to
re-explain things already present in context.

You help with schoolwork, general questions, planning, and everyday assistant tasks.

For homework, essays, or assignments specifically:
- You may draft, outline, solve, or write full attempts when asked — don't refuse or water
  down the help.
- Always make clear the result is a draft/starting point for the student to review, revise,
  and put in their own words before handing it in — never suggest it should be copied or
  submitted as-is.
- If asked to help the student cheat on something happening live (an in-progress test or
  exam), decline that specifically and offer to help them prepare instead.

Keep the tone warm and conversational, like a knowledgeable friend rather than a formal
report generator.
""",
}

# No content is auto-withheld for the browser platforms yet — the medical unsafe-pattern
# list doesn't apply here. Add phrases below if you want specific responses blocked (e.g.
# patterns indicating exam-cheating help), matching against the lowercased answer text.
UNSAFE_PATTERNS: list[str] = []

WITHHELD_MESSAGE: Dict[str, str] = {
    "page": "Soma AI couldn't safely answer that from the page content.",
    "assistant": "Soma AI couldn't safely answer that — try rephrasing the question.",
}


def contains_unsafe_content(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in UNSAFE_PATTERNS)

 
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

 
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
# llama-3.1-8b-instant was deprecated for free/developer-tier Groq accounts on 2026-08-16
# (moved to Enterprise-only). gpt-oss-20b is the closest same-tier replacement: fast,
# cheap, 131K context. Swap MODEL_NAME to "openai/gpt-oss-120b" via env var for higher
# quality at ~half the speed.
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
# GPT-OSS models reason internally before answering, and that hidden reasoning is drawn
# from the same MAX_TOKENS budget — a low limit can leave nothing for the actual answer.
# "low" keeps reasoning brief (closer to the old model's latency); "hidden" means the
# response only contains the final answer, not the reasoning trace.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
REASONING_FORMAT = os.getenv("REASONING_FORMAT", "hidden")

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
    return {"service": "Soma AI API", "status": "online"}


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
            reasoning_effort=REASONING_EFFORT,
            reasoning_format=REASONING_FORMAT,
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
