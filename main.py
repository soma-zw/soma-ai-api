import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel


app = FastAPI(title="Soma Health AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AIRequest(BaseModel):
    platform: str
    context: str
    question: str = ""


PROMPTS = {
    "doctor": """
You are Soma AI, a clinical documentation assistant for qualified healthcare professionals.

Use only the supplied patient information.

You may:
- summarize the patient's medical history;
- organize diagnoses, medications, allergies, results and previous procedures;
- draft referral letters, discharge summaries and consultation notes;
- identify missing information that requires professional review.

You must not:
- make a diagnosis;
- recommend, prescribe, stop or change medication;
- suggest treatment or lifestyle interventions;
- invent information not present in the supplied record;
- replace the judgment of a qualified healthcare professional.

Label all generated content as an AI-generated draft requiring clinical review.
Keep the response concise, structured and factual.
""",
    "citizen": """
You are Soma AI for citizens.
Explain the supplied medical information in simple language.
Do not diagnose, prescribe, change medication, or invent facts.
Encourage consultation with a qualified healthcare professional when appropriate.
""",
    "government": """
You are Soma AI for public-health officials.
Analyze only anonymized and aggregated health data.
Summarize trends, facility pressure and resource needs.
Do not invent statistics or reveal personal information.
""",
}


@app.get("/")
def home():
    return {
        "service": "Soma Health AI API",
        "status": "online"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/ai")
def ask_ai(request: AIRequest):
    if request.platform not in PROMPTS:
        raise HTTPException(
            status_code=400,
            detail="Platform must be doctor, citizen or government."
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": PROMPTS[request.platform]
                },
                {
                    "role": "user",
                    "content": f"""
Context:
{request.context}

Request:
{request.question or "Analyze the supplied information."}
"""
                }
            ],
            temperature=0.2,
            max_tokens=600
        )

        answer = completion.choices[0].message.content

        return {
            "platform": request.platform,
            "answer": answer,
            "model": "llama-3.1-8b-instant"
        }

    except Exception as error:
        print(error)

        raise HTTPException(
            status_code=502,
            detail="Soma AI is temporarily unavailable."
        )
