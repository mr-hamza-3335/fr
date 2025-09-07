from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import asyncio
import logging

# --- Try importing agent SDK (if installed) ---
try:
    from agents import Agent, Runner, trace
    from agents import AsyncOpenAI, RunConfig, OpenAIChatCompletionsModel
    AGENT_SDK_AVAILABLE = True
except Exception as e:
    # If SDK not available, we'll run in mock mode
    Agent = Runner = trace = None
    AsyncOpenAI = RunConfig = OpenAIChatCompletionsModel = None
    AGENT_SDK_AVAILABLE = False
    logging.warning("Agent SDK not available, running in mock mode. Error: %s", e)

load_dotenv()
logging.basicConfig(level=logging.INFO)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
ALLOWED_ORIGINS = (os.getenv("ALLOWED_ORIGINS") or "http://localhost:3000").split(",")

# --- If API key provided and SDK available, configure client ---
external_client = None
config = None
if GEMINI_API_KEY and AGENT_SDK_AVAILABLE:
    try:
        external_client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model = OpenAIChatCompletionsModel(
            model="gemini-2.0-flash",
            openai_client=external_client,
        )
        config = RunConfig(model=model, model_provider=external_client)
        logging.info("AI client configured.")
    except Exception as e:
        logging.exception("Failed to init AI client, falling back to mock.")
else:
    logging.info("No GEMINI/OPENAI key or SDK — mock mode enabled.")

# --- Define agents (if SDK exists) or just names for mock ---
if AGENT_SDK_AVAILABLE:
    proposal_agent = Agent(
        name="proposal_agent",
        instructions="""You are Proposal Agent. Input: job description and relevant details.
        Output: a concise, professional proposal with scope, timeline, deliverables, and CTA."""
    )
    chat_agent = Agent(
        name="chat_agent",
        instructions="You are Chat Helper Agent. Provide short professional replies."
    )
    project_agent = Agent(
        name="project_agent",
        instructions="You are Project Manager Agent. Manage tasks and deadlines."
    )

    parent_agent = Agent(
        name="parent_agent",
        instructions="You are Parent Agent. Route requests to proposal_agent, chat_agent, or project_agent.",
        handoffs=[proposal_agent, chat_agent, project_agent]
    )
else:
    proposal_agent = chat_agent = project_agent = parent_agent = None

# --- FastAPI app ---
app = FastAPI(title="FreelanceAI - Agents Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request schema ---
class RunRequest(BaseModel):
    text: str
    mock: bool | None = False
    force_agent: str | None = None

# --- Mock responses ---
def mock_response_for_agent(agent_name: str, text: str) -> str:
    if agent_name == "proposal_agent":
        return (
            "Proposal (mock):\n"
            "- Scope: Build responsive Next.js landing page and contact form.\n"
            "- Timeline: 7 days.\n"
            "- Deliverables: Responsive UI, contact form, SEO, CWV improvements.\n"
            "- CTA: 15-min kickoff call.\n"
            "Regards,\nHamza"
        )
    if agent_name == "chat_agent":
        return (
            "Reply suggestion (mock):\n"
            "Assalam o Alaikum! Thanks for the message. I can do this in 7 days. "
            "Proposed budget: $110. Can we schedule a quick call?"
        )
    if agent_name == "project_agent":
        return (
            "Task created (mock):\n"
            "- Task: Implement landing page\n"
            "- Deadline: 2025-09-07 18:00\n"
            "- Reminder: 24 hours before"
        )
    return "I cannot handle this request."

# --- Core run function ---
async def run_parent_and_get_result(text: str, force_agent: str | None, mock: bool):
    if mock or config is None:
        chosen = force_agent or (
            "proposal_agent" if any(k in text.lower() for k in ["proposal", "job description", "scope", "deliverable"])
            else "project_agent" if any(k in text.lower() for k in ["task", "deadline", "remind"])
            else "chat_agent"
        )
        return {"final_output": mock_response_for_agent(chosen, text), "last_agent": chosen}

    try:
        with trace("FreelanceAI Session"):
            result = await Runner.run(parent_agent, text, run_config=config)
            final = getattr(result, "final_output", None) or str(result)
            last = getattr(result, "last_agent", None)
            last_name = last.name if last is not None else "unknown"
            return {"final_output": final, "last_agent": last_name}
    except asyncio.TimeoutError:
        return {"final_output": "Generation timed out.", "last_agent": "timeout"}
    except Exception as e:
        logging.exception("Runner error")
        return {"final_output": f"Error during generation: {e}", "last_agent": "error"}

# --- API Routes ---
@app.get("/health")
async def health():
    return {"ok": True, "ai_enabled": (config is not None)}

@app.post("/run-agent")
async def run_agent(req: RunRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    return await run_parent_and_get_result(text, req.force_agent, bool(req.mock))

@app.post("/generate-proposal")
async def generate_proposal(req: RunRequest):
    return await run_parent_and_get_result(req.text, "proposal_agent", bool(req.mock))

@app.post("/generate-reply")
async def generate_reply(req: RunRequest):
    return await run_parent_and_get_result(req.text, "chat_agent", bool(req.mock))

@app.post("/add-task")
async def add_task(req: RunRequest):
    return await run_parent_and_get_result(req.text, "project_agent", bool(req.mock))

# --- Windows asyncio destructor workaround ---
if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        import asyncio.proactor_events
        try:
            orig = asyncio.proactor_events._ProactorBasePipeTransport.__del__
        except Exception:
            orig = None

        def silent_del(self):
            try:
                if orig:
                    orig(self)
            except RuntimeError as e:
                if "Event loop is closed" not in str(e):
                    raise

        asyncio.proactor_events._ProactorBasePipeTransport.__del__ = silent_del

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
