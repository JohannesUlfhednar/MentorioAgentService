"""Coach Majen Agent Service — FastAPI server.

V2 Architecture:
- Coach Majen is the master agent (GPT-4o) with consistent persona
- Sub-agents (GPT-4o-mini) are exposed as tools, not handoffs
- Input guardrail: blocks unsafe requests (medical, self-harm, steroids)
- Output guardrail: ensures Norwegian language
- Dynamic instructions: user data injected at runtime
- Tracing: enabled via OPENAI_AGENTS_TRACING_ENABLED env var
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import Runner, InputGuardrailTripwireTriggered

load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent.server")


# ── Tracing (optional, if key is set) ──────────────────────────────
if os.getenv("OPENAI_API_KEY"):
    os.environ.setdefault("OPENAI_AGENTS_TRACING_ENABLED", "true")


# ── FastAPI app ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Coach Majen Agent Service V2 starting ===")
    logger.info(f"Tracing: {os.getenv('OPENAI_AGENTS_TRACING_ENABLED', 'false')}")
    yield
    logger.info("=== Coach Majen Agent Service V2 shutting down ===")


app = FastAPI(
    title="Coach Majen Agent Service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    mentor_id: str
    message: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    agent_name: str = "Coach Majen"
    tools_called: list[str] = []
    processing_ms: int = 0
    guardrail_blocked: bool = False
    blocked_reason: str = ""


# ── Agent singleton ────────────────────────────────────────────────
_agent = None

def _get_agent():
    """Lazy-init the Coach Majen agent (singleton)."""
    global _agent
    if _agent is None:
        from agents_pkg.coach_majen import build_coach_majen
        _agent = build_coach_majen()
        logger.info("Coach Majen agent built")
    return _agent


# ── Chat endpoint ──────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint. Processes user message through Coach Majen."""
    start = time.time()
    logger.info(f"[chat] user={req.user_id} mentor={req.mentor_id} msg={req.message[:80]}...")

    try:
        # Build context with preloaded user data
        from agents_pkg.coach_majen import load_context
        ctx = load_context(req.user_id, req.mentor_id)

        # Build input messages
        messages: list[dict] = []

        # Add conversation history (last 20 messages)
        if req.conversation_history:
            recent = req.conversation_history[-20:]
            for msg in recent:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content.strip():
                    messages.append({"role": role, "content": content})

        # Add user data context as a system-like injection in the first user turn
        user_data_prefix = ""
        if ctx.onboarding_summary:
            user_data_prefix = (
                f"[SYSTEM: Brukerens profil — {ctx.onboarding_summary.replace(chr(10), ' | ')}] "
            )

        # Add the current message
        messages.append({
            "role": "user",
            "content": f"{user_data_prefix}{req.message}" if user_data_prefix and not messages else req.message,
        })

        # If we had history but no user data prefix was added, inject it in the first message
        if user_data_prefix and messages and len(messages) > 1:
            first_msg = messages[0]
            if first_msg["role"] == "user" and "[SYSTEM:" not in first_msg["content"]:
                messages[0] = {
                    "role": "user",
                    "content": f"{user_data_prefix}{first_msg['content']}",
                }

        # Run the agent
        agent = _get_agent()
        result = await Runner.run(agent, messages, context=ctx)

        # Collect tool names from the trace
        tools_called = []
        for item in result.new_items:
            item_type = getattr(item, "type", "")
            if item_type == "tool_call_item":
                name = getattr(item, "name", None) or getattr(getattr(item, "raw_item", None), "name", "")
                if name:
                    tools_called.append(name)

        response_text = str(result.final_output or "")
        elapsed = int((time.time() - start) * 1000)

        logger.info(
            f"[chat] done user={req.user_id} tools={tools_called} "
            f"len={len(response_text)} ms={elapsed}"
        )

        return ChatResponse(
            response=response_text,
            agent_name="Coach Majen",
            tools_called=tools_called,
            processing_ms=elapsed,
        )

    except InputGuardrailTripwireTriggered as e:
        elapsed = int((time.time() - start) * 1000)
        logger.warning(f"[chat] BLOCKED by safety guardrail user={req.user_id}")
        return ChatResponse(
            response=(
                "Beklager, men jeg kan ikke hjelpe med det du spør om. "
                "Jeg er en fitness-coach og kan hjelpe deg med trening, kosthold og mål. "
                "Hvis du trenger medisinsk hjelp, ta kontakt med legen din."
            ),
            agent_name="Coach Majen",
            processing_ms=elapsed,
            guardrail_blocked=True,
            blocked_reason="safety",
        )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        logger.error(f"[chat] ERROR user={req.user_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)[:200]}",
        )


# ── Debug endpoint — check if plans exist in DB ──────────────────

@app.get("/debug/plans/{user_id}")
async def debug_plans(user_id: str):
    """Check what training plans exist for a user. For debugging only."""
    from agents_pkg.db import get_db, TRAINING_PLAN_VERSIONS, NUTRITION_PLAN_VERSIONS
    try:
        tp = get_db().table(TRAINING_PLAN_VERSIONS) \
            .select("id, user_id, version, days, reason, created_at") \
            .eq("user_id", user_id) \
            .order("version", desc=True) \
            .limit(3) \
            .execute()

        np = get_db().table(NUTRITION_PLAN_VERSIONS) \
            .select("id, user_id, version, kcal, protein_grams, reason, created_at") \
            .eq("user_id", user_id) \
            .order("version", desc=True) \
            .limit(3) \
            .execute()

        return {
            "user_id": user_id,
            "training_plans": tp.data or [],
            "training_plan_count": len(tp.data or []),
            "nutrition_plans": np.data or [],
            "nutrition_plan_count": len(np.data or []),
        }
    except Exception as e:
        return {"error": str(e), "user_id": user_id}


@app.get("/debug/recent-plans")
async def debug_recent_plans():
    """Check the most recent training plans across all users. For debugging."""
    from agents_pkg.db import get_db, TRAINING_PLAN_VERSIONS
    try:
        tp = get_db().table(TRAINING_PLAN_VERSIONS) \
            .select("id, user_id, version, reason, created_at") \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        return {"recent_plans": tp.data or [], "count": len(tp.data or [])}
    except Exception as e:
        return {"error": str(e)}


# ── Health endpoint ────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "architecture": "agents-as-tools",
        "features": [
            "dynamic_instructions",
            "input_guardrails",
            "output_guardrails",
            "agent_delegation",
            "tracing",
            "debug_endpoints",
        ],
    }


# ── Run directly ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8100"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
