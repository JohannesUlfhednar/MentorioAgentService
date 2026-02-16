"""FastAPI server for the Coach Majen Agent Service.

This replaces the Node.js mentorAIService.ts with a proper multi-agent
system built on the OpenAI Agents SDK.

The Node.js backend calls POST /chat with user/mentor IDs and the message,
and this service runs the full agent pipeline and returns the response.
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents import Runner

from agents_pkg.coach_majen import (
    build_coach_majen,
    load_context,
    CoachContext,
    COACH_MAJEN_INSTRUCTIONS,
)
from agents_pkg.db import find_one, find_many, USERS, COACH_KNOWLEDGE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-service")


# ── App lifecycle ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent service starting...")
    yield
    logger.info("Agent service shutting down...")


app = FastAPI(
    title="MentorOS Agent Service",
    description="Multi-agent coaching system powered by OpenAI Agents SDK",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ───────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    mentor_id: str
    message: str
    conversation_history: list[dict] | None = None  # [{role, content}]


class ChatResponse(BaseModel):
    response: str
    agent_name: str
    tools_called: list[str]


# ── Agent cache (build once per mentor) ─────────────────────────────

_agent_cache: dict[str, any] = {}


async def get_or_build_agent(mentor_id: str):
    if mentor_id in _agent_cache:
        return _agent_cache[mentor_id]

    # Load mentor profile from DB
    mentor = find_one(USERS, {"id": mentor_id})
    mentor_name = "Coach Majen"
    extra_instructions = ""
    voice_tone = ""
    training_philosophy = ""
    nutrition_philosophy = ""

    if mentor:
        first = mentor.get("first_name", "")
        last = mentor.get("last_name", "")
        if first or last:
            mentor_name = f"Coach {first} {last}".strip()
        voice_tone = mentor.get("mentor_ai_voice_tone", "") or ""
        training_philosophy = mentor.get("mentor_ai_training_philosophy", "") or ""
        nutrition_philosophy = mentor.get("mentor_ai_nutrition_philosophy", "") or ""
        core = mentor.get("core_instructions", "") or ""
        if core:
            extra_instructions = core

    agent = build_coach_majen(
        mentor_name=mentor_name,
        extra_instructions=extra_instructions,
        voice_tone=voice_tone,
        training_philosophy=training_philosophy,
        nutrition_philosophy=nutrition_philosophy,
    )
    _agent_cache[mentor_id] = agent
    return agent


# ── Main chat endpoint ──────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        logger.info(f"Chat request: user={req.user_id}, mentor={req.mentor_id}, msg='{req.message[:80]}...'")

        # Build/cache the agent
        agent = await get_or_build_agent(req.mentor_id)

        # Load user context
        ctx = await load_context(req.user_id, req.mentor_id)

        # Build input with user data context
        input_parts = []

        # Inject user data into the conversation so the agent knows who they're talking to
        if ctx.user_name or ctx.onboarding_summary != "Ingen onboarding-data ennå":
            input_parts.append(
                f"[SYSTEM: Brukerens navn er '{ctx.user_name}'. "
                f"Onboarding-data:\n{ctx.onboarding_summary}\n"
                f"Kontekst:\n{ctx.user_context_summary}]"
            )

        input_parts.append(req.message)
        full_input = "\n\n".join(input_parts)

        # Build message list with conversation history
        messages = []
        if req.conversation_history:
            for msg in req.conversation_history[-15:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content.strip():
                    messages.append({"role": role, "content": content})

        # Add current message
        messages.append({"role": "user", "content": full_input})

        # Run the agent
        result = await Runner.run(
            agent,
            messages,
            context=ctx,
        )

        response_text = result.final_output or ""
        agent_name = result.last_agent.name if result.last_agent else agent.name

        # Collect tool call names from the run
        tools_called = []
        for item in result.new_items:
            if hasattr(item, "raw_item") and hasattr(item.raw_item, "name"):
                tools_called.append(item.raw_item.name)

        logger.info(f"Response from {agent_name}: {len(response_text)} chars, tools: {tools_called}")

        return ChatResponse(
            response=response_text,
            agent_name=agent_name,
            tools_called=tools_called,
        )

    except Exception as e:
        logger.exception(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "MentorOS Agent Service"}


# ── Run with uvicorn ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8100))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
