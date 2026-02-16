"""Guardrails for the Coach Majen agent system.

Input guardrails:  run BEFORE the agent, block inappropriate requests
Output guardrails: run AFTER the agent, validate response quality
"""

from __future__ import annotations

import logging

from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)

from .context import CoachContext

logger = logging.getLogger("agent.guardrails")


# ═══════════════════════════════════════════════════════════════════
# INPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

class SafetyCheckOutput(BaseModel):
    """Output from the safety guardrail agent."""
    is_unsafe: bool
    category: str = ""
    reasoning: str = ""

_safety_agent = Agent(
    name="Safety Checker",
    instructions=(
        "You are a safety classifier for a fitness coaching platform. "
        "Check if the user's message contains any of these unsafe categories:\n"
        "1. Medical advice requests (asking for diagnosis, medication, treatment of medical conditions)\n"
        "2. Mental health crisis (suicidal thoughts, self-harm)\n"
        "3. Harassment or abuse towards the coach\n"
        "4. Requests for illegal substances (steroids, banned drugs)\n"
        "5. Eating disorder encouragement (extreme restriction, purging)\n\n"
        "IMPORTANT: Normal fitness questions are SAFE. Questions about weight loss, "
        "muscle gain, training programs, nutrition, supplements (legal ones like creatine, "
        "protein), and general health are ALL SAFE.\n"
        "Only flag genuinely dangerous requests."
    ),
    output_type=SafetyCheckOutput,
    model="gpt-4o-mini",
)


@input_guardrail
async def safety_guardrail(
    ctx: RunContextWrapper[CoachContext],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Block messages that request medical advice, promote harmful behavior, etc."""
    result = await Runner.run(_safety_agent, input, context=ctx.context)
    output = result.final_output_as(SafetyCheckOutput)

    if output.is_unsafe:
        logger.warning(f"[safety] BLOCKED user={ctx.context.user_id} category={output.category} reason={output.reasoning[:100]}")

    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=output.is_unsafe,
    )


# ═══════════════════════════════════════════════════════════════════
# OUTPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

class LanguageCheckOutput(BaseModel):
    """Output from the language guardrail agent."""
    is_wrong_language: bool
    detected_language: str = ""
    reasoning: str = ""

_language_agent = Agent(
    name="Language Checker",
    instructions=(
        "You check if the given text is written in Norwegian (Bokmål or Nynorsk). "
        "The text SHOULD be in Norwegian. If it's in English, Swedish, Danish, or any other language, "
        "set is_wrong_language to true. "
        "Short words, names, or technical terms in English within a Norwegian text are OK. "
        "Only flag if the MAJORITY of the text is not Norwegian."
    ),
    output_type=LanguageCheckOutput,
    model="gpt-4o-mini",
)


@output_guardrail
async def language_guardrail(
    ctx: RunContextWrapper[CoachContext],
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """Ensure the agent's response is in Norwegian."""
    if not output or len(output.strip()) < 20:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    result = await Runner.run(_language_agent, f"Check this text: {output[:500]}", context=ctx.context)
    check = result.final_output_as(LanguageCheckOutput)

    if check.is_wrong_language:
        logger.warning(f"[language] Response not in Norwegian: {check.detected_language}")

    return GuardrailFunctionOutput(
        output_info=check,
        tripwire_triggered=check.is_wrong_language,
    )
