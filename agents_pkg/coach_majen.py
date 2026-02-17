"""Coach Majen V2 — Agents-as-Tools architecture.

Coach Majen is the MASTER agent. She never loses control of the conversation.
Specialized sub-agents are exposed as TOOLS that she calls when needed.
This keeps her persona consistent while delegating domain-specific work.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import Agent, Runner, RunContextWrapper, function_tool, ModelSettings

from .context import CoachContext
from .guardrails import safety_guardrail, language_guardrail
from . import tools

logger = logging.getLogger("agent.coach")


# ═══════════════════════════════════════════════════════════════════
# SUB-AGENTS (exposed as tools to Coach Majen)
# ═══════════════════════════════════════════════════════════════════

_body_tracker = Agent[CoachContext](
    name="Body Tracker Agent",
    instructions=(
        "Du er en spesialisert agent for vekt- og kroppslogging. "
        "Du logger vekt, henter vekthistorikk, og analyserer trender. "
        "Bruk alltid verktøyene du har tilgjengelig — ALDRI si at du har gjort noe uten å kalle verktøyet. "
        "Svar kort og presist med hva du gjorde."
    ),
    tools=[tools.log_weight, tools.get_weight_history],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.2),
)

_nutrition_agent = Agent[CoachContext](
    name="Nutrition Agent",
    instructions=(
        "Du er en spesialisert ernæringsagent. Du logger måltider, henter dagens ernæring, "
        "og lager kostholdsplaner. Når du lager en kostholdsplan, bruk save_nutrition_plan med "
        "ALLE detaljer (kcal, protein, karbs, fett). "
        "Bruk alltid verktøyene — ALDRI si at du har gjort noe uten å kalle verktøyet. "
        "Svar kort med hva du gjorde og resultatet."
    ),
    tools=[tools.log_meal, tools.get_today_nutrition, tools.save_nutrition_plan],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.3),
)

_training_planner = Agent[CoachContext](
    name="Training Planner Agent",
    instructions=(
        "Du er en spesialisert treningsplanlegger. Du lager treningsplaner og lagrer dem "
        "i Student Senteret. Når du mottar en plan å lagre, bruk save_training_plan med "
        "komplett days_json som inneholder ALLE dager og øvelser. "
        "ALDRI si at du har lagret noe uten å faktisk kalle save_training_plan. "
        "Svar kort med hva du gjorde."
    ),
    tools=[tools.save_training_plan, tools.get_current_training_plan],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.3),
)

_workout_logger = Agent[CoachContext](
    name="Workout Logger Agent",
    instructions=(
        "Du er en spesialisert agent for treningslogging. Du logger fullførte treningsøkter. "
        "Bruk alltid log_workout verktøyet. Svar kort med bekreftelse."
    ),
    tools=[tools.log_workout],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.2),
)

_goals_agent = Agent[CoachContext](
    name="Goals Agent",
    instructions=(
        "Du er en spesialisert agent for mål-setting. Du lagrer og oppdaterer brukerens "
        "fitness-mål. Bruk alltid save_goal verktøyet. Svar kort med bekreftelse."
    ),
    tools=[tools.save_goal],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.3),
)

_profile_agent = Agent[CoachContext](
    name="Profile & Memory Agent",
    instructions=(
        "Du er en spesialisert agent for profil- og minnehåndtering. "
        "Du oppdaterer brukerens profil og husker viktige fakta om dem. "
        "Bruk remember_fact for ting som skader, allergier, preferanser. "
        "Bruk update_profile for profildata. Svar kort med bekreftelse."
    ),
    tools=[tools.remember_fact, tools.update_profile],
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0.2),
)


# ═══════════════════════════════════════════════════════════════════
# AGENT-AS-TOOL WRAPPERS
# ═══════════════════════════════════════════════════════════════════
# These wrap each sub-agent as a tool Coach Majen can call.
# Coach Majen sends a task description; the sub-agent executes it.

@function_tool(timeout=30.0)
async def delegate_body_tracking(ctx: RunContextWrapper[CoachContext], task: str) -> str:
    """Log weight or get weight history. Send a clear task description.

    Args:
        task: What to do (e.g. 'Logg vekt 82 kg for i dag' or 'Hent vekthistorikk siste 14 dager').
    """
    result = await Runner.run(_body_tracker, task, context=ctx.context)
    return str(result.final_output)


@function_tool(timeout=30.0)
async def delegate_nutrition(ctx: RunContextWrapper[CoachContext], task: str) -> str:
    """Log meals, get today's nutrition, or save a nutrition plan. Send a clear task description.

    Args:
        task: What to do (e.g. 'Logg frokost: 2 egg og havregrøt, ca 400 kcal, 30g protein, 40g karbs, 15g fett'
              or 'Lagre kostholdsplan: 2800 kcal, 180g protein, 310g karbs, 85g fett').
    """
    result = await Runner.run(_nutrition_agent, task, context=ctx.context)
    return str(result.final_output)


@function_tool(timeout=45.0)
async def delegate_training_plan(ctx: RunContextWrapper[CoachContext], task: str) -> str:
    """Save or retrieve a training plan. Include ALL plan details in the task.

    Args:
        task: Complete plan to save including all days and exercises in detail.
              Example: 'Lagre treningsplan med 5 dager: Mandag=Bryst(Benkpress 4x6-8, Skrå hantelpress 3x8-10), ...'
    """
    logger.info(f"[delegate_training_plan] user={ctx.context.user_id} task_len={len(task)}")
    result = await Runner.run(_training_planner, task, context=ctx.context)

    # Log what sub-agent tools were called
    sub_tools = []
    for item in result.new_items:
        item_type = getattr(item, "type", "")
        if item_type == "tool_call_item":
            name = getattr(item, "name", None) or getattr(getattr(item, "raw_item", None), "name", "")
            if name:
                sub_tools.append(name)
        elif item_type == "tool_call_output_item":
            output = getattr(item, "output", "")
            if output:
                logger.info(f"[delegate_training_plan] tool_output: {str(output)[:300]}")

    logger.info(f"[delegate_training_plan] sub_tools_called={sub_tools} final_output_len={len(str(result.final_output))}")
    return str(result.final_output)


@function_tool(timeout=30.0)
async def delegate_workout_log(ctx: RunContextWrapper[CoachContext], task: str) -> str:
    """Log a completed workout session. Include description and exercises.

    Args:
        task: What to log (e.g. 'Logg trening for i dag: Bryst og triceps, 60 min').
    """
    result = await Runner.run(_workout_logger, task, context=ctx.context)
    return str(result.final_output)


@function_tool(timeout=30.0)
async def delegate_goals(ctx: RunContextWrapper[CoachContext], task: str) -> str:
    """Save or update the user's fitness goals.

    Args:
        task: Goal to save (e.g. 'Lagre mål: 80kg kroppsvekt innen 12 uker, benkpress 100kg').
    """
    result = await Runner.run(_goals_agent, task, context=ctx.context)
    return str(result.final_output)


@function_tool(timeout=30.0)
async def delegate_profile(ctx: RunContextWrapper[CoachContext], task: str) -> str:
    """Update user profile or remember important facts about the user.

    Args:
        task: What to save (e.g. 'Husk at brukeren har skulderproblemer høyre side'
              or 'Oppdater profil: trener 5 dager i uken').
    """
    result = await Runner.run(_profile_agent, task, context=ctx.context)
    return str(result.final_output)


# ═══════════════════════════════════════════════════════════════════
# DYNAMIC INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════

def build_instructions(ctx: RunContextWrapper[CoachContext], agent: Agent[CoachContext]) -> str:
    """Build Coach Majen's system prompt dynamically with user data."""
    c = ctx.context

    user_info_block = ""
    if c.onboarding_summary:
        user_info_block = f"""

## BRUKERENS DATA (fra onboarding og profil)
{c.onboarding_summary}
"""

    memory_block = ""
    if c.user_context_summary:
        memory_block = f"""

## TING JEG HUSKER OM DENNE BRUKEREN
{c.user_context_summary}
"""

    mentor_personality = ""
    if c.mentor_voice_tone or c.mentor_training_philosophy:
        mentor_personality = f"""

## MIN PERSONLIGHET OG FILOSOFI
{f"Tone: {c.mentor_voice_tone}" if c.mentor_voice_tone else ""}
{f"Treningsfilosofi: {c.mentor_training_philosophy}" if c.mentor_training_philosophy else ""}
{f"Ernæringsfilosofi: {c.mentor_nutrition_philosophy}" if c.mentor_nutrition_philosophy else ""}
{f"Spesielle instruksjoner: {c.mentor_core_instructions}" if c.mentor_core_instructions else ""}
"""

    return f"""Du er {c.mentor_name}, en profesjonell og varm online fitness-coach på Mentorio-plattformen.
Brukeren heter {c.user_name or 'brukeren'}.

# KJERNEREGLER

1. **ALLTID snakk norsk** (bokmål). Aldri svar på engelsk med mindre brukeren eksplisitt ber om det.
2. **Vær varm, motiverende og personlig** — bruk brukerens navn, vær entusiastisk men profesjonell.
3. **Ikke spør om informasjon du allerede har**. Se "BRUKERENS DATA" under. Bruk den informasjonen direkte.
4. **Vær handlingsorientert**. Når brukeren ber om en plan: LAG planen med en gang basert på data du har. Ikke still unødvendige spørsmål.
5. **Bruk verktøyene dine ALLTID** når du gjør noe konkret (logge vekt, lagre plan, etc.). ALDRI si "jeg har lagret" uten å faktisk kalle et verktøy.
{user_info_block}{memory_block}{mentor_personality}
# VERKTØY — DELEGER TIL SPESIALISERTE AGENTER

Du har tilgang til følgende verktøy for å utføre handlinger:

- **delegate_body_tracking**: Logg vekt, hent vekthistorikk
- **delegate_nutrition**: Logg måltider, hent dagens ernæring, lagre kostholdsplan
- **delegate_training_plan**: Lagre eller hent treningsplan
- **delegate_workout_log**: Logg fullført trening
- **delegate_goals**: Lagre eller oppdater mål
- **delegate_profile**: Oppdater profil, husk viktige fakta
- **get_user_stats**: Hent all brukerdata (vekt, mål, planer, etc.)

# OBLIGATORISK VERKTØYBRUK

Når noe av dette skjer, MÅ du kalle riktig verktøy:
- Bruker nevner vekt (f.eks. "veier 82 kg") → delegate_body_tracking("Logg vekt 82 kg for i dag")
- Bruker nevner et måltid → delegate_nutrition("Logg [beskrivelse med estimerte makroer]")
- Bruker godkjenner en plan → delegate_training_plan("Lagre treningsplan: [hele planen som JSON-kompatibel tekst]")
- Bruker vil ha en kostholdsplan lagret → delegate_nutrition("Lagre kostholdsplan: [detaljer]")
- Bruker nevner skade/allergi → delegate_profile("Husk at brukeren [detalje]")
- Du trenger oppdatert data → get_user_stats

# SAMTALEOPPSTART (første melding til ny bruker)

Når du møter en bruker for første gang:
1. Hent data med get_user_stats
2. Oppsummer det du vet om brukeren (fra onboarding-data + profil)
3. Forklar kort hva du kan hjelpe med
4. Spør om det er noe du mangler, eller om de vil starte med en plan
5. Vær KONKRET om neste steg

# NÅR BRUKEREN BER OM EN PLAN

1. IKKE spør om info du allerede har (treningsdager, mål, utstyr — sjekk brukerdata!)
2. LAG planen UMIDDELBART basert på det du vet
3. Presenter planen i et ryddig format
4. Spør: "Skal jeg lagre denne i Student Senteret ditt?"
5. Når brukeren sier ja/lagre/godkjent: KALL delegate_training_plan/delegate_nutrition MED HELE PLANEN

# KRITISK: NÅR BRUKEREN GODKJENNER

Når brukeren sier "ja", "lagre", "godkjent", "ser bra ut", "kjør på" eller lignende
etter at du har presentert en plan:

→ Du HUSKER planen du nettopp presenterte
→ Du KALLER verktøyet UMIDDELBART med KOMPLETT plandata
→ Du bekrefter at den er lagret og hvor brukeren finner den
→ ALDRI spør "hvilken plan?" eller "kan du gjenta?" — du VET dette
"""


# ═══════════════════════════════════════════════════════════════════
# BUILD COACH MAJEN
# ═══════════════════════════════════════════════════════════════════

def build_coach_majen() -> Agent[CoachContext]:
    """Build and return the Coach Majen master agent."""
    return Agent[CoachContext](
        name="Coach Majen",
        instructions=build_instructions,
        tools=[
            delegate_body_tracking,
            delegate_nutrition,
            delegate_training_plan,
            delegate_workout_log,
            delegate_goals,
            delegate_profile,
            tools.get_user_stats,
        ],
        model="gpt-4o",
        model_settings=ModelSettings(
            temperature=0.7,
            top_p=0.9,
        ),
        input_guardrails=[safety_guardrail],
        output_guardrails=[language_guardrail],
    )


# ═══════════════════════════════════════════════════════════════════
# CONTEXT LOADER
# ═══════════════════════════════════════════════════════════════════

def load_context(user_id: str, mentor_id: str) -> CoachContext:
    """Load user + mentor data from Supabase and build a CoachContext."""
    from .db import get_db, find_one, find_many, USERS, USER_PROFILES, USER_CONTEXT, COACH_KNOWLEDGE

    ctx = CoachContext(user_id=user_id, mentor_id=mentor_id)

    # Load user info
    user = find_one(USERS, {"id": user_id})
    if user:
        ctx.user_name = user.get("firstName") or user.get("first_name") or user.get("username") or ""

    # Load onboarding profile
    profile = find_one(USER_PROFILES, {"user_id": user_id})
    if profile:
        parts = []
        mapping = {
            "gender": "Kjønn", "age": "Alder", "height_cm": "Høyde",
            "current_weight_kg": "Vekt", "training_days_per_week": "Treningsdager/uke",
            "goals": "Mål", "fitness_level": "Treningsnivå",
            "training_location": "Treningssted", "available_equipment": "Utstyr",
            "injury_history": "Skader", "nutrition_preferences": "Matpreferanser",
        }
        for key, label in mapping.items():
            val = profile.get(key)
            if val:
                parts.append(f"- {label}: {val}")
        if parts:
            ctx.onboarding_summary = "\n".join(parts)

    # Load user context (remembered facts)
    context_rows = find_many(USER_CONTEXT, {"user_id": user_id}, select="key, value")
    if context_rows:
        ctx.user_context_summary = "\n".join(
            f"- {r['key']}: {r['value']}" for r in context_rows
        )

    # Load mentor info
    mentor = find_one(USERS, {"id": mentor_id})
    if mentor:
        ctx.mentor_name = f"Coach {mentor.get('firstName') or mentor.get('first_name') or 'Majen'}"

    # Load coach knowledge from DB
    knowledge = find_many(COACH_KNOWLEDGE, {"mentor_id": mentor_id}, select="key, value")
    for row in knowledge:
        key = row.get("key", "")
        val = row.get("value", "")
        if key == "voice_tone":
            ctx.mentor_voice_tone = val
        elif key == "training_philosophy":
            ctx.mentor_training_philosophy = val
        elif key == "nutrition_philosophy":
            ctx.mentor_nutrition_philosophy = val
        elif key == "core_instructions":
            ctx.mentor_core_instructions = val

    logger.info(
        f"[load_context] user={user_id} name={ctx.user_name} "
        f"has_profile={'yes' if ctx.onboarding_summary else 'no'} "
        f"has_memory={'yes' if ctx.user_context_summary else 'no'}"
    )
    return ctx
