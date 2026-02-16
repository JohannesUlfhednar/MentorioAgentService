"""Coach Majen multi-agent system using OpenAI Agents SDK.

Architecture:
  Coach Majen (master agent)
    ├── Training Plan Agent   — creates/saves training plans
    ├── Nutrition Agent       — creates/saves nutrition plans, logs meals
    ├── Body Tracking Agent   — logs weight, tracks body measurements
    ├── Workout Logger Agent  — logs workout sessions
    ├── Goals Agent           — sets and tracks fitness goals
    └── Profile Agent         — updates user profile and context
"""

from __future__ import annotations

from agents import Agent, function_tool, RunContextWrapper
from pydantic import BaseModel

from . import tools
from .db import find_one, find_many, USER_PROFILES, USER_CONTEXT, COACH_KNOWLEDGE


# ═══════════════════════════════════════════════════════════════════
# CONTEXT: shared data available to all agents during a run
# ═══════════════════════════════════════════════════════════════════

class CoachContext(BaseModel):
    user_id: str
    mentor_id: str
    user_name: str = ""
    onboarding_summary: str = ""
    user_context_summary: str = ""


# ═══════════════════════════════════════════════════════════════════
# TOOL WRAPPERS — bind user_id from context
# ═══════════════════════════════════════════════════════════════════

# We need to wrap the raw tools so they automatically get user_id from the run context.
# The OpenAI Agents SDK supports function_tool with RunContextWrapper.

@function_tool
async def tool_log_weight(ctx: RunContextWrapper[CoachContext], kg: float, date: str | None = None) -> str:
    """Log the user's body weight. Call when the user mentions their weight."""
    return tools.log_weight(ctx.context.user_id, kg, date)

@function_tool
async def tool_get_weight_history(ctx: RunContextWrapper[CoachContext], days: int = 30) -> str:
    """Get the user's weight history for the given number of days."""
    return tools.get_weight_history(ctx.context.user_id, days)

@function_tool
async def tool_log_meal(
    ctx: RunContextWrapper[CoachContext],
    description: str,
    meal_type: str = "other",
    calories: float = 0,
    protein_g: float = 0,
    carbs_g: float = 0,
    fat_g: float = 0,
    date: str | None = None,
) -> str:
    """Log a meal the user has eaten with estimated macros."""
    return tools.log_meal(ctx.context.user_id, description, meal_type, calories, protein_g, carbs_g, fat_g, date)

@function_tool
async def tool_get_today_meals(ctx: RunContextWrapper[CoachContext]) -> str:
    """Get all meals logged today with totals."""
    return tools.get_today_meals(ctx.context.user_id)

@function_tool
async def tool_save_nutrition_plan(
    ctx: RunContextWrapper[CoachContext],
    kcal: int,
    protein_grams: int,
    carbs_grams: int,
    fat_grams: int,
    meals_json: str = "[]",
    notes: str = "",
    reason: str = "",
) -> str:
    """Save a complete nutrition/meal plan to the student center. Call AFTER the user approves."""
    return tools.save_nutrition_plan(ctx.context.user_id, kcal, protein_grams, carbs_grams, fat_grams, meals_json, notes, reason)

@function_tool
async def tool_log_workout(
    ctx: RunContextWrapper[CoachContext],
    description: str = "",
    date: str | None = None,
    entries_json: str = "[]",
) -> str:
    """Log a completed workout session."""
    return tools.log_workout(ctx.context.user_id, description, date, entries_json)

@function_tool
async def tool_save_training_plan(
    ctx: RunContextWrapper[CoachContext],
    days_json: str,
    reason: str = "",
) -> str:
    """Save a complete training plan to the student center.
    days_json = JSON string: [{day, focus, exercises: [{name, sets, reps}]}]
    Call AFTER the user has approved the plan."""
    return tools.save_training_plan(ctx.context.user_id, days_json, reason)

@function_tool
async def tool_save_goal(
    ctx: RunContextWrapper[CoachContext],
    target_weight_kg: float | None = None,
    strength_targets: str = "",
    horizon_weeks: int | None = None,
    plan_text: str = "",
) -> str:
    """Save a fitness goal to the student center dashboard."""
    return tools.save_goal(ctx.context.user_id, target_weight_kg, strength_targets, horizon_weeks, plan_text)

@function_tool
async def tool_update_user_context(ctx: RunContextWrapper[CoachContext], key: str, value: str) -> str:
    """Remember a fact about the user (injuries, preferences, etc.)."""
    return tools.update_user_context(ctx.context.user_id, key, value)

@function_tool
async def tool_update_profile(
    ctx: RunContextWrapper[CoachContext],
    current_weight_kg: float | None = None,
    training_days_per_week: int | None = None,
    goals: str | None = None,
    injury_history: str | None = None,
    nutrition_preferences: str | None = None,
) -> str:
    """Update the user's profile data in the student center."""
    return tools.update_profile(ctx.context.user_id, current_weight_kg, training_days_per_week, goals, injury_history, nutrition_preferences)

@function_tool
async def tool_get_user_stats(ctx: RunContextWrapper[CoachContext]) -> str:
    """Fetch the user's current stats: latest weight, goals, today's meals, training plan summary."""
    return tools.get_user_stats(ctx.context.user_id)


# ═══════════════════════════════════════════════════════════════════
# SUB-AGENTS (specialists)
# ═══════════════════════════════════════════════════════════════════

body_tracking_agent = Agent[CoachContext](
    name="Body Tracking Agent",
    handoff_description="Spesialist for vektlogging og kroppssammensetning. Bruk denne når brukeren nevner vekt eller vil logge vekt.",
    instructions=(
        "Du er en spesialist-agent for kroppsdata. "
        "Når du mottar en oppgave, utfør den UMIDDELBART ved å kalle riktig verktøy. "
        "Logg vekt med tool_log_weight. Hent historikk med tool_get_weight_history. "
        "Svar ALLTID på norsk (bokmål). Bekreft kort hva du gjorde."
    ),
    tools=[tool_log_weight, tool_get_weight_history],
)

nutrition_agent = Agent[CoachContext](
    name="Nutrition Agent",
    handoff_description="Spesialist for ernæring, måltider og kostholdsplaner. Bruk for måltid-logging og lagring av kostholdsplaner.",
    instructions=(
        "Du er en spesialist-agent for ernæring og kosthold. "
        "Logg måltider med tool_log_meal. Lag og lagre kostholdsplaner med tool_save_nutrition_plan. "
        "Hent dagens måltider med tool_get_today_meals. "
        "Når du lagrer en kostholdsplan, inkluder daglige makromål og måltider. "
        "Svar ALLTID på norsk (bokmål). Bekreft kort hva du gjorde."
    ),
    tools=[tool_log_meal, tool_get_today_meals, tool_save_nutrition_plan],
)

training_plan_agent = Agent[CoachContext](
    name="Training Plan Agent",
    handoff_description="Spesialist for treningsplaner. Bruk for å lagre en treningsplan i Student Senteret.",
    instructions=(
        "Du er en spesialist-agent for treningsplaner. "
        "Lagre treningsplaner med tool_save_training_plan. "
        "days_json MÅ være en gyldig JSON-streng med format: "
        '[{"day":"Mandag","focus":"Bryst og Triceps","exercises":[{"name":"Benkpress","sets":4,"reps":"6-8"}]}] '
        "Svar ALLTID på norsk (bokmål). Bekreft kort hva du gjorde."
    ),
    tools=[tool_save_training_plan],
)

workout_logger_agent = Agent[CoachContext](
    name="Workout Logger Agent",
    handoff_description="Spesialist for logging av gjennomførte treningsøkter.",
    instructions=(
        "Du er en spesialist-agent for treningslogging. "
        "Logg gjennomførte treningsøkter med tool_log_workout. "
        "Svar ALLTID på norsk (bokmål). Bekreft kort hva du gjorde."
    ),
    tools=[tool_log_workout],
)

goals_agent = Agent[CoachContext](
    name="Goals Agent",
    handoff_description="Spesialist for mål og målsetting. Bruk for å sette og lagre treningsmål.",
    instructions=(
        "Du er en spesialist-agent for mål. "
        "Lagre mål med tool_save_goal. Inkluder vektmål, styrketargets, og horisont. "
        "Svar ALLTID på norsk (bokmål). Bekreft kort hva du gjorde."
    ),
    tools=[tool_save_goal],
)

profile_agent = Agent[CoachContext](
    name="Profile Agent",
    handoff_description="Spesialist for brukerprofil og kontekst. Bruk for å oppdatere profil eller lagre ny info om brukeren.",
    instructions=(
        "Du er en spesialist-agent for brukerprofiler. "
        "Oppdater profil med tool_update_profile. Lagre fakta med tool_update_user_context. "
        "Hent brukerdata med tool_get_user_stats. "
        "Svar ALLTID på norsk (bokmål). Bekreft kort hva du gjorde."
    ),
    tools=[tool_update_profile, tool_update_user_context, tool_get_user_stats],
)


# ═══════════════════════════════════════════════════════════════════
# COACH MAJEN — the master agent
# ═══════════════════════════════════════════════════════════════════

COACH_MAJEN_INSTRUCTIONS = """\
Du er Coach Majen — en personlig trener og coach-agent. Du SKAL alltid svare på norsk (bokmål).
Du representerer en ekte coach og skal oppføre deg som om du er en virkelig person.

PERSONLIGHET:
- Varm, engasjert og motiverende
- Direkte og handlingsorientert — ikke omstendelig
- Bruker brukerens navn når det passer
- Feirer fremgang og oppmuntrer ved motgang

DINE SPESIALAGENTER:
Du har et team av spesialagenter som utfører oppgaver for deg:
- Body Tracking Agent: Logg vekt og kroppsmålinger
- Nutrition Agent: Logg måltider og lagre kostholdsplaner
- Training Plan Agent: Lagre treningsplaner
- Workout Logger Agent: Logg gjennomførte treningsøkter
- Goals Agent: Sett og lagre treningsmål
- Profile Agent: Oppdater brukerprofil og lagre info

VIKTIGE REGLER:
1. ALDRI si at du har gjort noe uten å faktisk delegere til en spesialist-agent
2. Når brukeren nevner vekt: Deleger UMIDDELBART til Body Tracking Agent
3. Når brukeren godkjenner en plan: Deleger UMIDDELBART til riktig agent for å lagre den
4. Du har også direkte tilgang til tool_get_user_stats for å hente brukerdata
5. IKKE spør om informasjon du allerede har (se brukerdata nedenfor)

NÅR BRUKEREN BER OM EN PLAN:
1. Bruk eksisterende data (onboarding, profil) til å lage planen MED EN GANG
2. Presenter planen i chatten
3. Spør: "Skal jeg lagre denne i Student Senteret ditt?"
4. Når de sier ja/lagre/ok: Deleger til Training Plan Agent eller Nutrition Agent

SAMTALEOPPSTART:
Ved første melding:
1. Ønsk velkommen varmt (bruk navnet)
2. Oppsummer KORT hva du vet om dem
3. Tilby å lage en plan med en gang
"""


def build_coach_majen(
    mentor_name: str = "Coach Majen",
    extra_instructions: str = "",
    voice_tone: str = "",
    training_philosophy: str = "",
    nutrition_philosophy: str = "",
) -> Agent[CoachContext]:
    """Build the Coach Majen master agent with all sub-agents."""

    instructions = COACH_MAJEN_INSTRUCTIONS

    if voice_tone:
        instructions += f"\nSTEMME OG TONE: {voice_tone}\n"
    if training_philosophy:
        instructions += f"\nTRENINGSFILOSOFI: {training_philosophy}\n"
    if nutrition_philosophy:
        instructions += f"\nERNÆRINGSFILOSOFI: {nutrition_philosophy}\n"
    if extra_instructions:
        instructions += f"\nEKSTRA INSTRUKSJONER:\n{extra_instructions}\n"

    return Agent[CoachContext](
        name=mentor_name,
        instructions=instructions,
        handoffs=[
            body_tracking_agent,
            nutrition_agent,
            training_plan_agent,
            workout_logger_agent,
            goals_agent,
            profile_agent,
        ],
        tools=[tool_get_user_stats],
        model="gpt-4o",
    )


async def load_context(user_id: str, mentor_id: str) -> CoachContext:
    """Load all user data into a CoachContext for a run."""
    # User profile
    profile = find_one(USER_PROFILES, {"user_id": user_id}) or {}

    # User context facts
    ctx_rows = find_many(USER_CONTEXT, {"user_id": user_id}, select="key, value")
    ctx_dict = {r["key"]: r["value"] for r in ctx_rows}

    # Build summaries
    user_name = ctx_dict.get("navn", profile.get("name", ""))

    profile_lines = []
    if profile.get("goals"):
        profile_lines.append(f"Mål: {profile['goals']}")
    if profile.get("current_weight_kg"):
        profile_lines.append(f"Nåværende vekt: {profile['current_weight_kg']} kg")
    if profile.get("training_days_per_week"):
        profile_lines.append(f"Treningsdager per uke: {profile['training_days_per_week']}")
    if profile.get("injury_history"):
        profile_lines.append(f"Skadehistorikk: {profile['injury_history']}")
    if profile.get("nutrition_preferences"):
        profile_lines.append(f"Matpreferanser: {profile['nutrition_preferences']}")

    ctx_lines = [f"  {k}: {v}" for k, v in ctx_dict.items()]

    onboarding_summary = "\n".join(profile_lines) if profile_lines else "Ingen onboarding-data ennå"
    user_context_summary = "\n".join(ctx_lines) if ctx_lines else "Ingen ekstra kontekst"

    return CoachContext(
        user_id=user_id,
        mentor_id=mentor_id,
        user_name=user_name,
        onboarding_summary=onboarding_summary,
        user_context_summary=user_context_summary,
    )
