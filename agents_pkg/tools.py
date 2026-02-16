"""Tool functions for Coach Majen sub-agents.

Each function is a plain Python function decorated with @function_tool.
The SDK auto-generates JSON schemas from type hints + docstrings.
All tools receive RunContextWrapper[CoachContext] as first arg to get user_id.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Optional

from agents import RunContextWrapper, function_tool

from .context import CoachContext
from .db import (
    get_db, today_str, now_iso, upsert_row, insert_row,
    find_one, find_many, next_version,
    WEIGHT_ENTRIES, MEAL_LOGS, WORKOUT_LOGS, TRAINING_PLAN_VERSIONS,
    NUTRITION_PLAN_VERSIONS, GOALS, USER_PROFILES, USER_CONTEXT,
    CHANGE_EVENTS,
)

logger = logging.getLogger("agent.tools")


# ═══════════════════════════════════════════════════════════════════
# WEIGHT & BODY TOOLS
# ═══════════════════════════════════════════════════════════════════

@function_tool(timeout=15.0)
async def log_weight(ctx: RunContextWrapper[CoachContext], kg: float, log_date: str = "") -> str:
    """Log the user's body weight in kilograms.

    Args:
        kg: Weight in kilograms (must be between 20 and 500).
        log_date: Date in YYYY-MM-DD format. Leave empty for today.
    """
    uid = ctx.context.user_id
    d = log_date or today_str()
    if kg < 20 or kg > 500:
        return json.dumps({"success": False, "error": "Vekt må være mellom 20 og 500 kg"})

    upsert_row(WEIGHT_ENTRIES, {"user_id": uid, "date": d, "kg": kg}, "user_id,date")
    insert_row(CHANGE_EVENTS, {
        "user_id": uid, "type": "WEIGHT_LOG",
        "summary": f"Vekt logget: {kg} kg ({d})", "actor": "agent",
    })
    logger.info(f"[log_weight] user={uid} kg={kg} date={d}")
    return json.dumps({"success": True, "message": f"Vekt logget: {kg} kg ({d}). Synlig i Student Senteret."})


@function_tool(timeout=10.0)
async def get_weight_history(ctx: RunContextWrapper[CoachContext], days: int = 30) -> str:
    """Get the user's weight history for the given number of days.

    Args:
        days: Number of days of history to retrieve (default 30).
    """
    uid = ctx.context.user_id
    start = (date.today() - timedelta(days=days)).isoformat()
    rows = get_db().table(WEIGHT_ENTRIES) \
        .select("date, kg") \
        .eq("user_id", uid) \
        .gte("date", start) \
        .order("date", desc=False) \
        .execute().data or []
    return json.dumps({"entries": rows, "count": len(rows)})


# ═══════════════════════════════════════════════════════════════════
# MEAL / NUTRITION TOOLS
# ═══════════════════════════════════════════════════════════════════

@function_tool(timeout=15.0)
async def log_meal(
    ctx: RunContextWrapper[CoachContext],
    description: str,
    meal_type: str = "other",
    calories: float = 0,
    protein_g: float = 0,
    carbs_g: float = 0,
    fat_g: float = 0,
    log_date: str = "",
) -> str:
    """Log a meal the user has eaten with estimated macros.

    Args:
        description: What the user ate (e.g. '2 egg, havregrøt, juice').
        meal_type: One of: breakfast, lunch, dinner, snack, shake, other.
        calories: Estimated calories (kcal).
        protein_g: Estimated protein in grams.
        carbs_g: Estimated carbohydrates in grams.
        fat_g: Estimated fat in grams.
        log_date: Date in YYYY-MM-DD. Leave empty for today.
    """
    uid = ctx.context.user_id
    d = log_date or today_str()
    insert_row(MEAL_LOGS, {
        "user_id": uid, "date": d, "meal_type": meal_type,
        "description": description,
        "total_calories": calories, "total_protein_g": protein_g,
        "total_carbs_g": carbs_g, "total_fat_g": fat_g,
        "items": [],
    })
    logger.info(f"[log_meal] user={uid} desc={description[:40]} kcal={calories}")
    return json.dumps({
        "success": True,
        "message": f"Måltid logget: {description} ({calories} kcal, {protein_g}g P, {carbs_g}g K, {fat_g}g F). Synlig i Ernæring-fanen.",
    })


@function_tool(timeout=10.0)
async def get_today_nutrition(ctx: RunContextWrapper[CoachContext]) -> str:
    """Get all meals logged today with total macros."""
    uid = ctx.context.user_id
    meals = find_many(MEAL_LOGS, {"user_id": uid, "date": today_str()}, order_by="created_at")
    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    for m in meals:
        totals["calories"] += m.get("total_calories", 0) or 0
        totals["protein"] += m.get("total_protein_g", 0) or 0
        totals["carbs"] += m.get("total_carbs_g", 0) or 0
        totals["fat"] += m.get("total_fat_g", 0) or 0
    summaries = [f"- {m.get('description', '?')} ({m.get('total_calories', 0)} kcal)" for m in meals]
    return json.dumps({"meals": summaries, "totals": totals, "count": len(meals)})


@function_tool(timeout=20.0)
async def save_nutrition_plan(
    ctx: RunContextWrapper[CoachContext],
    kcal: int,
    protein_grams: int,
    carbs_grams: int,
    fat_grams: int,
    meals_json: str = "[]",
    notes: str = "",
    reason: str = "",
) -> str:
    """Save a complete nutrition plan to the student center Nutrition tab.
    Call AFTER the user has approved the plan.

    Args:
        kcal: Daily calorie target.
        protein_grams: Daily protein target in grams.
        carbs_grams: Daily carbohydrates target in grams.
        fat_grams: Daily fat target in grams.
        meals_json: JSON string array of meal objects with name, time, items.
        notes: General nutrition notes for the user.
        reason: Brief reason for plan creation/update.
    """
    uid = ctx.context.user_id
    try:
        meals = json.loads(meals_json)
    except Exception:
        meals = []

    version = next_version(NUTRITION_PLAN_VERSIONS, uid)
    insert_row(NUTRITION_PLAN_VERSIONS, {
        "user_id": uid, "version": version,
        "kcal": kcal, "protein_grams": protein_grams,
        "carbs_grams": carbs_grams, "fat_grams": fat_grams,
        "meals": meals, "notes": notes,
        "created_at": now_iso(),
    })
    insert_row(CHANGE_EVENTS, {
        "user_id": uid, "type": "NUTRITION_PLAN_SAVED",
        "summary": f"Kostholdsplan v{version}: {kcal} kcal, {protein_grams}g P",
        "actor": "agent", "after": {"version": version, "reason": reason},
    })
    logger.info(f"[save_nutrition_plan] user={uid} v{version} {kcal}kcal")
    return json.dumps({
        "success": True,
        "message": f"Kostholdsplan lagret (v{version}): {kcal} kcal, {protein_grams}g protein, {carbs_grams}g karbs, {fat_grams}g fett. Brukeren finner den i Ernæring-fanen i Student Senteret.",
    })


# ═══════════════════════════════════════════════════════════════════
# WORKOUT / TRAINING TOOLS
# ═══════════════════════════════════════════════════════════════════

@function_tool(timeout=15.0)
async def log_workout(
    ctx: RunContextWrapper[CoachContext],
    description: str = "",
    log_date: str = "",
    entries_json: str = "[]",
) -> str:
    """Log a completed workout session.

    Args:
        description: Brief description of the workout (e.g. 'Bryst og triceps, 60 min').
        log_date: Date in YYYY-MM-DD. Leave empty for today.
        entries_json: JSON string array of exercise entries with name, sets, reps, weight.
    """
    uid = ctx.context.user_id
    d = log_date or today_str()
    try:
        entries = json.loads(entries_json)
    except Exception:
        entries = []
    upsert_row(WORKOUT_LOGS, {"user_id": uid, "date": d, "entries": entries}, "user_id,date")
    logger.info(f"[log_workout] user={uid} date={d} desc={description[:40]}")
    return json.dumps({"success": True, "message": f"Trening logget for {d}: {description or 'økt registrert'}. Synlig i Aktivitet-fanen."})


@function_tool(timeout=20.0)
async def save_training_plan(
    ctx: RunContextWrapper[CoachContext],
    days_json: str,
    reason: str = "",
) -> str:
    """Save a complete training plan to the student center Activity tab.
    Call AFTER the user has approved the plan.

    Args:
        days_json: JSON string array of training days. Format:
            [{"day":"Mandag","focus":"Bryst og Triceps","exercises":[{"name":"Benkpress","sets":4,"reps":"6-8"},...]}]
        reason: Brief reason for plan creation/update.
    """
    uid = ctx.context.user_id
    try:
        days = json.loads(days_json)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Ugyldig JSON: {e}"})

    if not days or not isinstance(days, list):
        return json.dumps({"success": False, "error": "days_json må inneholde minst 1 treningsdag"})

    version = next_version(TRAINING_PLAN_VERSIONS, uid)
    insert_row(TRAINING_PLAN_VERSIONS, {
        "user_id": uid, "version": version, "days": days,
        "created_at": now_iso(),
    })
    insert_row(CHANGE_EVENTS, {
        "user_id": uid, "type": "TRAINING_PLAN_SAVED",
        "summary": f"Treningsplan v{version}: {len(days)} dager",
        "actor": "agent", "after": {"version": version, "reason": reason},
    })
    logger.info(f"[save_training_plan] user={uid} v{version} {len(days)} days")
    return json.dumps({
        "success": True,
        "message": f"Treningsplan lagret (v{version}) med {len(days)} treningsdager. Brukeren finner den i Aktivitet-fanen i Student Senteret.",
    })


@function_tool(timeout=10.0)
async def get_current_training_plan(ctx: RunContextWrapper[CoachContext]) -> str:
    """Get the user's current training plan from the student center."""
    uid = ctx.context.user_id
    tp = get_db().table(TRAINING_PLAN_VERSIONS).select("version, days, created_at") \
        .eq("user_id", uid).order("version", desc=True).limit(1).maybe_single().execute()
    if not tp.data:
        return json.dumps({"has_plan": False, "message": "Brukeren har ingen treningsplan ennå."})
    return json.dumps({"has_plan": True, "version": tp.data.get("version"), "days": tp.data.get("days", [])})


# ═══════════════════════════════════════════════════════════════════
# GOALS TOOLS
# ═══════════════════════════════════════════════════════════════════

@function_tool(timeout=15.0)
async def save_goal(
    ctx: RunContextWrapper[CoachContext],
    target_weight_kg: float | None = None,
    strength_targets: str = "",
    horizon_weeks: int | None = None,
    plan_text: str = "",
) -> str:
    """Save a fitness goal to the student center dashboard.

    Args:
        target_weight_kg: Target weight in kg (optional).
        strength_targets: Strength goals text (e.g. 'Benkpress 100kg, Markløft 150kg').
        horizon_weeks: Weeks until target (e.g. 12, 16).
        plan_text: Text description of the plan to reach the goal.
    """
    uid = ctx.context.user_id
    get_db().table(GOALS).update({"is_current": False}).eq("user_id", uid).execute()
    version = next_version(GOALS, uid)

    insert_row(GOALS, {
        "user_id": uid, "version": version, "is_current": True,
        "target_weight_kg": target_weight_kg,
        "strength_targets": strength_targets,
        "horizon_weeks": horizon_weeks,
        "plan": {"text": plan_text} if plan_text else None,
        "created_at": now_iso(),
    })
    parts = []
    if target_weight_kg:
        parts.append(f"vektmål {target_weight_kg} kg")
    if strength_targets:
        parts.append(f"styrke: {strength_targets}")
    if horizon_weeks:
        parts.append(f"{horizon_weeks} uker")
    logger.info(f"[save_goal] user={uid} v{version} {parts}")
    return json.dumps({
        "success": True,
        "message": f"Mål lagret: {', '.join(parts) or 'oppdatert'}. Synlig på Dashboard i Student Senteret.",
    })


# ═══════════════════════════════════════════════════════════════════
# PROFILE & MEMORY TOOLS
# ═══════════════════════════════════════════════════════════════════

@function_tool(timeout=10.0)
async def remember_fact(ctx: RunContextWrapper[CoachContext], key: str, value: str) -> str:
    """Remember a fact about the user permanently. Use for injuries, preferences, allergies, etc.

    Args:
        key: Category name (e.g. 'skader', 'allergier', 'matpreferanser').
        value: The fact to remember.
    """
    uid = ctx.context.user_id
    upsert_row(USER_CONTEXT, {
        "user_id": uid, "key": key, "value": value,
        "source": "agent", "updated_at": now_iso(),
    }, "user_id,key")
    logger.info(f"[remember_fact] user={uid} {key}={value[:50]}")
    return json.dumps({"success": True, "message": f"Lagret: {key} = {value}"})


@function_tool(timeout=10.0)
async def update_profile(
    ctx: RunContextWrapper[CoachContext],
    current_weight_kg: float | None = None,
    training_days_per_week: int | None = None,
    goals: str | None = None,
    injury_history: str | None = None,
    nutrition_preferences: str | None = None,
) -> str:
    """Update the user's profile data in the student center.

    Args:
        current_weight_kg: Current weight in kg.
        training_days_per_week: How many days per week the user trains.
        goals: Text description of goals.
        injury_history: Known injuries or limitations.
        nutrition_preferences: Dietary preferences or restrictions.
    """
    uid = ctx.context.user_id
    row: dict = {"user_id": uid, "updated_at": now_iso()}
    if current_weight_kg is not None:
        row["current_weight_kg"] = current_weight_kg
    if training_days_per_week is not None:
        row["training_days_per_week"] = training_days_per_week
    if goals is not None:
        row["goals"] = goals
    if injury_history is not None:
        row["injury_history"] = injury_history
    if nutrition_preferences is not None:
        row["nutrition_preferences"] = nutrition_preferences
    upsert_row(USER_PROFILES, row, "user_id")
    logger.info(f"[update_profile] user={uid} fields={list(row.keys())}")
    return json.dumps({"success": True, "message": "Profil oppdatert i Student Senteret."})


@function_tool(timeout=10.0)
async def get_user_stats(ctx: RunContextWrapper[CoachContext]) -> str:
    """Fetch the user's current stats: latest weight, goals, today's meals, and plan summaries.
    Call this when you need up-to-date info about the user."""
    uid = ctx.context.user_id
    stats: dict = {}

    rows = get_db().table(WEIGHT_ENTRIES).select("date, kg") \
        .eq("user_id", uid).order("date", desc=True).limit(5).execute().data or []
    if rows:
        stats["latest_weight"] = rows[0]
        stats["recent_weights"] = rows

    goal = find_one(GOALS, {"user_id": uid, "is_current": True})
    if goal:
        stats["current_goal"] = {
            k: goal.get(k) for k in ("target_weight_kg", "strength_targets", "horizon_weeks") if goal.get(k)
        }

    meals = find_many(MEAL_LOGS, {"user_id": uid, "date": today_str()})
    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    for m in meals:
        totals["calories"] += m.get("total_calories", 0) or 0
        totals["protein"] += m.get("total_protein_g", 0) or 0
        totals["carbs"] += m.get("total_carbs_g", 0) or 0
        totals["fat"] += m.get("total_fat_g", 0) or 0
    stats["today_meals"] = {"count": len(meals), "totals": totals}

    tp = get_db().table(TRAINING_PLAN_VERSIONS).select("version, days") \
        .eq("user_id", uid).order("version", desc=True).limit(1).maybe_single().execute()
    if tp.data:
        stats["training_plan"] = {"version": tp.data.get("version"), "days_count": len(tp.data.get("days", []))}

    np_res = get_db().table(NUTRITION_PLAN_VERSIONS).select("version, kcal, protein_grams, carbs_grams, fat_grams") \
        .eq("user_id", uid).order("version", desc=True).limit(1).maybe_single().execute()
    if np_res.data:
        stats["nutrition_plan"] = np_res.data

    ctx_rows = find_many(USER_CONTEXT, {"user_id": uid}, select="key, value")
    if ctx_rows:
        stats["user_context"] = {r["key"]: r["value"] for r in ctx_rows}

    profile = find_one(USER_PROFILES, {"user_id": uid})
    if profile:
        stats["profile"] = {
            k: v for k, v in profile.items()
            if k not in ("id", "created_at", "updated_at") and v is not None
        }

    return json.dumps(stats, default=str)
