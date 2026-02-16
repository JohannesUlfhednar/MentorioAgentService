"""All tool functions that sub-agents use to interact with the database.

Each function is a plain Python function that the OpenAI Agents SDK
automatically converts into an OpenAI function-calling tool.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Optional

from .db import (
    get_db, today_str, upsert_row, insert_row, find_one, find_many,
    WEIGHT_ENTRIES, MEAL_LOGS, WORKOUT_LOGS, TRAINING_PLAN_VERSIONS,
    NUTRITION_PLAN_VERSIONS, GOALS, USER_PROFILES, USER_CONTEXT,
    CHANGE_EVENTS, STUDENT_STATES,
)


# ═══════════════════════════════════════════════════════════════════
# WEIGHT & BODY TOOLS
# ═══════════════════════════════════════════════════════════════════

def log_weight(user_id: str, kg: float, date: Optional[str] = None) -> str:
    """Log the user's body weight. Call when the user mentions their weight."""
    d = date or today_str()
    if kg < 20 or kg > 500:
        return json.dumps({"success": False, "error": "Vekt må være mellom 20 og 500 kg"})
    upsert_row(WEIGHT_ENTRIES, {"user_id": user_id, "date": d, "kg": kg}, "user_id,date")
    insert_row(CHANGE_EVENTS, {
        "user_id": user_id, "type": "WEIGHT_LOG",
        "summary": f"Vekt logget: {kg} kg ({d})", "actor": "agent",
    })
    return json.dumps({"success": True, "message": f"Vekt logget: {kg} kg ({d})"})


def get_weight_history(user_id: str, days: int = 30) -> str:
    """Get the user's weight history for the given number of days."""
    start = (date.today() - timedelta(days=days)).isoformat()
    rows = get_db().table(WEIGHT_ENTRIES) \
        .select("date, kg") \
        .eq("user_id", user_id) \
        .gte("date", start) \
        .order("date", desc=False) \
        .execute().data or []
    return json.dumps({"entries": rows, "count": len(rows)})


# ═══════════════════════════════════════════════════════════════════
# MEAL / NUTRITION TOOLS
# ═══════════════════════════════════════════════════════════════════

def log_meal(
    user_id: str,
    description: str,
    meal_type: str = "other",
    calories: float = 0,
    protein_g: float = 0,
    carbs_g: float = 0,
    fat_g: float = 0,
    date: Optional[str] = None,
) -> str:
    """Log a meal the user has eaten with estimated macros."""
    d = date or today_str()
    row = insert_row(MEAL_LOGS, {
        "user_id": user_id, "date": d, "meal_type": meal_type,
        "description": description,
        "total_calories": calories, "total_protein_g": protein_g,
        "total_carbs_g": carbs_g, "total_fat_g": fat_g,
        "items": [],
    })
    return json.dumps({
        "success": True,
        "message": f"Måltid logget: {description} ({calories} kcal, {protein_g}g P, {carbs_g}g K, {fat_g}g F)",
    })


def get_today_meals(user_id: str) -> str:
    """Get all meals logged today with totals."""
    d = today_str()
    meals = find_many(MEAL_LOGS, {"user_id": user_id, "date": d}, order_by="created_at")
    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    for m in meals:
        totals["calories"] += m.get("total_calories", 0) or 0
        totals["protein"] += m.get("total_protein_g", 0) or 0
        totals["carbs"] += m.get("total_carbs_g", 0) or 0
        totals["fat"] += m.get("total_fat_g", 0) or 0
    return json.dumps({"meals": meals, "totals": totals, "count": len(meals)})


def save_nutrition_plan(
    user_id: str,
    kcal: int,
    protein_grams: int,
    carbs_grams: int,
    fat_grams: int,
    meals_json: str = "[]",
    notes: str = "",
    reason: str = "",
) -> str:
    """Save a complete nutrition/meal plan to the student center.
    Call AFTER the user has approved the plan.
    meals_json should be a JSON string array of meal objects."""
    try:
        meals = json.loads(meals_json)
    except Exception:
        meals = []

    # Get next version
    existing = get_db().table(NUTRITION_PLAN_VERSIONS) \
        .select("version").eq("user_id", user_id) \
        .order("version", desc=True).limit(1).maybe_single().execute()
    version = ((existing.data or {}).get("version", 0) or 0) + 1

    row = insert_row(NUTRITION_PLAN_VERSIONS, {
        "user_id": user_id, "version": version,
        "kcal": kcal, "protein_grams": protein_grams,
        "carbs_grams": carbs_grams, "fat_grams": fat_grams,
        "meals": meals, "notes": notes,
        "created_at": datetime.utcnow().isoformat(),
    })
    insert_row(CHANGE_EVENTS, {
        "user_id": user_id, "type": "NUTRITION_PLAN_SAVED",
        "summary": f"Kostholdsplan v{version}: {kcal} kcal, {protein_grams}g P",
        "actor": "agent", "after": {"version": version, "reason": reason},
    })
    return json.dumps({
        "success": True,
        "message": f"Kostholdsplan lagret (v{version}): {kcal} kcal, {protein_grams}g protein, {carbs_grams}g karbs, {fat_grams}g fett. Brukeren finner den i Ernæring-fanen.",
    })


# ═══════════════════════════════════════════════════════════════════
# WORKOUT TOOLS
# ═══════════════════════════════════════════════════════════════════

def log_workout(
    user_id: str,
    description: str = "",
    date: Optional[str] = None,
    entries_json: str = "[]",
) -> str:
    """Log a completed workout session."""
    d = date or today_str()
    try:
        entries = json.loads(entries_json)
    except Exception:
        entries = []
    upsert_row(WORKOUT_LOGS, {
        "user_id": user_id, "date": d, "entries": entries,
    }, "user_id,date")
    return json.dumps({"success": True, "message": f"Trening logget for {d}"})


def save_training_plan(
    user_id: str,
    days_json: str,
    reason: str = "",
) -> str:
    """Save a complete training plan to the student center.
    Call AFTER the user has approved the plan.
    days_json must be a JSON string: [{day, focus, exercises: [{name, sets, reps}]}]"""
    try:
        days = json.loads(days_json)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Ugyldig JSON for days: {e}"})

    if not days or not isinstance(days, list):
        return json.dumps({"success": False, "error": "days må være en liste med minst 1 dag"})

    # Get next version
    existing = get_db().table(TRAINING_PLAN_VERSIONS) \
        .select("version").eq("user_id", user_id) \
        .order("version", desc=True).limit(1).maybe_single().execute()
    version = ((existing.data or {}).get("version", 0) or 0) + 1

    row = insert_row(TRAINING_PLAN_VERSIONS, {
        "user_id": user_id, "version": version,
        "days": days,
        "created_at": datetime.utcnow().isoformat(),
    })
    insert_row(CHANGE_EVENTS, {
        "user_id": user_id, "type": "TRAINING_PLAN_SAVED",
        "summary": f"Treningsplan v{version}: {len(days)} dager",
        "actor": "agent", "after": {"version": version, "reason": reason},
    })
    return json.dumps({
        "success": True,
        "message": f"Treningsplan lagret (v{version}) med {len(days)} treningsdager. Brukeren finner den i Aktivitet-fanen i Student Senteret.",
    })


# ═══════════════════════════════════════════════════════════════════
# GOALS TOOLS
# ═══════════════════════════════════════════════════════════════════

def save_goal(
    user_id: str,
    target_weight_kg: Optional[float] = None,
    strength_targets: str = "",
    horizon_weeks: Optional[int] = None,
    plan_text: str = "",
) -> str:
    """Save a fitness goal to the student center dashboard."""
    # Mark old goals as not current
    get_db().table(GOALS).update({"is_current": False}).eq("user_id", user_id).execute()
    existing = get_db().table(GOALS) \
        .select("version").eq("user_id", user_id) \
        .order("version", desc=True).limit(1).maybe_single().execute()
    version = ((existing.data or {}).get("version", 0) or 0) + 1

    row = insert_row(GOALS, {
        "user_id": user_id, "version": version, "is_current": True,
        "target_weight_kg": target_weight_kg,
        "strength_targets": strength_targets,
        "horizon_weeks": horizon_weeks,
        "plan": {"text": plan_text} if plan_text else None,
        "created_at": datetime.utcnow().isoformat(),
    })
    parts = []
    if target_weight_kg:
        parts.append(f"vektmål {target_weight_kg} kg")
    if strength_targets:
        parts.append(f"styrke: {strength_targets}")
    if horizon_weeks:
        parts.append(f"{horizon_weeks} uker")
    return json.dumps({
        "success": True,
        "message": f"Mål lagret: {', '.join(parts) or 'oppdatert'}. Brukeren finner det på Dashboard i Student Senteret.",
    })


# ═══════════════════════════════════════════════════════════════════
# PROFILE & CONTEXT TOOLS
# ═══════════════════════════════════════════════════════════════════

def update_user_context(user_id: str, key: str, value: str) -> str:
    """Remember a fact about the user (injuries, preferences, etc.)."""
    upsert_row(USER_CONTEXT, {
        "user_id": user_id, "key": key, "value": value,
        "source": "agent", "updated_at": datetime.utcnow().isoformat(),
    }, "user_id,key")
    return json.dumps({"success": True, "message": f"Lagret: {key} = {value}"})


def update_profile(
    user_id: str,
    current_weight_kg: Optional[float] = None,
    training_days_per_week: Optional[int] = None,
    goals: Optional[str] = None,
    injury_history: Optional[str] = None,
    nutrition_preferences: Optional[str] = None,
) -> str:
    """Update the user's profile data in the student center."""
    row: dict = {"user_id": user_id, "updated_at": datetime.utcnow().isoformat()}
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
    return json.dumps({"success": True, "message": "Profil oppdatert"})


def get_user_stats(user_id: str) -> str:
    """Fetch the user's current stats: latest weight, goals, today's meals, training plan."""
    stats: dict = {}

    # Latest weight
    rows = get_db().table(WEIGHT_ENTRIES).select("date, kg") \
        .eq("user_id", user_id).order("date", desc=True).limit(5).execute().data or []
    if rows:
        stats["latest_weight"] = rows[0]
        stats["recent_weights"] = rows

    # Current goal
    goal = find_one(GOALS, {"user_id": user_id, "is_current": True})
    if goal:
        stats["current_goal"] = {
            "target_weight_kg": goal.get("target_weight_kg"),
            "strength_targets": goal.get("strength_targets"),
            "horizon_weeks": goal.get("horizon_weeks"),
        }

    # Today's meals
    meals = find_many(MEAL_LOGS, {"user_id": user_id, "date": today_str()})
    totals = {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}
    for m in meals:
        totals["calories"] += m.get("total_calories", 0) or 0
        totals["protein"] += m.get("total_protein_g", 0) or 0
        totals["carbs"] += m.get("total_carbs_g", 0) or 0
        totals["fat"] += m.get("total_fat_g", 0) or 0
    stats["today_meals"] = {"count": len(meals), "totals": totals}

    # Current training plan
    tp = get_db().table(TRAINING_PLAN_VERSIONS).select("version, days, created_at") \
        .eq("user_id", user_id).order("version", desc=True).limit(1).maybe_single().execute()
    if tp.data:
        stats["training_plan"] = {"version": tp.data.get("version"), "days_count": len(tp.data.get("days", []))}

    # Current nutrition plan
    np = get_db().table(NUTRITION_PLAN_VERSIONS).select("version, kcal, protein_grams, carbs_grams, fat_grams") \
        .eq("user_id", user_id).order("version", desc=True).limit(1).maybe_single().execute()
    if np.data:
        stats["nutrition_plan"] = np.data

    # User context
    ctx_rows = find_many(USER_CONTEXT, {"user_id": user_id}, select="key, value")
    if ctx_rows:
        stats["user_context"] = {r["key"]: r["value"] for r in ctx_rows}

    # Profile
    profile = find_one(USER_PROFILES, {"user_id": user_id})
    if profile:
        stats["profile"] = {
            k: v for k, v in profile.items()
            if k not in ("id", "created_at", "updated_at") and v is not None
        }

    return json.dumps(stats, default=str)
