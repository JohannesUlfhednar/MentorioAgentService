"""Supabase database helpers for agent tools."""

from __future__ import annotations

import os
import logging
from datetime import date, datetime
from typing import Any

from supabase import create_client, Client

logger = logging.getLogger("agent.db")

_client: Client | None = None


def get_db() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_SERVICE_KEY"]
        _client = create_client(url, key)
        logger.info("Supabase client initialized")
    return _client


def today_str() -> str:
    return date.today().isoformat()


def now_iso() -> str:
    return datetime.utcnow().isoformat()


# ── Generic helpers ─────────────────────────────────────────────────

def upsert_row(table: str, row: dict[str, Any], on_conflict: str = "user_id") -> dict | None:
    try:
        resp = get_db().table(table).upsert(row, on_conflict=on_conflict).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.error(f"upsert_row({table}) failed: {e}")
        return None


def insert_row(table: str, row: dict[str, Any]) -> dict | None:
    try:
        resp = get_db().table(table).insert(row).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.error(f"insert_row({table}) failed: {e}")
        return None


def find_one(table: str, filters: dict[str, Any], select: str = "*") -> dict | None:
    try:
        q = get_db().table(table).select(select)
        for k, v in filters.items():
            q = q.eq(k, v)
        resp = q.limit(1).maybe_single().execute()
        return resp.data
    except Exception as e:
        logger.error(f"find_one({table}) failed: {e}")
        return None


def find_many(table: str, filters: dict[str, Any], select: str = "*",
              order_by: str | None = None, ascending: bool = True,
              limit: int = 50) -> list[dict]:
    try:
        q = get_db().table(table).select(select)
        for k, v in filters.items():
            q = q.eq(k, v)
        if order_by:
            q = q.order(order_by, desc=not ascending)
        resp = q.limit(limit).execute()
        return resp.data or []
    except Exception as e:
        logger.error(f"find_many({table}) failed: {e}")
        return []


def next_version(table: str, user_id: str) -> int:
    """Get the next version number for a versioned table."""
    try:
        existing = get_db().table(table) \
            .select("version").eq("user_id", user_id) \
            .order("version", desc=True).limit(1).maybe_single().execute()
        return ((existing.data or {}).get("version", 0) or 0) + 1
    except Exception:
        return 1


# ── Table names ─────────────────────────────────────────────────────

WEIGHT_ENTRIES = "weight_entries"
MEAL_LOGS = "meal_logs"
WORKOUT_LOGS = "workout_logs"
TRAINING_PLAN_VERSIONS = "training_plan_versions"
NUTRITION_PLAN_VERSIONS = "nutrition_plan_versions"
GOALS = "goals"
USER_PROFILES = "user_profiles"
USER_CONTEXT = "user_context"
CHANGE_EVENTS = "change_events"
STUDENT_STATES = "student_states"
USERS = "users"
COACH_KNOWLEDGE = "coach_knowledge"
