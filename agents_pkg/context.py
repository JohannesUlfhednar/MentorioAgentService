"""Shared context for the Coach Majen agent system.

The context is created once per chat request and passed to all agents/tools.
It contains the user's identity and preloaded profile data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoachContext:
    """Dependency-injection context available to every agent and tool."""

    user_id: str
    mentor_id: str

    # Preloaded user data (injected into dynamic instructions)
    user_name: str = ""
    onboarding_summary: str = ""
    user_context_summary: str = ""
    mentor_name: str = "Coach Majen"
    mentor_voice_tone: str = ""
    mentor_training_philosophy: str = ""
    mentor_nutrition_philosophy: str = ""
    mentor_core_instructions: str = ""
