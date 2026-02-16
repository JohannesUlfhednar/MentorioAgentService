# MentorOS Agent Service

Multi-agent coaching system powered by the **OpenAI Agents SDK**.

## Architecture

```
Coach Majen (master agent)
├── Body Tracking Agent    — log weight, body measurements
├── Nutrition Agent        — log meals, save nutrition plans
├── Training Plan Agent    — save training plans
├── Workout Logger Agent   — log completed workouts
├── Goals Agent            — set and track fitness goals
└── Profile Agent          — update user profile and context
```

Coach Majen is the personality agent that handles all conversation. When actions need to happen (logging weight, saving a plan, etc.), Coach Majen delegates to the appropriate sub-agent via **handoffs**.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in:

- `OPENAI_API_KEY` — Your OpenAI API key
- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_SERVICE_KEY` — Supabase service role key

## Running

```bash
python main.py
```

Server starts on `http://localhost:8100`.

## API

### POST /chat

```json
{
  "user_id": "uuid",
  "mentor_id": "uuid",
  "message": "Hei coach!",
  "conversation_history": [
    {"role": "user", "content": "previous message"},
    {"role": "assistant", "content": "previous response"}
  ]
}
```

Response:
```json
{
  "response": "Hei! Velkommen...",
  "agent_name": "Coach Majen",
  "tools_called": ["log_weight"]
}
```

### GET /health

Returns `{"status": "ok"}`.

## Deployment (Railway)

1. Create a new Railway service
2. Connect to the MentorOSAgentService repo
3. Set environment variables
4. Add `AGENT_SERVICE_URL` to the Node.js backend pointing to this service
