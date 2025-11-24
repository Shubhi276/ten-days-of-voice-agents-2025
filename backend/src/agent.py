import logging
import json
import os
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    llm,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("health-agent")
load_dotenv(".env.local")

# --- CUSTOM CONFIGURATION ---
JOURNAL_FILE = "my_health_journal.json" # Unique filename

def read_journal_history():
    """Reads the last session to give context."""
    if not os.path.exists(JOURNAL_FILE):
        return "No previous journal entries found. Treat this as a first meeting."
    
    try:
        with open(JOURNAL_FILE, "r") as f:
            data = json.load(f)
            if data and len(data) > 0:
                last = data[-1]
                return f"PREVIOUS CONTEXT: On {last['date']}, User felt '{last['mood']}'. Goal was: '{last['focus']}'."
    except Exception:
        return "Error reading previous journal."
    return "Journal is empty."

class HealthBuddy(Agent):
    def __init__(self, user_context: str) -> None:
        super().__init__(
            instructions=f"""
            You are 'HealthBuddy', a warm and practical wellness assistant.
            
            CONTEXT FROM LAST TIME:
            {user_context}
            
            YOUR CONVERSATION FLOW:
            1. Check In: Ask how they are feeling physically and mentally today.
            2. Focus: Ask for ONE main focus or goal for today.
            3. Advice: Give one short, grounded tip (drink water, stretch, etc.).
            4. Save: Summarize their day and call the 'save_journal_entry' tool.
            
            Be concise. Don't sound robotic.
            """,
        )

    @function_tool
    def save_journal_entry(
        self,
        ctx: RunContext,
        mood_summary: Annotated[str, "Summary of user's mood"],
        main_focus: Annotated[str, "User's main goal/focus"],
        tip_given: Annotated[str, "The advice you gave"],
    ):
        """Saves the conversation to the JSON journal."""
        print(f"\n--- 💾 SAVING TO JOURNAL: {mood_summary} ---")
        
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mood": mood_summary, 
            "focus": main_focus, 
            "tip": tip_given
        }
        
        # Load existing data safely
        data = []
        if os.path.exists(JOURNAL_FILE):
            try:
                with open(JOURNAL_FILE, "r") as f:
                    content = f.read().strip()
                    if content: data = json.loads(content)
            except:
                data = [] # Reset if corrupt

        data.append(entry)

        # Write to file
        try:
            with open(JOURNAL_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print("--- ✅ SUCCESS: Written to my_health_journal.json ---")
            return "I've saved that to your journal. Have a great day!"
        except Exception as e:
            print(f"ERROR WRITING FILE: {e}")
            return "I had trouble writing to the file."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # 1. Load Context
    context_str = read_journal_history()
    print(f"🧠 Loaded Context: {context_str}")

    ctx.log_context_fields = {"room": ctx.room.name}
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"), # Using 2.5 as you requested
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=HealthBuddy(user_context=context_str),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))