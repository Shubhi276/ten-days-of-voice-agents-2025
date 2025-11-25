import logging
import json
import os

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
)
from livekit.plugins import murf, deepgram, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# -------------------------
# Load Course Content
# -------------------------
CONTENT_FILE = "shared-data/day4_tutor_content.json"
with open(CONTENT_FILE, "r") as f:
    COURSE_CONTENT = json.load(f)


def get_concept(concept_id):
    for c in COURSE_CONTENT:
        if c["id"] == concept_id:
            return c
    return None


# -------------------------
# Agent Class
# -------------------------
class TutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are an Active Recall Coach. "
                "The user will choose a mode: learn, quiz, or teach_back. "
                "Use the JSON course content when responding. "
                "Do NOT use special formatting like emojis or lists. "
            )
        )
        self.mode = None
        self.current_concept = "variables"  # default concept

    async def on_user_message(self, msg, ctx):
        text = msg.text.lower()

        # MODE SWITCHING
        if "learn" in text:
            self.mode = "learn"
            await ctx.send_message("Learn mode activated. Tell me a concept name to begin.")
            return

        if "quiz" in text:
            self.mode = "quiz"
            await ctx.send_message("Quiz mode activated. Which concept should I quiz you on.")
            return

        if "teach back" in text or "teachback" in text:
            self.mode = "teach_back"
            await ctx.send_message("Teach back mode activated. Which concept will you explain.")
            return

        # SELECT CONCEPT
        for c in COURSE_CONTENT:
            if c["id"] in text or c["title"].lower() in text:
                self.current_concept = c["id"]
                await ctx.send_message(f"Concept set to {c['title']}.")
                break

        concept = get_concept(self.current_concept)

        # -------------------------
        # LEARN MODE
        # -------------------------
        if self.mode == "learn":
            await ctx.send_message(f"Here is the explanation. {concept['summary']}")
            return

        # -------------------------
        # QUIZ MODE
        # -------------------------
        if self.mode == "quiz":
            await ctx.send_message(f"Here is your question. {concept['sample_question']}")
            return

        # -------------------------
        # TEACH BACK MODE
        # -------------------------
        if self.mode == "teach_back":
            # Give qualitative feedback to user's explanation
            user_answer = msg.text.strip()
            if len(user_answer) < 10:
                await ctx.send_message("Try giving a little more detailed explanation.")
                return

            await ctx.send_message("Thanks for explaining. You covered this concept reasonably well. Keep improving.")
            return

        # Default fallback
        await ctx.send_message(
            "Welcome to Teach the Tutor. Choose learn, quiz, or teach back to begin."
        )


# -------------------------
# Prewarm
# -------------------------
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


# -------------------------
# Entry Point
# -------------------------
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",  # Default for learn mode
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    tutor = TutorAgent()

    # Select different Murf voices based on mode
    @session.on("assistant_response")
    def _voice_switch(ev):
        mode = tutor.mode
        if mode == "learn":
            session.tts.voice = "en-US-matthew"
        elif mode == "quiz":
            session.tts.voice = "en-US-alicia"
        elif mode == "teach_back":
            session.tts.voice = "en-US-ken"

    await session.start(
        agent=tutor,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
