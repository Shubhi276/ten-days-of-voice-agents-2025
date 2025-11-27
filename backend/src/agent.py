import logging
import json
import os
from pathlib import Path

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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

BASE_DIR = Path(__file__).resolve().parent.parent
FRAUD_DB_PATH = BASE_DIR / "shared-data" / "fraud_cases.json"


class Assistant(Agent):
    def __init__(self) -> None:
        # Update instructions to teach the LLM the fraud flow and when to call the tools.
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. The user is interacting via voice.\n\n"
                "Special mode: Fraud Alert Flow\n"
                "If the user requests a fraud alert (phrases like 'start fraud alert', 'fraud alert for <name>', "
                "'check suspicious transaction', etc.), run the following safe, step-by-step flow by calling the provided tools:\n\n"
                "1) Call load_fraud_case(username) to load the case for the provided username. If no case is found, tell the user politely.\n"
                "2) Ask the user the stored security question (the tool returns it as part of the loaded case). Expect a spoken short answer.\n"
                "3) Call verify_security(answer) with the user's answer. If verification fails, call update_case(status='verification_failed', note=...) and tell the user you cannot proceed.\n"
                "4) If verification succeeds, read the suspicious transaction details (merchant, amount, masked card, time, location).\n"
                "5) Ask the user: 'Did you make this transaction? (yes or no)'. If user replies yes -> call update_case(status='confirmed_safe', note='Customer confirmed transaction as legitimate.'). If user replies no -> call update_case(status='confirmed_fraud', note='Customer denied the transaction. Card will be blocked and dispute raised (mock).').\n"
                "6) Always use calm, professional language. Never ask for card numbers, PINs, passwords, or other sensitive info. Use only the non-sensitive data in the case.\n\n"
                "When the user does not ask for fraud flow, behave as the normal assistant."
                "Provide the best possible answer to the user's request.\n\n"
                "IMPORTANT: For any security, fraud alert, or account-verification request, "
                "you MUST use the fraud tools (load_fraud_case, verify_security, update_case). "
                "NEVER create your own questions; always return the question from load_fraud_case."

            )
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

@function_tool
async def load_fraud_case(context: RunContext, username: str) -> dict:
    """
    Load the fraud case for the given username.
    Stores the case in context.run_state['fraud_case'] for later steps.
    Returns a small summary or an error message (as dict).
    """
    try:
        db_path = FRAUD_DB_PATH
        if not db_path.exists():
            return {"error": True, "message": f"Fraud database not found at {db_path}"}

        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("cases", [])
        for case in cases:
            if case.get("userName", "").lower() == username.lower():
                # store for the rest of the run
                context.run_state["fraud_case"] = case
                # Return the security question and a brief summary of transaction (no sensitive data)
                return {
                    "error": False,
                    "userName": case.get("userName"),
                    "securityQuestion": case.get("securityQuestion"),
                    "merchant": case.get("merchant"),
                    "transactionAmount": case.get("transactionAmount"),
                    "maskedCard": case.get("maskedCard"),
                    "timestamp": case.get("timestamp"),
                    "location": case.get("location"),
                    "status": case.get("status"),
                }

        return {"error": True, "message": f"No fraud case found for username '{username}'."}
    except Exception as e:
        logger.exception("Error loading fraud case")
        return {"error": True, "message": f"Error loading fraud case: {str(e)}"}


@function_tool
async def verify_security(context: RunContext, answer: str) -> dict:
    """
    Verify the user's answer to the stored security question.
    Returns { "verified": True/False, "expected": "<masked>" }.
    """
    try:
        case = context.run_state.get("fraud_case")
        if not case:
            return {"verified": False, "message": "No fraud case loaded in this session."}

        expected = case.get("securityAnswer", "")
        # Basic case-insensitive compare; in real app you'd use more robust checking
        verified = (str(answer).strip().lower() == str(expected).strip().lower())

        return {"verified": verified, "expected_masked": "[hidden]"}
    except Exception as e:
        logger.exception("Error verifying security")
        return {"verified": False, "message": f"Verification error: {str(e)}"}


@function_tool
async def update_case(context: RunContext, status: str, note: str) -> dict:
    """
    Update the fraud case status and write back to JSON file.
    status: e.g. 'confirmed_safe', 'confirmed_fraud', 'verification_failed'
    note: short outcome note
    """
    try:
        db_path = FRAUD_DB_PATH
        if not db_path.exists():
            return {"error": True, "message": f"Fraud DB not found at {db_path}"}

        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("cases", [])
        case_in_state = context.run_state.get("fraud_case")
        if not case_in_state:
            return {"error": True, "message": "No fraud case in run state to update."}

        # Find case by securityIdentifier or username
        updated = False
        for c in cases:
            if c.get("securityIdentifier") == case_in_state.get("securityIdentifier") or \
               c.get("userName", "").lower() == case_in_state.get("userName", "").lower():
                c["status"] = status
                c["outcomeNote"] = note
                updated = True
                # also sync run_state
                context.run_state["fraud_case"] = c
                break

        if not updated:
            # If not found by id, attempt to match by username
            for c in cases:
                if c.get("userName", "").lower() == case_in_state.get("userName", "").lower():
                    c["status"] = status
                    c["outcomeNote"] = note
                    updated = True
                    context.run_state["fraud_case"] = c
                    break

        if not updated:
            return {"error": True, "message": "Could not find matching fraud case to update."}

        # Write back
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Updated fraud case for {case_in_state.get('userName')} -> {status}: {note}")
        return {"error": False, "message": "Case updated", "status": status}
    except Exception as e:
        logger.exception("Error updating fraud case")
        return {"error": True, "message": f"Could not update case: {str(e)}"}


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
