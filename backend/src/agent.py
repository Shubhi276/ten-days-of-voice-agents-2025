import json
import logging

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
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")
# Paths for FAQ and Lead Storage
FAQ_PATH = "shared-data/day5_zomato_faq.json"
LEAD_PATH = "./lead_output.json"


def load_faq():
    try:
        with open(FAQ_PATH, "r") as f:
            return json.load(f)
    except:
        return {"faq": []}


faq_data = load_faq()


def match_faq(user_text: str):
    text = user_text.lower()
    for item in faq_data.get("faq", []):
        q_text = item["question"].lower()
        keywords = q_text.split()

        if any(k in text for k in keywords):
            return item["answer"]

    return None

lead_template = {
    "name": None,
    "company": None,
    "email": None,
    "role": None,
    "use_case": None,
    "team_size": None,
    "timeline": None,
}

lead_questions = [
    ("name", "May I have your name please?"),
    ("company", "Which company or restaurant do you represent?"),
    ("email", "What email can we contact you at?"),
    ("role", "What is your role in your company?"),
    ("use_case", "How do you plan to use Zomato’s services?"),
    ("team_size", "What is the size of your team?"),
    ("timeline", "When are you planning to get started? (Now / Soon / Later)"),
]

class ZomatoSDRAgent(Agent):
    def __init__(self):
        # 1. Convert the loaded JSON data into a text string the AI can read
        faq_text = json.dumps(faq_data, indent=2)
        
        super().__init__(
            instructions=f"""
            You are a Sales Development Representative for Zomato.
            
            HERE IS YOUR KNOWLEDGE BASE (FAQ):
            {faq_text}
            
            RULES:
            1. Answer questions using ONLY the Knowledge Base above.
            2. Speak in a friendly, helpful tone.
            3. If you don't know an answer based on the text above, say you will connect them to the team.
            4. Collect lead details naturally (Name, Email, Role).
            """
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


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
    agent = ZomatoSDRAgent()
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

    leads = lead_template.copy()
    ask_idx = 0

    await agent.llm_response(session, "Welcome to Zomato! How can I assist you today?")

    async for user_msg in session.iter_user_messages():
        text = user_msg.text
        lower = text.lower()

       
        if any(k in lower for k in ["thanks", "that's all", "ok bye", "done", "thank you"]):
            summary = ", ".join(
                f"{k}: {v}" for k, v in leads.items() if v
            )
            await agent.llm_response(session, f"Thank you. Here is your summary: {summary}")

            with open(LEAD_PATH, "w") as f:
                json.dump(leads, f, indent=2)

            await agent.llm_response(session, "A Zomato representative will reach out shortly. Have a great day!")
            break

        faq_answer = match_faq(text)
        if faq_answer:
            
            if ask_idx > 0:
                prev_key = lead_questions[ask_idx - 1][0]
                if leads[prev_key] is None:
                    leads[prev_key] = text

            await agent.llm_response(session, faq_answer)

        
            if ask_idx < len(lead_questions):
                key, question = lead_questions[ask_idx]
                if leads[key] is None:
                    await agent.llm_response(session, question)
                    ask_idx += 1

            continue

        if ask_idx > 0:
            prev_key = lead_questions[ask_idx - 1][0]
            if leads[prev_key] is None:
                leads[prev_key] = text

        if ask_idx < len(lead_questions):
            key, question = lead_questions[ask_idx]
            if leads[key] is None:
                await agent.llm_response(session, question)
                ask_idx += 1
            continue

        
        await agent.respond(session, text)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
