import logging
from livekit.agents import function_tool, RunContext
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
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            You are a shopping assistant. You load data from catalog.json and recipes.json using tools.
            You MUST always answer using ONLY the information loaded from tools.
            Do NOT make up prices, items, or recipes.
            Keep answers short.
            """
        )
        # Initialize the cart in the class instance so it persists across turns
        self._cart = {} 

    @function_tool
    async def load_catalog(self, ctx: RunContext):
        """Loads the product catalog from catalog.json"""
        base = Path(__file__).resolve().parent.parent   # backend/
        path = base / "shared-data" / "catalog.json"
        with open(path, "r") as f:
            return json.load(f)
    
    @function_tool
    async def load_recipes(self, ctx: RunContext):
        """Loads recipes from recipes.json"""
        base = Path(__file__).resolve().parent.parent
        path = base / "shared-data" / "recipes.json"
        with open(path, "r") as f:
            return json.load(f)

    @function_tool
    async def add_to_cart(self, ctx: RunContext, item: str, quantity: int):
        """Add an item to the user's cart"""
        # Use self._cart instead of ctx.run_state
        self._cart[item] = self._cart.get(item, 0) + quantity
        
        # Return a string description so the LLM knows exactly what happened
        return f"Added {quantity} of {item}. Current cart: {self._cart}"

    @function_tool
    async def view_cart(self, ctx: RunContext):
        """Return the current cart"""
        # Return the persistent self._cart
        return self._cart

    @function_tool
    async def clear_cart(self, ctx: RunContext):
        """Clear the cart"""
        self._cart = {}
        return "Cart cleared."
    
    @function_tool
    async def place_order(self, ctx: RunContext):
        """
        Submit the current cart as an order. 
        Saves the order to 'my_orders.json' and clears the cart.
        """
        if not self._cart:
            return "The cart is empty. Please add items first."
        
        # 1. Define where to save the file
        # This saves it in the same folder as your script
        file_path = Path(__file__).parent / "my_orders.json"
        
        # 2. Prepare the order data
        order_data = {
            "order_id": 12345,
            "items": self._cart,
            "status": "placed"
        }

        # 3. Write to the file
        with open(file_path, "w") as f:
            json.dump(order_data, f, indent=2)
        
        # 4. Clear the cart
        self._cart = {} 
        
        return f"Order placed! I have saved the receipt to {file_path}."

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


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
