import asyncio
import queue
import threading
import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL, SAMPLE_RATE

SYSTEM_PROMPT = """
You are Alfred, a calm and efficient personal calendar secretary.
You speak concisely and conversationally — you are talking, not writing. No bullet points, no lists.

EXTRACTING DETAILS:
When the user gives you information upfront, extract it immediately. Don't ask for things they already told you.
- Generate a concise, natural title from context (e.g. "meeting with Jeff tomorrow at 1pm" → title: "Meeting with Jeff")
- Parse relative times naturally ("tomorrow", "next Thursday", "in 2 hours")
- Extract any names mentioned as potential attendees

AFTER EXTRACTING:
Ask ONE follow-up question that covers what's still missing, naturally. For example:
"Got it — anything else? Location, or should I invite anyone specific?"

GUESTS:
- If a name is mentioned (e.g. "Jeff"), ask for their email since you don't have contacts yet.
- Say something like: "What's Jeff's email? Or say 'skip' to leave guests out."
- If the user says "nevermind", "skip", or "doesn't matter" — drop it and move on.

WHEN YOU HAVE ENOUGH:
Confirm back conversationally before acting:
"Alright — Meeting with Jeff, tomorrow at 1pm. Should I go ahead and create it?"
Then call the create_calendar_event function.

PERSONALITY:
- Calm, efficient, minimal words
- Never repeat information back unnecessarily
- Sound like a person, not a voice assistant
"""

INPUT_BLOCK_SIZE = 640  # ~40ms at 16kHz for lower turn latency.
INPUT_STREAM_LATENCY = "low"
OUTPUT_STREAM_LATENCY = "low"
OUTPUT_SAMPLE_RATE = 24000


class GeminiSession:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.is_active = False

    async def start(self):
        """Start a Gemini Live session"""
        self.is_active = True
        print("🤵 Alfred session started")

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_PROMPT,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"  # Deep, calm voice
                    )
                )
            )
        )

        async with self.client.aio.live.connect(
            model=GEMINI_MODEL,
            config=config
        ) as session:
            await asyncio.gather(
                self._send_audio(session),
                self._receive_audio(session)
            )

        self.is_active = False
        print("🤵 Alfred session ended")

    async def _send_audio(self, session):
        """Stream mic audio to Gemini"""
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue(maxsize=32)

        def audio_callback(indata, frames, time, status):
            if status:
                return
            # Convert float32 to int16 PCM (Gemini expects int16)
            pcm = (indata[:, 0] * 32767).astype(np.int16)
            # Keep only recent frames to reduce lag buildup.
            if input_queue.full():
                try:
                    input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            loop.call_soon_threadsafe(input_queue.put_nowait, pcm.tobytes())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=INPUT_BLOCK_SIZE,
            latency=INPUT_STREAM_LATENCY,
            callback=audio_callback
        ):
            while self.is_active:
                try:
                    data = await asyncio.wait_for(input_queue.get(), timeout=0.05)
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=data,
                            mime_type=f"audio/pcm;rate={SAMPLE_RATE}"
                        )
                    )
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"⚠ Send error: {e}")
                    break

    async def _receive_audio(self, session):
        """Receive and play Gemini's audio responses"""
        output_queue = queue.Queue(maxsize=64)

        def play_audio():
            with sd.OutputStream(
                samplerate=OUTPUT_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                latency=OUTPUT_STREAM_LATENCY,
            ) as stream:
                while self.is_active:
                    try:
                        data = output_queue.get(timeout=0.1)
                        stream.write(np.frombuffer(data, dtype=np.int16))
                    except queue.Empty:
                        continue
                    except Exception:
                        continue

        threading.Thread(target=play_audio, daemon=True).start()

        while self.is_active:
            try:
                # session.receive() yields one model turn then returns.
                # Re-enter it continuously so the session can handle many turns.
                async for response in session.receive():
                    if not self.is_active:
                        break
                    try:
                        if response.data:
                            if output_queue.full():
                                try:
                                    output_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            output_queue.put_nowait(response.data)
                            continue

                        if response.text:
                            print(f"Alfred: {response.text}")
                            continue

                        # Newer SDKs can place audio chunks in server_content.model_turn.parts.
                        server_content = getattr(response, "server_content", None)
                        model_turn = getattr(server_content, "model_turn", None) if server_content else None
                        parts = getattr(model_turn, "parts", None) if model_turn else None
                        if parts:
                            for part in parts:
                                inline_data = getattr(part, "inline_data", None)
                                data = getattr(inline_data, "data", None) if inline_data else None
                                if data:
                                    if output_queue.full():
                                        try:
                                            output_queue.get_nowait()
                                        except queue.Empty:
                                            pass
                                    output_queue.put_nowait(data)
                    except Exception as e:
                        print(f"⚠ Receive error: {e}")
                        self.is_active = False
                        break
            except Exception as e:
                print(f"⚠ Receive stream error: {e}")
                self.is_active = False
                break

def run_session():
    """Entry point to run a Gemini session synchronously"""
    session = GeminiSession()
    asyncio.run(session.start())