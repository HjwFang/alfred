from pathlib import Path

# App
APP_NAME = "Alfred"
DATA_DIR = Path.home() / ".alfred"
DATA_DIR.mkdir(exist_ok=True)

# Wake word
WAKE_WORD = "hey_jarvis"
WAKE_THRESHOLD = 0.35
WAKE_DEBUG_INTERVAL_SEC = 1.0

# Audio
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # ~80ms at 16kHz
INPUT_DEVICE_INDEX = None  # Set to an integer from sounddevice query, e.g. 15