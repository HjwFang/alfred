import time
from audio_loop import AudioListener
from config import APP_NAME


def _safe_print(unicode_text, ascii_text):
    try:
        print(unicode_text)
    except UnicodeEncodeError:
        print(ascii_text)


def on_wake_word():
    """Called when Alfred hears his name"""
    _safe_print("🤵 Alfred is listening...", "[WAKE] Alfred is listening...")
    # Phase 2: Gemini Live session starts here

def main():
    print(f"Starting {APP_NAME}...")
    
    listener = AudioListener(on_wake_word_callback=on_wake_word)
    listener.start()

    _safe_print(
        f"✓ {APP_NAME} is running. Say 'Hey Jarvis' to wake.",
        f"[OK] {APP_NAME} is running. Say 'Hey Jarvis' to wake.",
    )
    print("Press Ctrl+C to quit\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\nShutting down {APP_NAME}...")
        listener.stop()

if __name__ == "__main__":
    main()