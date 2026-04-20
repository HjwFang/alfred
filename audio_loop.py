import threading
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from config import (
    CHUNK_SIZE,
    INPUT_DEVICE_INDEX,
    SAMPLE_RATE,
    WAKE_THRESHOLD,
    WAKE_WORD,
)


def _safe_print(unicode_text, ascii_text):
    try:
        print(unicode_text)
    except UnicodeEncodeError:
        print(ascii_text)


class AudioListener:
    def __init__(self, on_wake_word_callback):
        """
        on_wake_word_callback: function to call when wake word fires
        """
        self.on_wake_word_callback = on_wake_word_callback
        self.model = None
        self.is_running = False
        self.thread = None
        self._prediction_key = None
        self._last_wake_time = 0.0
        self._silence_started_at = None
        self._silence_warned = False

    def start(self):
        """Start listening in background thread"""
        if self.is_running:
            return

        try:
            self.model = Model(wakeword_models=[WAKE_WORD])
            _safe_print(
                f"✓ Wake word model loaded: {WAKE_WORD}",
                f"[OK] Wake word model loaded: {WAKE_WORD}",
            )
        except Exception as e:
            _safe_print(f"✗ Failed to load model: {e}", f"[ERROR] Failed to load model: {e}")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        _safe_print("✓ Listening started", "[OK] Listening started")

    def stop(self):
        """Stop listening"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        _safe_print("✓ Listening stopped", "[OK] Listening stopped")

    def _listen_loop(self):
        """Main listening loop (background thread)"""

        def audio_callback(indata, frames, time_info, status):
            if status:
                _safe_print(f"⚠ Audio status: {status}", f"[WARN] Audio status: {status}")
                return

            try:
                # openwakeword expects 16-bit PCM samples at 16 kHz.
                mic_float = np.clip(indata[:, 0], -1.0, 1.0)
                audio_data = (mic_float * 32767).astype(np.int16)
                predictions = self.model.predict(audio_data)

                if self._prediction_key is None:
                    if WAKE_WORD in predictions:
                        self._prediction_key = WAKE_WORD
                    else:
                        target = WAKE_WORD.replace("_", "").lower()
                        for key in predictions:
                            normalized = key.replace("_", "").replace("-", "").lower()
                            if target in normalized:
                                self._prediction_key = key
                                break

                    if self._prediction_key:
                        _safe_print(
                            f"✓ Listening for wake key: {self._prediction_key}",
                            f"[OK] Listening for wake key: {self._prediction_key}",
                        )
                    else:
                        _safe_print(
                            f"⚠ Could not map wake word '{WAKE_WORD}' to model outputs: {list(predictions.keys())}",
                            f"[WARN] Could not map wake word '{WAKE_WORD}' to model outputs: {list(predictions.keys())}",
                        )
                        return

                score = float(predictions.get(self._prediction_key, 0.0))
                peak = float(np.max(np.abs(mic_float)))
                now = time.monotonic()

                if peak < 0.01:
                    if self._silence_started_at is None:
                        self._silence_started_at = now
                    elif (now - self._silence_started_at) > 5.0 and not self._silence_warned:
                        self._silence_warned = True
                        _safe_print(
                            "⚠ Mic level has stayed very low for 5s. Check selected input device and mic gain.",
                            "[WARN] Mic level has stayed very low for 5s. Check selected input device and mic gain.",
                        )
                else:
                    self._silence_started_at = None
                    self._silence_warned = False

                if score >= WAKE_THRESHOLD and (now - self._last_wake_time) > 1.0:
                    self._last_wake_time = now
                    _safe_print(
                        f"🎤 ALFRED DETECTED (score={score:.2f})",
                        f"[WAKE] ALFRED DETECTED (score={score:.2f})",
                    )
                    self.on_wake_word_callback()
            except Exception as e:
                # Keep stream alive even if a callback chunk fails.
                _safe_print(
                    f"⚠ Wake-word callback error: {e}",
                    f"[WARN] Wake-word callback error: {e}",
                )

        try:
            if INPUT_DEVICE_INDEX is not None:
                input_idx = INPUT_DEVICE_INDEX
            else:
                raw_default_device = sd.default.device
                try:
                    input_idx = raw_default_device[0]
                except (TypeError, IndexError):
                    input_idx = raw_default_device
            input_name = sd.query_devices(input_idx)["name"] if input_idx is not None else None
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                device=input_idx,
                callback=audio_callback,
            ):
                _safe_print(
                    f"🎙️ Input device: {input_name or 'default'} (index={input_idx})",
                    f"[INFO] Input device: {input_name or 'default'} (index={input_idx})",
                )
                _safe_print("✓ Mic stream opened", "[OK] Mic stream opened")
                while self.is_running:
                    time.sleep(0.05)
        except Exception as e:
            _safe_print(
                f"✗ Failed to open mic or stream error: {e}",
                f"[ERROR] Failed to open mic or stream error: {e}",
            )
        finally:
            self.is_running = False
