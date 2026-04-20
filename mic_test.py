# test_mic.py
import sounddevice as sd
import numpy as np  

def test_mic():
    print("Listening for 5 seconds... speak into your mic")
    
    recording = sd.rec(
        int(5 * 16000),
        samplerate=16000,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    
    peak = np.max(np.abs(recording))
    print(f"Peak amplitude: {peak:.4f}")
    
    if peak < 0.01:
        print("✗ Mic is too quiet or not picking up audio")
    else:
        print("✓ Mic is working")

test_mic()