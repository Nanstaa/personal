# Voice cloning play
# Ran on WSL:Ubuntu
# pip install f5-tts
# python3 test_clone.py


from f5_tts.api import F5TTS

# 1. Initialize the model (it will auto-download weights the first time)
tts = F5TTS()

# 2. Define your voice reference and the text you want to say
ref_audio = "nt_ref_shorter.wav"  # Path to your 15s voice sample
ref_text = "The quick brown fox jumps over the lazy dog. My voice is unique, and I'm recording this on my Surface to calibrate a local AI model." # What you said in ref.wav
#gen_text = "Hello! This is a voice clone of me.  Scary isn't it?  Yes, I'm freaking out too."
gen_text = "Processors act like your devices brain, telling other components what to do. The higher the number of cores, the more power you’ll have for processing tasks."

# 3. Generate the cloned speech
print("Cloning voice... this may take a minute on CPU.")
wav, sr, spect = tts.infer(
    ref_file=ref_audio,
    ref_text=ref_text,
    gen_text=gen_text,
    speed=0.9,
    cfg_strength=2.0,   # High CFG helps keep your voice profile stable
    sway_sampling_coef=-1.0 
)

# 4. Save the output
import soundfile as sf
sf.write("output.wav", wav, sr)
print("Done! Check 'output.wav' in your folder.")
