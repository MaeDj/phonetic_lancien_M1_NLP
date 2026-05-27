
import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

FS = 22050  # Sample rate required by assignment
CHANNELS = 1

sentences = [
    "Lucile a mis sa belle jupe sur la chaise.",
    "La poule picore la pâte rouge.",
    "Le loup gris passe par la salle vide.",
    "Arthur a vu une puce sur sa tête.",
    "C'est une bête avec de grandes pattes.",
    "La loupe de cette dame est très utile.",
    "Il fume sa pipe toute la journée.",
    "La chute de la tarte sur la table.",
    "Lucas boit une soupe chaude.",
    "La fête se passe dans la rue de la Harpe."
]

def record_speaker(speaker_name):
    os.makedirs('audio', exist_ok=True)
    
    print(f"\n--- Recording speaker: {speaker_name} ---")
    import numpy as np

    for idx, s in enumerate(sentences):
        filename = f"audio/{speaker_name}_sentence_{idx+1:02d}.wav"
        print(f"\nSentence {idx+1} of {len(sentences)}:")
        print(f"--> {s}")
        
        input("Press Enter to START recording...")
        print("Recording... Press Enter to STOP.")
        
        # Record up to 60 seconds per sentence
        recording = sd.rec(int(60 * FS), samplerate=FS, channels=CHANNELS)
        
        input()  # Wait for Enter
        sd.stop()
        
        # Strip trailing silence completely
        nz = np.nonzero(recording)[0]
        if len(nz) > 0:
            actual_record = recording[:nz[-1]+1]
        else:
            actual_record = recording
            
        write(filename, FS, actual_record)
        print(f"Saved {filename}")

if __name__ == "__main__":
    speaker = input("Enter speaker name (e.g., S1_Male, S2_Female): ")
    record_speaker(speaker)
