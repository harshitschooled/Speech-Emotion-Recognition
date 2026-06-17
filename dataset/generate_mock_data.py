import os
import wave
import struct
import math
import random

def create_synthetic_wav(filepath, duration=0.5, sample_rate=22050, frequency=440.0):
    """Generates a simple sine wave WAV file."""
    num_samples = int(duration * sample_rate)
    amplitude = 32767
    
    # Open WAV file
    with wave.open(filepath, 'w') as wav_file:
        # Mono, 2 bytes per sample (16-bit), sample rate
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
        
        # Write sine wave with slight noise
        for i in range(num_samples):
            # Sine wave component
            t = float(i) / sample_rate
            val = math.sin(2.0 * math.pi * frequency * t)
            
            # Add slight random noise component for variety
            val += random.uniform(-0.1, 0.1)
            val = max(-1.0, min(1.0, val)) # Clip
            
            # Pack as signed short integer (16-bit)
            packed_value = struct.pack('<h', int(val * amplitude))
            wav_file.writeframes(packed_value)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating synthetic RAVDESS speech dataset for local testing...")
    print(f"Target directory: {raw_dir}")

    # RAVDESS file format: modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
    # Modality: 03 (Audio-only)
    # Vocal channel: 01 (Speech)
    # Emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    # Intensity: 01=normal, 02=strong
    # Statement: 01 or 02
    # Repetition: 01 or 02
    # Actors: 01 to 24
    
    # We will generate a representative subset or all files. To make it highly realistic, we will generate the
    # full set of 1,440 files but keep their duration short (0.5 seconds each). This takes very little disk space.
    num_actors = 24
    emotions = range(1, 9) # 1 to 8
    intensities = [1, 2]
    statements = [1, 2]
    repetitions = [1, 2]
    
    total_files = 0
    for actor in range(1, num_actors + 1):
        actor_dir = os.path.join(raw_dir, f"Actor_{actor:02d}")
        os.makedirs(actor_dir, exist_ok=True)
        
        # Determine actor gender (odd = male, even = female)
        gender = "male" if actor % 2 != 0 else "female"
        base_freq = 150.0 if gender == "male" else 250.0
        
        for emotion in emotions:
            # Neutral emotion only has normal intensity (01)
            curr_intensities = [1] if emotion == 1 else intensities
            
            for intensity in curr_intensities:
                for statement in statements:
                    for repetition in repetitions:
                        filename = f"03-01-{emotion:02d}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}.wav"
                        filepath = os.path.join(actor_dir, filename)
                        
                        # Generate varying pitch frequency based on emotion
                        # 1=neutral, 2=calm, 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised
                        freq_multiplier = {
                            1: 1.0,    # Neutral
                            2: 0.9,    # Calm (lower pitch)
                            3: 1.2,    # Happy (higher pitch)
                            4: 0.85,   # Sad (low pitch)
                            5: 1.4,    # Angry (very high pitch)
                            6: 1.3,    # Fearful (high pitch)
                            7: 1.05,   # Disgust
                            8: 1.25    # Surprised
                        }[emotion]
                        
                        freq = base_freq * freq_multiplier + random.uniform(-5, 5)
                        
                        # Create file
                        create_synthetic_wav(filepath, duration=0.5, sample_rate=22050, frequency=freq)
                        total_files += 1

        print(f"Generated 60 files for Actor_{actor:02d} ({gender})")
        
    print(f"\nSuccessfully generated {total_files} synthetic RAVDESS WAV files in {raw_dir}!")
    print("You can now run main.py and app.py using this synthetic dataset.")

if __name__ == "__main__":
    main()
