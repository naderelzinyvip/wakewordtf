from pydub import AudioSegment
import os


# Function to change playback speed
def change_playback_speed(audio_path, speed=1.6):
    audio = AudioSegment.from_file(audio_path)
    slowed_audio = audio.speedup(playback_speed=speed)
    return slowed_audio

# Directory containing WAV files
input_directory = 'dataset/positve data/slowed'

# Output directory for slowed down audio files
output_directory = 'dataset/positve data/slowed_final'

# Ensure the output directory exists, create it if necessary
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over WAV files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        
        # Change the playback speed and save the modified audio
        slowed_audio = change_playback_speed(input_path)
        slowed_audio.export(output_path, format="wav")

print("Playback speed changed and files saved in", output_directory)
