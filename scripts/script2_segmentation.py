import os
import soundfile as sf

def cut_wav_files(input_folder, output_folder, duration_cut_seconds):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all WAV files in the input folder
    wav_files = [file for file in os.listdir(input_folder) if file.endswith('.wav')]

    for wav_file in wav_files:
        # Load the WAV file
        file_path = os.path.join(input_folder, wav_file)
        audio, sample_rate = sf.read(file_path)

        # Calculate the number of samples for the desired duration
        duration_cut_samples = int(duration_cut_seconds * sample_rate)

        # Calculate the number of segments
        num_segments = len(audio) // duration_cut_samples

        # Cut the audio into segments
        for i in range(num_segments):
            start_sample = i * duration_cut_samples
            end_sample = (i + 1) * duration_cut_samples

            # Extract the segment
            segment = audio[start_sample:end_sample]

            # Create a subfolder with the name of the WAV file
            subfolder = os.path.join(output_folder, os.path.splitext(wav_file)[0])
            os.makedirs(subfolder, exist_ok=True)

            # Save the segment into the subfolder
            sf.write(os.path.join(subfolder, f"{wav_file}_{i + 1:05d}.wav"), segment, sample_rate)

if __name__ == "__main__":
    # Set the input folder, output folder, and duration for cutting (in seconds)
    input_folder = "wav"
    output_folder = "output"
    duration_cut_seconds = 2  # Duration in seconds (adjust as needed)

    # Call the function to cut and save the WAV files
    cut_wav_files(input_folder, output_folder, duration_cut_seconds)
