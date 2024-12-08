import gradio as gr
from DeepDub.PreProcessing import Preprocessing
import os
import yaml
import tempfile
import json
import soundfile as sf
import zipfile
from DeepDub.logger import logger

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define accessible temporary directory for outputs
temp_output_dir = "/tmp/deepdub_outputs"
os.makedirs(temp_output_dir, exist_ok=True)

# Shared preprocessor instance
class VideoProcessorManager:
    def __init__(self):
        self.preprocessor = None
        self.step_results = {}

    def initialize(self, input_video):
        """
        Initialize the Preprocessing instance for the uploaded video.
        """
        video_path = os.path.abspath(input_video)
        output_dir = os.path.join(temp_output_dir, "separation_output")
        os.makedirs(output_dir, exist_ok=True)
        self.preprocessor = Preprocessing(
            input_video=video_path,
            output_dir=output_dir,
            audio_separator_model=config['audio_separator_model'],
            diarization_batch_size=config['diarization_batch_size'],
            device=config['device'],
            compute_type=config['compute_type'],
            HF_token=config['HF_token']
        )
        self.step_results = {}  # Reset results for a new video

manager = VideoProcessorManager()
def display_json_content(json_path, zip_path):
    try:
        # Read the JSON file
        with open(json_path, 'r') as file:
            json_content = json.load(file)
        return json.dumps(json_content, indent=4), zip_path
    except Exception as e:
        logger.error(f"Error reading JSON file: {e}")
        return f"Error: {e}", None

# Utility function to log and update results
def update_step_results(step, **results):
    manager.step_results[step] = results
    for key, value in results.items():
        logger.info(f"{step} result - {key}: {value}")

# Step 1: Split Audio and Video
def split_audio_video(input_video):
    try:
        if not manager.preprocessor or manager.preprocessor.input_video != os.path.abspath(input_video):
            manager.initialize(input_video)
        extracted_audio_path, video_no_audio_path = manager.preprocessor.split_audio_and_video()
        update_step_results("split", extracted_audio_path=extracted_audio_path, video_no_audio_path=video_no_audio_path)
        return video_no_audio_path, extracted_audio_path
    except Exception as e:
        logger.error(f"Error splitting audio and video: {e}")
        return f"Error: {e}", None

# Step 2: Separate Audio
def separate_audio(input_audio):
    try:
        if not input_audio:
            raise ValueError("No audio provided for separation.")
        
        # Convert Gradio audio tuple (sampling rate, numpy array) to a file path
        if isinstance(input_audio, tuple):
            sampling_rate, audio_data = input_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_output_dir) as temp_audio_file:
                sf.write(temp_audio_file.name, audio_data, sampling_rate)
                input_audio = temp_audio_file.name

        manager.preprocessor.extracted_audio_path = input_audio
        vocals_path, background_path = manager.preprocessor.separate_audio()

        # Generate and save spectrograms
        vocals_spectrogram_path = os.path.join(manager.preprocessor.base_output_dir, "vocals_spectrogram.png")
        background_spectrogram_path = os.path.join(manager.preprocessor.base_output_dir, "background_spectrogram.png")
        
        # Ensure spectrograms are generated and saved
        separator = manager.preprocessor.audio_separator
        separator.save_spectrogram(vocals_path, vocals_spectrogram_path)
        separator.save_spectrogram(background_path, background_spectrogram_path)

        update_step_results("separate", vocals_path=vocals_path, background_path=background_path,
                            vocals_spectrogram_path=vocals_spectrogram_path,
                            background_spectrogram_path=background_spectrogram_path)
        return vocals_path, background_path, vocals_spectrogram_path, background_spectrogram_path
    except Exception as e:
        logger.error(f"Error separating audio: {e}")
        return f"Error: {e}", None, None, None

# Step 3: Perform Diarization
def perform_diarization(input_audio):
    try:
        if not input_audio:
            raise ValueError("No vocals audio provided for diarization.")

        if isinstance(input_audio, tuple):
            sampling_rate, audio_data = input_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_output_dir) as temp_audio_file:
                sf.write(temp_audio_file.name, audio_data, sampling_rate)
                input_audio = temp_audio_file.name

        manager.preprocessor.vocals_path = input_audio
        diarization_data = manager.preprocessor.perform_diarization()

        # Compress the `speaker_audio` directory for download
        speaker_audio_dir = os.path.join(temp_output_dir, "speaker_audio")
        zip_path = os.path.join(temp_output_dir, "diarization_results.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, dirs, files in os.walk(speaker_audio_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, speaker_audio_dir)
                    zipf.write(full_path, arcname=relative_path)

        simplified_json_path = os.path.join(manager.preprocessor.base_output_dir, "diar_simple.json")
        update_step_results("diarization", simplified_json_path=simplified_json_path, zip_path=zip_path)
        return simplified_json_path, zip_path
    except Exception as e:
        logger.error(f"Error performing diarization: {e}")
        return f"Error: {e}", None

# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Video Processing Pipeline")

    # Smaller video upload box
    video_input = gr.Video(label="Upload Video", height=300)

    # Step 1: Split Audio and Video
    gr.Markdown("### Step 1: Split Audio and Video")
    split_button = gr.Button("Split Audio & Video")
    with gr.Row():
        video_without_audio = gr.Video(label="Video Without Audio", height=300)
        extracted_audio = gr.Audio(label="Extracted Audio")

    split_button.click(
        split_audio_video,
        inputs=video_input,
        outputs=[video_without_audio, extracted_audio]
    )

    # Step 2: Separate Audio
    gr.Markdown("### Step 2: Separate Audio")
    separate_button = gr.Button("Separate Audio")
    with gr.Row():
        vocals_audio = gr.Audio(label="Vocals")
        background_audio = gr.Audio(label="Background Music")
    with gr.Row():
        vocals_spectrogram = gr.Image(label="Vocals Spectrogram")
        background_spectrogram = gr.Image(label="Background Spectrogram")

    separate_button.click(
        separate_audio,
        inputs=extracted_audio,
        outputs=[vocals_audio, background_audio, vocals_spectrogram, background_spectrogram]
    )

    # Step 3: Perform Diarization
    gr.Markdown("### Step 3: Perform Diarization")
    diarize_button = gr.Button("Perform Diarization")
    diarization_json = gr.Textbox(label="Simplified Diarization Data", lines=10, interactive=False, max_lines=30)
    diarization_folder = gr.File(label="Diarization Results (ZIP)")

    diarize_button.click(
    lambda input_audio: display_json_content(*perform_diarization(input_audio)),
    inputs=vocals_audio,
    outputs=[diarization_json, diarization_folder]
)

demo.launch()