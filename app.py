import gradio as gr
from DeepDub.PreProcessing import Preprocessing
import os
import yaml
import tempfile
import json
import soundfile as sf
from DeepDub.logger import logger

from DeepDub.tm import translator

# Load configuration
with open('./DeepDub/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Define accessible temporary directory for outputs
temp_output_dir = "/tmp/deepdub_outputs"
os.makedirs(temp_output_dir, exist_ok=True)

# Shared preprocessor instance
class VideoProcessorManager:
    def __init__(self):
        self.preprocessor = None
        self.step_results = {}
        self.speaker_audio_structure = {}
        self.diar_simple_path = None

    def initialize(self, input_video=None):
        output_dir = os.path.join(temp_output_dir, "separation_output")
        os.makedirs(output_dir, exist_ok=True)
        self.preprocessor = Preprocessing(
            input_video=input_video,
            output_dir=output_dir,
            audio_separator_model=config['audio_separator_model'],
            diarization_batch_size=config['diarization_batch_size'],
            device=config['device'],
            compute_type=config['compute_type'],
            HF_token=config['HF_token']
        )
        self.step_results = {}
        self.speaker_audio_structure = {}
        self.diar_simple_path = None

manager = VideoProcessorManager()

def update_step_results(step, **results):
    manager.step_results[step] = results
    for key, value in results.items():
        logger.info(f"{step} result - {key}: {value}")

# Step 1: Split Audio and Video
def split_audio_video(input_video):
    try:
        if not manager.preprocessor or (manager.preprocessor.input_video and manager.preprocessor.input_video != os.path.abspath(input_video)):
            manager.initialize(input_video)
        extracted_audio_path, video_no_audio_path = manager.preprocessor.split_audio_and_video()
        update_step_results("split", extracted_audio_path=extracted_audio_path, video_no_audio_path=video_no_audio_path)
        return video_no_audio_path, extracted_audio_path
    except Exception as e:
        logger.error(f"Error splitting audio and video: {e}")
        return f"Error: {e}", None

def separate_audio(input_audio):
    try:
        if not manager.preprocessor:
            manager.initialize()
        if isinstance(input_audio, tuple):
            sampling_rate, audio_data = input_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_output_dir) as temp_audio_file:
                sf.write(temp_audio_file.name, audio_data, sampling_rate)
                input_audio = temp_audio_file.name
        manager.preprocessor.extracted_audio_path = input_audio
        vocals_path, background_path = manager.preprocessor.separate_audio()
        separator = manager.preprocessor.audio_separator
        vocals_spectrogram_path = os.path.join(manager.preprocessor.base_output_dir, "vocals_spectrogram.png")
        background_spectrogram_path = os.path.join(manager.preprocessor.base_output_dir, "background_spectrogram.png")
        separator.save_spectrogram(vocals_path, vocals_spectrogram_path)
        separator.save_spectrogram(background_path, background_spectrogram_path)

        update_step_results("separate",
                            vocals_path=vocals_path,
                            background_path=background_path,
                            vocals_spectrogram_path=vocals_spectrogram_path,
                            background_spectrogram_path=background_spectrogram_path)
        return vocals_path, background_path, vocals_spectrogram_path, background_spectrogram_path
    except Exception as e:
        logger.error(f"Error separating audio: {e}")
        return f"Error: {e}", None, None, None

# Step 3: Perform Diarization
def perform_diarization(input_audio):
    try:
        if not manager.preprocessor:
            manager.initialize()
        if isinstance(input_audio, tuple):
            sampling_rate, audio_data = input_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_output_dir) as temp_audio_file:
                sf.write(temp_audio_file.name, audio_data, sampling_rate)
                input_audio = temp_audio_file.name
        manager.preprocessor.vocals_path = input_audio
        diarization_data = manager.preprocessor.perform_diarization()

        diar_simple_path = diarization_data['diarization_data']
        manager.diar_simple_path = diar_simple_path

        speaker_audio_dir = diarization_data['speaker_audio_dir']
        manager.speaker_audio_structure = parse_speaker_audio_dir(speaker_audio_dir)

        update_step_results("diarization",
                            simplified_json_path=diar_simple_path,
                            speaker_audio_dir=speaker_audio_dir,
                            files_list=[])

        diar_json_content = load_json_file(diar_simple_path)
        # Return the diar_json_content and update the speakers dropdown choices
        return diar_json_content, gr.update(choices=list(manager.speaker_audio_structure.keys()))
    except Exception as e:
        logger.error(f"Error performing diarization: {e}")
        return f"Error: {e}", gr.update(choices=[])

def parse_speaker_audio_dir(speaker_audio_dir):
    speakers = {}
    if not os.path.exists(speaker_audio_dir):
        return speakers
    for item in os.listdir(speaker_audio_dir):
        sp_dir = os.path.join(speaker_audio_dir, item)
        if os.path.isdir(sp_dir) and item.startswith("SPEAKER_"):
            concatenated_wav = None
            concatenated_json = None
            segments = {}
            for f in os.listdir(sp_dir):
                full_path = os.path.join(sp_dir, f)
                if os.path.isfile(full_path):
                    if f.endswith("_concatenated.wav"):
                        concatenated_wav = f
                    elif f.endswith("_concatenated_metadata.json"):
                        concatenated_json = f
                elif os.path.isdir(full_path) and f.startswith("segment_"):
                    seg_audio = None
                    seg_json = None
                    for seg_f in os.listdir(full_path):
                        if seg_f.endswith(".wav"):
                            seg_audio = seg_f
                        elif seg_f.endswith(".json"):
                            seg_json = seg_f
                    if seg_audio and seg_json:
                        segments[f] = (seg_audio, seg_json)
            speakers[item] = {
                "concatenated": (concatenated_wav, concatenated_json),
                "segments": segments,
                "path": sp_dir
            }
    return speakers

def load_json_file(json_path):
    if not os.path.exists(json_path):
        return "Diarization file not found."
    with open(json_path, 'r') as f:
        content = json.load(f)
    return json.dumps(content, indent=4)

def update_speaker_files(speaker):
    if not speaker or speaker not in manager.speaker_audio_structure:
        return gr.update(choices=[], value=None)
    sp_data = manager.speaker_audio_structure[speaker]
    choices = []
    if sp_data['concatenated'][0] and sp_data['concatenated'][1]:
        choices.append("concatenated")
    sorted_segments = sorted(sp_data['segments'].keys(), key=lambda x: int(x.split("_")[1]))
    choices.extend(sorted_segments)
    return gr.update(choices=choices, value=None)

def display_speaker_file(speaker, selection):
    if not speaker or speaker not in manager.speaker_audio_structure:
        return "No valid speaker selected.", None
    sp_data = manager.speaker_audio_structure[speaker]
    sp_path = sp_data['path']
    if not selection:
        return "No file selected.", None
    
    if selection == "concatenated":
        wav, j = sp_data['concatenated']
        if not wav or not j:
            return "No concatenated files found.", None
        wav_path = os.path.join(sp_path, wav)
        json_path = os.path.join(sp_path, j)
        meta_content = "Metadata file not found."
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
            meta_content = json.dumps(meta, indent=4)
        audio_comp = wav_path if os.path.exists(wav_path) else None
        return meta_content, audio_comp
    else:
        segments = sp_data['segments']
        if selection not in segments:
            return "Segment not found.", None
        seg_audio, seg_json = segments[selection]
        seg_path = os.path.join(sp_path, selection)
        audio_path = os.path.join(seg_path, seg_audio)
        json_path = os.path.join(seg_path, seg_json)
        meta_content = "Metadata file not found."
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                meta = json.load(f)
            meta_content = json.dumps(meta, indent=4)
        audio_comp = audio_path if os.path.exists(audio_path) else None
        return meta_content, audio_comp

def save_diarization_data(json_text):
    try:
        new_data = json.loads(json_text)
        diar_simple_path = manager.diar_simple_path
        if not diar_simple_path or not os.path.exists(diar_simple_path):
            return "Error: No diarization path found or file doesn't exist."
        
        with open(diar_simple_path, 'w') as f:
            json.dump(new_data, f, indent=4)
        manager.preprocessor.diarization_data = new_data
        
        logger.info(f"Diarization JSON saved successfully at {diar_simple_path}!")
        return f"Diarization JSON saved successfully at {diar_simple_path}!"
    except json.JSONDecodeError as e:
        return f"JSON parsing error: {str(e)}"
    except Exception as e:
        return f"Error saving JSON: {str(e)}"


def translate_diar_json():
    """
    Called when user clicks 'Translate Diarization'.
    1) Reads diar_simple.json from manager.diar_simple_path (with any saved edits).
    2) translator.translate_json(...) writes 'diarization_translated.json' beside diar_simple.json
    3) Returns the new file's JSON content to display in Gradio.
    """
    try:
        diar_simple_path = manager.diar_simple_path
        if not diar_simple_path or not os.path.exists(diar_simple_path):
            return "Error: diar_simple.json path not found. Please run diarization first."

        # Load the current JSON from disk (with any saved edits)
        with open(diar_simple_path, "r", encoding="utf-8") as f:
            segments_data = json.load(f)
        if not isinstance(segments_data, list):
            return "Error: diar_simple.json is not a list of segments."

        # Translate & get path to diarization_translated.json
        translated_file_path = translator.translate_json(
            segments=segments_data,
            diar_json_path=diar_simple_path,  # no video path used
            target_language="French"
        )

        # Read back the translated JSON to show in a Gradio text box
        if not os.path.exists(translated_file_path):
            return f"Error: Translated file not found at {translated_file_path}"

        with open(translated_file_path, "r", encoding="utf-8") as tf:
            translated_content = json.load(tf)

        return json.dumps(translated_content, indent=4)

    except Exception as exc:
        logger.error(f"Error translating diarization JSON: {exc}")
        return f"Error translating diarization JSON: {str(exc)}"


####################
#   BUILD THE UI
####################

with gr.Blocks() as demo:
    gr.Markdown("## DeepDub - Demo")

    video_input = gr.Video(label="Upload Video", height=300)
    gr.Markdown("### Step 1: Split Audio and Video")
    split_button = gr.Button("Split Audio & Video")
    with gr.Row():
        video_without_audio = gr.Video(label="Video Without Audio", height=300)
        extracted_audio = gr.Audio(label="Extracted Audio", type="filepath", show_download_button=True, interactive=True)

    split_button.click(
        fn=split_audio_video,
        inputs=video_input,
        outputs=[video_without_audio, extracted_audio]
    )

    gr.Markdown("### Step 2: Separate Audio")
    separate_button = gr.Button("Separate Audio")
    with gr.Row():
        vocals_audio = gr.Audio(label="Vocals", type="filepath", show_download_button=True, interactive=True)
        background_audio = gr.Audio(label="Background Music", type="filepath", show_download_button=True, interactive=True)
    with gr.Row():
        vocals_spectrogram = gr.Image(label="Vocals Spectrogram")
        background_spectrogram = gr.Image(label="Background Spectrogram")

    separate_button.click(
        fn=separate_audio,
        inputs=extracted_audio,
        outputs=[vocals_audio, background_audio, vocals_spectrogram, background_spectrogram]
    )

    gr.Markdown("### Step 3: Perform Diarization")
    diarize_button = gr.Button("Perform Diarization")

    # This Textbox shows and allows editing of diar_simple.json
    diar_json_display = gr.Textbox(label="diar_simple.json Content", lines=15, interactive=True)

    speakers_dropdown = gr.Dropdown(label="Speakers", choices=[], interactive=True)
    speaker_file_dropdown = gr.Dropdown(label="Segments / Concatenated", choices=[], interactive=True)

    with gr.Row():
        metadata_display = gr.Textbox(label="Metadata JSON", lines=15, interactive=False)
        file_audio = gr.Audio(label="Audio Player", type="filepath", show_download_button=True, interactive=True)

    diarize_button.click(
        fn=perform_diarization,
        inputs=vocals_audio,
        outputs=[diar_json_display, speakers_dropdown]
    )

    speakers_dropdown.change(
        fn=update_speaker_files,
        inputs=speakers_dropdown,
        outputs=speaker_file_dropdown
    )

    speaker_file_dropdown.change(
        fn=display_speaker_file,
        inputs=[speakers_dropdown, speaker_file_dropdown],
        outputs=[metadata_display, file_audio]
    )

    # Button and textbox to confirm saving the JSON edits
    save_diar_button = gr.Button("Save Diarization Edits")
    save_status = gr.Textbox(label="Save Status", interactive=False)

    save_diar_button.click(
        fn=save_diarization_data,
        inputs=[diar_json_display],
        outputs=[save_status]
    )

    gr.Markdown("### Step 4: Translate Diarization JSON")
    translate_button = gr.Button("Translate Diarization")
    translated_json_display = gr.Textbox(label="Translated JSON", lines=15, interactive=False)

    translate_button.click(
        fn=translate_diar_json,
        inputs=None,
        outputs=[translated_json_display]
    )

demo.launch()