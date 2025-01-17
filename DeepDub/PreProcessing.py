import os
from DeepDub.AV_splitter import VideoProcessor
from DeepDub.AudioSeparator import AudioSeparator
from DeepDub.Diar import AudioDiarization
from DeepDub.logger import logger

class Preprocessing:
    def __init__(self, 
                 input_video=None, 
                 output_dir=None, 
                 audio_separator_model="Mel-RoFormer", 
                 diarization_batch_size=16, 
                 device="cpu", 
                 compute_type="int8", 
                 HF_token=None,
                 num_speakers=None,
                 language=None,
                 device_index=0,
                 metadata_dir=None   # <--- NEW ARG
                 ):
        """
        Initializes the Preprocessing class, storing references and defaults.
        """
        self.input_video = os.path.abspath(input_video) if input_video else None

        if output_dir:
            self.base_output_dir = os.path.abspath(output_dir)
        else:
            base_dir = os.path.dirname(self.input_video) if self.input_video else "."
            self.base_output_dir = os.path.join(base_dir, "separation_output")
        os.makedirs(self.base_output_dir, exist_ok=True)

        self.audio_separator = AudioSeparator(
            model=audio_separator_model, 
            output_dir=self.base_output_dir
        )
        self.diarization_batch_size = diarization_batch_size
        self.device = device
        self.compute_type = compute_type
        self.HF_token = HF_token
        self.num_speakers = num_speakers
        self.language = language
        self.device_index = device_index

        self.extracted_audio_path = None
        self.video_no_audio_path = None
        self.vocals_path = None
        self.background_path = None
        self.diarization_data = None
        self.speaker_audio_dir = None
        self.concatenated_audio_dir = None

        # NEW: store metadata_dir
        self.metadata_dir = metadata_dir

    def split_audio_and_video(self):
        if not self.input_video or not os.path.exists(self.input_video):
            logger.error("No valid input video provided for splitting.")
            raise ValueError("No valid input video provided.")
        logger.info(f"Splitting audio and video for: {self.input_video}")

        processor = VideoProcessor(self.input_video, processing_dir=self.base_output_dir)
        extracted_audio, video_no_audio = processor.split_audio_video()

        # Move them into base_output_dir if needed
        if extracted_audio:
            new_extracted = os.path.join(self.base_output_dir, os.path.basename(extracted_audio))
            if extracted_audio != new_extracted:
                os.rename(extracted_audio, new_extracted)
            self.extracted_audio_path = new_extracted

        if video_no_audio:
            new_video_no_audio = os.path.join(self.base_output_dir, os.path.basename(video_no_audio))
            if video_no_audio != new_video_no_audio:
                os.rename(video_no_audio, new_video_no_audio)
            self.video_no_audio_path = new_video_no_audio

        logger.info(f"Extracted audio path: {self.extracted_audio_path}")
        logger.info(f"Video without audio: {self.video_no_audio_path}")
        return self.extracted_audio_path, self.video_no_audio_path

    def separate_audio(self):
        if not self.extracted_audio_path or not os.path.exists(self.extracted_audio_path):
            logger.error("No valid audio provided for separation.")
            raise ValueError("Set 'extracted_audio_path' before calling separate_audio.")
        logger.info(f"Separating audio: {self.extracted_audio_path}")

        vocals_path, background_path = self.audio_separator.separate(self.extracted_audio_path)

        # Rename to a stable name in base_output_dir
        new_vocals_path = os.path.join(self.base_output_dir, "vocals.mp3")
        new_background_path = os.path.join(self.base_output_dir, "instrumental.mp3")

        if not os.path.exists(vocals_path):
            raise FileNotFoundError(f"Expected vocals file not found at: {vocals_path}")
        os.rename(vocals_path, new_vocals_path)
        self.vocals_path = new_vocals_path

        if not os.path.exists(background_path):
            raise FileNotFoundError(f"Expected background file not found at: {background_path}")
        os.rename(background_path, new_background_path)
        self.background_path = new_background_path

        logger.info(f"Vocals path: {self.vocals_path}")
        logger.info(f"Background path: {self.background_path}")
        return self.vocals_path, self.background_path

    def perform_diarization(self):
        if not self.vocals_path or not os.path.exists(self.vocals_path):
            logger.error("No valid vocals audio found for diarization.")
            raise ValueError("Set 'vocals_path' before calling perform_diarization.")

        logger.info(f"Performing diarization on: {self.vocals_path}")
        diarizer = AudioDiarization(
            audio_path=self.vocals_path,
            diarization_dir=os.path.join(self.base_output_dir, "diarization"),
            batch_size=self.diarization_batch_size,
            device=self.device,
            compute_type=self.compute_type,
            HF_token=self.HF_token,
            num_speakers=self.num_speakers,
            language=self.language,
            device_index=self.device_index,
            metadata_dir=self.metadata_dir  # pass along
        )
        diar_json_path = diarizer.perform_diarization()
        self.speaker_audio_dir = diarizer.extract_speaker_audio()
        self.concatenated_audio_dir = diarizer.concatenate_speaker_segments()

        self.diarization_data = diar_json_path
        logger.info(f"Diarization data path: {self.diarization_data}")
        return {
            "diarization_data": self.diarization_data,
            "speaker_audio_dir": self.speaker_audio_dir,
            "concatenated_audio_dir": self.concatenated_audio_dir
        }

    def get_paths(self):
        return {
            "extracted_audio_path": self.extracted_audio_path,
            "video_no_audio_path": self.video_no_audio_path,
            "vocals_path": self.vocals_path,
            "background_path": self.background_path,
            "diarization_folder": os.path.join(self.base_output_dir, "diarization")
        }

    def get_diarization_data(self):
        if not self.diarization_data or not os.path.exists(self.diarization_data):
            raise ValueError("No diarization data found.")
        return self.diarization_data