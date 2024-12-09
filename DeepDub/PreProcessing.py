import os
from DeepDub.AV_splitter import VideoProcessor
from DeepDub.AudioSeparator import AudioSeparator
from DeepDub.Diar import AudioDiarization

# Import logger
from DeepDub.logger import logger

class Preprocessing:
    def __init__(self, input_video=None, output_dir=None, 
                 audio_separator_model="Mel-RoFormer", 
                 diarization_batch_size=16, device="cpu", compute_type="int8", HF_token=None):
        """
        Initializes the Preprocessing class.
        
        Args:
            input_video (str): Path to the input video file.
            output_dir (str, optional): Base directory for output files. Defaults to a directory named 'separation_output' next to the input video.
            audio_separator_model (str): Model for audio separation. Defaults to "Mel-RoFormer".
            diarization_batch_size (int): Batch size for diarization.
            device (str): Device for model computation, e.g., 'cpu' or 'cuda'.
            compute_type (str): Precision type for WhisperX model.
            HF_token (str): Hugging Face token for diarization pipeline.
        """
        self.input_video = os.path.abspath(input_video) if input_video else None
        if output_dir:
            self.base_output_dir = os.path.abspath(output_dir)
        else:
            base_dir = os.path.dirname(self.input_video) if self.input_video else "."
            self.base_output_dir = os.path.join(base_dir, "separation_output")
        os.makedirs(self.base_output_dir, exist_ok=True)

        self.audio_separator = AudioSeparator(model=audio_separator_model, output_dir=self.base_output_dir)
        self.diarization_batch_size = diarization_batch_size
        self.device = device
        self.compute_type = compute_type
        self.HF_token = HF_token

        # Paths
        self.extracted_audio_path = None
        self.video_no_audio_path = None
        self.vocals_path = None
        self.background_path = None
        self.diarization_data = None
        self.speaker_audio_dir = None
        self.concatenated_audio_dir = None

    def split_audio_and_video(self):
        """
        Splits the input video into audio and video without audio.
        
        Returns:
            Tuple[str, str]: Paths to the extracted audio file and video without audio.
        """
        if not self.input_video or not os.path.exists(self.input_video):
            logger.error("No valid input video provided.")
            raise ValueError("No valid input video provided.")
        logger.info(f"Splitting audio and video for: {self.input_video}")
        
        # Use VideoProcessor to split audio and video
        processor = VideoProcessor(self.input_video)
        self.extracted_audio_path, self.video_no_audio_path = processor.split_audio_video()
        if self.extracted_audio_path:
            new_extracted = os.path.join(self.base_output_dir, os.path.basename(self.extracted_audio_path))
            if self.extracted_audio_path != new_extracted:
                os.rename(self.extracted_audio_path, new_extracted)
                self.extracted_audio_path = new_extracted
        if self.video_no_audio_path:
            new_video_no_audio = os.path.join(self.base_output_dir, os.path.basename(self.video_no_audio_path))
            if self.video_no_audio_path != new_video_no_audio:
                os.rename(self.video_no_audio_path, new_video_no_audio)
                self.video_no_audio_path = new_video_no_audio
        logger.info(f"Extracted audio: {self.extracted_audio_path}")
        logger.info(f"Video without audio: {self.video_no_audio_path}")
        return self.extracted_audio_path, self.video_no_audio_path

    def separate_audio(self):
        """
        Separates the extracted audio into vocals and background music.
        
        Returns:
            Tuple[str, str]: Paths to the vocals and background music files.
        """
        if not self.extracted_audio_path or not os.path.exists(self.extracted_audio_path):
            logger.error("No valid audio provided for separation.")
            raise ValueError("Set 'extracted_audio_path' to a valid audio file before calling separate_audio.")
        logger.info(f"Separating audio: {self.extracted_audio_path}")
        vocals_path, background_path = self.audio_separator.separate(self.extracted_audio_path)

        # Ensure paths are within base_output_dir
        new_vocals_path = os.path.join(self.base_output_dir, "vocals.mp3")
        new_background_path = os.path.join(self.base_output_dir, "instrumental.mp3")

        if not os.path.exists(vocals_path):
            logger.error(f"Expected vocals file not found at: {vocals_path}")
            raise FileNotFoundError(f"Expected vocals file not found at: {vocals_path}")
        os.rename(vocals_path, new_vocals_path)
        self.vocals_path = new_vocals_path

        if not os.path.exists(background_path):
            logger.error(f"Expected background file not found at: {background_path}")
            raise FileNotFoundError(f"Expected background file not found at: {background_path}")
        os.rename(background_path, new_background_path)
        self.background_path = new_background_path

        logger.info(f"Vocals path: {self.vocals_path}")
        logger.info(f"Background path: {self.background_path}")
        return self.vocals_path, self.background_path

    def perform_diarization(self):
        """
        Performs speaker diarization on the vocals audio.
        
        Returns:
            dict: Diarization results containing speaker segments.
        """
        if not self.vocals_path or not os.path.exists(self.vocals_path):
            logger.error("No valid vocals audio provided.")
            raise ValueError("Set 'vocals_path' to a valid audio file before calling perform_diarization.")
        logger.info(f"Performing diarization on vocals: {self.vocals_path}")
        diarizer = AudioDiarization(
            audio_path=self.vocals_path,
            diarization_dir=os.path.join(self.base_output_dir, "diarization"),
            batch_size=self.diarization_batch_size,
            device=self.device,
            compute_type=self.compute_type,
            HF_token=self.HF_token
        )

        self.diarization_data = diarizer.perform_diarization()
        self.speaker_audio_dir = diarizer.extract_speaker_audio()
        self.concatenated_audio_dir = diarizer.concatenate_speaker_segments()

        logger.info(f"Diarization results: {self.diarization_data}")
        return {
            "diarization_data": self.diarization_data,
            "speaker_audio_dir": self.speaker_audio_dir,
            "concatenated_audio_dir": self.concatenated_audio_dir
        }

    def get_paths(self):
        """
        Returns all the relevant paths generated during preprocessing.
        
        Returns:
            dict: Paths for extracted audio, video without audio, vocals, background, and diarization folder.
        """
        return {
            "extracted_audio_path": self.extracted_audio_path,
            "video_no_audio_path": self.video_no_audio_path,
            "vocals_path": self.vocals_path,
            "background_path": self.background_path,
            "diarization_folder": os.path.join(self.base_output_dir, "diarization")
        }

    def get_diarization_data(self):
        """
        Returns the speaker diarization results.
        
        Returns:
            dict: Diarization data, including speaker segments.
        """
        if not self.diarization_data or not os.path.exists(self.diarization_data):
            logger.error("Diarization not performed or no data found.")
            raise ValueError("Diarization not performed or no data found.")
        return self.diarization_data
