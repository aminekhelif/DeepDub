import os
import tempfile
import logging
from typing import Tuple, List
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import matplotlib
from audio_separator.separator import Separator 
from logger import logger

class AudioSeparator:
    """
    A class to handle audio separation using specified models with flexible input and output handling.
    """

    def __init__(self, model: str = "BS-RoFormer", output_dir: str = None, output_format: str = "mp3"):
        """
        Initializes the AudioSeparator with a specified model, output directory, and output format.

        Args:
            model (str): The model to use for separation. Defaults to "BS-RoFormer".
            output_dir (str): Directory to save output files. Defaults to tempfile.gettempdir().
            output_format (str): Format for output files. Defaults to "mp3".
        """
        self.output_dir = output_dir if output_dir else tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)
        self.sample_rate = 44100
        self.output_format = output_format 

        # Predefined model file paths
        self.model_paths = {
            "BS-RoFormer": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "Mel-RoFormer": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        }

        # Define available models
        self.available_models = list(self.model_paths.keys())

        # Initialize separators dictionary
        self.separators = {}

        # Set and load the initial model
        self.model = model

    @property
    def model(self) -> str:
        """Gets the current model."""
        return self._model

    @model.setter
    def model(self, model_name: str):
        """Sets the current model and loads it if not already loaded."""
        if model_name not in self.available_models:
            logger.error(f"Model '{model_name}' is not supported.")
            raise ValueError(f"Model '{model_name}' is not supported.")
        self._model = model_name
        if model_name not in self.separators:
            self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Loads the specified model."""
        model_path = self.model_paths[model_name]
        separator = Separator(output_dir=self.output_dir, output_format=self.output_format)
        for attempt in range(1, 4):
            try:
                logger.info(f"Loading model '{model_name}' from '{model_path}' (Attempt {attempt}/3)")
                separator.load_model(model_path)
                self.separators[model_name] = separator
                logger.info(f"Successfully loaded model '{model_name}'.")
                break
            except Exception as e:
                logger.error(f"Error loading model '{model_name}': {e}")
                if attempt == 3:
                    raise RuntimeError(f"Failed to load model '{model_name}' after 3 attempts.") from e

    def separate(self, audio_path: str) -> Tuple[str, str]:
        """
        Separates the audio into vocals and background music using the selected model.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            Tuple[str, str]: Paths to the vocals and background music files.
        """
        separator = self.separators[self.model]
        separated_files = separator.separate(audio_path)
        separated_paths = [os.path.join(self.output_dir, out) for out in separated_files]

        if self.model in ["BS-RoFormer", "Mel-RoFormer"]:
            vocals, bgm = separated_paths[1], separated_paths[0]
        else:
            logger.error(f"Unexpected output format for model '{self.model}'.")
            raise ValueError(f"Unexpected output format for model '{self.model}'.")

        return vocals, bgm

    def _merge(self, output_files: List[str]) -> str:
        """Merges multiple audio files into a single track."""
        audio_data = [sf.read(file)[0] for file in output_files]
        merged_audio = np.sum(np.array(audio_data), axis=0)
        merged_file = os.path.join(
            self.output_dir, f"{os.path.basename(output_files[0]).split('.')[0]}_merged.{self.output_format}"
        )
        sf.write(merged_file, merged_audio, self.sample_rate)
        return merged_file

    def plot_spectrogram(self, audio_path: str) -> matplotlib.figure.Figure:
        """
        Generates a Mel-frequency spectrogram for the given audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            matplotlib.figure.Figure: The spectrogram figure.
        """
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(15, 5))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel-frequency Spectrogram")
        fig.tight_layout()

        return fig