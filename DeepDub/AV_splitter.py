import os
import shutil
from moviepy.editor import VideoFileClip
from DeepDub.logger import logger

class VideoProcessor:
    def __init__(self, input_video, processing_dir=None):
        """
        Initialize the VideoProcessor with input video path and a central processing directory.
        
        Parameters:
        input_video (str): Path to the input video or audio file.
        processing_dir (str, optional): Central directory for saving outputs.
        """
        if not input_video or not os.path.exists(input_video):
            raise ValueError(f"Input file not found: {input_video}")
        self.input_video = os.path.abspath(input_video)
        
        self.processing_dir = (
            os.path.abspath(processing_dir)
            if processing_dir else
            os.path.dirname(self.input_video)
        )
        
        base_name = os.path.splitext(os.path.basename(self.input_video))[0]
        
        self.output_audio = os.path.join(self.processing_dir, f"{base_name}_audio.mp3")
        self.output_video_no_audio = os.path.join(self.processing_dir, f"{base_name}_no_audio.mp4")
        
        os.makedirs(self.processing_dir, exist_ok=True)

    def split_audio_video(self):
        """
        - If the input is a video, extracts the audio and generates a video without audio.
        - If the input is an audio file, copies it to the processing directory using the
          naming convention "<base_name>_audio.mp3" and does not produce a video.
        
        Returns:
        tuple: (path_to_audio, path_to_video_no_audio or None)
        """
        logger.info(f"Processing file: {self.input_video}")
        
        _, ext = os.path.splitext(self.input_video)
        ext = ext.lower()
        
        audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]
        
        if ext in audio_extensions:
            logger.info("Detected input as an audio file. Skipping video extraction.")
            shutil.copy2(self.input_video, self.output_audio)
            self.output_video_no_audio = None
            
            logger.info(f"Audio copied to: {self.output_audio}")
            logger.info("No 'video without audio' is generated because input is audio-only.")
            return self.output_audio, self.output_video_no_audio
        
        try:
            video = VideoFileClip(self.input_video)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise
        
        try:
            if video.audio:
                logger.info(f"Extracting audio to: {self.output_audio}")
                video.audio.write_audiofile(self.output_audio)
            else:
                logger.warning("No audio track found in the input video.")
                self.output_audio = None
            
            logger.info(f"Creating video without audio: {self.output_video_no_audio}")
            video_without_audio = video.without_audio()
            video_without_audio.write_videofile(
                self.output_video_no_audio, 
                codec='libx264',
                audio=False
            )
        finally:
            video.close()
            if 'video_without_audio' in locals():
                video_without_audio.close()
        
        logger.info("Audio and video split completed.")
        return self.output_audio, self.output_video_no_audio