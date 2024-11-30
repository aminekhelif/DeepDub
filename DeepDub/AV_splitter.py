import os
from moviepy.editor import VideoFileClip
from DeepDub.logger import logger

class VideoProcessor:
    def __init__(self, input_video, output_audio=None, output_video_no_audio=None):
        """
        Initialize the VideoProcessor with input video path and optional output paths.
        
        Parameters:
        input_video (str): Path to the input video file.
        output_audio (str, optional): Path to save the extracted audio file.
        output_video_no_audio (str, optional): Path to save the video without audio.
        """
        self.input_video = os.path.abspath(input_video)
        self.output_audio = output_audio
        self.output_video_no_audio = output_video_no_audio
        
        # Set default output paths if not provided
        if not self.output_audio:
            base_name = os.path.splitext(os.path.basename(self.input_video))[0]
            self.output_audio = os.path.join(os.path.dirname(self.input_video), f"{base_name}_audio.mp3")
        if not self.output_video_no_audio:
            base_name = os.path.splitext(os.path.basename(self.input_video))[0]
            self.output_video_no_audio = os.path.join(os.path.dirname(self.input_video), f"{base_name}_no_audio.mp4")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_audio)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def split_audio_video(self):
        """
        Extracts the audio from the input video and creates a version of the video without audio.
        
        Returns:
        tuple: A tuple containing the paths to the extracted audio file and the video without audio.
        """
        logger.info(f"Processing video: {self.input_video}")
        # Load the video
        try:
            video = VideoFileClip(self.input_video)
        except Exception as e:
            logger.error(f"Error loading video: {e}")
            raise
        try:
            # Extract and save the audio
            if video.audio:
                logger.info(f"Extracting audio to: {self.output_audio}")
                video.audio.write_audiofile(self.output_audio)
            else:
                logger.warning("No audio track found in the input video.")
                self.output_audio = None
            
            # Create and save the video without audio
            logger.info(f"Creating video without audio: {self.output_video_no_audio}")
            video_without_audio = video.without_audio()
            video_without_audio.write_videofile(self.output_video_no_audio, codec='libx264', audio=False)
        finally:
            # Ensure that resources are released properly
            video.close()
            video_without_audio.close()
        
        logger.info(f"Audio and video split completed.")
        return self.output_audio, self.output_video_no_audio