import os
from moviepy.editor import VideoFileClip
from DeepDub.logger import logger

class VideoProcessor:
    def __init__(self, input_video, processing_dir=None):
        """
        Initialize the VideoProcessor with input video path and a central processing directory.
        
        Parameters:
        input_video (str): Path to the input video file.
        processing_dir (str, optional): Central directory for saving outputs.
        """
        # Absolute path of the input video
        if not input_video or not os.path.exists(input_video):
            raise ValueError(f"Input video not found: {input_video}")
        self.input_video = os.path.abspath(input_video)
        
        # Set processing directory, defaulting to the input video's directory
        self.processing_dir = os.path.abspath(processing_dir) if processing_dir else os.path.dirname(self.input_video)
        
        # Derive output file paths relative to the processing directory
        base_name = os.path.splitext(os.path.basename(self.input_video))[0]
        self.output_audio = os.path.join(self.processing_dir, f"{base_name}_audio.mp3")
        self.output_video_no_audio = os.path.join(self.processing_dir, f"{base_name}_no_audio.mp4")
        
        # Ensure processing directory exists
        os.makedirs(self.processing_dir, exist_ok=True)

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
            if 'video_without_audio' in locals():
                video_without_audio.close()
        
        logger.info("Audio and video split completed.")
        return self.output_audio, self.output_video_no_audio