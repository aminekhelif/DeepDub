import os
from moviepy.editor import VideoFileClip

def split_audio_video(input_video):
    """
    Extracts the audio from the input video and creates a version of the video without audio.
    
    Parameters:
    input_video (str): Path to the input video file.
    
    Returns:
    tuple: A tuple containing the paths to the extracted audio file and the video without audio.
    """
    
    # Get the absolute path of the input video
    input_video_abs_path = os.path.abspath(input_video)
    
    # Extract the directory and base name (without extension) from the absolute path
    video_directory = os.path.dirname(input_video_abs_path)
    base_name = os.path.splitext(os.path.basename(input_video_abs_path))[0]
    
    # Create a folder at the same level as the original video to store the output files
    output_folder = os.path.join(video_directory, base_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Define output filenames based on the input video name and folder path
    output_audio = os.path.join(output_folder, f"{base_name}_audio.mp3")
    output_video_no_audio = os.path.join(output_folder, f"{base_name}_no_audio.mp4")
    
    # Load the video
    video = VideoFileClip(input_video_abs_path)
    
    try:
        # Extract and save the audio
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(output_audio)
        else:
            print("No audio track found in the input video.")
            output_audio = None
        
        # Create and save the video without audio
        video_no_audio = video.without_audio()
        video_no_audio.write_videofile(output_video_no_audio, codec='libx264', audio=False)
    finally:
        # Ensure that resources are released properly
        video.close()
        if 'video_no_audio' in locals():
            video_no_audio.close()
    
    # Return the paths to the audio and video files
    return output_audio, output_video_no_audio