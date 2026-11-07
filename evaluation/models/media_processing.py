# evaluation/models/media_processing.py
"""Shared media processing functions for all model wrappers"""

import base64
import io
import decord
import numpy as np
from PIL import Image
import os
from typing import List, Optional, Union

def load_image_from_path(image_path: str) -> str:
    """Load image from path and convert to base64 string"""
    try:
        image = Image.open(image_path)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

def load_video_frames_from_path(video_path: str, max_frames: int = 8) -> List[str]:
    """Extract frames from video using decord for speed"""
    try:
        # Load video with decord
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        # Sample frames evenly
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        
        # Extract frames
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Convert frames to base64
        frame_strings = []
        for frame in frames:
            # Convert RGB to PIL Image
            pil_image = Image.fromarray(frame)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            frame_strings.append(f"data:image/jpeg;base64,{img_str}")
        
        return frame_strings
        
    except Exception as e:
        raise ValueError(f"Error processing video {video_path}: {str(e)}")

def get_media_type(media_path: str) -> str:
    """Determine if media path is image or video"""
    _, ext = os.path.splitext(media_path)
    
    if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        return "video"
    elif ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return "image"
    else:
        raise ValueError(f"Unsupported media format: {ext}")

def process_media(media_path: str, max_frames: int = 8) -> Union[str, List[str]]:
    """Process media file and return appropriate format"""
    media_type = get_media_type(media_path)
    
    if media_type == "image":
        return load_image_from_path(media_path)
    elif media_type == "video":
        return load_video_frames_from_path(media_path, max_frames)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")
