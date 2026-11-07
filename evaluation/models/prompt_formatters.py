# evaluation/models/prompt_formatters.py
"""Model-specific prompt formatting functions"""

from typing import Union, List, Optional

def format_qwen_prompt(prompt: str, media: Optional[Union[str, List[str]]] = None) -> str:
    """Format prompt for Qwen 2.5 VL models"""
    if media is not None:
        if isinstance(media, list):
            # Video frames
            img_tags = "".join([f"<img>{frame}</img>" for frame in media])
            formatted_prompt = f"<|im_start|>user\n{img_tags}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Single image
            formatted_prompt = f"<|im_start|>user\n<img>{media}</img>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return formatted_prompt

def format_llava_prompt(prompt: str, media: Optional[Union[str, List[str]]] = None) -> str:
    """Format prompt for LLaVA models"""
    if media is not None:
        if isinstance(media, list):
            # Video frames - LLaVA typically uses first frame or specific format
            img_tag = f"<image>{media[0]}</image>"  # Use first frame for simplicity
            formatted_prompt = f"USER: {img_tag}\n{prompt}\nASSISTANT:"
        else:
            # Single image
            formatted_prompt = f"USER: <image>{media}</image>\n{prompt}\nASSISTANT:"
    else:
        formatted_prompt = f"USER: {prompt}\nASSISTANT:"
    
    return formatted_prompt

def format_generic_prompt(prompt: str, media: Optional[Union[str, List[str]]] = None) -> str:
    """Generic prompt format for models without specific templates"""
    if media is not None:
        if isinstance(media, list):
            # Video frames - simple concatenation
            media_str = "\n".join([f"Frame {i+1}: {frame}" for i, frame in enumerate(media)])
            formatted_prompt = f"{media_str}\n\n{prompt}"
        else:
            # Single image
            formatted_prompt = f"Image: {media}\n\n{prompt}"
    else:
        formatted_prompt = prompt
    
    return formatted_prompt

# Model-specific formatter registry
PROMPT_FORMATTERS = {
    "qwen": format_qwen_prompt,
    "llava": format_llava_prompt,
    "generic": format_generic_prompt
}

def get_prompt_formatter(model_family: str):
    """Get the appropriate prompt formatter for a model family"""
    return PROMPT_FORMATTERS.get(model_family.lower(), format_generic_prompt)
