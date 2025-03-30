from omegaconf import OmegaConf
from typing import Any, Dict, Union

def make_json_serializable(obj: Any) -> Any:
    """
    Converts OmegaConf DictConfig objects to regular Python dictionaries
    for JSON serialization.
    
    Args:
        obj: The object to convert, typically a DictConfig
        
    Returns:
        A JSON-serializable version of the input object
    """
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj