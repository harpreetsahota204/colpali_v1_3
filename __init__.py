import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import ColPali, ColPaliConfig 

logger = logging.getLogger(__name__)

def download_model(model_name, model_path):
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    snapshot_download(repo_id=model_name, local_dir=model_path)



def load_model(model_name, model_path, **kwargs):
    """Loads the ColPali model.
    
    Args:
        model_name: the name of the model
        model_path: the path to the model on disk
        **kwargs: additional keyword arguments including:
            - classes: list of class names for classification (optional)
            - pool_factor: token pooling compression factor (default: 3)
            - pooling_strategy: "mean" or "max" (default: "mean")
            - text_prompt: optional text prompt prefix for classification
            
    Returns:
        a ColPali instance
    """
    # Start with base configuration
    config_dict = {
        "model_path": model_path,
    }
    
    #Merge all kwargs into config_dict
    config_dict.update(kwargs)
    
    # Create config and model
    config = ColPaliConfig(config_dict)
    return ColPali(config)


def resolve_input(model_name, ctx):
    """Defines properties to collect the model's custom parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """

    inputs = types.Object()
    
    return types.Property(inputs)