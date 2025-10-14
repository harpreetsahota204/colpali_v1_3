import logging
import os
from packaging.version import Version
import warnings
import math
from PIL import Image

import numpy as np

import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

import torch
import torch.nn.functional as F
from colpali_engine.models import ColPali as ColPaliModel, ColPaliProcessor

logger = logging.getLogger(__name__)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ColPaliConfig(fout.TorchImageModelConfig):
    """
    This config class extends TorchImageModelConfig to provide specific parameters
    needed for ColPali used for visual document retrieval.
    
    ColPali is a multi-vector retrieval model that generates ColBERT-style 
    representations for both images and text queries.
    
    Args:
        model_path (str): Path to the model's weights on disk or HuggingFace model ID.
        
        text_prompt (str): Optional baseline text prompt to use for classification.
            Defaults to "".
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        super().__init__(d)
        
        # Path to model weights or HuggingFace model ID
        self.model_path = self.parse_string(d, "model_path", default="")
        
        # Optional base text prompt
        self.text_prompt = self.parse_string(d, "text_prompt", default="")


class ColPali(fout.TorchImageModel, fom.PromptMixin):
    """
    This model leverages ColPali, a Vision Language Model based on PaliGemma-3B,
    to create multi-vector embeddings for both images and text in a shared vector space,
    enabling visual document retrieval.
    
    Unlike single-vector models, ColPali generates multiple embedding vectors per input
    (similar to ColBERT's approach), allowing for more fine-grained matching between
    documents and queries.
    
    The model can:
    1. Embed images into multiple vectors
    2. Embed text queries into multiple vectors
    3. Calculate multi-vector similarity between images and text
    4. Support visual document retrieval
    
    It extends TorchImageModel for image processing capabilities and PromptMixin to
    enable text embedding capabilities.
    """
    
    def __init__(self, config):
        """Initialize the model.
        
        Args:
            config: A ColPaliConfig instance containing model parameters
        """
        # Initialize parent classes
        super().__init__(config)
        
        # Storage for text features and embeddings
        self._text_features = None  # Cached text features for classification
        self._last_computed_embeddings = None  # Last computed image embeddings
        self._last_computed_multi_vector_embeddings = None  # Store full multi-vector embeddings

    @property
    def has_embeddings(self):
        """Whether this instance can generate embeddings.
        
        Returns:
            bool: Always True for this model as embedding generation is supported
        """
        return True

    @property
    def can_embed_prompts(self):
        """Whether this instance can embed text prompts.
        
        Returns:
            bool: Always True for this model as text embedding is supported
        """
        return True

    def _load_model(self, config):
        """Load the model and processor from disk or HuggingFace.
        
        This method initializes both the processor (for tokenization and image
        preprocessing) and the model itself, configuring them for inference.

        Args:
            config: ColPaliConfig instance containing model parameters

        Returns:
            The loaded PyTorch model ready for inference
        """
        # Load the model from HuggingFace or local path
        model_path = config.model_path

        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self._device)
            
            # Use bfloat16 for Ampere or newer GPUs (capability >= 8.0)
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16
        
        # Initialize processor
        self.processor = ColPaliProcessor.from_pretrained(model_path)
        
        # Initialize model
        self.model = ColPaliModel.from_pretrained(
            model_path,
            **model_kwargs
        )

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        
        return self.model

    def _get_text_features(self):
        """Get or compute text features for the model's classification.
        
        This method caches the result for efficiency in repeated calls.
        Creates embeddings for each class by combining text_prompt with class names.
        
        Returns:
            torch.Tensor: Multi-vector text features for classification
        """
        # Check if text features are already computed and cached
        if self._text_features is None:
            # Create prompts for each class (following CLIP pattern)
            prompts = [
                "%s %s" % (self.config.text_prompt, c) for c in self.classes
            ]
            # Compute and cache the text features for all classes
            self._text_features = self._embed_prompts(prompts)
        
        # Return the cached features
        return self._text_features
    
    def _embed_prompts(self, prompts):
        """Embed text prompts for similarity search.
        
        Uses ColPali's multi-vector embedding approach where each prompt
        is represented by multiple embedding vectors.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            torch.Tensor: Multi-vector embeddings for the prompts with shape (batch, num_vectors, embedding_dim)
        """
        # Process text queries using ColPaliProcessor
        batch_queries = self.processor.process_queries(prompts).to(self.device)
        
        # Get query embeddings (multi-vector)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        
        # Return multi-vector embeddings
        return query_embeddings

    def embed_prompt(self, prompt):
        """Embed a single text prompt.
        
        Returns a flattened version of the multi-vector embedding for FiftyOne compatibility.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: Flattened embedding for the prompt
        """
        # Embed the single prompt
        embeddings = self._embed_prompts([prompt])
        
        # Flatten multi-vector to single vector for FiftyOne compatibility
        # Shape: (1, num_vectors, dim) -> (num_vectors * dim,)
        flattened = embeddings[0].flatten()
        
        # Return as CPU numpy array
        return flattened.detach().cpu().numpy()

    def embed_prompts(self, prompts):
        """Embed multiple text prompts.
        
        Returns flattened versions of the multi-vector embeddings for FiftyOne compatibility.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: Flattened embeddings for the prompts with shape (batch, num_vectors * dim)
        """
        # Embed prompts
        embeddings = self._embed_prompts(prompts)
        
        # Flatten multi-vector to single vector for each prompt
        # Shape: (batch, num_vectors, dim) -> (batch, num_vectors * dim)
        batch_size = embeddings.shape[0]
        flattened = embeddings.reshape(batch_size, -1)
        
        # Return as CPU numpy array
        return flattened.detach().cpu().numpy()

    def embed_images(self, imgs):
        """Embed a batch of images.
        
        Args:
            imgs: List of images to embed (PIL images)
            
        Returns:
            numpy array: Flattened embeddings for the images
        """
        # Process images using ColPaliProcessor
        batch_images = self.processor.process_images(imgs).to(self.device)
        
        # Get image embeddings (multi-vector)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        
        # Store the full multi-vector embeddings for scoring
        self._last_computed_multi_vector_embeddings = image_embeddings
        
        # Flatten multi-vector to single vector for FiftyOne compatibility
        # Shape: (batch, num_vectors, dim) -> (batch, num_vectors * dim)
        batch_size = image_embeddings.shape[0]
        flattened = image_embeddings.reshape(batch_size, -1)
        
        # Cache the flattened embeddings for get_embeddings() method
        self._last_computed_embeddings = flattened
        
        # Return as CPU numpy array for FiftyOne compatibility
        return flattened.detach().cpu().numpy()
    
    def embed(self, img):
        """Embed a single image.
        
        Implementation of TorchEmbeddingsMixin.embed() method.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: Flattened embedding for the image
        """
        # Convert single image to a list for batch processing
        imgs = [img]
        
        # Embed the single image using the batch method
        embeddings = self.embed_images(imgs)
        
        # Return the first (and only) embedding
        return embeddings[0]

    def embed_all(self, imgs):
        """Embed a batch of images.
        
        Implementation of TorchEmbeddingsMixin.embed_all() method.
        
        Args:
            imgs: List of images to embed (PIL images)
            
        Returns:
            numpy array: Flattened embeddings for the images
        """
        # Directly call embed_images which handles batch processing
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed embeddings.
        
        Required override for TorchEmbeddingsMixin to provide embeddings
        in the expected format for FiftyOne.
        
        Returns:
            numpy array: The last computed flattened embeddings
            
        Raises:
            ValueError: If no embeddings have been computed yet
        """
        # Check if embeddings capability is enabled
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        # Check if embeddings have been computed
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
            
        # Return the stored embeddings as a CPU numpy array
        return self._last_computed_embeddings.detach().cpu().numpy()

    def _get_class_logits(self, text_features, image_features):
        """Calculate multi-vector similarity scores between text and image features.
        
        Uses ColPali's multi-vector scoring approach similar to ColBERT's MaxSim operation.
        
        Args:
            text_features: Multi-vector text embeddings (torch.Tensor) with shape (num_classes, num_vectors, dim)
            image_features: Multi-vector image embeddings (torch.Tensor) with shape (num_images, num_vectors, dim)
            
        Returns:
            tuple: (logits_per_image, logits_per_text) following CLIP convention
                - logits_per_image: shape (num_images, num_classes)
                - logits_per_text: shape (num_classes, num_images)
        """
        with torch.no_grad():
            # Use ColPaliProcessor's scoring method for multi-vector similarity
            # Returns shape (num_classes, num_images)
            logits_per_text = self.processor.score_multi_vector(text_features, image_features)
            
            # Transpose to get (num_images, num_classes) for FiftyOne
            logits_per_image = logits_per_text.t()
            
            return logits_per_image, logits_per_text

    def _predict_all(self, imgs):
        """Run prediction on a batch of images.
        
        Used for zero-shot classification by comparing image embeddings
        to text embeddings of class names using multi-vector similarity.
        
        Args:
            imgs: List of images to classify
            
        Returns:
            numpy array: Similarity scores (logits)
        """
        # Get image embeddings (stores multi-vector embeddings internally)
        _ = self.embed_images(imgs)
        
        # Get multi-vector image embeddings
        image_features = self._last_computed_multi_vector_embeddings
        
        # Get multi-vector text embeddings for classes
        text_features = self._get_text_features()
        
        # Calculate multi-vector similarity (following CLIP pattern)
        output, _ = self._get_class_logits(text_features, image_features)
        
        # Get frame size for output processor
        if isinstance(imgs[0], torch.Tensor):
            height, width = imgs[0].size()[-2:]
        elif hasattr(imgs[0], 'size'):  # PIL Image
            width, height = imgs[0].size
        else:
            height, width = imgs[0].shape[:2]  # numpy array
        
        frame_size = (width, height)
        
        if self.has_logits:
            self._output_processor.store_logits = self.store_logits
        
        return self._output_processor(
            output, 
            frame_size, 
            confidence_thresh=self.config.confidence_thresh
        )
