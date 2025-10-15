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
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

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
        
        pool_factor (int): Token pooling compression factor. Default is 3 (optimal).
            Higher values = more compression, lower accuracy.
        
        pooling_strategy (str): Final pooling strategy after token pooling.
            Options: "mean" (default) or "max".
            - "mean": Average pooling, good for holistic document matching
            - "max": Max pooling, good for specific content/keyword matching
    """

    def __init__(self, d):
        """Initialize the configuration.

        Args:
            d: A dictionary containing the configuration parameters
        """
        # ColPaliProcessor handles all preprocessing, so we use raw inputs
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        
        super().__init__(d)
        
        # Path to model weights or HuggingFace model ID
        self.model_path = self.parse_string(d, "model_path", default="")
        
        # Optional base text prompt
        self.text_prompt = self.parse_string(d, "text_prompt", default="")
        
        # Token pooling configuration
        self.pool_factor = self.parse_int(d, "pool_factor", default=3)
        self.pooling_strategy = self.parse_string(
            d, "pooling_strategy", default="mean"
        )
        
        # Validate pooling strategy
        if self.pooling_strategy not in ["mean", "max"]:
            raise ValueError(
                f"pooling_strategy must be 'mean' or 'max', "
                f"got '{self.pooling_strategy}'"
            )


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
        
        # Initialize token pooler for compression
        self.token_pooler = HierarchicalTokenPooler()
        self.pool_factor = config.pool_factor
        self.pooling_strategy = config.pooling_strategy

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
    
    def _apply_final_pooling(self, embeddings):
        """Apply final pooling strategy to token-pooled embeddings.
        
        Reduces multi-vector embeddings to fixed-dimension vectors for FiftyOne compatibility.
        
        Args:
            embeddings: Token-pooled embeddings with shape (batch, reduced_vectors, dim)
            
        Returns:
            torch.Tensor: Fixed-dimension pooled embeddings with shape (batch, dim)
        """
        if self.pooling_strategy == "mean":
            # Average across all vectors
            pooled = embeddings.mean(dim=1)  # (batch, dim)
            print(f"[DEBUG _apply_final_pooling] Mean pooling: {embeddings.shape} → {pooled.shape}")
            return pooled
        elif self.pooling_strategy == "max":
            # Take maximum across all vectors
            pooled = embeddings.max(dim=1)[0]  # (batch, dim)
            print(f"[DEBUG _apply_final_pooling] Max pooling: {embeddings.shape} → {pooled.shape}")
            return pooled
        else:
            raise ValueError(f"Unknown pooling_strategy: {self.pooling_strategy}")

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
        """Embed a single text prompt with token pooling.
        
        Uses token pooling to reduce sequence length while retaining ~98% performance.
        
        Args:
            prompt: Text prompt to embed
            
        Returns:
            numpy array: Token-pooled embedding with shape (reduced_num_vectors, dim)
        """
        print(f"[DEBUG embed_prompt] Embedding prompt: {prompt[:50]}...")
        
        # Embed the single prompt
        embeddings = self._embed_prompts([prompt])
        print(f"[DEBUG embed_prompt] Raw multi-vector shape: {embeddings.shape}")
        
        # Apply token pooling to reduce sequence length
        pooled_embeddings = self.token_pooler.pool_embeddings(
            embeddings,
            pool_factor=self.pool_factor,
            padding=True,
            padding_side=self.processor.tokenizer.padding_side,
        )
        print(f"[DEBUG embed_prompt] After token pooling (factor={self.pool_factor}): {pooled_embeddings.shape}")
        
        # Apply final pooling strategy (always produces fixed dimension)
        final_embeddings = self._apply_final_pooling(pooled_embeddings)
        
        # Return first (and only) embedding: (1, dim) -> (dim,)
        result = final_embeddings[0].detach().cpu().numpy()
        print(f"[DEBUG embed_prompt] Final shape: {result.shape}")
        return result

    def embed_prompts(self, prompts):
        """Embed multiple text prompts with token pooling.
        
        Uses token pooling to reduce sequence length while retaining ~98% performance.
        
        Args:
            prompts: List of text prompts to embed
            
        Returns:
            numpy array: Token-pooled embeddings with shape (batch, reduced_num_vectors, dim)
        """
        print(f"[DEBUG embed_prompts] Embedding {len(prompts)} prompts")
        
        # Embed prompts
        embeddings = self._embed_prompts(prompts)
        print(f"[DEBUG embed_prompts] Raw multi-vector shape: {embeddings.shape}")
        
        # Apply token pooling to reduce sequence length
        pooled_embeddings = self.token_pooler.pool_embeddings(
            embeddings,
            pool_factor=self.pool_factor,
            padding=True,
            padding_side=self.processor.tokenizer.padding_side,
        )
        print(f"[DEBUG embed_prompts] After token pooling (factor={self.pool_factor}): {pooled_embeddings.shape}")
        
        # Apply final pooling strategy
        final_embeddings = self._apply_final_pooling(pooled_embeddings)
        
        # Return as numpy array
        result = final_embeddings.detach().cpu().numpy()
        print(f"[DEBUG embed_prompts] Final shape: {result.shape}")
        return result

    def embed_images(self, imgs):
        """Embed a batch of images.
        
        With raw_inputs=True, FiftyOne passes images in their original format
        (PIL, numpy array, or tensor). ColPaliProcessor requires PIL Images.
        
        Returns raw multi-vector embeddings.
        
        Args:
            imgs: List of images to embed (PIL images, numpy arrays (HWC), or tensors (CHW))
            
        Returns:
            numpy array: Multi-vector embeddings for the images with shape (batch, num_vectors, dim)
        """
        print(f"[DEBUG embed_images] Embedding {len(imgs)} images")
        # Convert to PIL Images if needed (ColPaliProcessor requirement)
        pil_images = []
        for img in imgs:
            if isinstance(img, Image.Image):
                # Already PIL Image
                pil_images.append(img)
            elif isinstance(img, torch.Tensor):
                # Raw tensor (CHW, uint8) → PIL Image
                img_np = img.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype != np.uint8:
                    img_np = img_np.astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                # Numpy array (HWC, uint8) → PIL Image
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        # Process images using ColPaliProcessor
        batch_images = self.processor.process_images(pil_images).to(self.device)
        
        # Get image embeddings (multi-vector)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        
        print(f"[DEBUG embed_images] Raw multi-vector shape: {image_embeddings.shape}")
        
        # Store the full multi-vector embeddings for classification scoring (before pooling)
        self._last_computed_multi_vector_embeddings = image_embeddings
        
        # Apply token pooling to reduce sequence length
        pooled_embeddings = self.token_pooler.pool_embeddings(
            image_embeddings,
            pool_factor=self.pool_factor,
            padding=True,
            padding_side=self.processor.tokenizer.padding_side,
        )
        print(f"[DEBUG embed_images] After token pooling (factor={self.pool_factor}): {pooled_embeddings.shape}")
        
        # Apply final pooling strategy
        final_embeddings = self._apply_final_pooling(pooled_embeddings)
        
        # Cache final embeddings for get_embeddings() method
        self._last_computed_embeddings = final_embeddings
        
        # Return as numpy array
        result = final_embeddings.detach().cpu().numpy()
        print(f"[DEBUG embed_images] Final shape: {result.shape}")
        return result
    
    def embed(self, img):
        """Embed a single image.
        
        Implementation of TorchEmbeddingsMixin.embed() method.
        
        Args:
            img: PIL image to embed
            
        Returns:
            numpy array: Multi-vector embedding for the image with shape (num_vectors, dim)
        """
        print("[DEBUG embed] Embedding single image")
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
            numpy array: Multi-vector embeddings for the images with shape (batch, num_vectors, dim)
        """
        print(f"[DEBUG embed_all] Embedding {len(imgs)} images")
        # Directly call embed_images which handles batch processing
        return self.embed_images(imgs)
    
    def get_embeddings(self):
        """Get the last computed embeddings.
        
        Required override for TorchEmbeddingsMixin to provide embeddings
        in the expected format for FiftyOne.
        
        Returns:
            numpy array: The last computed multi-vector embeddings
            
        Raises:
            ValueError: If no embeddings have been computed yet
        """
        print("[DEBUG get_embeddings] Called")
        
        # Check if embeddings capability is enabled
        if not self.has_embeddings:
            raise ValueError("This model instance does not expose embeddings")
        
        # Check if embeddings have been computed
        if self._last_computed_embeddings is None:
            raise ValueError("No embeddings have been computed yet")
        
        print(f"[DEBUG get_embeddings] Cached embeddings shape: {self._last_computed_embeddings.shape}")
            
        # Return the stored embeddings as a CPU numpy array
        result = self._last_computed_embeddings.detach().cpu().numpy()
        print(f"[DEBUG get_embeddings] Returning shape: {result.shape}")
        return result

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
            logits_per_text = self.processor.score_single_vector(text_features, image_features)
            
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
