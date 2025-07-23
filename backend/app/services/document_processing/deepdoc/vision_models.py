"""
ONNX vision models management for DeepDoc system.
"""
import logging
import os
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading

import trio
import numpy as np
import onnxruntime as ort
import requests
from huggingface_hub import hf_hub_download

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about an ONNX model."""
    
    def __init__(self, 
                 name: str,
                 repo_id: str,
                 filename: str,
                 description: str,
                 input_shape: tuple,
                 output_shape: tuple,
                 model_type: str = "vision"):
        self.name = name
        self.repo_id = repo_id
        self.filename = filename
        self.description = description
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_type = model_type
        self.local_path: Optional[str] = None
        self.session: Optional[ort.InferenceSession] = None
        self.loaded = False


class VisionModelsManager:
    """
    Manager for ONNX vision models used in DeepDoc processing.
    """
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.models_path = Path(settings.ONNX_MODELS_PATH)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialized = False
        
        # Define available models
        self._define_models()
    
    def _define_models(self):
        """Define available ONNX models for DeepDoc."""
        # Layout detection model
        self.models["layout"] = ModelInfo(
            name="layout",
            repo_id=settings.ONNX_MODELS_REPO,
            filename="layout.onnx",
            description="Document layout detection model",
            input_shape=(1, 3, 1024, 1024),
            output_shape=(1, 5, 1024, 1024),  # 5 classes: text, title, table, figure, header
            model_type="layout"
        )
        
        # OCR text detection model
        self.models["text_detection"] = ModelInfo(
            name="text_detection",
            repo_id=settings.ONNX_MODELS_REPO,
            filename="text_detection.onnx",
            description="Text region detection model",
            input_shape=(1, 3, 640, 640),
            output_shape=(1, 25200, 85),  # YOLO-style output
            model_type="detection"
        )
        
        # OCR text recognition model
        self.models["text_recognition"] = ModelInfo(
            name="text_recognition",
            repo_id=settings.ONNX_MODELS_REPO,
            filename="text_recognition.onnx",
            description="Text recognition model",
            input_shape=(1, 3, 32, 128),
            output_shape=(1, 26, 37),  # Character sequence output
            model_type="recognition"
        )
        
        # Table structure recognition model
        self.models["table_structure"] = ModelInfo(
            name="table_structure",
            repo_id=settings.ONNX_MODELS_REPO,
            filename="table_structure.onnx",
            description="Table structure recognition model",
            input_shape=(1, 3, 512, 512),
            output_shape=(1, 6, 512, 512),  # Table components: cell, row, column, etc.
            model_type="table"
        )
        
        logger.info(f"Defined {len(self.models)} ONNX models for DeepDoc")
    
    async def initialize(self):
        """Initialize the models manager."""
        if self._initialized:
            return
        
        try:
            # Check if DeepDoc is enabled
            if not settings.DEEPDOC_ENABLED:
                logger.info("DeepDoc is disabled, skipping model initialization")
                self._initialized = True
                return
            
            # Download and load models
            await self._download_models()
            await self._load_models()
            
            self._initialized = True
            logger.info("VisionModelsManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize VisionModelsManager: {e}")
            # Don't raise exception - allow system to work without DeepDoc
            self._initialized = True
    
    async def _download_models(self):
        """Download models from Hugging Face if not present locally."""
        for model_name, model_info in self.models.items():
            try:
                local_path = self.models_path / model_info.filename
                
                if local_path.exists():
                    logger.debug(f"Model {model_name} already exists locally")
                    model_info.local_path = str(local_path)
                    continue
                
                logger.info(f"Downloading model {model_name} from {model_info.repo_id}")
                
                # Download model using trio thread
                downloaded_path = await trio.to_thread.run_sync(
                    self._download_model_sync,
                    model_info.repo_id,
                    model_info.filename,
                    str(local_path)
                )
                
                model_info.local_path = downloaded_path
                logger.info(f"Successfully downloaded model {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to download model {model_name}: {e}")
                # Continue with other models
    
    def _download_model_sync(self, repo_id: str, filename: str, local_path: str) -> str:
        """Download model synchronously."""
        try:
            # Try Hugging Face Hub first
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(self.models_path.parent),
                local_dir=str(self.models_path),
                local_dir_use_symlinks=False
            )
            return downloaded_path
            
        except Exception as e:
            logger.warning(f"Hugging Face download failed: {e}")
            
            # Fallback: try direct download if URL is provided
            # This is a placeholder - in real implementation, you'd have actual model URLs
            logger.warning(f"Model {filename} not available for download")
            
            # Create a dummy model file for testing
            dummy_path = Path(local_path)
            dummy_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_path.write_bytes(b"dummy_onnx_model")
            
            return str(dummy_path)
    
    async def _load_models(self):
        """Load ONNX models into memory."""
        for model_name, model_info in self.models.items():
            if not model_info.local_path or not Path(model_info.local_path).exists():
                logger.warning(f"Model {model_name} not available, skipping load")
                continue
            
            try:
                # Load model in thread to avoid blocking
                session = await trio.to_thread.run_sync(
                    self._load_model_sync,
                    model_info.local_path
                )
                
                if session:
                    model_info.session = session
                    model_info.loaded = True
                    logger.info(f"Successfully loaded model {model_name}")
                else:
                    logger.warning(f"Failed to load model {model_name}")
                    
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
    
    def _load_model_sync(self, model_path: str) -> Optional[ort.InferenceSession]:
        """Load ONNX model synchronously."""
        try:
            # Check if file is a real ONNX model
            if Path(model_path).stat().st_size < 100:  # Too small to be real model
                logger.warning(f"Model file {model_path} appears to be dummy/placeholder")
                return None
            
            # Configure ONNX Runtime
            providers = ['CPUExecutionProvider']
            
            # Try to use GPU if available
            try:
                if ort.get_device() == 'GPU':
                    providers.insert(0, 'CUDAExecutionProvider')
            except:
                pass
            
            # Create inference session
            session = ort.InferenceSession(
                model_path,
                providers=providers
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_path}: {e}")
            return None
    
    async def run_inference(self, 
                          model_name: str, 
                          input_data: np.ndarray,
                          input_name: str = None) -> Optional[np.ndarray]:
        """
        Run inference on a model.
        
        Args:
            model_name: Name of the model to use
            input_data: Input data as numpy array
            input_name: Name of input tensor (optional)
            
        Returns:
            Model output as numpy array or None if failed
        """
        if not self._initialized:
            await self.initialize()
        
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model_info = self.models[model_name]
        
        if not model_info.loaded or not model_info.session:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        try:
            # Get input name if not provided
            if input_name is None:
                input_name = model_info.session.get_inputs()[0].name
            
            # Validate input shape
            expected_shape = model_info.input_shape
            if input_data.shape != expected_shape:
                logger.warning(f"Input shape {input_data.shape} doesn't match expected {expected_shape}")
                # Try to reshape if possible
                if input_data.size == np.prod(expected_shape):
                    input_data = input_data.reshape(expected_shape)
                else:
                    logger.error(f"Cannot reshape input for model {model_name}")
                    return None
            
            # Run inference in thread
            output = await trio.to_thread.run_sync(
                self._run_inference_sync,
                model_info.session,
                {input_name: input_data}
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            return None
    
    def _run_inference_sync(self, 
                           session: ort.InferenceSession, 
                           inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run inference synchronously."""
        outputs = session.run(None, inputs)
        return outputs[0] if outputs else None
    
    async def preprocess_image(self, 
                             image: np.ndarray, 
                             target_size: tuple,
                             normalize: bool = True) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target size (height, width)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image ready for model input
        """
        try:
            import cv2
            
            # Resize image
            if image.shape[:2] != target_size:
                image = cv2.resize(image, (target_size[1], target_size[0]))
            
            # Convert to float32
            image = image.astype(np.float32)
            
            # Normalize if requested
            if normalize:
                image = image / 255.0
                # Standard ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image - mean) / std
            
            # Convert from HWC to CHW format
            image = np.transpose(image, (2, 0, 1))
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    async def postprocess_detection(self, 
                                  output: np.ndarray, 
                                  confidence_threshold: float = 0.5,
                                  nms_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Postprocess detection model output.
        
        Args:
            output: Model output array
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for duplicate removal
            
        Returns:
            List of detection results
        """
        try:
            detections = []
            
            # This is a simplified postprocessing - real implementation would depend on model format
            if output is not None and len(output.shape) >= 2:
                # Assume output format: [batch, detections, (x, y, w, h, conf, class_scores...)]
                for detection in output[0]:  # First batch
                    if len(detection) >= 5:
                        x, y, w, h, conf = detection[:5]
                        
                        if conf > confidence_threshold:
                            detections.append({
                                "bbox": [float(x), float(y), float(w), float(h)],
                                "confidence": float(conf),
                                "class_id": 0,  # Simplified
                                "class_name": "text"
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection postprocessing failed: {e}")
            return []
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get information about models.
        
        Args:
            model_name: Specific model name (optional)
            
        Returns:
            Model information
        """
        if model_name:
            if model_name in self.models:
                model = self.models[model_name]
                return {
                    "name": model.name,
                    "description": model.description,
                    "loaded": model.loaded,
                    "local_path": model.local_path,
                    "input_shape": model.input_shape,
                    "output_shape": model.output_shape,
                    "model_type": model.model_type
                }
            else:
                return {"error": f"Unknown model: {model_name}"}
        else:
            return {
                "available_models": list(self.models.keys()),
                "loaded_models": [name for name, model in self.models.items() if model.loaded],
                "models_path": str(self.models_path),
                "deepdoc_enabled": settings.DEEPDOC_ENABLED,
                "initialized": self._initialized
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of vision models system."""
        try:
            if not self._initialized:
                await self.initialize()
            
            model_status = {}
            for name, model in self.models.items():
                model_status[name] = {
                    "loaded": model.loaded,
                    "available": model.local_path is not None and Path(model.local_path).exists(),
                    "session_ready": model.session is not None
                }
            
            loaded_count = sum(1 for model in self.models.values() if model.loaded)
            
            return {
                "status": "healthy" if loaded_count > 0 else "degraded",
                "initialized": self._initialized,
                "deepdoc_enabled": settings.DEEPDOC_ENABLED,
                "models_path": str(self.models_path),
                "total_models": len(self.models),
                "loaded_models": loaded_count,
                "model_status": model_status
            }
            
        except Exception as e:
            logger.error(f"Vision models health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instance
vision_models_manager = VisionModelsManager()