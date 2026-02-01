"""
Audio handling utilities for ground truth evaluation.

Supports loading, encoding, and validating audio files
for ASR and TTS evaluation.

Provides integration with:
- OpenAI Whisper for ASR transcription
- UTMOS for neural MOS prediction
- Speaker embeddings for voice similarity
"""

import os
import base64
import struct
import wave
import logging
import tempfile
from typing import Optional, Tuple, BinaryIO, Any, Union

logger = logging.getLogger(__name__)


def load_audio(path: str) -> Optional[bytes]:
    """
    Load audio file and return bytes.
    
    Args:
        path: Path to audio file
    
    Returns:
        Audio bytes or None
    """
    if not os.path.exists(path):
        return None
    
    with open(path, "rb") as f:
        return f.read()


def get_audio_duration(audio: bytes, format: str = "wav") -> float:
    """
    Get audio duration in seconds.
    
    Args:
        audio: Audio bytes
        format: Audio format (wav, mp3, etc.)
    
    Returns:
        Duration in seconds
    """
    if format.lower() == "wav":
        return _get_wav_duration(audio)
    
    # For other formats, estimate based on file size
    # This is a rough estimate - real implementation would use pydub/ffmpeg
    return len(audio) / 16000  # Assume 16kHz mono


def _get_wav_duration(audio: bytes) -> float:
    """Get duration of WAV audio."""
    try:
        # Parse WAV header
        if len(audio) < 44:
            return 0.0
        
        # RIFF header
        if audio[:4] != b'RIFF':
            return 0.0
        
        # Find fmt chunk
        fmt_offset = audio.find(b'fmt ')
        if fmt_offset < 0:
            return 0.0
        
        # Read format info
        fmt_offset += 4
        chunk_size = struct.unpack('<I', audio[fmt_offset:fmt_offset+4])[0]
        fmt_offset += 4
        
        # Audio format, channels, sample rate, byte rate, block align, bits per sample
        audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
            '<HHIIHH', audio[fmt_offset:fmt_offset+16]
        )
        
        # Find data chunk
        data_offset = audio.find(b'data', fmt_offset)
        if data_offset < 0:
            return 0.0
        
        data_offset += 4
        data_size = struct.unpack('<I', audio[data_offset:data_offset+4])[0]
        
        # Calculate duration
        if byte_rate > 0:
            return data_size / byte_rate
        
        return 0.0
        
    except Exception:
        return 0.0


def encode_audio_base64(audio: bytes) -> str:
    """
    Encode audio as base64 string.
    
    Args:
        audio: Audio bytes
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(audio).decode('utf-8')


def decode_audio_base64(encoded: str) -> bytes:
    """
    Decode base64 audio string.
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Audio bytes
    """
    return base64.b64decode(encoded)


def validate_audio_format(audio: bytes) -> Tuple[bool, str]:
    """
    Validate audio format and return format type.
    
    Args:
        audio: Audio bytes
    
    Returns:
        Tuple of (is_valid, format_name)
    """
    if len(audio) < 12:
        return False, "unknown"
    
    # Check for WAV
    if audio[:4] == b'RIFF' and audio[8:12] == b'WAVE':
        return True, "wav"
    
    # Check for MP3 (ID3 or frame sync)
    if audio[:3] == b'ID3' or (audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0):
        return True, "mp3"
    
    # Check for FLAC
    if audio[:4] == b'fLaC':
        return True, "flac"
    
    # Check for OGG
    if audio[:4] == b'OggS':
        return True, "ogg"
    
    return False, "unknown"


def get_audio_info(audio: bytes) -> dict:
    """
    Get audio file information.
    
    Args:
        audio: Audio bytes
    
    Returns:
        Dict with audio info
    """
    is_valid, format_name = validate_audio_format(audio)
    duration = get_audio_duration(audio, format_name) if is_valid else 0.0
    
    return {
        "valid": is_valid,
        "format": format_name,
        "duration_seconds": duration,
        "size_bytes": len(audio),
    }


# =============================================================================
# Whisper ASR Integration
# =============================================================================

class WhisperASR:
    """
    Whisper-based ASR for round-trip TTS evaluation.
    
    Supports multiple backends:
    1. OpenAI Whisper API (via LiteLLM)
    2. Local Whisper model
    3. faster-whisper for improved performance
    """
    
    def __init__(
        self,
        model: str = "whisper-1",
        use_api: bool = True,
        llm_caller: Any = None,
        local_model_size: str = "base"
    ):
        """
        Initialize Whisper ASR.
        
        Args:
            model: Model name for API or local model size
            use_api: Whether to use OpenAI API (via llm_caller)
            llm_caller: LiteLLM instance for API calls
            local_model_size: Local model size (tiny, base, small, medium, large)
        """
        self.model = model
        self.use_api = use_api
        self.llm_caller = llm_caller
        self.local_model_size = local_model_size
        self._local_model = None
    
    async def transcribe(
        self,
        audio: Union[bytes, str],
        language: str = None
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio bytes or path to audio file
            language: Language hint (optional)
        
        Returns:
            Transcribed text
        """
        # Handle path input
        if isinstance(audio, str):
            audio = load_audio(audio)
            if audio is None:
                raise ValueError(f"Could not load audio from path")
        
        if self.use_api and self.llm_caller:
            return await self._transcribe_api(audio, language)
        else:
            return self._transcribe_local(audio, language)
    
    async def _transcribe_api(
        self,
        audio: bytes,
        language: str = None
    ) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            # Write audio to temp file for API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                temp_path = f.name
            
            try:
                # Use LiteLLM transcription if available
                if hasattr(self.llm_caller, 'atranscription'):
                    result = await self.llm_caller.atranscription(
                        model=self.model,
                        file=open(temp_path, "rb"),
                        language=language
                    )
                elif hasattr(self.llm_caller, 'transcription'):
                    result = self.llm_caller.transcription(
                        model=self.model,
                        file=open(temp_path, "rb"),
                        language=language
                    )
                else:
                    # Fallback to openai client directly
                    import openai
                    client = openai.OpenAI()
                    with open(temp_path, "rb") as audio_file:
                        result = client.audio.transcriptions.create(
                            model=self.model,
                            file=audio_file,
                            language=language
                        )
                    return result.text
                
                return result.text if hasattr(result, 'text') else str(result)
                
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Whisper API transcription failed: {e}")
            raise
    
    def _transcribe_local(
        self,
        audio: bytes,
        language: str = None
    ) -> str:
        """Transcribe using local Whisper model."""
        try:
            model = self._get_local_model()
            
            # Write audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                temp_path = f.name
            
            try:
                result = model.transcribe(temp_path, language=language)
                return result.get("text", "")
            finally:
                os.unlink(temp_path)
                
        except ImportError:
            logger.error("Local Whisper not available. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Local Whisper transcription failed: {e}")
            raise
    
    def _get_local_model(self):
        """Lazy-load local Whisper model."""
        if self._local_model is None:
            try:
                import whisper
                self._local_model = whisper.load_model(self.local_model_size)
            except ImportError:
                # Try faster-whisper
                try:
                    from faster_whisper import WhisperModel
                    self._local_model = WhisperModel(self.local_model_size)
                except ImportError:
                    raise ImportError(
                        "No Whisper implementation found. Install with:\n"
                        "  pip install openai-whisper  # or\n"
                        "  pip install faster-whisper"
                    )
        return self._local_model


# =============================================================================
# UTMOS Neural MOS Prediction
# =============================================================================

class UTMOSPredictor:
    """
    UTMOS neural MOS prediction for TTS quality assessment.
    
    Predicts Mean Opinion Score (MOS) for audio quality on 1-5 scale.
    Uses the UTMOS model from the SpeechMOS library.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize UTMOS predictor.
        
        Args:
            device: Device to run model on (cpu, cuda)
        """
        self.device = device
        self._model = None
        self._sample_rate = 16000
    
    async def predict(self, audio: Union[bytes, str]) -> float:
        """
        Predict MOS score for audio.
        
        Args:
            audio: Audio bytes or path
        
        Returns:
            MOS score (1.0-5.0)
        """
        return self.predict_sync(audio)
    
    def predict_sync(self, audio: Union[bytes, str]) -> float:
        """Synchronous MOS prediction."""
        try:
            model = self._get_model()
            
            # Handle path input
            if isinstance(audio, str):
                audio_bytes = load_audio(audio)
                if audio_bytes is None:
                    return 3.0
            else:
                audio_bytes = audio
            
            # Write to temp file for model
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            try:
                import torch
                import torchaudio
                
                # Load audio
                waveform, sr = torchaudio.load(temp_path)
                
                # Resample if needed
                if sr != self._sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self._sample_rate)
                    waveform = resampler(waveform)
                
                # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Get prediction
                with torch.no_grad():
                    score = model(waveform.to(self.device))
                
                return float(score.mean().item())
                
            finally:
                os.unlink(temp_path)
                
        except ImportError as e:
            logger.warning(f"UTMOS dependencies not available: {e}")
            return 3.0  # Neutral score
        except Exception as e:
            logger.error(f"UTMOS prediction failed: {e}")
            return 3.0
    
    def _get_model(self):
        """Lazy-load UTMOS model."""
        if self._model is None:
            try:
                # Try speechmos library first
                from speechmos import UTMOS
                self._model = UTMOS(device=self.device)
            except ImportError:
                try:
                    # Fallback to direct torch hub
                    import torch
                    self._model = torch.hub.load(
                        "tarepan/SpeechMOS:v1.2.0",
                        "utmos22_strong",
                        trust_repo=True
                    ).to(self.device)
                except Exception as e:
                    logger.warning(f"Could not load UTMOS model: {e}")
                    # Return a dummy model that returns neutral score
                    class DummyUTMOS:
                        def __call__(self, *args, **kwargs):
                            import torch
                            return torch.tensor([3.0])
                    self._model = DummyUTMOS()
        return self._model


# =============================================================================
# Speaker Embedding Comparison
# =============================================================================

class SpeakerEncoder:
    """
    Speaker embedding encoder for voice similarity comparison.
    
    Uses speech embedding models to compare voice characteristics
    between generated TTS audio and reference audio.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize speaker encoder.
        
        Args:
            device: Device to run model on
        """
        self.device = device
        self._model = None
    
    async def encode(self, audio: Union[bytes, str]) -> "numpy.ndarray":
        """
        Encode audio to speaker embedding.
        
        Args:
            audio: Audio bytes or path
        
        Returns:
            Speaker embedding vector
        """
        return self.encode_sync(audio)
    
    def encode_sync(self, audio: Union[bytes, str]) -> "numpy.ndarray":
        """Synchronous speaker encoding."""
        import numpy as np
        
        try:
            model = self._get_model()
            
            if isinstance(audio, str):
                audio_bytes = load_audio(audio)
            else:
                audio_bytes = audio
            
            if audio_bytes is None:
                return np.zeros(256)
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            
            try:
                embedding = model.embed_utterance_from_file(temp_path)
                return embedding
            finally:
                os.unlink(temp_path)
                
        except ImportError:
            logger.warning("Speaker encoder dependencies not available")
            return np.zeros(256)
        except Exception as e:
            logger.error(f"Speaker encoding failed: {e}")
            return np.zeros(256)
    
    def compute_similarity(
        self,
        embedding1: "numpy.ndarray",
        embedding2: "numpy.ndarray"
    ) -> float:
        """
        Compute cosine similarity between speaker embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
        
        Returns:
            Similarity score (0-1)
        """
        import numpy as np
        
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))
    
    def _get_model(self):
        """Lazy-load speaker encoder model."""
        if self._model is None:
            try:
                # Try resemblyzer (lightweight)
                from resemblyzer import VoiceEncoder
                self._model = VoiceEncoder(device=self.device)
            except ImportError:
                try:
                    # Try speechbrain
                    from speechbrain.pretrained import EncoderClassifier
                    self._model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="pretrained_models/spkrec-ecapa-voxceleb"
                    )
                except ImportError:
                    logger.warning(
                        "No speaker encoder available. Install with:\n"
                        "  pip install resemblyzer  # or\n"
                        "  pip install speechbrain"
                    )
                    # Return dummy encoder
                    class DummyEncoder:
                        def embed_utterance_from_file(self, path):
                            import numpy as np
                            return np.zeros(256)
                    self._model = DummyEncoder()
        return self._model


# =============================================================================
# SNR Computation
# =============================================================================

def compute_snr(audio: bytes) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) for audio.
    
    Uses a simple energy-based approach to estimate SNR.
    
    Args:
        audio: Audio bytes
    
    Returns:
        SNR in dB (higher is better)
    """
    try:
        import numpy as np
        
        # Write to temp file for analysis
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            temp_path = f.name
        
        try:
            # Try torchaudio first
            try:
                import torchaudio
                waveform, sr = torchaudio.load(temp_path)
                samples = waveform.numpy().flatten()
            except ImportError:
                # Fallback to scipy
                from scipy.io import wavfile
                sr, samples = wavfile.read(temp_path)
                samples = samples.astype(np.float32) / 32768.0
            
            # Simple energy-based SNR estimation
            # Assume signal is in louder segments, noise in quieter
            frame_size = int(sr * 0.025)  # 25ms frames
            hop_size = int(sr * 0.010)    # 10ms hop
            
            energies = []
            for i in range(0, len(samples) - frame_size, hop_size):
                frame = samples[i:i + frame_size]
                energy = np.mean(frame ** 2)
                energies.append(energy)
            
            if not energies:
                return 20.0
            
            energies = np.array(energies)
            
            # Estimate signal as mean of top 50% energies
            # Estimate noise as mean of bottom 20% energies
            sorted_energies = np.sort(energies)
            
            signal_energy = np.mean(sorted_energies[int(len(sorted_energies) * 0.5):])
            noise_energy = np.mean(sorted_energies[:int(len(sorted_energies) * 0.2)]) + 1e-10
            
            snr_db = 10 * np.log10(signal_energy / noise_energy)
            
            return float(max(0, min(60, snr_db)))
            
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logger.warning(f"SNR computation failed: {e}")
        return 20.0  # Default moderate SNR

