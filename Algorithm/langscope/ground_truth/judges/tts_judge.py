"""
TTS (Text-to-Speech) Ground Truth Judge.

Evaluates TTS output quality using:
- Round-trip WER (TTS → ASR → compare)
- UTMOS/DNSMOS (neural MOS prediction)
- Speaker similarity
- Composite scoring

Integrates with:
- Whisper for ASR transcription
- UTMOS for neural MOS prediction
- Speaker embedding models for voice similarity
"""

from typing import Dict, List, Optional, Any
import logging

from langscope.ground_truth.judge import GroundTruthJudge, GroundTruthScore, EvaluationMode
from langscope.ground_truth.metrics import (
    word_error_rate,
    compute_tts_composite_score,
)
from langscope.ground_truth.utils.audio import (
    WhisperASR,
    UTMOSPredictor,
    SpeakerEncoder,
    compute_snr,
)

logger = logging.getLogger(__name__)


class TTSGroundTruthJudge(GroundTruthJudge):
    """
    Specialized judge for TTS evaluation.
    
    Uses a combination of:
    1. Round-trip WER: TTS audio transcribed back to text, compared to input
    2. UTMOS: Neural MOS prediction for naturalness
    3. SNR: Signal-to-noise ratio for audio quality
    4. Speaker similarity: How well voice matches reference
    
    Supports multiple ASR backends:
    - OpenAI Whisper API (via LiteLLM)
    - Local Whisper model
    - faster-whisper
    """
    
    def __init__(
        self,
        asr_caller: Any = None,
        utmos_model: Any = None,
        speaker_encoder: Any = None,
        llm_caller: Any = None,
        use_whisper_api: bool = True,
        whisper_model: str = "whisper-1",
        local_whisper_size: str = "base",
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize TTS judge.
        
        Args:
            asr_caller: ASR model/API for transcription (legacy, use llm_caller)
            utmos_model: UTMOS model for MOS prediction (legacy, auto-created)
            speaker_encoder: Model for speaker embeddings (legacy, auto-created)
            llm_caller: LiteLLM instance for API calls
            use_whisper_api: Whether to use Whisper API vs local model
            whisper_model: Whisper model name for API
            local_whisper_size: Local whisper model size
            device: Device for local models (cpu, cuda)
        """
        super().__init__(domain="tts", **kwargs)
        
        # Use provided caller or create new with llm_caller
        if asr_caller is not None:
            self.asr_caller = asr_caller
        else:
            self.asr_caller = WhisperASR(
                model=whisper_model,
                use_api=use_whisper_api,
                llm_caller=llm_caller or self.llm_caller,
                local_model_size=local_whisper_size
            )
        
        # UTMOS model
        if utmos_model is not None:
            self.utmos_model = utmos_model
        else:
            self.utmos_model = UTMOSPredictor(device=device)
        
        # Speaker encoder
        if speaker_encoder is not None:
            self.speaker_encoder = speaker_encoder
        else:
            self.speaker_encoder = SpeakerEncoder(device=device)
    
    async def _compute_metrics_async(
        self,
        audio_bytes: bytes,
        input_text: str,
        sample: Dict
    ) -> Dict[str, float]:
        """
        Compute TTS metrics asynchronously.
        
        Args:
            audio_bytes: Generated audio
            input_text: Original text that was synthesized
            sample: Sample metadata including reference audio
        
        Returns:
            Dict of metric values
        """
        metrics = {}
        
        # 1. Round-trip WER
        round_trip_wer = await self._compute_round_trip_wer(audio_bytes, input_text)
        metrics["round_trip_wer"] = round_trip_wer
        
        # 2. UTMOS (Neural MOS prediction)
        utmos = await self._compute_utmos(audio_bytes)
        metrics["utmos"] = utmos
        
        # 3. SNR
        snr = self._compute_snr(audio_bytes)
        metrics["snr"] = snr
        
        # 4. Speaker similarity (if reference provided)
        ref_audio = sample.get("reference_audio_bytes")
        if ref_audio:
            speaker_sim = await self._compute_speaker_similarity(audio_bytes, ref_audio)
            metrics["speaker_similarity"] = speaker_sim
        else:
            metrics["speaker_similarity"] = None
        
        # 5. Composite score
        metrics["composite_tts"] = compute_tts_composite_score(
            round_trip_wer=round_trip_wer,
            utmos=utmos,
            snr=snr,
            speaker_sim=metrics.get("speaker_similarity"),
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        response: Any,
        ground_truth: Any,
        sample: Dict
    ) -> Dict[str, float]:
        """
        Synchronous wrapper - returns placeholder metrics.
        For TTS, prefer using evaluate_audio() which is async.
        """
        # If response is already metrics dict (from async path)
        if isinstance(response, dict) and "round_trip_wer" in response:
            return response
        
        # Otherwise return placeholder
        return {
            "round_trip_wer": 0.5,
            "utmos": 3.0,
            "snr": 20.0,
            "composite_tts": 0.5,
        }
    
    async def evaluate_audio(
        self,
        audio_bytes: bytes,
        input_text: str,
        sample: Dict,
        model_id: str = ""
    ) -> GroundTruthScore:
        """
        Evaluate TTS audio output.
        
        Args:
            audio_bytes: Generated TTS audio
            input_text: Original text that was synthesized
            sample: Sample metadata
            model_id: ID of the model being evaluated
        
        Returns:
            GroundTruthScore with TTS metrics
        """
        sample_id = sample.get("sample_id", "")
        
        try:
            metrics = await self._compute_metrics_async(
                audio_bytes, input_text, sample
            )
            
            return GroundTruthScore(
                model_id=model_id,
                sample_id=sample_id,
                metrics=metrics,
                overall=metrics.get("composite_tts", 0.0),
                evaluation_mode="hybrid",  # TTS uses round-trip ASR
            )
        except Exception as e:
            logger.error(f"TTS evaluation failed: {e}")
            return GroundTruthScore(
                model_id=model_id,
                sample_id=sample_id,
                metrics={},
                overall=0.0,
                error=str(e),
            )
    
    async def _compute_round_trip_wer(
        self,
        audio_bytes: bytes,
        original_text: str
    ) -> float:
        """
        Transcribe audio and compute WER against original text.
        
        Uses Whisper ASR (API or local) for transcription.
        """
        if not self.asr_caller:
            logger.warning("No ASR caller configured, returning placeholder WER")
            return 0.3  # Placeholder
        
        try:
            # Transcribe audio using Whisper
            transcription = await self.asr_caller.transcribe(audio_bytes)
            
            # Compute WER
            return word_error_rate(transcription, original_text)
        except Exception as e:
            logger.error(f"Round-trip WER computation failed: {e}")
            return 1.0  # Worst case
    
    async def _compute_utmos(self, audio_bytes: bytes) -> float:
        """
        Compute UTMOS score for audio quality.
        
        Uses UTMOS neural MOS predictor (1-5 scale).
        """
        if not self.utmos_model:
            logger.warning("No UTMOS model configured, returning placeholder")
            return 3.0  # Neutral MOS
        
        try:
            return await self.utmos_model.predict(audio_bytes)
        except Exception as e:
            logger.error(f"UTMOS computation failed: {e}")
            return 3.0
    
    def _compute_snr(self, audio_bytes: bytes) -> float:
        """
        Compute signal-to-noise ratio using energy-based analysis.
        """
        try:
            return compute_snr(audio_bytes)
        except Exception as e:
            logger.error(f"SNR computation failed: {e}")
            return 20.0  # Default moderate SNR
    
    async def _compute_speaker_similarity(
        self,
        generated_audio: bytes,
        reference_audio: bytes
    ) -> float:
        """
        Compute speaker embedding similarity between generated and reference audio.
        
        Uses speaker embedding models (resemblyzer or speechbrain).
        """
        if not self.speaker_encoder:
            logger.warning("No speaker encoder configured")
            return 0.5
        
        try:
            # Get embeddings
            gen_embedding = await self.speaker_encoder.encode(generated_audio)
            ref_embedding = await self.speaker_encoder.encode(reference_audio)
            
            # Compute similarity
            return self.speaker_encoder.compute_similarity(gen_embedding, ref_embedding)
        except Exception as e:
            logger.error(f"Speaker similarity computation failed: {e}")
            return 0.5
    
    def get_quality_breakdown(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Get human-readable quality breakdown.
        
        Args:
            metrics: Computed TTS metrics
        
        Returns:
            Breakdown with ratings for each aspect
        """
        def rate(value: float, thresholds: tuple) -> str:
            if value >= thresholds[0]:
                return "Excellent"
            elif value >= thresholds[1]:
                return "Good"
            elif value >= thresholds[2]:
                return "Fair"
            else:
                return "Poor"
        
        breakdown = {}
        
        # Intelligibility (WER: lower is better)
        wer = metrics.get("round_trip_wer", 1.0)
        if wer <= 0.1:
            breakdown["intelligibility"] = "Excellent"
        elif wer <= 0.2:
            breakdown["intelligibility"] = "Good"
        elif wer <= 0.4:
            breakdown["intelligibility"] = "Fair"
        else:
            breakdown["intelligibility"] = "Poor"
        
        # Naturalness (UTMOS: higher is better, 1-5 scale)
        utmos = metrics.get("utmos", 3.0)
        breakdown["naturalness"] = rate(utmos, (4.0, 3.5, 3.0))
        
        # Audio quality (SNR: higher is better)
        snr = metrics.get("snr", 0.0)
        breakdown["audio_quality"] = rate(snr, (30, 20, 10))
        
        # Speaker match
        speaker_sim = metrics.get("speaker_similarity")
        if speaker_sim is not None:
            breakdown["speaker_match"] = rate(speaker_sim, (0.9, 0.7, 0.5))
        
        # Overall
        composite = metrics.get("composite_tts", 0.0)
        breakdown["overall"] = rate(composite, (0.8, 0.6, 0.4))
        
        return breakdown

