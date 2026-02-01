/**
 * Model Type Definitions and Smart Mapper
 * 
 * This module provides a unified type system for handling different AI model types
 * (LLM, ASR, TTS, VLM, V2V, etc.) with automatic configuration of:
 * - API fields required for each type
 * - Evaluation pipelines (subjective vs ground truth)
 * - Relevant metrics and domains
 */

// =============================================================================
// Core Model Types
// =============================================================================

export type ModelType = 
  | 'LLM'       // Large Language Model (text-to-text)
  | 'ASR'       // Automatic Speech Recognition (audio-to-text)
  | 'TTS'       // Text-to-Speech (text-to-audio)
  | 'VLM'       // Vision-Language Model (image+text-to-text)
  | 'V2V'       // Video-to-Video (video processing)
  | 'STT'       // Speech-to-Text (alias for ASR, more specific)
  | 'ImageGen'  // Image Generation (text-to-image)
  | 'VideoGen'  // Video Generation (text-to-video)
  | 'Embedding' // Embedding Model (text-to-vector)
  | 'Reranker'  // Reranking Model (documents-to-ranked)

export type InputFormat = 'text' | 'audio' | 'image' | 'video' | 'multimodal' | 'documents'
export type OutputFormat = 'text' | 'audio' | 'image' | 'video' | 'vector' | 'ranking'
export type EvaluationType = 'subjective' | 'ground_truth' | 'both'

// =============================================================================
// API Field Configuration
// =============================================================================

export interface APIFieldConfig {
  name: string
  label: string
  type: 'text' | 'password' | 'url' | 'select' | 'number' | 'checkbox'
  required: boolean
  placeholder?: string
  helpText?: string
  options?: { value: string; label: string }[]
  defaultValue?: string | number | boolean
  validation?: {
    pattern?: string
    minLength?: number
    maxLength?: number
    min?: number
    max?: number
  }
}

// =============================================================================
// Model Type Configuration
// =============================================================================

export interface ModelTypeConfig {
  type: ModelType
  displayName: string
  description: string
  icon: string
  inputFormat: InputFormat
  outputFormat: OutputFormat
  evaluationType: EvaluationType
  apiFields: APIFieldConfig[]
  metrics: MetricConfig[]
  groundTruthDomains: string[]
  subjectiveDomains: string[]
  supportedLanguages?: string[]
  maxContextLength?: number
  streamingSupported: boolean
  batchSupported: boolean
}

export interface MetricConfig {
  id: string
  name: string
  description: string
  unit?: string
  direction: 'higher' | 'lower'  // Is higher better or lower better?
  primary?: boolean  // Is this a primary metric for this model type?
}

// =============================================================================
// Common API Fields
// =============================================================================

const COMMON_API_FIELDS: Record<string, APIFieldConfig> = {
  endpoint: {
    name: 'endpoint',
    label: 'API Endpoint',
    type: 'url',
    required: true,
    placeholder: 'https://api.example.com/v1',
    helpText: 'The base URL for your model API'
  },
  apiKey: {
    name: 'apiKey',
    label: 'API Key',
    type: 'password',
    required: true,
    placeholder: 'sk-...',
    helpText: 'Your API key (stored securely, never shared)'
  },
  modelId: {
    name: 'modelId',
    label: 'Model ID',
    type: 'text',
    required: true,
    placeholder: 'gpt-4, llama-3.1-70b, etc.',
    helpText: 'The model identifier used in API calls'
  },
  language: {
    name: 'language',
    label: 'Primary Language',
    type: 'select',
    required: false,
    options: [
      { value: 'en', label: 'English' },
      { value: 'hi', label: 'Hindi' },
      { value: 'bn', label: 'Bengali' },
      { value: 'ta', label: 'Tamil' },
      { value: 'te', label: 'Telugu' },
      { value: 'mr', label: 'Marathi' },
      { value: 'multilingual', label: 'Multilingual' },
    ],
    helpText: 'Primary language the model is optimized for'
  },
  maxTokens: {
    name: 'maxTokens',
    label: 'Max Tokens',
    type: 'number',
    required: false,
    defaultValue: 4096,
    validation: { min: 1, max: 200000 },
    helpText: 'Maximum context length in tokens'
  },
  temperature: {
    name: 'temperature',
    label: 'Default Temperature',
    type: 'number',
    required: false,
    defaultValue: 0.7,
    validation: { min: 0, max: 2 },
    helpText: 'Default sampling temperature'
  },
  sampleRate: {
    name: 'sampleRate',
    label: 'Sample Rate',
    type: 'select',
    required: false,
    options: [
      { value: '8000', label: '8 kHz' },
      { value: '16000', label: '16 kHz' },
      { value: '22050', label: '22.05 kHz' },
      { value: '44100', label: '44.1 kHz' },
      { value: '48000', label: '48 kHz' },
    ],
    defaultValue: '16000',
    helpText: 'Audio sample rate for input/output'
  },
  voiceId: {
    name: 'voiceId',
    label: 'Voice ID',
    type: 'text',
    required: false,
    placeholder: 'alloy, nova, etc.',
    helpText: 'Voice identifier for TTS models'
  },
  imageSize: {
    name: 'imageSize',
    label: 'Default Image Size',
    type: 'select',
    required: false,
    options: [
      { value: '256x256', label: '256x256' },
      { value: '512x512', label: '512x512' },
      { value: '1024x1024', label: '1024x1024' },
      { value: '1792x1024', label: '1792x1024 (Landscape)' },
      { value: '1024x1792', label: '1024x1792 (Portrait)' },
    ],
    defaultValue: '1024x1024',
    helpText: 'Default output image dimensions'
  },
  embeddingDimension: {
    name: 'embeddingDimension',
    label: 'Embedding Dimension',
    type: 'number',
    required: false,
    defaultValue: 1536,
    validation: { min: 64, max: 8192 },
    helpText: 'Output vector dimension'
  }
}

// =============================================================================
// Model Type Configurations
// =============================================================================

export const MODEL_TYPE_CONFIGS: Record<ModelType, ModelTypeConfig> = {
  LLM: {
    type: 'LLM',
    displayName: 'Large Language Model',
    description: 'Text generation and chat models (GPT, Claude, Llama, etc.)',
    icon: 'ph ph-chat-centered-text',
    inputFormat: 'text',
    outputFormat: 'text',
    evaluationType: 'both',
    streamingSupported: true,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.maxTokens,
      COMMON_API_FIELDS.temperature,
      {
        name: 'apiFormat',
        label: 'API Format',
        type: 'select',
        required: true,
        options: [
          { value: 'openai', label: 'OpenAI Compatible' },
          { value: 'anthropic', label: 'Anthropic' },
          { value: 'google', label: 'Google AI' },
          { value: 'custom', label: 'Custom' },
        ],
        defaultValue: 'openai',
        helpText: 'API format/compatibility'
      }
    ],
    metrics: [
      { id: 'raw_quality', name: 'Raw Quality', description: 'Overall response quality', direction: 'higher', primary: true },
      { id: 'cost_adjusted', name: 'Cost-Adjusted', description: 'Quality per dollar spent', direction: 'higher' },
      { id: 'latency', name: 'Latency', description: 'Time to complete response', unit: 'ms', direction: 'lower' },
      { id: 'ttft', name: 'Time to First Token', description: 'Time until first token streams', unit: 'ms', direction: 'lower' },
      { id: 'consistency', name: 'Consistency', description: 'Variance in response quality', direction: 'higher' },
      { id: 'token_efficiency', name: 'Token Efficiency', description: 'Quality per token generated', direction: 'higher' },
      { id: 'instruction_following', name: 'Instruction Following', description: 'Adherence to format and constraints', direction: 'higher' },
      { id: 'hallucination_resistance', name: 'Hallucination Resistance', description: 'Factual accuracy', direction: 'higher' },
      { id: 'long_context', name: 'Long Context', description: 'Quality at large context lengths', direction: 'higher' },
    ],
    groundTruthDomains: ['needle_in_haystack', 'long_document_qa', 'long_context_code_completion', 'multi_document_reasoning'],
    subjectiveDomains: ['medical', 'legal', 'financial', 'customer_support', 'education', 'code', 'creative_writing', 'translation']
  },

  ASR: {
    type: 'ASR',
    displayName: 'Speech Recognition',
    description: 'Audio to text transcription (Whisper, Deepgram, AssemblyAI, etc.)',
    icon: 'ph ph-microphone',
    inputFormat: 'audio',
    outputFormat: 'text',
    evaluationType: 'ground_truth',
    streamingSupported: true,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.language,
      COMMON_API_FIELDS.sampleRate,
      {
        name: 'punctuation',
        label: 'Auto Punctuation',
        type: 'checkbox',
        required: false,
        defaultValue: true,
        helpText: 'Automatically add punctuation'
      },
      {
        name: 'diarization',
        label: 'Speaker Diarization',
        type: 'checkbox',
        required: false,
        defaultValue: false,
        helpText: 'Identify different speakers'
      }
    ],
    metrics: [
      { id: 'wer', name: 'Word Error Rate', description: 'Percentage of word errors', unit: '%', direction: 'lower', primary: true },
      { id: 'cer', name: 'Character Error Rate', description: 'Percentage of character errors', unit: '%', direction: 'lower' },
      { id: 'latency', name: 'Latency', description: 'Processing time per second of audio', unit: 'ms', direction: 'lower' },
      { id: 'rtf', name: 'Real-Time Factor', description: 'Processing time / audio duration', direction: 'lower' },
    ],
    groundTruthDomains: ['asr'],
    subjectiveDomains: [],
    supportedLanguages: ['en', 'hi', 'bn', 'ta', 'te', 'mr', 'multilingual']
  },

  TTS: {
    type: 'TTS',
    displayName: 'Text-to-Speech',
    description: 'Convert text to natural speech audio (ElevenLabs, OpenAI TTS, etc.)',
    icon: 'ph ph-speaker-high',
    inputFormat: 'text',
    outputFormat: 'audio',
    evaluationType: 'ground_truth',
    streamingSupported: true,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.language,
      COMMON_API_FIELDS.voiceId,
      COMMON_API_FIELDS.sampleRate,
      {
        name: 'speed',
        label: 'Speed',
        type: 'number',
        required: false,
        defaultValue: 1.0,
        validation: { min: 0.25, max: 4.0 },
        helpText: 'Speech speed multiplier'
      }
    ],
    metrics: [
      { id: 'mos', name: 'Mean Opinion Score', description: 'Predicted human quality rating', direction: 'higher', primary: true },
      { id: 'intelligibility', name: 'Intelligibility', description: 'Round-trip WER (TTS→ASR)', unit: '%', direction: 'lower' },
      { id: 'naturalness', name: 'Naturalness', description: 'How natural the speech sounds', direction: 'higher' },
      { id: 'speaker_similarity', name: 'Speaker Similarity', description: 'Voice cloning accuracy', direction: 'higher' },
      { id: 'latency', name: 'Latency', description: 'Time to generate audio', unit: 'ms', direction: 'lower' },
    ],
    groundTruthDomains: ['tts'],
    subjectiveDomains: [],
    supportedLanguages: ['en', 'hi', 'bn', 'ta', 'te', 'mr', 'multilingual']
  },

  VLM: {
    type: 'VLM',
    displayName: 'Vision-Language Model',
    description: 'Models that understand images and text (GPT-4V, Claude Vision, LLaVA, etc.)',
    icon: 'ph ph-eye',
    inputFormat: 'multimodal',
    outputFormat: 'text',
    evaluationType: 'both',
    streamingSupported: true,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.maxTokens,
      {
        name: 'imageDetail',
        label: 'Image Detail Level',
        type: 'select',
        required: false,
        options: [
          { value: 'auto', label: 'Auto' },
          { value: 'low', label: 'Low (faster)' },
          { value: 'high', label: 'High (detailed)' },
        ],
        defaultValue: 'auto',
        helpText: 'Level of detail for image analysis'
      }
    ],
    metrics: [
      { id: 'accuracy', name: 'Accuracy', description: 'Visual QA accuracy', unit: '%', direction: 'higher', primary: true },
      { id: 'raw_quality', name: 'Raw Quality', description: 'Overall response quality', direction: 'higher' },
      { id: 'latency', name: 'Latency', description: 'Time to process image + generate', unit: 'ms', direction: 'lower' },
      { id: 'spatial_reasoning', name: 'Spatial Reasoning', description: 'Understanding of spatial relationships', direction: 'higher' },
    ],
    groundTruthDomains: ['visual_qa', 'document_extraction', 'image_captioning', 'ocr'],
    subjectiveDomains: ['medical_imaging', 'document_analysis']
  },

  V2V: {
    type: 'V2V',
    displayName: 'Video-to-Video',
    description: 'Video understanding and processing models',
    icon: 'ph ph-film-strip',
    inputFormat: 'video',
    outputFormat: 'video',
    evaluationType: 'both',
    streamingSupported: false,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      {
        name: 'maxDuration',
        label: 'Max Video Duration',
        type: 'number',
        required: false,
        defaultValue: 60,
        validation: { min: 1, max: 3600 },
        helpText: 'Maximum video duration in seconds'
      },
      {
        name: 'fps',
        label: 'Output FPS',
        type: 'select',
        required: false,
        options: [
          { value: '24', label: '24 fps' },
          { value: '30', label: '30 fps' },
          { value: '60', label: '60 fps' },
        ],
        defaultValue: '30'
      }
    ],
    metrics: [
      { id: 'temporal_consistency', name: 'Temporal Consistency', description: 'Frame-to-frame coherence', direction: 'higher', primary: true },
      { id: 'quality', name: 'Visual Quality', description: 'Output video quality', direction: 'higher' },
      { id: 'latency', name: 'Processing Time', description: 'Time per second of video', unit: 'ms', direction: 'lower' },
    ],
    groundTruthDomains: [],
    subjectiveDomains: ['video_editing', 'video_effects']
  },

  STT: {
    type: 'STT',
    displayName: 'Speech-to-Text',
    description: 'Real-time speech transcription (alias for ASR)',
    icon: 'ph ph-waveform',
    inputFormat: 'audio',
    outputFormat: 'text',
    evaluationType: 'ground_truth',
    streamingSupported: true,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.language,
      COMMON_API_FIELDS.sampleRate,
    ],
    metrics: [
      { id: 'wer', name: 'Word Error Rate', description: 'Percentage of word errors', unit: '%', direction: 'lower', primary: true },
      { id: 'latency', name: 'Streaming Latency', description: 'Delay in real-time transcription', unit: 'ms', direction: 'lower' },
    ],
    groundTruthDomains: ['asr'],
    subjectiveDomains: []
  },

  ImageGen: {
    type: 'ImageGen',
    displayName: 'Image Generation',
    description: 'Text-to-image models (DALL-E, Stable Diffusion, Midjourney, etc.)',
    icon: 'ph ph-paint-brush',
    inputFormat: 'text',
    outputFormat: 'image',
    evaluationType: 'subjective',
    streamingSupported: false,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.imageSize,
      {
        name: 'steps',
        label: 'Inference Steps',
        type: 'number',
        required: false,
        defaultValue: 30,
        validation: { min: 1, max: 150 },
        helpText: 'Number of diffusion steps'
      },
      {
        name: 'guidanceScale',
        label: 'Guidance Scale',
        type: 'number',
        required: false,
        defaultValue: 7.5,
        validation: { min: 1, max: 30 },
        helpText: 'How closely to follow the prompt'
      }
    ],
    metrics: [
      { id: 'prompt_adherence', name: 'Prompt Adherence', description: 'How well output matches prompt', direction: 'higher', primary: true },
      { id: 'aesthetic_quality', name: 'Aesthetic Quality', description: 'Visual appeal', direction: 'higher' },
      { id: 'latency', name: 'Generation Time', description: 'Time to generate image', unit: 'ms', direction: 'lower' },
      { id: 'fid', name: 'FID Score', description: 'Fréchet Inception Distance', direction: 'lower' },
    ],
    groundTruthDomains: [],
    subjectiveDomains: ['image_generation']
  },

  VideoGen: {
    type: 'VideoGen',
    displayName: 'Video Generation',
    description: 'Text-to-video models (Sora, Runway, etc.)',
    icon: 'ph ph-video-camera',
    inputFormat: 'text',
    outputFormat: 'video',
    evaluationType: 'subjective',
    streamingSupported: false,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      {
        name: 'duration',
        label: 'Video Duration',
        type: 'number',
        required: false,
        defaultValue: 5,
        validation: { min: 1, max: 60 },
        helpText: 'Output video duration in seconds'
      },
      {
        name: 'resolution',
        label: 'Resolution',
        type: 'select',
        required: false,
        options: [
          { value: '720p', label: '720p' },
          { value: '1080p', label: '1080p' },
          { value: '4k', label: '4K' },
        ],
        defaultValue: '1080p'
      }
    ],
    metrics: [
      { id: 'prompt_adherence', name: 'Prompt Adherence', description: 'How well video matches prompt', direction: 'higher', primary: true },
      { id: 'temporal_consistency', name: 'Temporal Consistency', description: 'Frame coherence over time', direction: 'higher' },
      { id: 'quality', name: 'Visual Quality', description: 'Overall video quality', direction: 'higher' },
      { id: 'latency', name: 'Generation Time', description: 'Time to generate video', unit: 'ms', direction: 'lower' },
    ],
    groundTruthDomains: [],
    subjectiveDomains: ['video_generation']
  },

  Embedding: {
    type: 'Embedding',
    displayName: 'Embedding Model',
    description: 'Text embedding models for search and retrieval (OpenAI Embeddings, Cohere, etc.)',
    icon: 'ph ph-vector-three',
    inputFormat: 'text',
    outputFormat: 'vector',
    evaluationType: 'ground_truth',
    streamingSupported: false,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      COMMON_API_FIELDS.embeddingDimension,
      {
        name: 'normalize',
        label: 'Normalize Vectors',
        type: 'checkbox',
        required: false,
        defaultValue: true,
        helpText: 'Normalize output vectors to unit length'
      }
    ],
    metrics: [
      { id: 'mteb_avg', name: 'MTEB Average', description: 'Massive Text Embedding Benchmark average', direction: 'higher', primary: true },
      { id: 'retrieval_ndcg', name: 'Retrieval NDCG@10', description: 'Retrieval quality', direction: 'higher' },
      { id: 'clustering', name: 'Clustering Score', description: 'Cluster separation quality', direction: 'higher' },
      { id: 'latency', name: 'Latency', description: 'Time to embed', unit: 'ms', direction: 'lower' },
    ],
    groundTruthDomains: ['embedding_retrieval'],
    subjectiveDomains: []
  },

  Reranker: {
    type: 'Reranker',
    displayName: 'Reranker Model',
    description: 'Document reranking models for search (Cohere Rerank, etc.)',
    icon: 'ph ph-sort-ascending',
    inputFormat: 'documents',
    outputFormat: 'ranking',
    evaluationType: 'ground_truth',
    streamingSupported: false,
    batchSupported: true,
    apiFields: [
      COMMON_API_FIELDS.endpoint,
      COMMON_API_FIELDS.apiKey,
      COMMON_API_FIELDS.modelId,
      {
        name: 'topK',
        label: 'Top K Results',
        type: 'number',
        required: false,
        defaultValue: 10,
        validation: { min: 1, max: 100 },
        helpText: 'Number of documents to return'
      }
    ],
    metrics: [
      { id: 'ndcg', name: 'NDCG@10', description: 'Normalized Discounted Cumulative Gain', direction: 'higher', primary: true },
      { id: 'mrr', name: 'MRR', description: 'Mean Reciprocal Rank', direction: 'higher' },
      { id: 'latency', name: 'Latency', description: 'Time to rerank', unit: 'ms', direction: 'lower' },
    ],
    groundTruthDomains: ['reranking'],
    subjectiveDomains: []
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get configuration for a model type
 */
export function getModelTypeConfig(type: ModelType): ModelTypeConfig {
  return MODEL_TYPE_CONFIGS[type]
}

/**
 * Get all available model types
 */
export function getAllModelTypes(): ModelType[] {
  return Object.keys(MODEL_TYPE_CONFIGS) as ModelType[]
}

/**
 * Get model types by evaluation type
 */
export function getModelTypesByEvaluation(evalType: EvaluationType): ModelType[] {
  return getAllModelTypes().filter(type => {
    const config = MODEL_TYPE_CONFIGS[type]
    return config.evaluationType === evalType || config.evaluationType === 'both'
  })
}

/**
 * Get model types by input format
 */
export function getModelTypesByInput(format: InputFormat): ModelType[] {
  return getAllModelTypes().filter(type => MODEL_TYPE_CONFIGS[type].inputFormat === format)
}

/**
 * Get model types by output format
 */
export function getModelTypesByOutput(format: OutputFormat): ModelType[] {
  return getAllModelTypes().filter(type => MODEL_TYPE_CONFIGS[type].outputFormat === format)
}

/**
 * Get primary metric for a model type
 */
export function getPrimaryMetric(type: ModelType): MetricConfig | undefined {
  return MODEL_TYPE_CONFIGS[type].metrics.find(m => m.primary)
}

/**
 * Check if a model type supports ground truth evaluation
 */
export function supportsGroundTruth(type: ModelType): boolean {
  const config = MODEL_TYPE_CONFIGS[type]
  return config.evaluationType === 'ground_truth' || config.evaluationType === 'both'
}

/**
 * Check if a model type supports subjective evaluation
 */
export function supportsSubjective(type: ModelType): boolean {
  const config = MODEL_TYPE_CONFIGS[type]
  return config.evaluationType === 'subjective' || config.evaluationType === 'both'
}

/**
 * Group model types by category for display
 */
export function getModelTypesByCategory(): Record<string, ModelType[]> {
  return {
    'Text': ['LLM', 'Embedding', 'Reranker'],
    'Speech': ['ASR', 'STT', 'TTS'],
    'Vision': ['VLM', 'ImageGen'],
    'Video': ['V2V', 'VideoGen'],
  }
}

/**
 * Get icon for a model type
 */
export function getModelTypeIcon(type: ModelType): string {
  return MODEL_TYPE_CONFIGS[type].icon
}

/**
 * Get display name for a model type
 */
export function getModelTypeDisplayName(type: ModelType): string {
  return MODEL_TYPE_CONFIGS[type].displayName
}
