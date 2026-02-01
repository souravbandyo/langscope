import type { TaskType } from '@/components/arena'

export interface MockResponse {
  id: string
  modelId: string // Hidden until reveal
  content: string
}

// Task type definitions with Input → Output combinations
export interface TaskTypeDefinition {
  id: TaskType
  label: string
  inputType: 'text' | 'audio' | 'image' | 'video'
  outputType: 'text' | 'code' | 'audio' | 'image' | 'video'
  icon: string // Phosphor icon class
  description: string
  defaultPrompt: string
  suggestedDomains: string[]
}

// All available task types as Input → Output combinations
export const taskTypes: TaskTypeDefinition[] = [
  // Text Input Tasks
  {
    id: 'text_to_text',
    label: 'Text → Text',
    inputType: 'text',
    outputType: 'text',
    icon: 'ph ph-article',
    description: 'Reasoning, summarization, Q&A',
    defaultPrompt: 'Explain the impact of AI on modern society',
    suggestedDomains: ['reasoning', 'writing', 'summarization', 'translation'],
  },
  {
    id: 'text_to_code',
    label: 'Text → Code',
    inputType: 'text',
    outputType: 'code',
    icon: 'ph ph-code',
    description: 'Code generation from natural language',
    defaultPrompt: 'Write a function to calculate the Fibonacci sequence',
    suggestedDomains: ['code', 'code_generation'],
  },
  {
    id: 'text_to_audio',
    label: 'Text → Audio',
    inputType: 'text',
    outputType: 'audio',
    icon: 'ph ph-speaker-high',
    description: 'Text-to-speech synthesis',
    defaultPrompt: 'Hello, welcome to our service. How can I help you today?',
    suggestedDomains: ['speech', 'tts', 'multilingual'],
  },
  {
    id: 'text_to_image',
    label: 'Text → Image',
    inputType: 'text',
    outputType: 'image',
    icon: 'ph ph-image',
    description: 'Image generation from text prompts',
    defaultPrompt: 'A serene mountain landscape at sunset with a lake reflection',
    suggestedDomains: ['vision', 'creative', 'multimodal'],
  },
  {
    id: 'text_to_video',
    label: 'Text → Video',
    inputType: 'text',
    outputType: 'video',
    icon: 'ph ph-video-camera',
    description: 'Video generation from text prompts',
    defaultPrompt: 'A butterfly gracefully landing on a flower',
    suggestedDomains: ['vision', 'creative', 'multimodal'],
  },
  // Audio Input Tasks
  {
    id: 'audio_to_text',
    label: 'Audio → Text',
    inputType: 'audio',
    outputType: 'text',
    icon: 'ph ph-microphone',
    description: 'Speech recognition, transcription',
    defaultPrompt: 'Transcribe the following audio',
    suggestedDomains: ['speech', 'asr', 'multilingual'],
  },
  {
    id: 'audio_to_audio',
    label: 'Audio → Audio',
    inputType: 'audio',
    outputType: 'audio',
    icon: 'ph ph-waveform',
    description: 'Voice conversion, audio enhancement',
    defaultPrompt: 'Convert voice style while preserving content',
    suggestedDomains: ['speech', 'audio'],
  },
  // Image Input Tasks
  {
    id: 'image_to_text',
    label: 'Image → Text',
    inputType: 'image',
    outputType: 'text',
    icon: 'ph ph-eye',
    description: 'Image captioning, OCR, visual Q&A',
    defaultPrompt: 'Describe what you see in this image',
    suggestedDomains: ['vision', 'multimodal', 'ocr'],
  },
  {
    id: 'image_to_image',
    label: 'Image → Image',
    inputType: 'image',
    outputType: 'image',
    icon: 'ph ph-images',
    description: 'Image editing, style transfer',
    defaultPrompt: 'Apply artistic style transformation',
    suggestedDomains: ['vision', 'creative'],
  },
  // Video Input Tasks
  {
    id: 'video_to_text',
    label: 'Video → Text',
    inputType: 'video',
    outputType: 'text',
    icon: 'ph ph-film-strip',
    description: 'Video captioning, analysis',
    defaultPrompt: 'Describe what happens in this video',
    suggestedDomains: ['vision', 'multimodal'],
  },
  {
    id: 'video_to_video',
    label: 'Video → Video',
    inputType: 'video',
    outputType: 'video',
    icon: 'ph ph-film-slate',
    description: 'Video editing, enhancement',
    defaultPrompt: 'Enhance video quality and stabilize',
    suggestedDomains: ['vision', 'creative'],
  },
]

// Get task type definition by ID
export function getTaskType(id: TaskType): TaskTypeDefinition | undefined {
  return taskTypes.find(t => t.id === id)
}

// Get output type for rendering
export function getOutputType(taskTypeId: TaskType): 'text' | 'code' | 'audio' | 'image' | 'video' {
  const taskType = getTaskType(taskTypeId)
  return taskType?.outputType || 'text'
}

// Generate varied mock responses for different task types
export function generateMockResponses(
  taskType: TaskType,
  modelIds: string[],
  prompt: string
): MockResponse[] {
  const outputType = getOutputType(taskType)
  
  const generators: Record<string, (modelId: string, index: number) => string> = {
    code: generateCodeResponse,
    text: generateTextResponse,
    audio: generateAudioResponse,
    image: generateImageResponse,
    video: generateVideoResponse,
  }

  const generator = generators[outputType] || generators.text

  return modelIds.map((modelId, index) => ({
    id: `response_${index}`,
    modelId,
    content: generator(modelId, index),
  }))
}

// Code responses - different implementations of the same task
function generateCodeResponse(_modelId: string, index: number): string {
  const codeExamples = [
    // Response A - Recursive Fibonacci
`def fibonacci(n):
    """Calculate the nth Fibonacci number recursively."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")`,

    // Response B - Iterative with memoization
`def fibonacci(n, memo={}):
    """Calculate Fibonacci with memoization for efficiency."""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

# Generate first 10 numbers
results = [fibonacci(i) for i in range(10)]
print("Fibonacci sequence:", results)`,

    // Response C - Generator approach
`def fibonacci_generator(limit):
    """Generate Fibonacci sequence up to limit."""
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

# Using the generator
fib_seq = list(fibonacci_generator(10))
print(f"First 10 Fibonacci numbers: {fib_seq}")`,

    // Response D - Matrix exponentiation (advanced)
`import numpy as np

def fibonacci_matrix(n):
    """Calculate Fibonacci using matrix exponentiation O(log n)."""
    if n <= 0:
        return 0
    
    F = np.array([[1, 1], [1, 0]], dtype=object)
    result = matrix_power(F, n - 1)
    return result[0][0]

def matrix_power(M, n):
    if n == 1:
        return M
    if n % 2 == 0:
        half = matrix_power(M, n // 2)
        return np.dot(half, half)
    else:
        return np.dot(M, matrix_power(M, n - 1))

print([fibonacci_matrix(i) for i in range(1, 11)])`,

    // Response E - Dynamic programming
`def fibonacci_dp(n):
    """Dynamic programming approach with O(n) time and O(1) space."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1

# Test the function
print("Fibonacci sequence (DP):")
for i in range(10):
    print(f"  F({i}) = {fibonacci_dp(i)}")`,

    // Response F - Closed-form formula
`import math

def fibonacci_formula(n):
    """Binet's formula - closed form solution."""
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2
    return int((phi**n - psi**n) / math.sqrt(5))

# Note: May have floating point errors for large n
results = [fibonacci_formula(i) for i in range(10)]
print("Using Binet's formula:", results)`,
  ]

  return codeExamples[index % codeExamples.length]
}

// Text responses - different writing styles
function generateTextResponse(_modelId: string, index: number): string {
  const textExamples = [
    // Response A - Formal academic style
`The implications of artificial intelligence on modern society are profound and multifaceted. As we witness the rapid advancement of machine learning technologies, it becomes increasingly important to consider both the opportunities and challenges that emerge.

From an economic perspective, AI promises to revolutionize productivity across virtually every sector. Manufacturing, healthcare, finance, and creative industries are all experiencing transformative changes driven by intelligent automation and predictive analytics.

However, these advancements also raise significant ethical questions regarding employment displacement, algorithmic bias, and the concentration of technological power among a small number of corporations.`,

    // Response B - Conversational style
`So here's the thing about AI - it's changing everything, and I mean everything! Think about how we search for information, how doctors diagnose diseases, even how we create art. It's wild!

But let's be real for a second. Not everyone's super excited about these changes. Some folks are worried about their jobs, others are concerned about privacy. And honestly? Those are totally valid concerns.

The key is finding that sweet spot where we can enjoy all the cool benefits of AI while also making sure we don't leave anyone behind. It's a balance, you know?`,

    // Response C - Technical explanation
`Artificial intelligence systems operate through complex neural network architectures that process information in layers. Each layer extracts increasingly abstract features from the input data, enabling the system to recognize patterns and make predictions.

Key components include:
1. Input layer - receives raw data
2. Hidden layers - perform feature extraction
3. Output layer - produces final predictions

Training involves backpropagation of errors and gradient descent optimization to minimize the loss function. Modern approaches utilize attention mechanisms and transformer architectures for improved performance.`,

    // Response D - Storytelling approach
`Imagine waking up in 2040. Your AI assistant has already optimized your morning routine based on your sleep patterns. As you drive to work (in your self-driving car, of course), the traffic system coordinates thousands of vehicles seamlessly.

But Maria, a factory worker in Ohio, remembers when things were different. When she lost her job to automation, she thought her world was ending. Now, after retraining in AI systems maintenance, she earns twice what she made before.

This is the future we're building - not perfect, but full of possibility.`,

    // Response E - Bullet-point analysis
`Key Impacts of AI on Society:

• Economic Transformation
  - Automation of routine tasks
  - New job categories emerging
  - Shifts in required skills

• Healthcare Revolution
  - Early disease detection
  - Personalized treatment plans
  - Drug discovery acceleration

• Ethical Considerations
  - Data privacy concerns
  - Algorithmic fairness
  - Accountability questions

• Social Implications
  - Digital divide risks
  - Educational changes needed
  - Human-AI collaboration models`,

    // Response F - Philosophical perspective
`At its core, the question of artificial intelligence is fundamentally a question about consciousness, agency, and what it means to be human. When we create systems that can learn, adapt, and seemingly make decisions, we force ourselves to confront our own assumptions about intelligence and creativity.

The ancient Greeks spoke of the "daemons" - intermediary beings between humans and gods. In some ways, our AI systems occupy a similar conceptual space: powerful tools that extend human capability while remaining fundamentally different from human consciousness.`,
  ]

  return textExamples[index % textExamples.length]
}

// Audio responses - placeholder URLs (would be real audio in production)
function generateAudioResponse(_modelId: string, index: number): string {
  // Using placeholder data URLs or mock paths
  // In production, these would be actual generated audio files
  const audioPlaceholders = [
    'mock_audio_a.mp3',
    'mock_audio_b.mp3',
    'mock_audio_c.mp3',
    'mock_audio_d.mp3',
    'mock_audio_e.mp3',
    'mock_audio_f.mp3',
  ]
  return audioPlaceholders[index % audioPlaceholders.length]
}

// Image responses - using picsum.photos for demo
function generateImageResponse(_modelId: string, index: number): string {
  // Different seeds produce different images
  const seeds = ['ai-art-1', 'ai-art-2', 'ai-art-3', 'ai-art-4', 'ai-art-5', 'ai-art-6']
  const seed = seeds[index % seeds.length]
  return `https://picsum.photos/seed/${seed}/600/400`
}

// Video responses - placeholder paths
function generateVideoResponse(_modelId: string, index: number): string {
  const videoPlaceholders = [
    'mock_video_a.mp4',
    'mock_video_b.mp4',
    'mock_video_c.mp4',
    'mock_video_d.mp4',
    'mock_video_e.mp4',
    'mock_video_f.mp4',
  ]
  return videoPlaceholders[index % videoPlaceholders.length]
}

// Legacy task type config for backward compatibility
export const taskTypeConfig: Record<string, {
  label: string
  icon: string
  description: string
  defaultPrompt: string
}> = taskTypes.reduce((acc, t) => {
  acc[t.id] = {
    label: t.label,
    icon: t.icon,
    description: t.description,
    defaultPrompt: t.defaultPrompt,
  }
  return acc
}, {} as Record<string, { label: string; icon: string; description: string; defaultPrompt: string }>)
