"""
Needle in Haystack sample generator.

Generates test samples with hidden facts (needles) at various
positions within long context (haystack).
"""

import random
import string
from typing import Dict, List, Optional, Tuple


# Sample needle templates
NEEDLE_TEMPLATES = {
    "code": [
        "The authorization code for {subject} is {value}.",
        "The secret password for {subject} is {value}.",
        "The access code for {subject} is {value}.",
    ],
    "fact": [
        "{subject} was established in the year {value}.",
        "The population of {subject} is {value}.",
        "{subject} is located at coordinates {value}.",
    ],
    "number": [
        "The account number for {subject} is {value}.",
        "The order ID for {subject} is {value}.",
        "The transaction ID for {subject} is {value}.",
    ],
    "date": [
        "The deadline for {subject} is {value}.",
        "{subject} was completed on {value}.",
        "The launch date for {subject} is {value}.",
    ],
    "name": [
        "The contact person for {subject} is {value}.",
        "The lead developer for {subject} is {value}.",
        "The owner of {subject} is {value}.",
    ],
}

QUESTION_TEMPLATES = {
    "code": "What is the authorization code for {subject}?",
    "fact": "When was {subject} established?",
    "number": "What is the account number for {subject}?",
    "date": "What is the deadline for {subject}?",
    "name": "Who is the contact person for {subject}?",
}

SUBJECTS = [
    "Project Aurora", "Project Beta", "Project Omega",
    "the Vault", "the Archive", "the Repository",
    "Operation Sunrise", "Operation Eclipse", "Operation Phoenix",
]


def generate_random_value(needle_type: str) -> str:
    """Generate a random value for a needle type."""
    if needle_type == "code":
        # Alphanumeric code like X7-DELTA-9
        prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
        middle = ''.join(random.choices(string.ascii_uppercase, k=5))
        suffix = str(random.randint(1, 9))
        return f"{prefix}-{middle}-{suffix}"
    
    elif needle_type == "number":
        # Numeric ID
        return str(random.randint(100000, 999999))
    
    elif needle_type == "date":
        # Random date
        year = random.randint(2024, 2026)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        return f"{year}-{month:02d}-{day:02d}"
    
    elif needle_type == "name":
        first_names = ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    else:  # fact
        return str(random.randint(1900, 2024))


def generate_needle(
    needle_type: str = "code",
    subject: str = None
) -> Tuple[str, str, str]:
    """
    Generate a needle (hidden fact).
    
    Args:
        needle_type: Type of needle (code, fact, number, date, name)
        subject: Subject of the needle (optional)
    
    Returns:
        Tuple of (needle_text, question, expected_answer)
    """
    if subject is None:
        subject = random.choice(SUBJECTS)
    
    value = generate_random_value(needle_type)
    
    template = random.choice(NEEDLE_TEMPLATES.get(needle_type, NEEDLE_TEMPLATES["code"]))
    needle_text = template.format(subject=subject, value=value)
    
    question_template = QUESTION_TEMPLATES.get(needle_type, QUESTION_TEMPLATES["code"])
    question = question_template.format(subject=subject)
    
    return needle_text, question, value


def generate_distractors(
    needle_type: str,
    count: int = 5
) -> List[str]:
    """
    Generate distractor facts (similar but different).
    
    Args:
        needle_type: Type matching the main needle
        count: Number of distractors
    
    Returns:
        List of distractor sentences
    """
    distractors = []
    used_subjects = set()
    
    for _ in range(count):
        # Pick a different subject
        available = [s for s in SUBJECTS if s not in used_subjects]
        if not available:
            available = SUBJECTS
        
        subject = random.choice(available)
        used_subjects.add(subject)
        
        value = generate_random_value(needle_type)
        template = random.choice(NEEDLE_TEMPLATES.get(needle_type, NEEDLE_TEMPLATES["code"]))
        distractor = template.format(subject=subject, value=value)
        distractors.append(distractor)
    
    return distractors


class NeedleGenerator:
    """
    Generator for Needle in Haystack samples.
    
    Creates samples with:
    - Configurable context lengths (4K to 128K+ tokens)
    - Configurable needle positions (0.0 to 1.0)
    - Multiple needle types (code, fact, number, date, name)
    - Distractors (similar but wrong facts)
    """
    
    def __init__(
        self,
        haystack_sources: List[str] = None,
        default_haystack_text: str = None
    ):
        """
        Initialize generator.
        
        Args:
            haystack_sources: Paths to text files for haystack content
            default_haystack_text: Default text to use if no sources
        """
        self.haystack_sources = haystack_sources or []
        self.default_haystack = default_haystack_text or self._get_default_haystack()
    
    def generate_sample(
        self,
        context_length: int,
        needle_position: float,
        needle_type: str = "code",
        distractor_count: int = 5
    ) -> Dict:
        """
        Generate a needle in haystack sample.
        
        Args:
            context_length: Target context length in tokens (approx)
            needle_position: Position of needle (0.0 = start, 1.0 = end)
            needle_type: Type of needle
            distractor_count: Number of distractors
        
        Returns:
            Sample dict with haystack, needle, question, answer
        """
        # Generate needle
        needle_text, question, expected_answer = generate_needle(needle_type)
        
        # Generate distractors
        distractors = generate_distractors(needle_type, distractor_count)
        
        # Generate haystack
        haystack = self._generate_haystack(
            context_length,
            needle_position,
            needle_text,
            distractors
        )
        
        # Generate sample ID
        sample_id = f"needle_{context_length}_{int(needle_position * 100):03d}_{needle_type}"
        
        return {
            "sample_id": sample_id,
            "domain": "needle_in_haystack",
            "category": "long_context",
            "context_length": context_length,
            "needle_position": needle_position,
            "needle_type": needle_type,
            "haystack": haystack,
            "needle": needle_text,
            "question": question,
            "expected_answer": expected_answer,
            "answer_variants": self._generate_answer_variants(expected_answer, needle_type),
            "distractors": distractors,
            "difficulty": self._compute_difficulty(context_length, needle_position),
        }
    
    def _generate_haystack(
        self,
        context_length: int,
        needle_position: float,
        needle_text: str,
        distractors: List[str]
    ) -> str:
        """Generate haystack with needle inserted at position."""
        # Approximate tokens to characters (1 token ≈ 4 chars)
        target_chars = context_length * 4
        
        # Get base text
        base_text = self._get_base_text(target_chars)
        
        # Calculate insertion point
        insert_pos = int(len(base_text) * needle_position)
        
        # Find paragraph boundary
        if insert_pos > 0:
            # Look for paragraph break
            para_break = base_text.rfind("\n\n", 0, insert_pos)
            if para_break > 0:
                insert_pos = para_break + 2
        
        # Insert distractors throughout (before needle position)
        text_with_distractors = base_text[:insert_pos]
        
        if distractors and insert_pos > 0:
            chunk_size = insert_pos // (len(distractors) + 1)
            for i, distractor in enumerate(distractors):
                pos = chunk_size * (i + 1)
                para = text_with_distractors.rfind("\n\n", 0, pos)
                if para > 0:
                    text_with_distractors = (
                        text_with_distractors[:para + 2] +
                        distractor + "\n\n" +
                        text_with_distractors[para + 2:]
                    )
        
        # Insert needle
        haystack = (
            text_with_distractors +
            "\n\n" + needle_text + "\n\n" +
            base_text[insert_pos:]
        )
        
        return haystack
    
    def _get_base_text(self, target_chars: int) -> str:
        """Get base text for haystack."""
        # Try to load from sources
        for source in self.haystack_sources:
            try:
                with open(source, "r", encoding="utf-8") as f:
                    text = f.read()
                    if len(text) >= target_chars:
                        return text[:target_chars]
            except Exception:
                continue
        
        # Use default, repeating if needed
        text = self.default_haystack
        while len(text) < target_chars:
            text = text + "\n\n" + self.default_haystack
        
        return text[:target_chars]
    
    def _get_default_haystack(self) -> str:
        """Get default haystack text."""
        # Generic text about various topics
        return """
The development of artificial intelligence has been one of the most significant technological 
advancements of the 21st century. From early rule-based systems to modern deep learning models, 
the field has evolved dramatically over the past few decades.

Machine learning, a subset of AI, enables computers to learn from data without being explicitly 
programmed. This approach has led to breakthroughs in image recognition, natural language 
processing, and game playing.

Neural networks, inspired by biological neurons, form the foundation of deep learning. These 
networks can have millions or even billions of parameters, allowing them to model complex 
patterns in data.

The transformer architecture, introduced in 2017, revolutionized natural language processing. 
Models like GPT and BERT have achieved remarkable performance on a wide range of language tasks.

Large language models are trained on vast amounts of text data from the internet. They learn 
patterns in language that allow them to generate coherent and contextually appropriate responses.

The ethics of AI development is an increasingly important topic. Researchers and policymakers 
are working to ensure that AI systems are developed and deployed responsibly.

Climate change poses one of the greatest challenges facing humanity. Scientists agree that 
human activities are causing global temperatures to rise at an unprecedented rate.

Renewable energy sources such as solar, wind, and hydroelectric power are becoming increasingly 
cost-competitive with fossil fuels. Many countries are investing heavily in clean energy 
infrastructure.

Space exploration has entered a new era with the involvement of private companies. Reusable 
rockets have dramatically reduced the cost of reaching orbit.

The human genome was fully sequenced in 2003, opening new possibilities for personalized 
medicine. Gene therapy shows promise for treating previously incurable genetic diseases.
""" * 10  # Repeat to ensure enough text
    
    def _generate_answer_variants(
        self,
        answer: str,
        needle_type: str
    ) -> List[str]:
        """Generate acceptable answer variants."""
        variants = [answer]
        
        if needle_type == "code":
            # Add without hyphens
            variants.append(answer.replace("-", " "))
            variants.append(answer.replace("-", ""))
        
        elif needle_type == "number":
            # Add with commas
            try:
                num = int(answer)
                variants.append(f"{num:,}")
            except ValueError:
                pass
        
        elif needle_type == "date":
            # Add various date formats
            try:
                parts = answer.split("-")
                if len(parts) == 3:
                    year, month, day = parts
                    variants.append(f"{month}/{day}/{year}")
                    variants.append(f"{day}/{month}/{year}")
            except Exception:
                pass
        
        return variants
    
    def _compute_difficulty(
        self,
        context_length: int,
        needle_position: float
    ) -> str:
        """Compute difficulty level."""
        # Longer context = harder
        # Middle position = harder
        
        length_score = 0
        if context_length >= 64000:
            length_score = 2
        elif context_length >= 16000:
            length_score = 1
        
        position_score = 0
        if 0.3 <= needle_position <= 0.7:
            position_score = 1  # Middle is harder
        
        total = length_score + position_score
        
        if total >= 3:
            return "hard"
        elif total >= 1:
            return "medium"
        else:
            return "easy"


def generate_needle_sample(
    context_length: int,
    needle_position: float,
    needle_type: str = "code"
) -> Dict:
    """
    Convenience function to generate a single sample.
    
    Args:
        context_length: Target context length
        needle_position: Position (0.0-1.0)
        needle_type: Type of needle
    
    Returns:
        Sample dict
    """
    generator = NeedleGenerator()
    return generator.generate_sample(context_length, needle_position, needle_type)


