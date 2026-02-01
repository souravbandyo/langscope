"""
Domain-specific prompt templates.

Provides prompt generation utilities for different evaluation scenarios.
"""

from typing import Dict, List, Optional
from langscope.domain.domain_config import Domain, DomainPrompts


def format_case_prompt(
    prompts: DomainPrompts,
    difficulty: str = "medium",
    topic: str = None
) -> str:
    """
    Format case generation prompt.
    
    Args:
        prompts: Domain prompts
        difficulty: Difficulty level
        topic: Optional specific topic
    
    Returns:
        Formatted prompt
    """
    prompt = prompts.case_generation.format(
        difficulty=difficulty
    )
    
    if topic:
        prompt += f"\n\nFocus on the topic: {topic}"
    
    return prompt


def format_question_prompt(
    prompts: DomainPrompts,
    case_text: str
) -> str:
    """
    Format question generation prompt.
    
    Args:
        prompts: Domain prompts
        case_text: Case text
    
    Returns:
        Formatted prompt
    """
    return prompts.question_generation.format(case_text=case_text)


def format_answer_prompt(
    prompts: DomainPrompts,
    case_text: str,
    question_text: str
) -> str:
    """
    Format answer generation prompt.
    
    Args:
        prompts: Domain prompts
        case_text: Case text
        question_text: Question text
    
    Returns:
        Formatted prompt
    """
    return prompts.answer_generation.format(
        case_text=case_text,
        question_text=question_text
    )


def format_judge_ranking_prompt(
    prompts: DomainPrompts,
    case_text: str,
    question_text: str,
    responses: Dict[str, str],
    labels: Dict[str, str] = None
) -> str:
    """
    Format judge ranking prompt.
    
    Args:
        prompts: Domain prompts
        case_text: Case text
        question_text: Question text
        responses: Dict of {model_id: response_text}
        labels: Optional labels for anonymization
    
    Returns:
        Formatted prompt
    """
    # Generate labels if not provided
    if labels is None:
        label_chars = list("ABCDEFGHIJ")
        labels = {
            model_id: label_chars[i]
            for i, model_id in enumerate(responses.keys())
        }
    
    # Format responses
    response_text_parts = []
    for model_id, response in responses.items():
        label = labels.get(model_id, model_id)
        response_text_parts.append(f"### Response {label}\n{response}")
    
    responses_formatted = "\n\n".join(response_text_parts)
    
    return prompts.judge_ranking.format(
        case_text=case_text,
        question_text=question_text,
        responses=responses_formatted,
        n_responses=len(responses)
    )


def get_system_prompt_for_role(role: str, domain: str = None) -> str:
    """
    Get system prompt for a specific role.
    
    Args:
        role: Role name (competitor, judge, case_creator, question_creator)
        domain: Optional domain for specialization
    
    Returns:
        System prompt
    """
    base_prompts = {
        "competitor": "You are a helpful AI assistant providing accurate and comprehensive responses.",
        
        "judge": """You are an expert evaluator tasked with ranking AI responses.
Your rankings should be based on:
- Accuracy and correctness
- Completeness and depth
- Clarity and organization
- Relevance to the question

Provide only rankings, no explanations.""",
        
        "case_creator": """You are an expert content creator.
Generate challenging but fair evaluation cases that:
- Are realistic and well-structured
- Have clear objectives
- Allow for meaningful differentiation between responses""",
        
        "question_creator": """You are an expert question designer.
Create questions that:
- Test deep understanding
- Require synthesis of information
- Have clear evaluation criteria"""
    }
    
    return base_prompts.get(role, base_prompts["competitor"])


def create_comparison_prompt(
    response_a: str,
    response_b: str,
    case_text: str,
    question_text: str
) -> str:
    """
    Create prompt for pairwise comparison (if needed).
    
    Args:
        response_a: First response
        response_b: Second response
        case_text: Case text
        question_text: Question text
    
    Returns:
        Comparison prompt
    """
    return f"""Compare the following two responses to the given case and question.

## Case
{case_text}

## Question
{question_text}

## Response A
{response_a}

## Response B
{response_b}

## Task
Which response is better? Reply with only "A" or "B"."""


def validate_prompt_template(template: str, required_vars: List[str]) -> bool:
    """
    Validate that a prompt template has required variables.
    
    Args:
        template: Prompt template string
        required_vars: List of required variable names
    
    Returns:
        True if all variables are present
    """
    for var in required_vars:
        if f"{{{var}}}" not in template:
            return False
    return True


def merge_prompts(
    base: DomainPrompts,
    override: DomainPrompts
) -> DomainPrompts:
    """
    Merge two prompt sets, with override taking precedence.
    
    Args:
        base: Base prompts
        override: Override prompts
    
    Returns:
        Merged prompts
    """
    return DomainPrompts(
        case_generation=override.case_generation or base.case_generation,
        question_generation=override.question_generation or base.question_generation,
        answer_generation=override.answer_generation or base.answer_generation,
        judge_ranking=override.judge_ranking or base.judge_ranking,
    )


