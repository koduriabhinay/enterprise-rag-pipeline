"""Answer generation: extractive by default, optional LLM abstractive mode."""

from generation.answer import RAGPipeline, Answer
from generation.extractive import generate_extractive, GeneratedAnswer, Citation
from generation.llm import LlmConfig, generate_abstractive

__all__ = [
    "RAGPipeline",
    "Answer",
    "generate_extractive",
    "GeneratedAnswer",
    "Citation",
    "LlmConfig",
    "generate_abstractive",
]
