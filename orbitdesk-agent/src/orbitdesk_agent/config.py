"""Central configuration: model names, revisions and tunable thresholds.

Keeping these in one place makes the "state the exact model names and
revisions used" requirement easy to satisfy and easy to change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "8b3219a92973c328a8e22fadcfa821b5dc75636" 


    generation_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    generation_model_revision: str = "main"

    max_new_tokens: int = 300
    generation_temperature: float = 0.2


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 4

    min_confidence_for_answer: float = 0.30


@dataclass(frozen=True)
class GraphConfig:
   
    max_generation_attempts: int = 2
    
    recursion_limit: int = 25


MODEL_CONFIG = ModelConfig()
RETRIEVAL_CONFIG = RetrievalConfig()
GRAPH_CONFIG = GraphConfig()
