from typing import Dict, Any, Optional, Callable
from functools import cached_property
from pydantic import (
    BaseModel, computed_field, ConfigDict,
    UUID4, field_validator, ValidationInfo,
)
from langchain.embeddings.base import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
# from ..constants import CredentialTypeEnum
# from ..core.config import get_settings

# settings = get_settings()
class CredentialTypeEnum(Enum):
    """Credential types"""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    BUDSERVE = "budserve"

class CacheMetricConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enable_metrics: bool = os.getenv("ENABLE_CACHE_METRIC")
    request_start_time: Optional[float] = None
    metric_request_id: Optional[UUID4] = None
    endpoint_id: Optional[UUID4] = None
    project_id: Optional[UUID4] = None
    model_id: Optional[UUID4] = None
    engine: Optional[CredentialTypeEnum] = None
    api_endpoint: Optional[str] = None

    @field_validator("enable_metrics")
    @classmethod
    def validate_enable_metrics(cls, value: bool, info: ValidationInfo) -> bool:
        return value and bool(info.data["metric_request_id"])


class EvictionPolicy(BaseModel):
    policy: str = os.environ("CACHE_EVICTION_POLICY")
    max_size: int = os.environ("CACHE_MAX_SIZE")
    ttl: Optional[int] = os.environ("CACHE_TTL")


class BudServeCacheConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    embedding_model: Optional[str] = os.environ("CACHE_EMBEDDING_MODEL")
    eviction_policy: Optional[EvictionPolicy] = EvictionPolicy()
    score_threshold: float = os.environ("CACHE_SCORE_THRESHOLD")

    metric_config: Optional[CacheMetricConfig] = None

    @computed_field
    @cached_property
    def embeddings(self) -> Optional[Embeddings]:
        if not self.embedding_model:
            return None
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
        )
  


def create_cache_config(cache_config: Dict[str, Any]) -> Dict[str, Any]:
    llm_cache_config = {}
    llm_cache_config["embedding_model"] = (
        cache_config.get("embedding_model") or os.environ("CACHE_EMBEDDING_MODEL")
    )
    llm_cache_config["eviction_policy"] = (
        cache_config.get("eviction_policy") or os.environ("CACHE_EVICTION_POLICY")
    )
    llm_cache_config["max_size"] = (
        cache_config.get("max_size") or os.environ("CACHE_MAX_SIZE")
    )
    llm_cache_config["ttl"] = cache_config.get("ttl") or os.environ("CACHE_TTL")
    llm_cache_config["score_threshold"] = (
        cache_config.get("score_threshold") or os.environ("CACHE_SCORE_THRESHOLD")
    )
    return llm_cache_config


if __name__ == "__main__":
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    eviction_policy = EvictionPolicy()
    cache_config = BudServeCacheConfig(
        embedding_model=embedding_model, eviction_policy=eviction_policy
    )
    print(cache_config.model_dump())
