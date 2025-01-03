#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""Manages application and secret configurations, utilizing environment variables and Dapr's configuration store for syncing."""

from pathlib import Path
from typing import Optional

from budmicroframe.commons.config import BaseAppConfig, BaseSecretsConfig, register_settings, enable_periodic_sync_from_store
from pydantic import DirectoryPath, Field

from litellm.__about__ import __version__


class AppConfig(BaseAppConfig):
    name: str = __version__.split("@")[0]
    version: str = __version__.split("@")[-1]
    description: str = Field("Bud-Litellm is a proxy server for LLM requests.", alias="DOCS_DESCRIPTION")
    api_root: str = Field("", alias="SERVER_ROOT_PATH")

    # Base Directory
    base_dir: DirectoryPath = Path(__file__).parent.parent.parent.resolve()

    # Bud-Litellm env
    litellm_log: str = Field("DEBUG", alias="LITELLM_LOG")
    litellm_master_key: str = Field("sk-1234", alias="LITELLM_MASTER_KEY")
    litellm_salt_key: str = Field("litellm_salt_key", alias="LITELLM_SALT_KEY")
    database_url: str = Field(..., alias="DATABASE_URL")
    store_model_in_db: bool = Field(True, alias="STORE_MODEL_IN_DB")
    budserve_app_baseurl: str = Field("http://127.0.0.1:8050", alias="BUDSERVE_APP_BASEURL")
    
    # Redis Config
    redis_host: str = Field("localhost", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_username: str = Field("", alias="REDIS_USERNAME")
    redis_password: str = Field("", alias="REDIS_PASSWORD")
    redis_db: int = Field(0, alias="REDIS_DB")
    
    # Cache Config
    enable_cache: bool = Field(False, alias="ENABLE_CACHE")
    enable_cache_metric: bool = Field(False, alias="ENABLE_CACHE_METRIC")
    cache_eviction_policy: str = Field("LRU", alias="CACHE_EVICTION_POLICY")
    cache_max_size: int = Field(1000, alias="CACHE_MAX_SIZE")
    cache_ttl: int = Field(3600, alias="CACHE_TTL")
    cache_score_threshold: float = Field(0.8, alias="CACHE_SCORE_THRESHOLD")
    cache_embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", alias="CACHE_EMBEDDING_MODEL")

    # Metrics App and Topic
    budmetrics_app_name: str = Field("budMetrics", alias="BUDMETRICS_APP_NAME")
    budmetrics_topic_name: str = Field("budMetricsMessages", alias="BUDMETRICS_TOPIC_NAME")
    

class SecretsConfig(BaseSecretsConfig):
    name: str = __version__.split("@")[0]
    version: str = __version__.split("@")[-1]
    
    # Database
    psql_user: Optional[str] = Field(
        None,
        alias="PSQL_USER",
        json_schema_extra=enable_periodic_sync_from_store(is_global=True),
    )
    psql_password: Optional[str] = Field(
        None,
        alias="PSQL_PASSWORD",
        json_schema_extra=enable_periodic_sync_from_store(is_global=True),
    )


app_settings = AppConfig()
secrets_settings = SecretsConfig()

register_settings(app_settings, secrets_settings)