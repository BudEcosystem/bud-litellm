import json

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body


class BudServeMiddleware(BaseHTTPMiddleware):
    llm_request_list = [
        "/chat/completions",
        "/completions",
        "/embeddings",
        "/images/generation",
        "/audio/speech",
        "/audio/transcriptions",
    ]

    async def get_api_key(self, request):
        authorization_header = request.headers.get("Authorization")
        api_key = authorization_header.split(" ")[1]
        return api_key

    async def dispatch(
        self,
        request,
        call_next,
    ):
        """
        Steps to prepare user_config

        1. api_key and model (endpoint_name) fetch all endpoint details : model_list
        2. Using models involved in endpoint details, fetch proprietary credentials
        3. Create user_config using model_configuration (endpoint model) and router_config (project model)
        4. Add validations for fallbacks
        """
        route: str = get_request_route(request=request)
        verbose_proxy_logger.info(f"Request: {route}")
        run_through_middleware = any(
            each_route in route for each_route in self.llm_request_list
        )
        verbose_proxy_logger.info(f"Run Through Middleware: {run_through_middleware}")
        if not run_through_middleware:
            return await call_next(request)

        # get the request body
        request_data = await _read_request_body(request=request)
        api_key = await self.get_api_key(request)
        endpoint_name = request_data.get("model")

        # get endpoint details to fill cache_params
        # redis connection params we will set as kubernetes env variables
        # can be fetched using os.getenv
        import os

        request_data["user_config"] = {
            "cache_responses": False,
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": os.getenv("REDIS_PORT", 6379),
            "redis_password": os.getenv("REDIS_PASSWORD", ""),
            "endpoint_cache_settings": {
                "cache": False,
                "type": "redis-semantic",  # gpt_cache_redis
                "cache_params": {
                    "host": os.getenv("REDIS_HOST", "localhost"),
                    "port": os.getenv("REDIS_PORT", 6379),
                    "password": os.getenv("REDIS_PASSWORD", ""),
                    "similarity_threshold": 0.8,
                    "redis_semantic_cache_use_async": False,
                    "redis_semantic_cache_embedding_model": "sentence-transformers/all-mpnet-base-v2",
                    "eviction_policy": {"policy": "ttl", "max_size": 100, "ttl": 600},
                },
            },
            "model_list": [
                {
                    "model_name": "gpt4",
                    "litellm_params": {
                        "model": "openai/gpt-3.5-turbo",
                        "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
                        "rpm": 100,
                        "request_timeout": 120,
                    },
                    "model_info": {"id": "model_id:123"},
                },
                {
                    "model_name": "gpt4",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": os.getenv("OPENAI_API_KEY", "dummy"),
                        "tpm": 10000,
                    },
                    "model_info": {"id": "model_id:456"},
                },
            ],
        }
        request._body = json.dumps(request_data).encode("utf-8")
        return await call_next(request)
