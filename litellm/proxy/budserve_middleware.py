import json
import httpx
import os

from typing import Optional, Any

from litellm.commons.config import app_settings, secrets_settings

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy._types import ProxyException
from fastapi.responses import JSONResponse


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
        x_api_key_header = request.headers.get("X-Api-Key")
        api_key_header = request.headers.get("Api-Key")
        if not authorization_header and not x_api_key_header and not api_key_header:
            raise ProxyException(
                message="Authorization/X-Api-Key/Api-Key header is missing",
                type="unauthorized",
                param="Authorization",
                code=401,
            )
        if authorization_header:
            api_key = authorization_header.split(" ")[1]
        elif x_api_key_header:
            api_key = x_api_key_header
        elif api_key_header:
            api_key = api_key_header
        return api_key

    async def extract_cache_preference(self, request):
        return request.headers.get("Cache-Preference", "false").lower() == "true"

    async def fetch_user_config(
        self, api_key: Optional[str], endpoint_name: str, user_jwt: Optional[str], project_id: Optional[str]
    ):
        # redis key : router_config:{api_key}:{endpoint_name}
        budserve_app_baseurl = os.getenv("BUDSERVE_APP_BASEURL", "http://localhost:9000")
        url = f"{budserve_app_baseurl}/credentials/router-config"

        # Build params
        params = {"endpoint_name": endpoint_name}
        if api_key:
            params["api_key"] = api_key
        if project_id:
            params["project_id"] = project_id

        # Build headers
        headers = {"Content-Type": "application/json"}
        if user_jwt:
            headers["Authorization"] = f"Bearer {user_jwt}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers, follow_redirects=True)
                verbose_proxy_logger.debug(f"Response: {response}")
                response_data = response.json()
                if response_data.get("success", False):
                    return response_data.get("result", None)
                else:
                    raise ProxyException(
                        message=response_data.get("message", "Error fetching user config"),
                        type="not_found",
                        param=endpoint_name,
                        code=404,
                    )
        except Exception as e:
            verbose_proxy_logger.error(f"Error fetching user config from {url}: {e}")
            if isinstance(e, ProxyException):
                raise e
            else:
                raise ProxyException(
                    message=f"Error fetching user config from {url}: {e}",
                    type="internal_server_error",
                    param=endpoint_name,
                    code=500,
                )

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
        if request.method == "OPTIONS":
            origin = request.headers.get("Origin")  # Get the request's Origin header
            allowed_origins = app_settings.cors_origins.split(",")

            # Check if the request origin is in the allowed list
            if origin in allowed_origins:
                allow_origin = origin  # Allow only the matching origin
            else:
                allow_origin = "null"  # Block if origin is not in the allowed list

            return JSONResponse(
                content={"message": "CORS preflight request successful."},
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": allow_origin,
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "Authorization, Content-Type, *",
                    "Access-Control-Allow-Credentials": "true",
                },
            )
        route: str = get_request_route(request=request)
        verbose_proxy_logger.info(f"Request: {route}")
        run_through_middleware = any(each_route in route for each_route in self.llm_request_list)
        verbose_proxy_logger.info(f"Run Through Middleware: {run_through_middleware}")
        if not run_through_middleware:
            return await call_next(request)

        # get the request body
        request_data = await _read_request_body(request=request)
        request.state.original_body = json.dumps(request_data)
        enable_cache = await self.extract_cache_preference(request)
        api_key = await self.get_api_key(request)
        endpoint_name = request_data.get("model")
        user_jwt = await _get_user_jwt(request)
        project_id = await _get_project_id(request)
        # project_id = await _get_project_id_from_body(request_data)
        # if user_jwt is present, api_key change to None
        if user_jwt:
            api_key = None

        # get endpoint details to fill cache_params
        user_config = await self.fetch_user_config(api_key, endpoint_name, user_jwt, project_id)

        if enable_cache:
            user_config["cache_configuration"] = {
                "score_threshold": 0.5,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "eviction_policy": "LRU",
                "max_size": 1000,
                "ttl": None,
            }

        request_data["metadata"] = {
            "project_id": user_config.get("project_id"),
            "project_name": user_config.get("project_name"),
        }

        # redis connection params we will set as kubernetes env variables
        # can be fetched using os.getenv
        request_data["user_config"] = {
            "cache_responses": False if not user_config.get("cache_configuration") else True,
            "redis_host": app_settings.redis_host,
            "redis_port": app_settings.redis_port,
            "redis_password": secrets_settings.redis_password,
            "endpoint_cache_settings": {
                "cache": False if not user_config.get("cache_configuration") else True,
                "type": "gpt_cache_redis",  # redis-semantic
                "cache_params": {
                    "host": app_settings.cache_redis_host,
                    "port": app_settings.cache_redis_port,
                    "password": secrets_settings.cache_redis_password,
                    "similarity_threshold": user_config.get("cache_configuration", {}).get("score_threshold")
                    if user_config.get("cache_configuration")
                    else app_settings.cache_score_threshold,
                    "redis_semantic_cache_use_async": False,
                    "redis_semantic_cache_embedding_model": user_config.get("cache_configuration", {}).get(
                        "embedding_model"
                    )
                    if user_config.get("cache_configuration")
                    else app_settings.cache_embedding_model,
                    "eviction_policy": {
                        "policy": user_config.get("cache_configuration", {}).get("eviction_policy")
                        if user_config.get("cache_configuration")
                        else app_settings.cache_eviction_policy,
                        "max_size": user_config.get("cache_configuration", {}).get("max_size")
                        if user_config.get("cache_configuration")
                        else app_settings.cache_max_size,
                        "ttl": user_config.get("cache_configuration", {}).get("ttl")
                        if user_config.get("cache_configuration")
                        else app_settings.cache_ttl,
                    },
                },
            },
            "routing_strategy_args": {"routing_policy": user_config.get("routing_policy") or {}},
            "model_list": user_config.get("model_configuration", []),
        }
        request_data['model'] = user_config.get("model_configuration", {}).get("model_name")
        
        request._body = json.dumps(request_data).encode("utf-8")
        return await call_next(request)


async def _get_user_jwt(request: Request):
    """Get the user jwt from the request headers"""
    authorization_header = request.headers.get("Authorization")
    if authorization_header:
        auth_token = authorization_header.split(" ")[1]

        if auth_token and isinstance(auth_token, str):
            auth_token_len = auth_token.split(".")
            if len(auth_token_len) == 3:
                return auth_token

    return None


async def _get_project_id(request: Request):
    """Get the project id from the request headers"""
    return request.headers.get("Project-Id")


async def _get_project_id_from_body(request_data: dict[str, Any]):
    """Get the project id from the request headers"""
    return request_data.get("metadata", {}).get("project_id")
