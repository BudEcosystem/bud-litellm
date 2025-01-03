import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from uuid import UUID, uuid4

from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.commons.config import app_settings
from litellm._logging import verbose_logger
from budmicroframe.commons.schemas import CloudEventBase
from budmicroframe.shared.dapr_service import DaprService

# error in budserve_middleware.py
# Scenario 1: if user sends wrong api key
# Scenario 2: if user sends wrong model param

# Keys i won't have:
# project_id, project_name, endpoint_id, endpoint_name (what user has sent), endpoint_path,
# model_id, provider, modality, model_name
# Keys i can set:
# request_arrival_time == request_forwarded_time == response_start_time == response_end_time
# request_body, response_body, cost = 0, is_cache_hit = False, is_success = False, is_streaming = False



class RequestMetrics(CloudEventBase):
    request_id: UUID
    request_ip: Optional[str] = None
    project_id: Optional[UUID]
    project_name: Optional[str]
    endpoint_id: Optional[UUID]
    endpoint_name: Optional[str]
    endpoint_path: Optional[str]
    model_id: Optional[UUID]
    model_name: Optional[str]
    provider: Optional[str]
    modality: Optional[str]
    request_arrival_time: datetime
    request_forwarded_time: datetime
    response_start_time: datetime
    response_end_time: datetime
    request_body: Dict[str, Any]
    response_body: Union[Dict[str, Any], List[Dict[str, Any]]]
    cost: Optional[float] = None
    is_cache_hit: bool
    is_streaming: bool = False
    is_success: bool

    def validate_intervals(self) -> "RequestMetrics":
        if self.response_start_time > self.response_end_time:
            raise ValueError("Response start time cannot be after response end time.")
        if self.request_arrival_time > self.response_start_time:
            raise ValueError("Request arrival time cannot be after response start time.")
        if self.request_forwarded_time > self.response_start_time:
            raise ValueError("Request forwarded time cannot be after response start time.")
        if self.request_arrival_time > self.response_end_time:
            raise ValueError("Request arrival time cannot be after response end time.")
        return self
   
# This file includes the custom callbacks for LiteLLM Proxy
# Once defined, these can be passed in proxy_config.yaml
class MyCustomHandler(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs): 
        verbose_logger.info("Pre-API Call")
    
    def log_post_api_call(self, kwargs, response_obj, start_time, end_time): 
        verbose_logger.info("Post-API Call")

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.info("On Stream")
        
    def log_success_event(self, kwargs, response_obj, start_time, end_time): 
        verbose_logger.info("On Success")

    def log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        verbose_logger.info("On Failure")
        
    def get_request_metrics(self, kwargs, response_obj, start_time, end_time, failure=False) -> RequestMetrics:
        # log: key, user, model, prompt, response, tokens, cost
        # Access kwargs passed to litellm.completion()
        verbose_logger.info(f"\nkwargs : {kwargs}")
        # verbose_logger.info(f"\nresponse_obj : {response_obj}")

        model = kwargs.get("model", None)
        is_cache_hit = kwargs.get("cache_hit")
        response_body = copy.deepcopy(kwargs.get("standard_logging_object", {}).get("response", {})) if not failure else {
            "exception": str(kwargs.get("exception", None)),
            "traceback": kwargs.get("traceback_exception", None) 
        }
        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        proxy_server_request = litellm_params.get("proxy_server_request", {})
        if proxy_server_request and proxy_server_request.get("body"):
            # To handle unserializable metadata in proxy_server_request and circular dependencies
            proxy_server_request["body"].pop("metadata", None)
        if not proxy_server_request:
            proxy_server_request["body"] = {
                "model": model,
                "messages": kwargs.get("messages", []),
                "stream": kwargs.get("stream", False)
            }
        model_info = copy.deepcopy(litellm_params.get("model_info", {}))
        metadata = litellm_params.get("metadata", {})
        endpoint = metadata.get("endpoint", "")
        api_route = urlparse(str(endpoint)).path
        if litellm_params.get("api_base"):
            api_route = f"{litellm_params['api_base']}{api_route}"
        
        # Calculate cost using  litellm.completion_cost()
        response_obj = response_obj or {}
        cost = litellm.completion_cost(completion_response=response_obj) if not failure else 0

        usage = response_obj.get("usage", None) or {}
        if isinstance(usage, litellm.Usage):
            usage = dict(usage)
        
        metrics_data = RequestMetrics(
            request_id=kwargs.get("litellm_call_id", uuid4()),
            project_id=metadata.get("project_id", None),
            project_name=metadata.get("project_name", None),
            endpoint_id=model_info["metadata"]["endpoint_id"] if model_info else None,
            endpoint_name=model,
            endpoint_path=api_route,
            model_id=model_info["id"] if model_info else None,
            model_name=model_info["metadata"]["name"] if model_info else None,
            provider=model_info["metadata"]["provider"] if model_info else None,
            modality=model_info["metadata"]["modality"] if model_info else None,
            request_arrival_time=start_time,
            request_forwarded_time=kwargs.get("api_call_start_time") or start_time,
            response_start_time=kwargs.get("completion_start_time") or end_time,
            response_end_time=end_time,
            request_body=proxy_server_request.get("body", {}),
            response_body=response_body,
            cost=cost,
            is_cache_hit=is_cache_hit or False,
            is_streaming=kwargs.get("stream", False),
            is_success=not failure,
        )
        # verbose_logger.info(f"\n\nMetrics Data: {metrics_data}\n\n")
        return metrics_data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.info("On Async Success!")
        metrics_data = self.get_request_metrics(kwargs, response_obj, start_time, end_time)
        metrics_data_json = metrics_data.model_dump(mode="json")
        verbose_logger.info(f"Metrics Data JSON: {metrics_data_json}")
        with DaprService() as dapr_service:
            dapr_service.publish_to_topic(
                data=metrics_data_json,
                target_topic_name=app_settings.budmetrics_topic_name,
                target_name=app_settings.budmetrics_app_name,
                event_type="add_request_metrics",
            )
        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        try:
            verbose_logger.info("On Async Failure !")
            metrics_data = self.get_request_metrics(kwargs, response_obj, start_time, end_time, failure=True)
            metrics_data_json = metrics_data.model_dump(mode="json")
            verbose_logger.info(f"Metrics Data JSON: {metrics_data_json}")
            with DaprService() as dapr_service:
                dapr_service.publish_to_topic(
                    data=metrics_data_json,
                    target_topic_name=app_settings.budmetrics_topic_name,
                    target_name=app_settings.budmetrics_app_name,
                    event_type="add_request_metrics",
                )
        except Exception as e:
            # TODO: what metrics data to log here?
            import traceback
            verbose_logger.info(f"Exception: {e}")
            verbose_logger.info(f"Traceback: {traceback.format_exc()}")

proxy_handler_instance = MyCustomHandler()

# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy