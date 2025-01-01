from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.commons.config import app_settings
from litellm._logging import verbose_logger
from budmicroframe.commons.schemas import CloudEventBase
from budmicroframe.shared.dapr_service import DaprService


class RequestMetrics(CloudEventBase):
    request_id: UUID
    request_ip: Optional[str] = None
    project_id: UUID
    project_name: str
    endpoint_id: UUID | str
    endpoint_name: str
    endpoint_path: str
    model_id: UUID | str
    provider: str
    modality: str
    request_arrival_time: datetime
    request_forwarded_time: datetime
    response_start_time: datetime
    response_end_time: datetime
    request_body: Dict[str, Any]
    response_body: Union[Dict[str, Any], List[Dict[str, Any]]]
    cost: Optional[float] = None
    is_cache_hit: bool
    is_success: bool

    _model_name: Optional[str] = None
    _input_tokens: Optional[int] = None
    _output_tokens: Optional[int] = None
    _response_analysis: Optional[Dict[str, Any]] = None
    _is_streaming: Optional[bool] = None
    
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

    @property
    def model_name(self) -> Optional[str]:
        return self._model_name

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
        self._model_name = value

    @property
    def input_tokens(self) -> Optional[int]:
        return self._input_tokens

    @input_tokens.setter
    def input_tokens(self, value: Optional[int]) -> None:
        self._input_tokens = value
        
    @property
    def output_tokens(self) -> Optional[int]:
        return self._output_tokens

    @output_tokens.setter
    def output_tokens(self, value: Optional[int]) -> None:
        self._output_tokens = value

    @property
    def response_analysis(self) -> Optional[Dict[str, Any]]:
        return self._response_analysis

    @response_analysis.setter
    def response_analysis(self, value: Optional[Dict[str, Any]]) -> None:
        self._response_analysis = value

    @property
    def is_streaming(self) -> Optional[bool]:
        return self._is_streaming

    @is_streaming.setter
    def is_streaming(self, value: Optional[bool]) -> None:
        self._is_streaming = value
        
    @property
    def ttft(self) -> float:
        if self.is_streaming and self.response_start_time > self.request_arrival_time:
            return round((self.response_start_time - self.request_arrival_time).total_seconds(), 3)

    @property
    def latency(self) -> float:
        if self.response_end_time > self.request_arrival_time:
            return round((self.response_end_time - self.request_arrival_time).total_seconds(), 3)

    @property
    def throughput(self) -> Optional[float]:
        if self.output_tokens is not None and self.latency is not None:
            return round(self.output_tokens / self.latency, 3)
       
 
class UpdateRequestMetrics(CloudEventBase):
    request_id: UUID
    cost: Optional[float] = None


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
        verbose_logger.info(f"\nresponse_obj : {response_obj}")

        model = kwargs.get("model", None)
        is_cache_hit = kwargs.get("cache_hit")
        response_body = kwargs.get("standard_logging_object", {}).get("response", {})
        if failure:
            response_body = {
                "exception": str(kwargs.get("exception", None)),
                "traceback": kwargs.get("traceback_exception", None) 
            }
        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        proxy_server_request = litellm_params.get("proxy_server_request", {})
        if not proxy_server_request:
            proxy_server_request["body"] = {
                "model": model,
                "messages": kwargs.get("messages", []),
                "stream": kwargs.get("stream", False)
            }
        model_info = litellm_params.get("model_info", {})
        metadata = litellm_params.get("metadata", {})   # headers passed to LiteLLM proxy, can be found here

        # Calculate cost using  litellm.completion_cost()
        response_obj = response_obj or {}
        cost = litellm.completion_cost(completion_response=response_obj) if not failure else 0

        usage = response_obj.get("usage", None) or {}
        if isinstance(usage, litellm.Usage):
            usage = dict(usage)
        
        metrics_data = RequestMetrics(
            request_id=kwargs.get("litellm_call_id", None),
            project_id=metadata.get("project_id", None),
            project_name=metadata.get("project_name", None),
            endpoint_id=model_info["metadata"]["endpoint_id"] if model_info else "",
            endpoint_name=model,
            endpoint_path=litellm_params["api_base"] if litellm_params else None,
            model_id=model_info["id"] if model_info else "",
            provider=model_info["metadata"]["provider"] if model_info else "",
            modality=model_info["metadata"]["modality"] if model_info else "",
            request_arrival_time=start_time,
            request_forwarded_time=kwargs.get("api_call_start_time") or start_time,
            response_start_time=kwargs.get("completion_start_time") or end_time,
            response_end_time=end_time,
            request_body=proxy_server_request.get("body", {}),
            response_body=response_body,
            cost=cost,
            is_cache_hit=is_cache_hit or False,
            is_success=not failure,
            # model_name=model_info["metadata"]["name"] if model_info else "",
            # is_streaming=kwargs.get("stream", False)
        )
        verbose_logger.info(f"\n\nMetrics Data: {metrics_data}\n\n")
        return metrics_data

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        verbose_logger.info("On Async Success!")
        metrics_data = self.get_request_metrics(kwargs, response_obj, start_time, end_time)
        with DaprService() as dapr_service:
            dapr_service.publish_to_topic(
                data=metrics_data.model_dump(mode="json"),
                target_topic_name=app_settings.budmetrics_topic_name,
                target_name=app_settings.budmetrics_app_name,
                event_type="add_request_metrics",
            )
        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time): 
        try:
            verbose_logger.info("On Async Failure !")
            metrics_data = self.get_request_metrics(kwargs, response_obj, start_time, end_time, failure=True)
            with DaprService() as dapr_service:
                dapr_service.publish_to_topic(
                    data=metrics_data.model_dump(mode="json"),
                    target_topic_name=app_settings.budmetrics_topic_name,
                    target_name=app_settings.budmetrics_app_name,
                    event_type="add_request_metrics",
                )
        except Exception as e:
            # TODO: what metrics data to log here?
            verbose_logger.info(f"Exception: {e}")

proxy_handler_instance = MyCustomHandler()

# Set litellm.callbacks = [proxy_handler_instance] on the proxy
# need to set litellm.callbacks = [proxy_handler_instance] # on the proxy