import copy
import sys
import time
from datetime import datetime
from unittest import mock

from dotenv import load_dotenv

from litellm.types.utils import StandardCallbackDynamicParams

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, headers
from litellm.proxy.utils import (
    duration_in_seconds,
    _extract_from_regex,
    get_last_day_of_month,
)
from litellm.utils import (
    check_valid_key,
    create_pretrained_tokenizer,
    create_tokenizer,
    function_to_dict,
    get_llm_provider,
    get_max_tokens,
    get_supported_openai_params,
    get_token_count,
    get_valid_models,
    token_counter,
    trim_messages,
    validate_environment,
)
from unittest.mock import AsyncMock, MagicMock, patch


# Assuming your trim_messages, shorten_message_to_fit_limit, and get_token_count functions are all in a module named 'message_utils'


# Test 1: Check trimming of normal message
def test_basic_trimming():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that definitely exceeds the token limit.",
        }
    ]
    trimmed_messages = trim_messages(messages, model="claude-2", max_tokens=8)
    print("trimmed messages")
    print(trimmed_messages)
    # print(get_token_count(messages=trimmed_messages, model="claude-2"))
    assert (get_token_count(messages=trimmed_messages, model="claude-2")) <= 8


# test_basic_trimming()


def test_basic_trimming_no_max_tokens_specified():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that is definitely under the token limit.",
        }
    ]
    trimmed_messages = trim_messages(messages, model="gpt-4")
    print("trimmed messages for gpt-4")
    print(trimmed_messages)
    # print(get_token_count(messages=trimmed_messages, model="claude-2"))
    assert (
        get_token_count(messages=trimmed_messages, model="gpt-4")
    ) <= litellm.model_cost["gpt-4"]["max_tokens"]


# test_basic_trimming_no_max_tokens_specified()


def test_multiple_messages_trimming():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that will exceed the token limit.",
        },
        {
            "role": "user",
            "content": "This is another long message that will also exceed the limit.",
        },
    ]
    trimmed_messages = trim_messages(
        messages=messages, model="gpt-3.5-turbo", max_tokens=20
    )
    # print(get_token_count(messages=trimmed_messages, model="gpt-3.5-turbo"))
    assert (get_token_count(messages=trimmed_messages, model="gpt-3.5-turbo")) <= 20


# test_multiple_messages_trimming()


def test_multiple_messages_no_trimming():
    messages = [
        {
            "role": "user",
            "content": "This is a long message that will exceed the token limit.",
        },
        {
            "role": "user",
            "content": "This is another long message that will also exceed the limit.",
        },
    ]
    trimmed_messages = trim_messages(
        messages=messages, model="gpt-3.5-turbo", max_tokens=100
    )
    print("Trimmed messages")
    print(trimmed_messages)
    assert messages == trimmed_messages


# test_multiple_messages_no_trimming()


def test_large_trimming_multiple_messages():
    messages = [
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."},
    ]
    trimmed_messages = trim_messages(messages, max_tokens=20, model="gpt-4-0613")
    print("trimmed messages")
    print(trimmed_messages)
    assert (get_token_count(messages=trimmed_messages, model="gpt-4-0613")) <= 20


# test_large_trimming()


def test_large_trimming_single_message():
    messages = [
        {"role": "user", "content": "This is a singlelongwordthatexceedsthelimit."}
    ]
    trimmed_messages = trim_messages(messages, max_tokens=5, model="gpt-4-0613")
    assert (get_token_count(messages=trimmed_messages, model="gpt-4-0613")) <= 5
    assert (get_token_count(messages=trimmed_messages, model="gpt-4-0613")) > 0


def test_trimming_with_system_message_within_max_tokens():
    # This message is 33 tokens long
    messages = [
        {"role": "system", "content": "This is a short system message"},
        {
            "role": "user",
            "content": "This is a medium normal message, let's say litellm is awesome.",
        },
    ]
    trimmed_messages = trim_messages(
        messages, max_tokens=30, model="gpt-4-0613"
    )  # The system message should fit within the token limit
    assert len(trimmed_messages) == 2
    assert trimmed_messages[0]["content"] == "This is a short system message"


def test_trimming_with_system_message_exceeding_max_tokens():
    # This message is 33 tokens long. The system message is 13 tokens long.
    messages = [
        {"role": "system", "content": "This is a short system message"},
        {
            "role": "user",
            "content": "This is a medium normal message, let's say litellm is awesome.",
        },
    ]
    trimmed_messages = trim_messages(messages, max_tokens=12, model="gpt-4-0613")
    assert len(trimmed_messages) == 1


def test_trimming_with_tool_calls():
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        },
        Message(
            content=None,
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "San Francisco, CA", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_G11shFcS024xEKjiAOSt6Tc9",
                    type="function",
                ),
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "Tokyo, Japan", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_e0ss43Bg7H8Z9KGdMGWyZ9Mj",
                    type="function",
                ),
                ChatCompletionMessageToolCall(
                    function=Function(
                        arguments='{"location": "Paris, France", "unit": "celsius"}',
                        name="get_current_weather",
                    ),
                    id="call_nRjLXkWTJU2a4l9PZAf5as6g",
                    type="function",
                ),
            ],
            function_call=None,
        ),
        {
            "tool_call_id": "call_G11shFcS024xEKjiAOSt6Tc9",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
        },
        {
            "tool_call_id": "call_e0ss43Bg7H8Z9KGdMGWyZ9Mj",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "Tokyo", "temperature": "10", "unit": "celsius"}',
        },
        {
            "tool_call_id": "call_nRjLXkWTJU2a4l9PZAf5as6g",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "Paris", "temperature": "22", "unit": "celsius"}',
        },
    ]
    result = trim_messages(messages=messages, max_tokens=1, return_response_tokens=True)

    print(result)

    assert len(result[0]) == 3  # final 3 messages are tool calls


def test_trimming_should_not_change_original_messages():
    messages = [
        {"role": "system", "content": "This is a short system message"},
        {
            "role": "user",
            "content": "This is a medium normal message, let's say litellm is awesome.",
        },
    ]
    messages_copy = copy.deepcopy(messages)
    trimmed_messages = trim_messages(messages, max_tokens=12, model="gpt-4-0613")
    assert messages == messages_copy


@pytest.mark.parametrize("model", ["gpt-4-0125-preview", "claude-3-opus-20240229"])
def test_trimming_with_model_cost_max_input_tokens(model):
    messages = [
        {"role": "system", "content": "This is a normal system message"},
        {
            "role": "user",
            "content": "This is a sentence" * 100000,
        },
    ]
    trimmed_messages = trim_messages(messages, model=model)
    assert (
        get_token_count(trimmed_messages, model=model)
        < litellm.model_cost[model]["max_input_tokens"]
    )


def test_aget_valid_models():
    old_environ = os.environ
    os.environ = {"OPENAI_API_KEY": "temp"}  # mock set only openai key in environ

    valid_models = get_valid_models()
    print(valid_models)

    # list of openai supported llms on litellm
    expected_models = (
        litellm.open_ai_chat_completion_models + litellm.open_ai_text_completion_models
    )

    assert valid_models == expected_models

    # reset replicate env key
    os.environ = old_environ

    # GEMINI
    expected_models = litellm.gemini_models
    old_environ = os.environ
    os.environ = {"GEMINI_API_KEY": "temp"}  # mock set only openai key in environ

    valid_models = get_valid_models()

    print(valid_models)
    assert valid_models == expected_models

    # reset replicate env key
    os.environ = old_environ


# test_get_valid_models()


def test_bad_key():
    key = "bad-key"
    response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
    print(response, key)
    assert response == False


def test_good_key():
    key = os.environ["OPENAI_API_KEY"]
    response = check_valid_key(model="gpt-3.5-turbo", api_key=key)
    assert response == True


# test validate environment


def test_validate_environment_empty_model():
    api_key = validate_environment()
    if api_key is None:
        raise Exception()


def test_validate_environment_api_key():
    response_obj = validate_environment(model="gpt-3.5-turbo", api_key="sk-my-test-key")
    assert (
        response_obj["keys_in_environment"] is True
    ), f"Missing keys={response_obj['missing_keys']}"


def test_validate_environment_api_base_dynamic():
    for provider in ["ollama", "ollama_chat"]:
        kv = validate_environment(provider + "/mistral", api_base="https://example.com")
        assert kv["keys_in_environment"]
        assert kv["missing_keys"] == []


@mock.patch.dict(os.environ, {"OLLAMA_API_BASE": "foo"}, clear=True)
def test_validate_environment_ollama():
    for provider in ["ollama", "ollama_chat"]:
        kv = validate_environment(provider + "/mistral")
        assert kv["keys_in_environment"]
        assert kv["missing_keys"] == []


@mock.patch.dict(os.environ, {}, clear=True)
def test_validate_environment_ollama_failed():
    for provider in ["ollama", "ollama_chat"]:
        kv = validate_environment(provider + "/mistral")
        assert not kv["keys_in_environment"]
        assert kv["missing_keys"] == ["OLLAMA_API_BASE"]


def test_function_to_dict():
    print("testing function to dict for get current weather")

    def get_current_weather(location: str, unit: str):
        """Get the current weather in a given location

        Parameters
        ----------
        location : str
            The city and state, e.g. San Francisco, CA
        unit : {'celsius', 'fahrenheit'}
            Temperature unit

        Returns
        -------
        str
            a sentence indicating the weather
        """
        if location == "Boston, MA":
            return "The weather is 12F"

    function_json = litellm.utils.function_to_dict(get_current_weather)
    print(function_json)

    expected_output = {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit",
                    "enum": "['fahrenheit', 'celsius']",
                },
            },
            "required": ["location", "unit"],
        },
    }
    print(expected_output)

    assert function_json["name"] == expected_output["name"]
    assert function_json["description"] == expected_output["description"]
    assert function_json["parameters"]["type"] == expected_output["parameters"]["type"]
    assert (
        function_json["parameters"]["properties"]["location"]
        == expected_output["parameters"]["properties"]["location"]
    )

    # the enum can change it can be - which is why we don't assert on unit
    # {'type': 'string', 'description': 'Temperature unit', 'enum': "['fahrenheit', 'celsius']"}
    # {'type': 'string', 'description': 'Temperature unit', 'enum': "['celsius', 'fahrenheit']"}

    assert (
        function_json["parameters"]["required"]
        == expected_output["parameters"]["required"]
    )

    print("passed")


# test_function_to_dict()


def test_token_counter():
    try:
        messages = [{"role": "user", "content": "hi how are you what time is it"}]
        tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
        print("gpt-35-turbo")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="claude-2", messages=messages)
        print("claude-2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="gemini/chat-bison", messages=messages)
        print("gemini/chat-bison")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="ollama/llama2", messages=messages)
        print("ollama/llama2")
        print(tokens)
        assert tokens > 0

        tokens = token_counter(model="anthropic.claude-instant-v1", messages=messages)
        print("anthropic.claude-instant-v1")
        print(tokens)
        assert tokens > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_token_counter()


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-3.5-turbo", True),
        ("azure/gpt-4-1106-preview", True),
        ("groq/gemma-7b-it", True),
        ("anthropic.claude-instant-v1", False),
        ("gemini/gemini-1.5-flash", True),
    ],
)
def test_supports_function_calling(model, expected_bool):
    try:
        assert litellm.supports_function_calling(model=model) == expected_bool
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_get_max_token_unit_test():
    """
    More complete testing in `test_completion_cost.py`
    """
    model = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

    max_tokens = get_max_tokens(
        model
    )  # Returns a number instead of throwing an Exception

    assert isinstance(max_tokens, int)


def test_get_supported_openai_params() -> None:
    # Mapped provider
    assert isinstance(get_supported_openai_params("gpt-4"), list)

    # Unmapped provider
    assert get_supported_openai_params("nonexistent") is None


def test_redact_msgs_from_logs():
    """
    Tests that turn_off_message_logging does not modify the response_obj

    On the proxy some users were seeing the redaction impact client side responses
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.redact_messages import (
        redact_message_input_output_from_logging,
    )

    litellm.turn_off_message_logging = True

    response_obj = litellm.ModelResponse(
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner.",
                    "role": "assistant",
                },
            }
        ]
    )

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )

    # Assert the response_obj content is NOT modified
    assert (
        response_obj.choices[0].message.content
        == "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner."
    )

    litellm.turn_off_message_logging = False
    print("Test passed")


def test_redact_msgs_from_logs_with_dynamic_params():
    """
    Tests redaction behavior based on standard_callback_dynamic_params setting:
    In all tests litellm.turn_off_message_logging is True


    1. When standard_callback_dynamic_params.turn_off_message_logging is False (or not set): No redaction should occur. User has opted out of redaction.
    2. When standard_callback_dynamic_params.turn_off_message_logging is True: Redaction should occur. User has opted in to redaction.
    3. standard_callback_dynamic_params.turn_off_message_logging not set, litellm.turn_off_message_logging is True: Redaction should occur.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.redact_messages import (
        redact_message_input_output_from_logging,
    )

    litellm.turn_off_message_logging = True
    test_content = "I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner."
    response_obj = litellm.ModelResponse(
        choices=[
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": test_content,
                    "role": "assistant",
                },
            }
        ]
    )

    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id="1234",
        start_time=datetime.now(),
        function_id="1234",
    )

    # Test Case 1: standard_callback_dynamic_params = False (or not set)
    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        turn_off_message_logging=False
    )
    litellm_logging_obj.model_call_details["standard_callback_dynamic_params"] = (
        standard_callback_dynamic_params
    )
    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )
    # Assert no redaction occurred
    assert _redacted_response_obj.choices[0].message.content == test_content

    # Test Case 2: standard_callback_dynamic_params = True
    standard_callback_dynamic_params = StandardCallbackDynamicParams(
        turn_off_message_logging=True
    )
    litellm_logging_obj.model_call_details["standard_callback_dynamic_params"] = (
        standard_callback_dynamic_params
    )
    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )
    # Assert redaction occurred
    assert _redacted_response_obj.choices[0].message.content == "redacted-by-litellm"

    # Test Case 3: standard_callback_dynamic_params does not override litellm.turn_off_message_logging
    # since litellm.turn_off_message_logging is True redaction should occur
    standard_callback_dynamic_params = StandardCallbackDynamicParams()
    litellm_logging_obj.model_call_details["standard_callback_dynamic_params"] = (
        standard_callback_dynamic_params
    )
    _redacted_response_obj = redact_message_input_output_from_logging(
        result=response_obj,
        model_call_details=litellm_logging_obj.model_call_details,
    )
    # Assert no redaction occurred
    assert _redacted_response_obj.choices[0].message.content == "redacted-by-litellm"

    # Reset settings
    litellm.turn_off_message_logging = False
    print("Test passed")


@pytest.mark.parametrize(
    "duration, unit",
    [("7s", "s"), ("7m", "m"), ("7h", "h"), ("7d", "d"), ("7mo", "mo")],
)
def test_extract_from_regex(duration, unit):
    value, _unit = _extract_from_regex(duration=duration)

    assert value == 7
    assert _unit == unit


def test_duration_in_seconds():
    """
    Test if duration int is correctly calculated for different str
    """
    import time

    now = time.time()
    current_time = datetime.fromtimestamp(now)

    if current_time.month == 12:
        target_year = current_time.year + 1
        target_month = 1
    else:
        target_year = current_time.year
        target_month = current_time.month + 1

    # Determine the day to set for next month
    target_day = current_time.day
    last_day_of_target_month = get_last_day_of_month(target_year, target_month)

    if target_day > last_day_of_target_month:
        target_day = last_day_of_target_month

    next_month = datetime(
        year=target_year,
        month=target_month,
        day=target_day,
        hour=current_time.hour,
        minute=current_time.minute,
        second=current_time.second,
        microsecond=current_time.microsecond,
    )

    # Calculate the duration until the first day of the next month
    duration_until_next_month = next_month - current_time
    expected_duration = int(duration_until_next_month.total_seconds())

    value = duration_in_seconds(duration="1mo")

    assert value - expected_duration < 2


def test_get_llm_provider_ft_models():
    """
    All ft prefixed models should map to OpenAI
    gpt-3.5-turbo-0125 (recommended),
    gpt-3.5-turbo-1106,
    gpt-3.5-turbo,
    gpt-4-0613 (experimental)
    gpt-4o-2024-05-13.
    babbage-002, davinci-002,

    """
    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-0125")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo-1106")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-4-0613")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-3.5-turbo")
    assert custom_llm_provider == "openai"

    model, custom_llm_provider, _, _ = get_llm_provider(model="ft:gpt-4o-2024-05-13")
    assert custom_llm_provider == "openai"


@pytest.mark.parametrize("langfuse_trace_id", [None, "my-unique-trace-id"])
@pytest.mark.parametrize(
    "langfuse_existing_trace_id", [None, "my-unique-existing-trace-id"]
)
def test_logging_trace_id(langfuse_trace_id, langfuse_existing_trace_id):
    """
    - Unit test for `_get_trace_id` function in Logging obj
    """
    from litellm.litellm_core_utils.litellm_logging import Logging

    litellm.success_callback = ["langfuse"]
    litellm_call_id = "my-unique-call-id"
    litellm_logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="acompletion",
        litellm_call_id=litellm_call_id,
        start_time=datetime.now(),
        function_id="1234",
    )

    metadata = {}

    if langfuse_trace_id is not None:
        metadata["trace_id"] = langfuse_trace_id
    if langfuse_existing_trace_id is not None:
        metadata["existing_trace_id"] = langfuse_existing_trace_id

    litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey how's it going?"}],
        mock_response="Hey!",
        litellm_logging_obj=litellm_logging_obj,
        metadata=metadata,
    )

    time.sleep(3)
    assert litellm_logging_obj._get_trace_id(service_name="langfuse") is not None

    ## if existing_trace_id exists
    if langfuse_existing_trace_id is not None:
        assert (
            litellm_logging_obj._get_trace_id(service_name="langfuse")
            == langfuse_existing_trace_id
        )
    ## if trace_id exists
    elif langfuse_trace_id is not None:
        assert (
            litellm_logging_obj._get_trace_id(service_name="langfuse")
            == langfuse_trace_id
        )
    ## if existing_trace_id exists
    else:
        assert (
            litellm_logging_obj._get_trace_id(service_name="langfuse")
            == litellm_call_id
        )


def test_convert_model_response_object():
    """
    Unit test to ensure model response object correctly handles openrouter errors.
    """
    args = {
        "response_object": {
            "id": None,
            "choices": None,
            "created": None,
            "model": None,
            "object": None,
            "service_tier": None,
            "system_fingerprint": None,
            "usage": None,
            "error": {
                "message": '{"type":"error","error":{"type":"invalid_request_error","message":"Output blocked by content filtering policy"}}',
                "code": 400,
            },
        },
        "model_response_object": litellm.ModelResponse(
            id="chatcmpl-b88ce43a-7bfc-437c-b8cc-e90d59372cfb",
            choices=[
                litellm.Choices(
                    finish_reason="stop",
                    index=0,
                    message=litellm.Message(content="default", role="assistant"),
                )
            ],
            created=1719376241,
            model="openrouter/anthropic/claude-3.5-sonnet",
            object="chat.completion",
            system_fingerprint=None,
            usage=litellm.Usage(),
        ),
        "response_type": "completion",
        "stream": False,
        "start_time": None,
        "end_time": None,
        "hidden_params": None,
    }

    try:
        litellm.convert_to_model_response_object(**args)
        pytest.fail("Expected this to fail")
    except Exception as e:
        assert hasattr(e, "status_code")
        assert e.status_code == 400
        assert hasattr(e, "message")
        assert (
            e.message
            == '{"type":"error","error":{"type":"invalid_request_error","message":"Output blocked by content filtering policy"}}'
        )


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("vertex_ai/gemini-1.5-pro", True),
        ("gemini/gemini-1.5-pro", True),
        ("predibase/llama3-8b-instruct", True),
        ("gpt-3.5-turbo", False),
        ("groq/llama3-70b-8192", True),
    ],
)
def test_supports_response_schema(model, expected_bool):
    """
    Unit tests for 'supports_response_schema' helper function.

    Should be true for gemini-1.5-pro on google ai studio / vertex ai AND predibase models
    Should be false otherwise
    """
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import supports_response_schema

    response = supports_response_schema(model=model, custom_llm_provider=None)

    assert expected_bool == response


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-3.5-turbo", True),
        ("gpt-4", True),
        ("command-nightly", False),
        ("gemini-pro", True),
    ],
)
def test_supports_function_calling_v2(model, expected_bool):
    """
    Unit test for 'supports_function_calling' helper function.
    """
    from litellm.utils import supports_function_calling

    response = supports_function_calling(model=model, custom_llm_provider=None)
    assert expected_bool == response


@pytest.mark.parametrize(
    "model, expected_bool",
    [
        ("gpt-4-vision-preview", True),
        ("gpt-3.5-turbo", False),
        ("claude-3-opus-20240229", True),
        ("gemini-pro-vision", True),
        ("command-nightly", False),
    ],
)
def test_supports_vision(model, expected_bool):
    """
    Unit test for 'supports_vision' helper function.
    """
    from litellm.utils import supports_vision

    response = supports_vision(model=model, custom_llm_provider=None)
    assert expected_bool == response


def test_usage_object_null_tokens():
    """
    Unit test.

    Asserts Usage obj always returns int.

    Fixes https://github.com/BerriAI/litellm/issues/5096
    """
    usage_obj = litellm.Usage(prompt_tokens=2, completion_tokens=None, total_tokens=2)

    assert usage_obj.completion_tokens == 0


def test_is_base64_encoded():
    import base64

    import requests

    litellm.set_verbose = True
    url = "https://dummyimage.com/100/100/fff&text=Test+image"
    response = requests.get(url)
    file_data = response.content

    encoded_file = base64.b64encode(file_data).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_file}"

    from litellm.utils import is_base64_encoded

    assert is_base64_encoded(s=base64_image) is True


@mock.patch("httpx.AsyncClient")
@mock.patch.dict(
    os.environ,
    {"SSL_VERIFY": "/certificate.pem", "SSL_CERTIFICATE": "/client.pem"},
    clear=True,
)
def test_async_http_handler(mock_async_client):
    import httpx

    timeout = 120
    event_hooks = {"request": [lambda r: r]}
    concurrent_limit = 2

    AsyncHTTPHandler(timeout, event_hooks, concurrent_limit)

    mock_async_client.assert_called_with(
        cert="/client.pem",
        transport=None,
        event_hooks=event_hooks,
        headers=headers,
        limits=httpx.Limits(
            max_connections=concurrent_limit,
            max_keepalive_connections=concurrent_limit,
        ),
        timeout=timeout,
        verify="/certificate.pem",
    )


@mock.patch("httpx.AsyncClient")
@mock.patch.dict(os.environ, {}, clear=True)
def test_async_http_handler_force_ipv4(mock_async_client):
    """
    Test AsyncHTTPHandler when litellm.force_ipv4 is True

    This is prod test - we need to ensure that httpx always uses ipv4 when litellm.force_ipv4 is True
    """
    import httpx
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    # Set force_ipv4 to True
    litellm.force_ipv4 = True

    try:
        timeout = 120
        event_hooks = {"request": [lambda r: r]}
        concurrent_limit = 2

        AsyncHTTPHandler(timeout, event_hooks, concurrent_limit)

        # Get the call arguments
        call_args = mock_async_client.call_args[1]

        ############# IMPORTANT ASSERTION #################
        # Assert transport exists and is configured correctly for using ipv4
        assert isinstance(call_args["transport"], httpx.AsyncHTTPTransport)
        print(call_args["transport"])
        assert call_args["transport"]._pool._local_address == "0.0.0.0"
        ####################################

        # Assert other parameters match
        assert call_args["event_hooks"] == event_hooks
        assert call_args["headers"] == headers
        assert isinstance(call_args["limits"], httpx.Limits)
        assert call_args["limits"].max_connections == concurrent_limit
        assert call_args["limits"].max_keepalive_connections == concurrent_limit
        assert call_args["timeout"] == timeout
        assert call_args["verify"] is True
        assert call_args["cert"] is None

    finally:
        # Reset force_ipv4 to default
        litellm.force_ipv4 = False


@pytest.mark.parametrize(
    "model, expected_bool", [("gpt-3.5-turbo", False), ("gpt-4o-audio-preview", True)]
)
def test_supports_audio_input(model, expected_bool):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    from litellm.utils import supports_audio_input, supports_audio_output

    supports_pc = supports_audio_input(model=model)

    assert supports_pc == expected_bool


def test_is_base64_encoded_2():
    from litellm.utils import is_base64_encoded

    assert (
        is_base64_encoded(
            s="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/x+AAwMCAO+ip1sAAAAASUVORK5CYII="
        )
        is True
    )

    assert is_base64_encoded(s="Dog") is False


@pytest.mark.parametrize(
    "messages, expected_bool",
    [
        ([{"role": "user", "content": "hi"}], True),
        ([{"role": "user", "content": [{"type": "text", "text": "hi"}]}], True),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "url": "https://example.com/image.png"}
                    ],
                }
            ],
            True,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image",
                            "source": {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "1234",
                                },
                            },
                        },
                    ],
                }
            ],
            False,
        ),
    ],
)
def test_validate_chat_completion_user_messages(messages, expected_bool):
    from litellm.utils import validate_chat_completion_user_messages

    if expected_bool:
        ## Valid message
        validate_chat_completion_user_messages(messages=messages)
    else:
        ## Invalid message
        with pytest.raises(Exception):
            validate_chat_completion_user_messages(messages=messages)


def test_models_by_provider():
    """
    Make sure all providers from model map are in the valid providers list
    """
    from litellm import models_by_provider

    providers = set()
    for k, v in litellm.model_cost.items():
        if "_" in v["litellm_provider"] and "-" in v["litellm_provider"]:
            continue
        elif k == "sample_spec":
            continue
        elif (
            v["litellm_provider"] == "sagemaker"
            or v["litellm_provider"] == "bedrock_converse"
        ):
            continue
        else:
            providers.add(v["litellm_provider"])

    for provider in providers:
        assert provider in models_by_provider.keys()


@pytest.mark.parametrize(
    "litellm_params, disable_end_user_cost_tracking, expected_end_user_id",
    [
        ({}, False, None),
        ({"proxy_server_request": {"body": {"user": "123"}}}, False, "123"),
        ({"proxy_server_request": {"body": {"user": "123"}}}, True, None),
    ],
)
def test_get_end_user_id_for_cost_tracking(
    litellm_params, disable_end_user_cost_tracking, expected_end_user_id
):
    from litellm.utils import get_end_user_id_for_cost_tracking

    litellm.disable_end_user_cost_tracking = disable_end_user_cost_tracking
    assert (
        get_end_user_id_for_cost_tracking(litellm_params=litellm_params)
        == expected_end_user_id
    )


@pytest.mark.parametrize(
    "litellm_params, disable_end_user_cost_tracking_prometheus_only, expected_end_user_id",
    [
        ({}, False, None),
        ({"proxy_server_request": {"body": {"user": "123"}}}, False, "123"),
        ({"proxy_server_request": {"body": {"user": "123"}}}, True, None),
    ],
)
def test_get_end_user_id_for_cost_tracking_prometheus_only(
    litellm_params, disable_end_user_cost_tracking_prometheus_only, expected_end_user_id
):
    from litellm.utils import get_end_user_id_for_cost_tracking

    litellm.disable_end_user_cost_tracking_prometheus_only = (
        disable_end_user_cost_tracking_prometheus_only
    )
    assert (
        get_end_user_id_for_cost_tracking(
            litellm_params=litellm_params, service_type="prometheus"
        )
        == expected_end_user_id
    )


def test_is_prompt_caching_enabled_error_handling():
    """
    Assert that `is_prompt_caching_valid_prompt` safely handles errors in `token_counter`.
    """
    with patch(
        "litellm.utils.token_counter",
        side_effect=Exception(
            "Mocked error, This should not raise an error. Instead is_prompt_caching_valid_prompt should return False."
        ),
    ):
        result = litellm.utils.is_prompt_caching_valid_prompt(
            messages=[{"role": "user", "content": "test"}],
            tools=None,
            custom_llm_provider="anthropic",
            model="anthropic/claude-3-5-sonnet-20240620",
        )

        assert result is False  # Should return False when an error occurs


def test_is_prompt_caching_enabled_return_default_image_dimensions():
    """
    Assert that `is_prompt_caching_valid_prompt` calls token_counter with use_default_image_token_count=True
    when processing messages containing images

    IMPORTANT: Ensures Get token counter does not make a GET request to the image url
    """
    with patch("litellm.utils.token_counter") as mock_token_counter:
        litellm.utils.is_prompt_caching_valid_prompt(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://www.gstatic.com/webp/gallery/1.webp",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            tools=None,
            custom_llm_provider="openai",
            model="gpt-4o-mini",
        )

        # Assert token_counter was called with use_default_image_token_count=True
        args_to_mock_token_counter = mock_token_counter.call_args[1]
        print("args_to_mock", args_to_mock_token_counter)
        assert args_to_mock_token_counter["use_default_image_token_count"] is True


def test_token_counter_with_image_url_with_detail_high():
    """
    Assert that token_counter does not make a GET request to the image url when `use_default_image_token_count=True`

    PROD TEST this is importat - Can impact latency very badly
    """
    from litellm.constants import DEFAULT_IMAGE_TOKEN_COUNT
    from litellm._logging import verbose_logger
    import logging

    verbose_logger.setLevel(logging.DEBUG)

    _tokens = litellm.utils.token_counter(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.gstatic.com/webp/gallery/1.webp",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        model="gpt-4o-mini",
        use_default_image_token_count=True,
    )
    print("tokens", _tokens)
    assert _tokens == DEFAULT_IMAGE_TOKEN_COUNT + 7