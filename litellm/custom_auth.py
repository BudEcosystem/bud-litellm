from datetime import datetime, timedelta
from pydantic import BaseModel

import litellm
from litellm.proxy._types import *
from litellm.proxy.auth.auth_utils import (
    get_request_route,
    pre_db_read_auth_checks,
)
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth: 
    """
    Custom Auth dependency for User API Key Authentication
    We receive budserve ap key and check if it is valid

    Steps:

    1. Check api-key in cache
    2. Get api-key details from db
    3. Check expiry
    4. Check budget
    5. Check model budget
    """
    try:
        from litellm.proxy.proxy_server import user_api_key_cache, master_key
        
        route: str = get_request_route(request=request)
        # get the request body
        request_data = await _read_request_body(request=request)
        await pre_db_read_auth_checks(
            request_data=request_data,
            request=request,
            route=route,
        )
        
        # look for info is user_api_key_auth cache
        valid_token: Optional[UserAPIKeyAuth] = await user_api_key_cache.async_get_cache(
            key=hash_token(api_key)
        )
        # OR
        # valid_token: Optional[UserAPIKeyAuth] = user_api_key_cache.get_cache(  # type: ignore
        #     key=api_key
        # )
        if valid_token is None:
            # getting token details from authentication service
            _valid_token = BaseModel(
                api_key=api_key,
                expires=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                budget=100,
                model_max_budget={},
                model_spend={},
                spend=50,
            )
            valid_token = UserAPIKeyAuth(
                **_valid_token.model_dump(exclude_none=True)
            )
        if valid_token is not None:
            if valid_token.expires is not None:
                current_time = datetime.now(timezone.utc)
                expiry_time = datetime.fromisoformat(valid_token.expires)
                if (
                    expiry_time.tzinfo is None
                    or expiry_time.tzinfo.utcoffset(expiry_time) is None
                ):
                    expiry_time = expiry_time.replace(tzinfo=timezone.utc)
                verbose_proxy_logger.debug(
                    f"Checking if token expired, expiry time {expiry_time} and current time {current_time}"
                )
                if expiry_time < current_time:
                    # Token exists but is expired.
                    raise ProxyException(
                        message=f"Authentication Error - Expired Key. Key Expiry time {expiry_time} and current time {current_time}",
                        type=ProxyErrorTypes.expired_key,
                        code=400,
                        param=api_key,
                    )
            if valid_token.spend is not None and valid_token.max_budget is not None:
                if valid_token.spend >= valid_token.max_budget:
                    raise litellm.BudgetExceededError(
                        current_cost=valid_token.spend,
                        max_budget=valid_token.max_budget,
                    )
                max_budget_per_model = valid_token.model_max_budget
                current_model = request_data.get("model", None)
                if (
                    max_budget_per_model is not None
                    and isinstance(max_budget_per_model, dict)
                    and len(max_budget_per_model) > 0
                    and prisma_client is not None
                    and current_model is not None
                    and valid_token.token is not None
                ):
                    ## GET THE SPEND FOR THIS MODEL
                    twenty_eight_days_ago = datetime.now() - timedelta(days=28)
                    model_spend = await prisma_client.db.litellm_spendlogs.group_by(
                        by=["model"],
                        sum={"spend": True},
                        where={
                            "AND": [
                                {"api_key": valid_token.token},
                                {"startTime": {"gt": twenty_eight_days_ago}},
                                {"model": current_model},
                            ]
                        },  # type: ignore
                    )
                    if (
                        len(model_spend) > 0
                        and max_budget_per_model.get(current_model, None) is not None
                    ):
                        if (
                            "model" in model_spend[0]
                            and model_spend[0].get("model") == current_model
                            and "_sum" in model_spend[0]
                            and "spend" in model_spend[0]["_sum"]
                            and model_spend[0]["_sum"]["spend"]
                            >= max_budget_per_model[current_model]
                        ):
                            current_model_spend = model_spend[0]["_sum"]["spend"]
                            current_model_budget = max_budget_per_model[current_model]
                            raise litellm.BudgetExceededError(
                                current_cost=current_model_spend,
                                max_budget=current_model_budget,
                            )
            # Add hashed token to cache
            await user_api_key_cache.async_set_cache(
                key=api_key,
                value=valid_token,
            )
        else:
            # No token was found when looking up in the DB
            raise Exception("Invalid proxy server token passed")
        
    except Exception as e: 
        if isinstance(e, litellm.BudgetExceededError):
            raise ProxyException(
                message=e.message,
                type=ProxyErrorTypes.budget_exceeded,
                param=None,
                code=400,
            )
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "detail", f"Authentication Error({str(e)})"),
                type=ProxyErrorTypes.auth_error,
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_401_UNAUTHORIZED),
            )
        elif isinstance(e, ProxyException):
            raise e
        raise ProxyException(
            message="Authentication Error, " + str(e),
            type=ProxyErrorTypes.auth_error,
            param=getattr(e, "param", "None"),
            code=status.HTTP_401_UNAUTHORIZED,
        )