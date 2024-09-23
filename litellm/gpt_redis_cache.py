import os
import ast
import hashlib
import cachetools
import json
import inspect
from typing import List, Optional, Any, Dict, Callable, Tuple

from gptcache import Cache, Config
from gptcache.manager import manager_factory
from gptcache.manager.eviction.memory_cache import MemoryCacheEviction, popitem_wrapper
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import LangChain, Onnx
from gptcache.similarity_evaluation import SbertCrossencoderEvaluation
from langchain_community.cache import GPTCache
from langchain.embeddings.base import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .caching import BaseCache, print_verbose


class BudServeMemoryCacheEviction(MemoryCacheEviction):
    """ This class `BudServeMemoryCacheEviction` is a subclass of `MemoryCacheEviction`
    that implements memory cache eviction policies such as LRU and TTL with customizable
    parameters. """

    def __init__(
        self,
        policy: str = "LRU",
        maxsize: int = 1000,
        clean_size: int = 0,
        on_evict: Callable[[List[Any]], None] = None,
        ttl: Optional[int] = None,
        **kwargs,
    ):
        try:
            super().__init__(policy, maxsize, clean_size, on_evict, **kwargs)
        except ValueError:
            self._policy = policy.upper()
            if self._policy == "TTL":
                if not ttl:
                    raise ValueError("TTL policy requires ttl parameter")
                self._cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl, **kwargs)
            else:
                raise ValueError(f"Unknown policy {policy}")
            self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)


def init_gptcache_redis(cache_obj: Cache, hashed_llm: str, endpoint_cache_config: dict, embedding_model: str, similarity_threshold: float):
    """Initialise the GPT cache object."""
    print_verbose(f"gptcache redis semantic-cache init_gptcache_redis, hashed llm string: {hashed_llm}")
    endpoint_id = endpoint_cache_config.get("endpoint_id", "1234")
    embedding_model: str = endpoint_cache_config.get("embedding_model", embedding_model)
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
        )
        embeddings = LangChain(embeddings=embeddings)
    except Exception as exc:
        print_verbose(f"gptcache redis semantic-cache embeddings error: {str(exc)}")
        print_verbose(f"gptcache redis semantic-cache using Onnx embeddings")
        embeddings = Onnx()
    eviction_policy = endpoint_cache_config.get("eviction_policy", {})

    # data manager __init__ puts already present ids in eviction base
    # since we are setting eviction_base after data manager initialisation
    # we dont want that to happen in __init__ of SSDataManager class
    # Therefore we are setting eviction manager as no_op_eviction
    data_manager = manager_factory(
        "redis,redis",
        # data_dir=f"similar_cache_{endpoint_id}_{hashed_llm}",
        scalar_params={
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": os.getenv("REDIS_PORT", "6342"),
            "password": os.getenv("REDIS_PASSWORD", "budpassword"),
            "global_key_prefix": f"cache_{endpoint_id}_{hashed_llm}",
        },
        vector_params={
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": os.getenv("REDIS_PORT", "6342"),
            "password": os.getenv("REDIS_PASSWORD", "budpassword"),
            "dimension": embeddings.dimension,
            "top_k": 1,
            "collection_name": f"index_{endpoint_id}_{hashed_llm}",
            "namespace": f"namespace_{endpoint_id}_{hashed_llm}",
        },
        eviction_manager="no_op_eviction"
    )

    # add value for on_evict parameter of eviction base
    eviction_params={
        "maxsize": eviction_policy.get("max_size", 100),
        "policy": eviction_policy.get("policy", "LRU"),
        "clean_size": int(eviction_policy.get("max_size", 100) * 0.2) or 1,
        "ttl": eviction_policy.get("ttl"),
        "on_evict": data_manager._clear,
    }
    data_manager.eviction_base = BudServeMemoryCacheEviction(**eviction_params)

    # initialise the eviction base
    ids = data_manager.s.get_ids(deleted=False)
    data_manager.eviction_base.put(ids)

    # https://github.com/zilliztech/GPTCache/blob/acc20f05400dabdcde451194e9bb73b986747685/gptcache/adapter/api.py#L134
    init_similar_cache(
        cache_obj=cache_obj,
        embedding=embeddings,
        data_manager=data_manager,
        evaluation=SbertCrossencoderEvaluation(),
        config=Config(
            similarity_threshold=endpoint_cache_config.get("score_threshold", similarity_threshold),
            auto_flush=1,
            # log_time_func=run_async_log_time_func,
        )
    )


class BudServeGPTCache(GPTCache, BaseCache):
    def __init__(
        self,
        host=None,
        port=None,
        password=None,
        redis_url=None,
        similarity_threshold=None,
        use_async=False,
        embedding_model="text-embedding-ada-002",
        **kwargs,
    ):
        print_verbose(
            "gptcache redis semantic-cache initializing..."
        )
        GPTCache.__init__(self, init_gptcache_redis)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        if redis_url is None:
            # if no url passed, check if host, port and password are passed, if not raise an Exception
            if host is None or port is None or password is None:
                # try checking env for host, port and password
                import os

                host = os.getenv("REDIS_HOST")
                port = os.getenv("REDIS_PORT")
                password = os.getenv("REDIS_PASSWORD")
                if host is None or port is None or password is None:
                    raise Exception("Redis host, port, and password must be provided")

            redis_url = "redis://:" + password + "@" + host + ":" + port
        print_verbose(f"gptcache redis semantic-cache redis_url: {redis_url}")
        if use_async == False:
            print_verbose("gptcache redis semantic-cache using sync redis client")

    def _get_gptcache(self, llm_string: str, cache_config: dict) -> Any:
        """Get a cache object.

        When the corresponding llm model cache does not exist, it will be created."""
        _gptcache = self.gptcache_dict.get(llm_string, None)
        if not _gptcache:
            _gptcache = self._new_gptcache(llm_string, cache_config)
        return _gptcache

    def _new_gptcache(self, llm_string: str, cache_config: dict) -> Any:
        """New gptcache object"""
        _gptcache = Cache()
        if self.init_gptcache_func is not None:
            sig = inspect.signature(self.init_gptcache_func)
            if len(sig.parameters) == 5:
                self.init_gptcache_func(_gptcache, llm_string, cache_config, self.embedding_model, self.similarity_threshold)
            elif len(sig.parameters) == 3:
                self.init_gptcache_func(_gptcache, llm_string, cache_config)
            elif len(sig.parameters) == 2:
                self.init_gptcache_func(_gptcache, llm_string)  # type: ignore[call-arg]
            else:
                self.init_gptcache_func(_gptcache)  # type: ignore[call-arg]
        else:
            raise ValueError("init_gptcache_func is not defined.")

        self.gptcache_dict[llm_string] = _gptcache
        return _gptcache

    def _get_cache_logic(self, cached_response: Any):
        """
        Common 'get_cache_logic' across sync + async redis client implementations
        """
        if cached_response is None:
            return cached_response

        # check if cached_response is bytes
        if isinstance(cached_response, bytes):
            cached_response = cached_response.decode("utf-8")

        try:
            cached_response = json.loads(
                cached_response
            )  # Convert string to dictionary
        except:
            cached_response = ast.literal_eval(cached_response)
        return cached_response

    def set_cache(self, key, value, **kwargs):
        import time
        from gptcache.adapter.api import put
        
        endpoint_cache_config = kwargs.get("endpoint_cache_config", {})
        endpoint_cache_config["metric_config"] = {"request_start_time" : time.time()}
        
        llm_cache = self._get_gptcache(key, endpoint_cache_config)

        print_verbose(f"gptcache redis set_cache, kwargs: {kwargs}")
        print_verbose(f"gptcache redis set_cache type : {type(value)}, value: {value}")

        # get the prompt
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)

        cache_metric = put(
            prompt,
            value if isinstance(value, str) else json.dumps(value),
            cache_obj=llm_cache,
            cache_metric_config=endpoint_cache_config.get("metric_config", {})
        )

        return

    def get_cache(self, key, **kwargs):
        # send endpoint_cache_config in kwargs
        print_verbose(f"sync gptcache redis get_cache, kwargs: {kwargs}")
        import time
        from gptcache.adapter.api import get
        
        endpoint_cache_config = kwargs.get("endpoint_cache_config", {})
        endpoint_cache_config["metric_config"] = {"request_start_time" : time.time()}
        
        llm_cache = self._get_gptcache(key, endpoint_cache_config)
        
        # query
        # get the messages
        messages = kwargs["messages"]
        prompt = "".join(message["content"] for message in messages)
        
        results, cache_metric = get(
            prompt,
            cache_obj=llm_cache,
            cache_metric_config=endpoint_cache_config.get("metric_config", {})
        )
        
        print_verbose(f"sync gptcache redis get_cache, results: {results}")
        
        results = [json.loads(results)] if results is not None else None

        if results == None:
            return None
        
        if isinstance(results, list):
            if len(results) == 0:
                return None

        cached_value = results[0]["response"]
        return self._get_cache_logic(cached_response=cached_value)

    async def async_set_cache(self, key, value, **kwargs):
        return self.set_cache(key, value, **kwargs)

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key, **kwargs)
