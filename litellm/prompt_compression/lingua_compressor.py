import hashlib
import json
from threading import Lock
from llmlingua import PromptCompressor

class PromptCompressorSingleton:
    """Singleton class to initialize and reuse the PromptCompressor instance based on configuration."""
    
    _instances = {}  # Stores instances using a hash key
    _lock = Lock()

    @classmethod
    def _generate_config_hash(cls, **kwargs):
        """Generates a unique hash based on dynamic configuration parameters."""
        config_str = json.dumps(kwargs, sort_keys=True)  # Convert to string for hashing
        return hashlib.md5(config_str.encode()).hexdigest()  # Generate a unique hash

    def __new__(cls, **kwargs):
        """Create or reuse an instance based on the hash of the configuration."""
        config_hash = cls._generate_config_hash(**kwargs)
        with cls._lock:
            if config_hash not in cls._instances:
                cls._instances[config_hash] = super().__new__(cls)
                cls._instances[config_hash]._initialize(**kwargs)
        return cls._instances[config_hash]

    def _initialize(self, **kwargs):
        """Initializes the Prompt Compressor with the given configuration."""
        self.compressor = PromptCompressor(**kwargs)

    def compress(self, prompt: str, rate: float = 0.33, force_tokens=['\n', '?']):
        """Compresses the given prompt."""
        return self.compressor.compress_prompt(prompt, rate=rate, force_tokens=force_tokens)
