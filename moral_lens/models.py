from abc import ABC, abstractmethod
import gc
from typing import List, Tuple, Optional, Any

import torch
import asyncio

try:
    from aiolimiter import AsyncLimiter
except ImportError:
    AsyncLimiter = None

from tqdm.asyncio import tqdm


# Provider-specific client libraries - import only if installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = AutoModelForCausalLM = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None


from moral_lens.utils import load_yaml_file, fuzzy_match_decisions, parse_reasoning_and_decision
from moral_lens.config import ApiConfig, ModelConfig, RateLimitConfig
from moral_lens.data_models import Prompt, LLMResponse

APIKEYS = ApiConfig()
print(APIKEYS.summary())  # Ensure API keys are loaded and print summary

SEED = 42

MAX_RETRIES = 10

RATE_LIMITS = {
    "openai":       RateLimitConfig(num_concurrent=50, max_rate=20, period=1),
    "anthropic":    RateLimitConfig(num_concurrent=10, max_rate=50, period=60),
    "gemini":       RateLimitConfig(num_concurrent=100, max_rate=50, period=1),
    "openrouter":   RateLimitConfig(num_concurrent=50, max_rate=20, period=1),
    "huggingface":  None, # HuggingFace does not require rate limiting
}

MODEL_DEFAULTS = {
    "temperature": 0.0,
    "max_completion_tokens": 2048,
    "reasoning_model": False,
    "accepts_system_message": True,
}


class BaseModel(ABC):
    """
    Abstract base class defining a unified interface for various models.
    Provides both synchronous and asynchronous (parallel) request support.
    """
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 2048,
        reasoning_model: bool = False,
        accepts_system_message: bool = True,
        provider_routing: Optional[List[str]] = None,
        reasoning_effort: Optional[str] = None,
        rate_limits: RateLimitConfig = RateLimitConfig(1, 1, 1),
        **kwargs,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_model = reasoning_model
        self.accepts_system_message = accepts_system_message
        self.provider_routing = provider_routing
        self.reasoning_effort = reasoning_effort

        # Default rate limits
        self.num_concurrent = rate_limits.num_concurrent
        self.max_rate = rate_limits.max_rate
        self.time_period = rate_limits.period



    @abstractmethod
    def ask(
        self,
        prompt: Prompt,
    ) -> LLMResponse:
        """
        Synchronous method to send a prompt and receive a (thinking, output) tuple.
        """
        pass

    async def ask_async(
        self,
        prompt: Prompt,
    ) -> LLMResponse:
        """
        Asynchronous wrapper for the ask() method.
        """
        return await asyncio.to_thread(self.ask, prompt)


    async def ask_async_with_retry(
        self,
        prompts: List[Prompt],
        validation_fn: Optional[Any] = None
    ) -> List[LLMResponse]:
        """
        Asynchronous method to send multiple prompts with retry logic and rate limiting.
        Args:
            prompts (List[Prompt]): List of prompts to send.
            **hard coded** validation_fn (Optional[Any]): Custom validation function to validate responses.

        Returns:
            List[LLMResponse]: A list of valid LLM responses for the provided prompts.
        """
        results = [None] * len(prompts)
        attempts = 0
        pbar = tqdm(total=len(prompts), desc=f"Attempt {attempts+1}: Valid responses received", ascii=True)

        # if self.max_rate > self.num_concurrent:
        #     self.max_rate = self.num_concurrent

        # Ensure there are max concurency_limit tasks running at any time
        sem = asyncio.Semaphore(self.num_concurrent)

        # Ensure there are max num_concurrent requests in any 1 second time period
        limiter = AsyncLimiter(max_rate=self.max_rate, time_period=self.time_period)

        async def process_with_semaphore(
            i: int, prompt: Prompt
        ) -> Tuple[int, bool, Optional[LLMResponse]]:
            async with sem, limiter:
                try:
                    response = await self.ask_async(prompt)
                    if validation_fn and not validation_fn(response):
                        return i, False, None
                    # if not is_valid_response(response):
                    #     return i, False, None
                    response.attempts = attempts + 1
                    return i, True, response
                except Exception:
                    return i, False, None

        # Continue processing until all prompts are handled or max retries are exhausted
        while attempts < MAX_RETRIES and sum(1 for x in results if x is not None) < len(prompts):
            # Create tasks for all prompts that haven't been processed yet
            tasks = [
                asyncio.create_task(process_with_semaphore(i, prompts[i]))
                for i, res in enumerate(results)
                if res is None
            ]

            # If a task is successful, update the results and the progress bar
            for completed in asyncio.as_completed(tasks):
                i, success, response = await completed
                if success:
                    results[i] = response
                    pbar.update(1)

            pbar.set_description(f"Attempt {attempts+1}: Valid responses received")
            attempts += 1

            if self.temperature == 0.0:
                # If the temperature is 0, we can break early
                break

        for i, res in enumerate(results):
            if res is None:
                results[i] = LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    content="",
                    attempts=MAX_RETRIES if self.temperature > 0 else 1,
                )

        pbar.close()
        return results


# ------------------------------
# OpenAI
# ------------------------------
class OpenAIModel(BaseModel):
    """
    Wrapper for OpenAI models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if OpenAI is None:
            raise ImportError(
                "OpenAI library is not installed. Please install it to use OpenAI models."
            )
        if not APIKEYS.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=ApiConfig.OPENAI_API_KEY)

        print(f"[INFO] OpenAI model {self.model_id} loaded.")

    def ask(
        self, prompt: Prompt
    ) -> LLMResponse:
        # Format the prompt for OpenAI
        messages = prompt.openai_format()

        if not self.accepts_system_message:
            messages = messages[1:]

        body = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "seed": SEED,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        # Query the OpenAI API
        completion = self.client.chat.completions.create(**body)

        response_obj = LLMResponse(
            model_id=self.model_id,
            completion=completion,
            content=getattr(completion.choices[0].message, "content", ""),
            thinking_content=getattr(completion.choices[0].message, "reasoning", ""),
            two_choices=list(prompt.messages[-1].content.split('"')[1::2]), # TODO: a bit hacky
        )

        return response_obj


# ------------------------------
# Anthropic
# ------------------------------
class AnthropicModel(BaseModel):
    """
    Wrapper for Anthropic models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if Anthropic is None:
            raise ImportError(
                "Anthropic library is not installed. Please install it to use Anthropic models."
            )

        self.client = Anthropic(api_key=APIKEYS.ANTHROPIC_API_KEY)

        self.thinking_config = {"type": "disabled"}
        if self.reasoning_model:
            self.thinking_config = {
                "type": "enabled",
                "budget_tokens": int(self.max_completion_tokens * 0.8)
            }

        print(f"[INFO] Anthropic model {self.model_id} loaded.")

    def ask(
        self, prompt: Prompt
    ) -> LLMResponse:
        # Format the prompt for Anthropic
        system_prompt, messages = prompt.anthropic_format()
        if not self.accepts_system_message:
            system_prompt = None

        # Query the Anthropic API
        completion = self.client.messages.create(
            model=self.model_id,
            system=system_prompt,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_completion_tokens,
            thinking=self.thinking_config,
        )

        response_obj = LLMResponse(
            model_id=self.model_id,
            completion=completion,
            content=getattr(completion.content[-1], "text", ""),
            thinking_content=getattr(completion.content[0], "thinking", ""),
            two_choices=list(prompt.messages[-1].content.split('"')[1::2]), # TODO: a bit hacky
        )

        return response_obj


# ------------------------------
# Gemini
# ------------------------------
class GeminiModel(BaseModel):
    """
    Wrapper for Gemini (Google GenAI) models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if genai is None or types is None:
            raise ImportError(
                "Google GenAI library is not installed. Please install it to use Gemini models."
            )
        if not APIKEYS.GOOGLE_API_KEY:
            raise ValueError(
                "Gemini API key is not set. Please set the GOOGLE_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=APIKEYS.GOOGLE_API_KEY)

        print(f"[INFO] Gemini model {self.model_id} loaded.")

    def ask(
        self, prompt: Prompt
    ) -> LLMResponse:
        # Format the prompt for Gemini
        system_prompt, messages = prompt.gemini_format()
        if not self.accepts_system_message:
            system_prompt = None

        # Query the Gemini API
        completion = self.client.models.generate_content(
            model=self.model_id,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_completion_tokens,
                seed=SEED,
                http_options=types.HttpOptions(api_version="v1alpha"),
            ),
        )

        response_obj = LLMResponse(
            model_id=self.model_id,
            completion=completion,
            content=getattr(completion, "text", ""),
            thinking_content=getattr(completion, "reasoning", ""),
            two_choices=list(prompt.messages[-1].content.split('"')[1::2]), # TODO: a bit hacky
        )

        return response_obj


# ------------------------------
# OpenRouter
# ------------------------------
class OpenRouterModel(BaseModel):
    """
    Wrapper for OpenRouter models, which allow specifying provider routing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if OpenAI is None:
            raise ImportError(
                "OpenAI library is not installed. Please install it to use OpenRouter models."
            )
        if not APIKEYS.OPENROUTER_API_KEY:
            raise ValueError(
                "OpenRouter API key is not set. Please set the OPENROUTER_API_KEY environment variable."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=APIKEYS.OPENROUTER_API_KEY,
        )

        self.provider_routing = {
            "order": self.provider_routing,
            "allow_fallbacks": False,
        } if self.provider_routing else None

        print(f"[INFO] OpenRouter model {self.model_id} loaded.")

    def ask(
        self, prompt: Prompt
    ) -> LLMResponse:
        # Use the OpenAI format for compatibility with OpenRouter
        messages = prompt.openai_format()
        if not self.accepts_system_message:
            messages = messages[1:]

        body = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_completion_tokens,
            "seed": SEED,
        }
        if self.provider_routing:
            body["extra_body"] = {"provider": self.provider_routing}

        # Query the OpenRouter API
        completion = self.client.chat.completions.create(**body)

        response_obj = LLMResponse(
            model_id=self.model_id,
            completion=completion,
            content=getattr(completion.choices[0].message, "content", ""),
            thinking_content=getattr(completion.choices[0].message, "reasoning", ""),
            two_choices=list(prompt.messages[-1].content.split('"')[1::2]), # TODO: a bit hacky
        )

        return response_obj


# ------------------------------
# HuggingFace
# ------------------------------
class HuggingFaceModel(BaseModel):
    """
    Local HuggingFace model wrapper. Supports batching natively.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "transformers library is not installed. Please install it to use HuggingFace models."
            )

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            token=APIKEYS.HF_TOKEN,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",
            local_files_only=True,
            token=APIKEYS.HF_TOKEN,
            trust_remote_code=True,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Check if the model supports a chat template
        self.chat_model = hasattr(self.tokenizer, "chat_template") and (
            self.tokenizer.chat_template is not None
        )

        print(f"[INFO] HuggingFace model {self.model_id} loaded on {self.model.device}.")

    def unload(self):
        """
        Unload the model from GPU memory and clear CUDA cache to free memory.
        This should be called when the model is no longer needed.
        """
        # Delete model and tokenizer references
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] HuggingFace model {self.model_id} unloaded.")

    def ask(
        self, prompt: Prompt
    ) -> LLMResponse:
        # Build input based on whether the model is chat-based
        if self.chat_model:
            messages = prompt.openai_format()

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                continue_final_message=False,
            )
        else:
            # Standard tokenization for non-chat models (not tested)
            text = prompt[-1].content
            input_ids = self.tokenizer(
                text, return_tensors="pt", padding=True
            )["input_ids"]

        # Generate tokens with or without sampling
        if self.temperature == 0.0:
            outputs = self.model.generate(
                input_ids.to(self.model.device),
                max_new_tokens=self.max_completion_tokens,
                do_sample=False,
            )
        else:
            outputs = self.model.generate(
                input_ids.to(self.model.device),
                max_new_tokens=self.max_completion_tokens,
                do_sample=True,
                temperature=self.temperature,
            )

        # Decode outputs and format the result
        completion = self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        response_obj = LLMResponse(
            model_id=self.model_id,
            completion=outputs,
            content=completion[0].strip() if completion else "",
            thinking_content="",  # Our HuggingFace models do not provide reasoning/thinking
            two_choices=list(prompt.messages[-1].content.split('"')[1::2]), # TODO: a bit hacky
        )

        return response_obj


    def ask_batch(self, prompts: List[Prompt]) -> List[LLMResponse]:
        messages_list = [p.openai_format() for p in prompts]
        inputs = self.tokenizer.apply_chat_template(
            messages_list,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            return_dict=False,
            tokenize=True
        ).to(self.model.device)

        if self.temperature == 0.0:
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_completion_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        else:
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_completion_tokens,
                do_sample=True,
                temperature=self.temperature,
            )

        decoded = self.tokenizer.batch_decode(
            outputs[:, inputs.shape[1]:],
            skip_special_tokens=True,
        )

        return [
            LLMResponse(
                model_id=self.model_id,
                completion=decoded[i],
                content=decoded[i].strip(),
                thinking_content="",
                two_choices=list(p.messages[-1].content.split('"')[1::2]),
            )
            for i, p in enumerate(prompts)
        ]


    def ask_batch_with_retry(
        self,
        prompts: List[Prompt],
        validation_fn: Optional[Any] = None,
        batch_size: int = 1
    ) -> List[LLMResponse]:
        results = [None] * len(prompts)
        attempts = 0
        pbar = tqdm(total=len(prompts), desc=f"Attempt {attempts+1}: Valid responses received", ascii=True)

        while attempts < MAX_RETRIES and sum(1 for x in results if x is not None) < len(prompts):
            to_retry_indices = [i for i, r in enumerate(results) if r is None]

            # Process prompts in batches
            for i in range(0, len(to_retry_indices), batch_size):
                batch_indices = to_retry_indices[i:i + batch_size]
                batch_prompts = [prompts[idx] for idx in batch_indices]

                try:
                    batch_responses = self.ask_batch(batch_prompts)
                except Exception:
                    continue

                for rel_idx, abs_idx in enumerate(batch_indices):
                    response = batch_responses[rel_idx]
                    if validation_fn and not validation_fn(response):
                        continue
                    response.attempts = attempts + 1
                    results[abs_idx] = response
                    pbar.update(1)

            pbar.set_description(f"Attempt {attempts+1}: Valid responses received")
            attempts += 1
            if self.temperature == 0.0:
                break

        # Fill in any missing results
        for i, res in enumerate(results):
            if res is None:
                results[i] = LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    content="",
                    attempts=MAX_RETRIES,
                )

        pbar.close()
        return results


class ModelFactory:
    @staticmethod
    def get_model(
        model: ModelConfig | str
    ) -> BaseModel:
        """
        Factory method to create an instance of the appropriate model class based on the provider.
        Args:
            model_config (ModelConfig): The configuration object for the model.
        Returns:
            BaseModel: An instance of the appropriate model class based on the provider in the model_config.
        """
        if isinstance(model, str):
            # If a string is provided, load the model configuration from the YAML file.
            model = load_model_config(model)

        if isinstance(model, ModelConfig) is False:
            raise ValueError(
                "The provided model must be a ModelConfig instance or a valid model ID string."
            )

        provider = model.provider.lower()
        model_kwargs = model.to_dict()

        rate_limits: RateLimitConfig = RATE_LIMITS.get(provider)

        if model_kwargs.get('num_concurrent', None):
            rate_limits.num_concurrent = model_kwargs['num_concurrent']
        if model_kwargs.get('max_rate', None):
            rate_limits.max_rate = model_kwargs['max_rate']
        if model_kwargs.get('time_period', None):
            rate_limits.period = model_kwargs['time_period']

        if provider == "openai":
            return OpenAIModel(**model_kwargs, rate_limits=rate_limits)

        elif provider == "anthropic":
            return AnthropicModel(**model_kwargs, rate_limits=rate_limits)

        elif provider == "gemini":
            return GeminiModel(**model_kwargs, rate_limits=rate_limits)

        elif provider == "openrouter":
            return OpenRouterModel(**model_kwargs, rate_limits=rate_limits)

        elif provider == "huggingface":
            return HuggingFaceModel(**model_kwargs)
        else:
            raise ValueError(
                "Provider must be one of: openai, anthropic, gemini, openrouter, huggingface."
            )

# Cache for the models configuration
_MODELS_CONFIG_CACHE = {}

def load_model_config(model_id: str, path: str = "moral_lens/config/models.yaml", disable_cache: bool = False) -> ModelConfig:
    """
    Load a model configuration from a YAML file and return a ModelConfig instance.

    models.yaml is formated as follows:
    ```yaml
    gpt-4o-mini-2024-07-18:
      model_name: GPT-4o mini
      provider: openai
      release_date: 2024-07-18

    gpt-4o-2024-08-06:
      model_name: GPT-4o
      provider: openai
      release_date: 2024-08-06
    ...

    Args:
        model_id (str): The model ID to load from the YAML file.
        path (str): The path to the YAML file. Default is 'moral_lens/config/models.yaml'.
        disable_cache (bool): If True, bypass the cache and load directly from file. Default is True.

    Returns:
        ModelConfig: An instance of ModelConfig containing the loaded model configuration.
    ```
    """
    # Load the YAML file if not cached or if cache is disabled
    if disable_cache or path not in _MODELS_CONFIG_CACHE:
        yaml_obj = load_yaml_file(path)
        if not disable_cache:
            _MODELS_CONFIG_CACHE[path] = yaml_obj
    else:
        yaml_obj = _MODELS_CONFIG_CACHE[path]

    model_data = yaml_obj.get(model_id, None)
    if model_data is None:
        # Fallback looking for model name as the last part of a path
        for key, value in yaml_obj.items():
            if key.split("/")[-1] == model_id.split("/")[-1]: # e.g., `google/gemma-3-27b-it` vs `path/to/gemma-3-27b-it`
                model_data = value
                break
    if model_data is None:
        raise ValueError(f"Model '{model_id}' not found in the YAML configuration.")

    save_id = model_id.split("/")[-1]  # e.g., `google/gemma-3-27b-it` or `path/to/gemma-3-27b-it` --> `gemma-3-27b-it`
    model_id = model_id.split(":")[0]  # e.g., `o3-mini-2025-01-31:high` --> `o3-mini-2025-01-31`

    cfg = ModelConfig(
        model_id=model_id,
        save_id=save_id,
        model_name=model_data['model_name'],
        provider=model_data['provider'],
        release_date=model_data['release_date'],
        developer=model_data.get('developer', "Unknown"),

        reasoning_model=model_data.get('reasoning_model', MODEL_DEFAULTS['reasoning_model']),
        accepts_system_message=model_data.get('accepts_system_message', MODEL_DEFAULTS['accepts_system_message']),
        temperature=model_data.get('temperature', MODEL_DEFAULTS['temperature']),
        max_completion_tokens=model_data.get('max_completion_tokens', MODEL_DEFAULTS['max_completion_tokens']),
        provider_routing=model_data.get('provider_routing', None),
        reasoning_effort=model_data.get('reasoning_effort', None),

        num_concurrent=model_data.get('num_concurrent', None),
        max_rate=model_data.get('max_rate', None),
        time_period=model_data.get('time_period', None),
    )
    return cfg