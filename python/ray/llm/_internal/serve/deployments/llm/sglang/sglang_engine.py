from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Union

from starlette.requests import Request

import ray
from ray.llm._internal.common.utils.import_utils import try_import
from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
)
from ray.llm._internal.serve.configs.server_models import (
    DiskMultiplexConfig,
    LLMConfig,
)
from ray.llm._internal.serve.deployments.llm.llm_engine import LLMEngine
from ray.llm._internal.serve.deployments.utils.node_initialization_utils import (
    InitializeNodeOutput,
    initialize_node,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    pass

sglang = try_import("sglang")
logger = get_logger(__name__)


def _get_sglang_engine_config(
    llm_config: LLMConfig,
) -> Any:
    """Convert LLMConfig to SGLang ServerArgs."""
    if sglang is None:
        raise ImportError(
            "SGLang is not installed. Please install it with `pip install sglang`."
        )

    from sglang.srt.server_args import ServerArgs

    # Get engine configuration - this will be the general config from LLMConfig
    engine_kwargs = llm_config.engine_kwargs
    model_loading_config = llm_config.model_loading_config

    # Convert Ray LLM config to SGLang ServerArgs
    server_args = ServerArgs(
        model_path=model_loading_config.model_id,
        tokenizer_path=engine_kwargs.get("tokenizer", None),
        tokenizer_mode=engine_kwargs.get("tokenizer_mode", "auto"),
        trust_remote_code=engine_kwargs.get("trust_remote_code", False),
        dtype=engine_kwargs.get("dtype", "auto"),
        kv_cache_dtype=engine_kwargs.get("kv_cache_dtype", "auto"),
        quantization=engine_kwargs.get("quantization", None),
        context_length=engine_kwargs.get("max_model_len", None),
        device=engine_kwargs.get("device", None),
        served_model_name=engine_kwargs.get("served_model_name", None),
        chat_template=engine_kwargs.get("chat_template", None),
        is_embedding=engine_kwargs.get("is_embedding", False),
        revision=model_loading_config.revision,
        host="127.0.0.1",
        port=30000,
        mem_fraction_static=engine_kwargs.get("gpu_memory_utilization", None),
        max_running_requests=engine_kwargs.get("max_num_seqs", None),
        max_total_tokens=engine_kwargs.get("max_total_tokens", None),
        max_prefill_tokens=engine_kwargs.get("max_prefill_tokens", 16384),
        tp_size=engine_kwargs.get("tensor_parallel_size", 1),
        stream_interval=1,
        random_seed=engine_kwargs.get("seed", None),
        log_level="info",
        log_requests=False,
        show_time_cost=False,
        enable_metrics=False,
        decode_log_interval=40,
        disable_log_stats=engine_kwargs.get("disable_log_stats", False),
        disable_log_requests=engine_kwargs.get("disable_log_requests", True),
    )

    return server_args


class SGLangEngine(LLMEngine):
    def __init__(
        self,
        llm_config: LLMConfig,
    ):
        """Create a SGLang Engine class

        Args:
            llm_config: The llm configuration for this engine
        """
        super().__init__(llm_config)

        self.llm_config = llm_config

        if sglang is None:
            raise ImportError(
                "SGLang is not installed. Please install it with `pip install sglang`."
            )

        self.llm_config.setup_engine_backend()

        self._running = False

        # SGLang Integration points. Will be set through .start()
        self._engine_client: Optional[Any] = None

    async def start(self) -> None:
        """Start the SGLang engine.

        If the engine is already running, do nothing.
        """

        if self._running:
            # The engine is already running!
            logger.info("Skipping engine restart because the engine is already running")
            return

        node_initialization = await initialize_node(self.llm_config)

        sglang_server_args = self._prepare_engine_config(node_initialization)

        # Apply checkpoint info to the llm_config.
        # This is needed for capturing model capabilities
        # (e.g. supports vision, etc.) on the llm_config.
        config = self.llm_config.get_engine_config()
        self.llm_config.apply_checkpoint_info(
            config.actual_hf_model_id,
            trust_remote_code=config.trust_remote_code,
        )

        self._engine_client = self._start_sglang_engine(
            sglang_server_args,
            node_initialization.placement_group,
        )

        self._validate_engine_client()

        self._running = True

        logger.info("Started SGLang engine.")

    def _validate_engine_client(self):
        assert self._engine_client is not None, "engine_client is not initialized"
        assert hasattr(
            self._engine_client, "generate"
        ), "engine_client must have a generate attribute"
        assert hasattr(
            self._engine_client, "async_generate"
        ), "engine_client must have an async_generate attribute"

    def _prepare_engine_config(self, node_initialization: InitializeNodeOutput) -> Any:
        """Prepare the engine config to start the engine.

        Args:
            node_initialization: The node initialization output.

        Returns:
            sglang_server_args: The SGLang's internal server arguments.
        """

        engine_config = self.llm_config.get_engine_config()

        if hasattr(engine_config, "use_gpu"):
            use_gpu = engine_config.use_gpu
        else:
            use_gpu = True  # Default to GPU usage

        if use_gpu:
            # Create engine config on a task with access to GPU,
            # as GPU capability may be queried.
            ref = (
                ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    accelerator_type=self.llm_config.accelerator_type,
                )(_get_sglang_engine_config)
                .options(
                    runtime_env=node_initialization.runtime_env,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=node_initialization.placement_group,
                    ),
                )
                .remote(self.llm_config)
            )
            sglang_server_args = ray.get(ref)
        else:
            sglang_server_args = _get_sglang_engine_config(self.llm_config)

        return sglang_server_args

    def _start_sglang_engine(
        self,
        server_args: Any,
        placement_group: PlacementGroup,
    ) -> Any:
        """Creates a SGLang engine from the server arguments."""
        if sglang is None:
            raise ImportError(
                "SGLang is not installed. Please install it with `pip install sglang`."
            )

        from sglang.srt.entrypoints.engine import Engine

        # Create the SGLang engine
        engine_client = Engine(server_args=server_args)

        return engine_client

    async def resolve_lora(self, disk_lora_model: DiskMultiplexConfig):
        """SGLang doesn't currently support LoRA adapter loading like vLLM.

        This is a placeholder for future LoRA support.
        """
        logger.warning(
            "LoRA adapter loading is not yet supported in SGLang engine. "
            f"Ignoring LoRA model: {disk_lora_model.model_id}"
        )

    def _create_raw_request(
        self,
        request: Union[CompletionRequest, ChatCompletionRequest, EmbeddingRequest],
        path: str,
    ) -> Request:
        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"x-request-id", getattr(request, "request_id", "").encode())],
            "query_string": b"",
        }
        return Request(scope)

    async def chat(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[Union[str, ChatCompletionResponse, ErrorResponse], None]:
        """Run a ChatCompletion with the SGLang engine."""

        # Convert request to SGLang format
        sglang_request = self._convert_chat_request(request)

        try:
            if request.stream:
                # Handle streaming response
                generator = self._engine_client.async_generate(**sglang_request)
                async for chunk in generator:
                    # Convert SGLang chunk to OpenAI format
                    yield self._convert_chat_chunk_to_openai(chunk, request)
            else:
                # Handle non-streaming response
                result = await self._engine_client.async_generate(**sglang_request)
                response = self._convert_chat_response_to_openai(result, request)
                yield ChatCompletionResponse(**response)

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            yield ErrorResponse(message=str(e), type="InternalServerError", code=500)

    async def completions(
        self, request: CompletionRequest
    ) -> AsyncGenerator[Union[str, CompletionResponse, ErrorResponse], None]:
        """Run a Completion with the SGLang engine."""

        # Convert request to SGLang format
        sglang_request = self._convert_completion_request(request)

        try:
            if request.stream:
                # Handle streaming response
                generator = self._engine_client.async_generate(**sglang_request)
                async for chunk in generator:
                    # Convert SGLang chunk to OpenAI format
                    yield self._convert_completion_chunk_to_openai(chunk, request)
            else:
                # Handle non-streaming response
                result = await self._engine_client.async_generate(**sglang_request)
                response = self._convert_completion_response_to_openai(result, request)
                yield CompletionResponse(**response)

        except Exception as e:
            logger.error(f"Error in completion: {e}")
            yield ErrorResponse(message=str(e), type="InternalServerError", code=500)

    async def embeddings(
        self, request: EmbeddingRequest
    ) -> AsyncGenerator[Union[EmbeddingResponse, ErrorResponse], None]:
        """Run an Embedding with the SGLang engine."""

        # Convert request to SGLang format
        sglang_request = self._convert_embedding_request(request)

        try:
            # SGLang embeddings are not streaming
            result = await self._engine_client.async_generate(**sglang_request)
            response = self._convert_embedding_response_to_openai(result, request)
            yield EmbeddingResponse(**response)

        except Exception as e:
            logger.error(f"Error in embedding: {e}")
            yield ErrorResponse(message=str(e), type="InternalServerError", code=500)

    def _convert_chat_request(self, request: ChatCompletionRequest) -> dict:
        """Convert OpenAI ChatCompletionRequest to SGLang format."""

        # Convert messages to prompt text
        if request.messages:
            # For now, simple conversion - join messages with newlines
            # In production, this should use proper chat templates
            prompt_parts = []
            for msg in request.messages:
                if msg.role == "system":
                    prompt_parts.append(f"System: {msg.content}")
                elif msg.role == "user":
                    prompt_parts.append(f"Human: {msg.content}")
                elif msg.role == "assistant":
                    prompt_parts.append(f"Assistant: {msg.content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
        else:
            prompt = ""

        sglang_params = {
            "prompt": prompt,
            "stream": request.stream,
        }

        # Convert sampling parameters
        if request.temperature is not None:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["temperature"] = request.temperature

        if request.max_tokens is not None:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["max_new_tokens"] = request.max_tokens

        if request.top_p is not None:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["top_p"] = request.top_p

        if request.stop:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["stop"] = request.stop

        return sglang_params

    def _convert_completion_request(self, request: CompletionRequest) -> dict:
        """Convert OpenAI CompletionRequest to SGLang format."""

        sglang_params = {
            "prompt": request.prompt,
            "stream": request.stream,
        }

        # Convert sampling parameters
        if request.temperature is not None:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["temperature"] = request.temperature

        if request.max_tokens is not None:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["max_new_tokens"] = request.max_tokens

        if request.top_p is not None:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["top_p"] = request.top_p

        if request.stop:
            sglang_params["sampling_params"] = sglang_params.get("sampling_params", {})
            sglang_params["sampling_params"]["stop"] = request.stop

        return sglang_params

    def _convert_embedding_request(self, request: EmbeddingRequest) -> dict:
        """Convert OpenAI EmbeddingRequest to SGLang format."""

        # Handle input as string or list of strings
        if isinstance(request.input, str):
            prompt = request.input
        elif isinstance(request.input, list):
            prompt = request.input[0] if request.input else ""
        else:
            prompt = str(request.input)

        sglang_params = {
            "prompt": prompt,
            "stream": False,  # Embeddings are not streamed
        }

        return sglang_params

    def _convert_chat_chunk_to_openai(
        self, chunk: dict, request: ChatCompletionRequest
    ) -> str:
        """Convert SGLang streaming chunk to OpenAI format string."""
        # This is a simplified conversion - in production this should
        # properly handle the SGLang streaming format
        import json

        openai_chunk = {
            "id": "chatcmpl-" + str(hash(str(chunk))),
            "object": "chat.completion.chunk",
            "created": int(__import__("time").time()),
            "model": request.model or "sglang",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk.get("text", "")},
                    "finish_reason": chunk.get("finish_reason"),
                }
            ],
        }

        return f"data: {json.dumps(openai_chunk)}\n\n"

    def _convert_completion_chunk_to_openai(
        self, chunk: dict, request: CompletionRequest
    ) -> str:
        """Convert SGLang streaming chunk to OpenAI completion format string."""
        import json

        openai_chunk = {
            "id": "cmpl-" + str(hash(str(chunk))),
            "object": "text_completion",
            "created": int(__import__("time").time()),
            "model": request.model or "sglang",
            "choices": [
                {
                    "index": 0,
                    "text": chunk.get("text", ""),
                    "finish_reason": chunk.get("finish_reason"),
                }
            ],
        }

        return f"data: {json.dumps(openai_chunk)}\n\n"

    def _convert_chat_response_to_openai(
        self, result: dict, request: ChatCompletionRequest
    ) -> dict:
        """Convert SGLang response to OpenAI ChatCompletionResponse format."""

        return {
            "id": "chatcmpl-" + str(hash(str(result))),
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": request.model or "sglang",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.get("text", "")},
                    "finish_reason": result.get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": result.get("meta_info", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("meta_info", {}).get(
                    "completion_tokens", 0
                ),
                "total_tokens": result.get("meta_info", {}).get("prompt_tokens", 0)
                + result.get("meta_info", {}).get("completion_tokens", 0),
            },
        }

    def _convert_completion_response_to_openai(
        self, result: dict, request: CompletionRequest
    ) -> dict:
        """Convert SGLang response to OpenAI CompletionResponse format."""

        return {
            "id": "cmpl-" + str(hash(str(result))),
            "object": "text_completion",
            "created": int(__import__("time").time()),
            "model": request.model or "sglang",
            "choices": [
                {
                    "index": 0,
                    "text": result.get("text", ""),
                    "finish_reason": result.get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": result.get("meta_info", {}).get("prompt_tokens", 0),
                "completion_tokens": result.get("meta_info", {}).get(
                    "completion_tokens", 0
                ),
                "total_tokens": result.get("meta_info", {}).get("prompt_tokens", 0)
                + result.get("meta_info", {}).get("completion_tokens", 0),
            },
        }

    def _convert_embedding_response_to_openai(
        self, result: dict, request: EmbeddingRequest
    ) -> dict:
        """Convert SGLang response to OpenAI EmbeddingResponse format."""

        # Extract embeddings from result
        # This is a placeholder - actual implementation depends on SGLang's embedding output format
        embeddings = result.get("embeddings", [])
        if not embeddings:
            # Generate dummy embedding if not available
            embeddings = [[0.0] * 768]  # Default 768-dimensional embedding

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": embeddings[0] if embeddings else [0.0] * 768,
                }
            ],
            "model": request.model or "sglang",
            "usage": {
                "prompt_tokens": result.get("meta_info", {}).get("prompt_tokens", 0),
                "total_tokens": result.get("meta_info", {}).get("prompt_tokens", 0),
            },
        }

    async def check_health(self) -> None:
        """Check the health of the SGLang engine."""
        assert self._engine_client is not None, "engine_client is not initialized"

        try:
            # Try a simple health check by attempting a small generation
            test_result = await self._engine_client.async_generate(
                prompt="test", sampling_params={"max_new_tokens": 1}
            )
            if test_result is None:
                raise RuntimeError("Engine returned None for health check")
        except BaseException as e:
            logger.error("Healthcheck failed. The replica will be restarted")
            raise e from None
