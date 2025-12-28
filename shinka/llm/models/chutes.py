import logging

import backoff
import openai

from .pricing import CHUTES_MODELS
from .result import QueryResult

logger = logging.getLogger(__name__)


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.warning(
            f"Chutes.ai - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def estimate_input_tokens(messages) -> int:
    """
    Estimate input tokens for a list of messages.
    Uses simple heuristic: 1 token ≈ 4 characters for most languages.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        Estimated token count
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg:
            content = msg['content']
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle multi-modal content
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        total_chars += len(item['text'])

    # Add overhead for message structure (role, formatting, etc.)
    # Roughly 10 tokens per message
    overhead = len(messages) * 10

    # 1 token ≈ 4 chars (conservative estimate)
    estimated_tokens = (total_chars // 4) + overhead

    return estimated_tokens


def calculate_safe_max_tokens(
    model: str,
    input_tokens: int,
    default_max_tokens: int = 65536,
    safety_buffer: int = 2000,
) -> int:
    """
    Calculate safe max_tokens based on model context length and input size.

    Args:
        model: Model name
        input_tokens: Estimated input token count
        default_max_tokens: Default max tokens from config (default: 65536)
        safety_buffer: Safety buffer in tokens (default: 2000)

    Returns:
        Safe max_tokens value
    """
    # Get model context length
    model_info = CHUTES_MODELS.get(model, {})
    max_context = model_info.get("max_context_length", 128000)  # Default 128K

    # Calculate available tokens for output
    available_tokens = max_context - input_tokens - safety_buffer

    # Ensure available tokens is positive
    if available_tokens <= 0:
        logger.warning(
            f"Input tokens ({input_tokens}) exceed model context length ({max_context}). "
            f"Using minimum max_tokens of 1024."
        )
        return 1024

    # Cap at chutes.ai streaming limit (65536) and default config
    max_allowed = min(65536, default_max_tokens)
    safe_max_tokens = min(available_tokens, max_allowed)

    logger.debug(
        f"Token calculation - Model: {model}, Context: {max_context}, "
        f"Input: {input_tokens}, Available: {available_tokens}, "
        f"Safe max_tokens: {safe_max_tokens}"
    )

    return safe_max_tokens


@backoff.on_exception(
    backoff.expo,
    (
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
    ),
    max_tries=20,
    max_value=20,
    on_backoff=backoff_handler,
)
def query_chutes(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query chutes.ai model using OpenAI-compatible Chat Completions API."""
    # Build messages list
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(msg_history)
    messages.append({"role": "user", "content": msg})

    # Estimate input tokens and adjust max_tokens dynamically
    estimated_input_tokens = estimate_input_tokens(messages)

    # Get max_tokens from kwargs, default to 65536
    requested_max_tokens = kwargs.get('max_tokens', 65536)

    # Calculate safe max_tokens
    safe_max_tokens = calculate_safe_max_tokens(
        model=model,
        input_tokens=estimated_input_tokens,
        default_max_tokens=requested_max_tokens,
        safety_buffer=2000,
    )

    # Update kwargs with safe max_tokens
    kwargs_copy = kwargs.copy()
    kwargs_copy['max_tokens'] = safe_max_tokens

    # Log if max_tokens was adjusted
    if safe_max_tokens < requested_max_tokens:
        logger.warning(
            f"max_tokens adjusted from {requested_max_tokens} to {safe_max_tokens} "
            f"(input: {estimated_input_tokens} tokens, model: {model})"
        )

    # Call the OpenAI-compatible API
    # chutes.ai limits: streaming=65536 tokens, non-streaming=6144 tokens
    if output_model is None:
        # Use streaming for standard output to support up to 65536 tokens
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, **kwargs_copy
        )

        # Collect content from streaming chunks
        content = ""
        reasoning_content = ""
        usage = None

        # Limits for reasoning content to prevent unbounded growth
        MAX_REASONING_LENGTH = 100000  # 100KB (~25K tokens)
        reasoning_truncated = False

        for chunk in stream:
            # Skip chunks without choices (common in streaming)
            if not chunk.choices or len(chunk.choices) == 0:
                # Some APIs return usage in the last chunk without choices
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage = chunk.usage
                continue

            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content

            # Handle reasoning models with length limit
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                if len(reasoning_content) < MAX_REASONING_LENGTH:
                    reasoning_content += delta.reasoning_content
                elif not reasoning_truncated:
                    reasoning_content += f"\n... [reasoning truncated at {MAX_REASONING_LENGTH} chars] ..."
                    reasoning_truncated = True
                    logger.warning(
                        f"Reasoning content truncated at {MAX_REASONING_LENGTH} chars "
                        f"to prevent unbounded growth"
                    )
            elif hasattr(delta, 'reasoning') and delta.reasoning:
                if len(reasoning_content) < MAX_REASONING_LENGTH:
                    reasoning_content += delta.reasoning
                elif not reasoning_truncated:
                    reasoning_content += f"\n... [reasoning truncated at {MAX_REASONING_LENGTH} chars] ..."
                    reasoning_truncated = True
                    logger.warning(
                        f"Reasoning content truncated at {MAX_REASONING_LENGTH} chars "
                        f"to prevent unbounded growth"
                    )

            # Some APIs return usage in chunks with choices too
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = chunk.usage

        # Use reasoning content if main content is empty
        if not content and reasoning_content:
            content = reasoning_content
            logger.info(
                f"Using reasoning_content from streaming "
                f"(length: {len(content)}, truncated: {reasoning_truncated})"
            )

        # Get token counts from usage if available, otherwise estimate
        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
        else:
            # Estimate: 1 token ≈ 4 chars (rough approximation)
            input_tokens = len(str(messages)) // 4
            output_tokens = len(content) // 4
            logger.debug(f"Estimated tokens: input={input_tokens}, output={output_tokens}")
    else:
        # Structured output with instructor - must use non-streaming (6144 token limit)
        # instructor doesn't support streaming
        response = client.chat.completions.create(
            model=model, messages=messages, response_model=output_model, stream=False, **kwargs
        )
        content = response
        # Get usage from response
        input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0

    # Update message history
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": str(content)},
    ]

    # Calculate costs (use pricing from CHUTES_MODELS)
    # Token counts are already set above (either from usage or estimated)
    input_cost = CHUTES_MODELS[model]["input_price"] * input_tokens
    output_cost = CHUTES_MODELS[model]["output_price"] * output_tokens

    result = QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=input_cost + output_cost,
        input_cost=input_cost,
        output_cost=output_cost,
        thought="",
        model_posteriors=model_posteriors,
    )
    return result
