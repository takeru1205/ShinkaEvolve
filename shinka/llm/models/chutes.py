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

    # Call the OpenAI-compatible API
    # chutes.ai limits: streaming=65536 tokens, non-streaming=6144 tokens
    if output_model is None:
        # Use streaming for standard output to support up to 65536 tokens
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, **kwargs
        )

        # Collect content from streaming chunks
        content = ""
        reasoning_content = ""
        usage = None
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            # Handle reasoning models
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            elif hasattr(delta, 'reasoning') and delta.reasoning:
                reasoning_content += delta.reasoning
            # Some APIs return usage in the last chunk
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = chunk.usage

        # Use reasoning content if main content is empty
        if not content and reasoning_content:
            content = reasoning_content
            logger.info(f"Using reasoning_content from streaming (length: {len(content)})")

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
