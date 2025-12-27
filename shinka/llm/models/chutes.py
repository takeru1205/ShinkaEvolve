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
    # Explicitly disable streaming to avoid max_tokens=6144 limit
    # Non-streaming requests support up to 163,840 tokens
    if output_model is None:
        response = client.chat.completions.create(
            model=model, messages=messages, stream=False, **kwargs
        )

        # Get content from message
        message = response.choices[0].message
        content = message.content

        # For reasoning models (like GLM-4.7-TEE), content might be None
        # In that case, use reasoning_content or reasoning field
        if content is None:
            # Try reasoning_content first (standard field name)
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                content = message.reasoning_content
                logger.info(f"Using reasoning_content (length: {len(content)})")
            # Fall back to reasoning field
            elif hasattr(message, 'reasoning') and message.reasoning:
                content = message.reasoning
                logger.info(f"Using reasoning field (length: {len(content)})")
            else:
                logger.error(
                    f"Content is None and no reasoning found. Message: {message}"
                )
    else:
        # Structured output with instructor
        # Explicitly disable streaming to avoid max_tokens=6144 limit
        response = client.chat.completions.create(
            model=model, messages=messages, response_model=output_model, stream=False, **kwargs
        )
        content = response

    # Update message history
    new_msg_history = msg_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": str(content)},
    ]

    # Calculate costs (use pricing from CHUTES_MODELS)
    # OpenAI API uses prompt_tokens and completion_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
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
