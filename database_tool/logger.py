import logging

# Set up logging for console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def log_token_usage(result):
    """
    Extracts token usage details from the API response and logs it.
    """
    try:
        # Extract token usage details from response_metadata
        usage_data = result.response_metadata.get("usage", {})
        usage_metadata = result.usage_metadata if hasattr(result, "usage_metadata") else {}

        input_tokens = usage_data.get("input_tokens", 0) or usage_metadata.get("input_tokens", 0)
        output_tokens = usage_data.get("output_tokens", 0) or usage_metadata.get("output_tokens", 0)
        total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)

        # Log token usage
        logger.info(f"Model: claude-3-5-sonnet-20241022 | Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | Total Tokens: {total_tokens}")
        
        return input_tokens, output_tokens, total_tokens
    except Exception as e:
        logger.error(f"Error extracting token usage: {str(e)}")
        return 0, 0, 0  # Return 0 if extraction fails
