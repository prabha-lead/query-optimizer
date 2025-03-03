import logging

# Set up logging for console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def log_token_usage(result):
    """
    Extracts token usage details from the OpenAI API response and logs it.
    """
    try:
        # Extract token usage details from the result object
        usage_data = result.usage

        input_tokens = usage_data.prompt_tokens
        output_tokens = usage_data.completion_tokens
        total_tokens = usage_data.total_tokens

        # Log token usage
        logger.info(f"Model: gpt-4o-mini | Input Tokens: {input_tokens} | Output Tokens: {output_tokens} | Total Tokens: {total_tokens}")
        
        return input_tokens, output_tokens, total_tokens
    except AttributeError as e:
        logger.error(f"Error extracting token usage: {str(e)}")
        return 0, 0, 0  # Return 0 if extraction fails
