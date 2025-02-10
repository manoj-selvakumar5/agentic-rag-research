import json
import time
import logging
import boto3
from botocore.config import Config
from transformers import AutoTokenizer
import os

os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
if not os.path.exists("/tmp/huggingface_cache"):
    os.makedirs("/tmp/huggingface_cache")


# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

# Hardcoded configuration values
MODEL_ID = "arn:aws:bedrock:us-west-2:533267284022:imported-model/satgxdjbohr8"
HF_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
REGION_INFO = "us-west-2"

# Initialize the tokenizer using the specified Hugging Face model
logger.info("Loading tokenizer for model: %s", HF_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

# Initialize the Bedrock Runtime client with hardcoded region information
session = boto3.Session()
client = session.client(
    service_name='bedrock-runtime',
    region_name=REGION_INFO,
    config=Config(
        connect_timeout=300,  # 5 minutes
        read_timeout=300,     # 5 minutes
        retries={'max_attempts': 3}
    )
)

def generate(messages, temperature=0.3, max_tokens=4096, top_p=0.9, continuation=False, max_retries=10):
    """
    Generate response using the reasoning model with proper tokenization and a retry mechanism.
    
    Parameters:
        messages (list): List of message dictionaries with 'role' and 'content'.
        temperature (float): Controls randomness.
        max_tokens (int): Maximum number of tokens to generate.
        top_p (float): Nucleus sampling parameter.
        continuation (bool): Whether this is a continuation of previous generation.
        max_retries (int): Maximum number of retry attempts.
    
    Returns:
        dict: Model response containing generated text and metadata.
    """
    # Prepare the prompt from the messages using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not continuation
    )
    
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps({
                    'prompt': prompt,
                    'temperature': temperature,
                    'max_gen_len': max_tokens,
                    'top_p': top_p
                }),
                accept='application/json',
                contentType='application/json'
            )
            result = json.loads(response['body'].read().decode('utf-8'))
            return result
        except Exception as e:
            logger.error("Attempt %d failed: %s", attempt + 1, str(e))
            attempt += 1
            if attempt < max_retries:
                time.sleep(30)
    raise Exception("Failed to get response after maximum retries")

def auto_generate(messages, **kwargs):
    """
    Handle longer responses that exceed token limits by iteratively generating until a complete answer is obtained.
    
    Parameters:
        messages (list): List of message dictionaries.
        **kwargs: Additional parameters for the generate function.
    
    Returns:
        dict: Enhanced response including the final answer and any reasoning provided.
    """
    res = generate(messages, **kwargs)
    
    # Continue generation if the response indicates it was truncated due to token length limits
    while res.get("stop_reason") == "length":
        for v in messages:
            if v.get("role") == "user":
                v["content"] += res.get("generation", "")
        res = generate(messages, **kwargs, continuation=True)

    # Parse out the reasoning if present (e.g., wrapped in <think> tags)
    for v in messages:
        if v.get("role") == "user":
            gen = v["content"] + res.get("generation", "")
            if "<think>" in gen and "</think>" in gen:
                # Extract the reasoning portion between the <think> tags
                think = gen.split("</think>")[0].split("<think>")[-1]
                # The final answer is assumed to be after the </think> tag
                answer = gen.split("</think>")[-1]
            else:
                think = ""
                answer = res.get("generation", "")
            res = {**res, "generation": gen, "answer": answer, "think": think}
            return res

def populate_function_response(event, response_body):
    """
    Populate the response in a format compatible with the downstream integration.
    
    Parameters:
        event (dict): The original Lambda event.
        response_body (dict): The response body to include.
    
    Returns:
        dict: A structured response object.
    """
    return {
        'response': {
            'actionGroup': event.get('actionGroup', ''),
            'function': event.get('function', ''),
            'functionResponse': {
                'responseBody': {
                    'TEXT': {
                        'body': str(response_body)
                    }
                }
            }
        }
    }

def lambda_handler(event, context):
    """
    AWS Lambda handler that extracts a prompt, calls the reasoning model, and returns the generated result.
    
    Expected event format:
    {
      "parameters": [
          {"name": "prompt", "value": "Your question or prompt text"}
      ],
      "actionGroup": "...",
      "function": "...",
      "sessionAttributes": { ... },
      "promptSessionAttributes": { ... }
    }
    """
    logger.info("Received event: %s", json.dumps(event))
    session_attributes = event.get('sessionAttributes', {})
    prompt_session_attributes = event.get('promptSessionAttributes', {})
    response_body = {'TEXT': {'body': ""}}

    try:
        # Extract the prompt from the event parameters
        prompt = next(
            (param["value"] for param in event.get('parameters', [])
             if param.get("name") == "prompt"), None
        )
        if not prompt:
            raise ValueError("Missing 'prompt' parameter")
        
        # Prepare messages for the reasoning model
        messages = [{"role": "user", "content": prompt}]
        
        # Call auto_generate to get the full model response (including reasoning if available)
        result = auto_generate(messages)
        logger.info("Generated model response: %s", json.dumps(result))
        response_body['TEXT']['body'] = result
    except Exception as e:
        logger.error("Error during model generation: %s", str(e), exc_info=True)
        response_body['TEXT']['body'] = f"Error: {str(e)}"

    # Build and return the final response object
    final_response = {
        'messageVersion': '1.0',
        **populate_function_response(event, response_body),
        'sessionAttributes': session_attributes,
        'promptSessionAttributes': prompt_session_attributes
    }
    
    logger.info("Final response: %s", json.dumps(final_response))
    return final_response
