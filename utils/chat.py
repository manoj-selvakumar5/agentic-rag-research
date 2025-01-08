import logging
import sys  # <-- For attaching the stream handler to sys.stdout
import boto3
import json
import random
import time
import zipfile
from io import BytesIO
import uuid
import os

from typing import List, Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

# -----------------------------------------------------------------------------
# 1. Configure Logging to Print in Jupyter
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a StreamHandler that writes to stdout (which Jupyter will display)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

# Set a log format (time, log level, message)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
stream_handler.setFormatter(formatter)

# Prevent adding multiple handlers if the logger is already configured
if logger.hasHandlers():
    logger.handlers.clear()

# Finally, add our custom handler
logger.addHandler(stream_handler)


# -----------------------------------------------------------------------------
# 2. Centralized AppContext
# -----------------------------------------------------------------------------
class AppContext:
    """
    Holds a shared boto3 Session, region, and account info. Also provides 
    common utility methods, such as interactive_sleep for user-friendly sleeps.
    """

    def __init__(self):
        """
        Initialize the shared context with a new boto3 session, automatically
        detecting region and account info from the AWS environment.
        """
        self.session = boto3.Session()
        self.region: str = self.session.region_name
        self.account_number: str = self.session.client('sts').get_caller_identity()['Account']

    def client(self, service_name: str):
        """
        Create a boto3 client for the specified service, 
        using the stored session and region.
        
        Args:
            service_name (str): The AWS service for which to create a client.
        
        Returns:
            boto3.Client: Configured AWS client.
        """
        return self.session.client(service_name, region_name=self.region)

    def resource(self, service_name: str):
        """
        Create a boto3 resource for the specified service.
        
        Args:
            service_name (str): The AWS service for which to create a resource.
        
        Returns:
            boto3.Resources: Configured AWS resource object.
        """
        return self.session.resource(service_name, region_name=self.region)

    def interactive_sleep(self, seconds: int) -> None:
        """
        Sleep for a certain number of seconds, printing dots for user feedback.
        Useful for progress in a Jupyter Notebook.

        Args:
            seconds (int): Number of seconds to sleep.
        """
        dots = ''
        for _ in range(seconds):
            dots += '.'
            print(dots, end='\r')
            time.sleep(1)
        print("")  # Move to the next line after sleeping

# -----------------------------------------------------------------------------
# 3. BedrockChat Class
# -----------------------------------------------------------------------------
class BedrockChat:
    """
    Provides methods to create, update, and delete Bedrock Agents, along with 
    helper methods to manage agent action groups and optional code interpreter 
    integration. Agents can also be associated with knowledge bases.
    """

    def __init__(self):
        """
        Initializes BedrockChat with:
            - A shared AppContext for region/account info
            - Clients for bedrock-agent, bedrock-agent-runtime, IAM, STS, and Lambda
        """
        self.context = AppContext()
        self.bedrock_agent_client = self.context.client('bedrock-agent')
        self.bedrock_agent_runtime_client = self.context.client('bedrock-agent-runtime')
        self.iam_client = self.context.client('iam')
        self.sts_client = self.context.client('sts')
        self.lambda_client = self.context.client('lambda')
        self.bedrock_runtime_client = self.context.client('bedrock-runtime')


    def converse(
        self,
        messages: List[Dict[str, Any]],
        system_prompts: List[Dict[str, str]] = [],
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        temperature: float = 0.5,
        top_k: int = 200,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Implements the Bedrock Converse API for chat-based interactions.
        
        Args:
            system_prompts: List of system prompts to guide model behavior
            messages: List of conversation messages
            model_id: Bedrock model identifier
            temperature: Control randomness (0-1)
            top_k: Number of tokens to consider
            verbose: Enable detailed logging
            
        Returns:
            Dictionary containing model response and usage statistics
        """
        try:
            # Configure inference parameters
            inference_config = {"temperature": temperature}
            additional_fields = {"top_k": top_k}
    
            # Call Bedrock converse API
            response = self.bedrock_runtime_client.converse(
                modelId=model_id,
                messages=messages,
                system=system_prompts,
                inferenceConfig=inference_config,
                additionalModelRequestFields=additional_fields
            )
    
            if verbose:
                # Log token usage
                token_usage = response['usage']
                logger.info(f"Input tokens: {token_usage['inputTokens']}")
                logger.info(f"Output tokens: {token_usage['outputTokens']}")
                logger.info(f"Total tokens: {token_usage['totalTokens']}")
                logger.info(f"Stop reason: {response['stopReason']}")
                logger.info(f"Latency(ms): {response['metrics']['latencyMs']}")
    
            return response
    
        except Exception as e:
            error_msg = f"Error in converse: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    # -----

# -----------------------------------------------------------------------------
# 4. SyntheticDataGenerator Class
# -----------------------------------------------------------------------------
class SyntheticDataGenerator:
    """
    A utility class to generate synthetic data for different use cases.
    """

    def __init__(self):
        """
        Initialize the SyntheticDataGenerator with a Bedrock chat client.

        """
        self.bedrock_chat = BedrockChat()

    def generate_forecasting_instructions(self, file_name: str) -> str:
        """
        Generates step-by-step instructions for energy forecasting and writes them to a file.
        Skips generation if the file already exists.

        Args:
            file_name (str): The full path to the file where instructions will be saved.

        Returns:
            str: Path to the file or a message indicating that the file already exists.
        """
        # Check if the file already exists
        if os.path.exists(file_name):
            logger.info(f"File already exists: {file_name}. Skipping generation.")
            return f"File already exists: {file_name}. Skipping generation."

        # Define the energy forecasting instructions
        energy_forecasting_instructions = '''
        You will act as a data scientist who knows how to perform machine learning
        forecasting using Python and scikit-learn. You will generate a step-by-step
        guide on how to create a forecast process for a time-series dataset.

        The dataset has the following JSON structure:
        {
            "customer_id": "1",
            "day": "2024/06/01",
            "sumPowerReading": "120.0",
            "kind": "measured"
        }

        Choose a forecast algorithm that works with scikit-learn, explain the details,
        and provide a step-by-step guide with code samples, showcasing how to perform
        the forecast on this dataset.

        Include explanations on how to interpret the forecasted values and how to
        identify the factors driving those values.

        Answer only with the step-by-step guide. Avoid phrases like:
        "OK, I can generate it," or "Yes, please find the following example."
        Be direct and provide only the step-by-step instructions.
        '''

        # Prepare the initial message for the API call
        messages = [{
            "role": "user",
            "content": [{
                "text": energy_forecasting_instructions,
            }]
        }]

        try:
            # Call the Bedrock Converse API to generate the content
            response = self.bedrock_chat.converse(messages=messages, verbose=True)

            # Extract the generated text from the response
            generated_text = response['output']['message']['content'][0]['text']

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            # Write the generated text to a file
            with open(file_name, 'w') as f:
                f.write(generated_text)

            logger.info(f"Instructions generated and saved to: {file_name}")
            return f"Instructions generated and saved to: {file_name}"

        except Exception as e:
            error_msg = f"Failed to generate forecasting instructions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
    def generate_solar_panel_instructions(self, file_name: str) -> str:
        """
        Generates step-by-step instructions for installing solar panels and writes them to a file.
        Skips generation if the file already exists.

        Args:
            file_name (str): The full path to the file where instructions will be saved.

        Returns:
            str: Path to the file or a message indicating that the file already exists.
        """
        # Check if the file already exists
        if os.path.exists(file_name):
            logger.info(f"File already exists: {file_name}. Skipping generation.")
            return f"File already exists: {file_name}. Skipping generation."

        # Define the solar panel installation instructions
        text_generation_energy_instructions = '''
        You will be act as an expert on clean energy.
        You will generate a step-by-step on how to install a solar panel at home.
        You know the following fictional solar panel models: Sunpower X, Sunpower Y
        and Sunpower double-X. For each one of those models, provide some general
        model description and its features. Next provide a numbered list describing
        how to install each model of solar panel. Include information about how to
        ensure compliance with energy rules.

        Answer only with the instructions and solar panel descriptions.
        Avoid answer with afirmations like: "OK, I can generate it,",
        "As an expert on clean energy, I ", or "Yes, please find following example."
        Be direct and only reply the instructions and descriptions.
        '''

        # Prepare the initial message for the API call
        messages = [{
            "role": "user",
            "content": [{
                "text": text_generation_energy_instructions,
            }]
        }]

        try:
            # Call the Bedrock Converse API to generate the content
            response = self.bedrock_chat.converse(messages=messages, verbose=True)

            # Extract the generated text from the response
            generated_text = response['output']['message']['content'][0]['text']

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_name), exist_ok=True)

            # Write the generated text to a file
            with open(file_name, 'w') as f:
                f.write(generated_text)

            logger.info(f"Instructions generated and saved to: {file_name}")
            return f"Instructions generated and saved to: {file_name}"

        except Exception as e:
            error_msg = f"Failed to generate solar panel instructions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def generate_solar_panel_maintenance_instructions(self, file_name: str) -> str:
        """
        Generates step-by-step instructions for solar panel maintenance and writes them to a file.
        Skips generation if the file already exists.
    
        Args:
            file_name (str): The full path to the file where instructions will be saved.
    
        Returns:
            str: Path to the file or a message indicating that the file already exists.
        """
        # Check if the file already exists
        if os.path.exists(file_name):
            logger.info(f"File already exists: {file_name}. Skipping generation.")
            return f"File already exists: {file_name}. Skipping generation."
    
        # Define the solar panel maintenance instructions
        text_generation_energy_instructions = '''
        You will be act as an expert on clean energy.
        You know the following fictional solar panel models: Sunpower X, Sunpower Y
        and Sunpower double-X. Here is are some descriptions of the different
        models and how to install them:
        <description_and_instructions>
        {description_and_instructions}
        </description_and_instructions>
        Generate a step-by-step instructions on how to do maintenance on each of
        those models at a regular home. Include information about how to
        ensure consistent compliance with energy rules.
        Just answer in a numbered list.
        '''
    
        # Prepare the initial message for the API call
        messages = [{
            "role": "user",
            "content": [{
                "text": text_generation_energy_instructions,
            }]
        }]
    
        try:
            # Call the Bedrock Converse API to generate the content
            response = self.bedrock_chat.converse(messages=messages, verbose=True)
    
            # Extract the generated text from the response
            generated_text = response['output']['message']['content'][0]['text']
    
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
            # Write the generated text to a file
            with open(file_name, 'w') as f:
                f.write(generated_text)
    
            logger.info(f"Instructions generated and saved to: {file_name}")
            return f"Instructions generated and saved to: {file_name}"
    
        except Exception as e:
            error_msg = f"Failed to generate solar panel maintenance instructions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)