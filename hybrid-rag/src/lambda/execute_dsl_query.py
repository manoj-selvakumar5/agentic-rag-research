import json
import boto3
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():  # Prevent duplicate handlers in AWS Lambda
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

class OpenSearchManager:
    """Handles OpenSearch connection and queries."""
    
    def __init__(self, host):
        self.session = boto3.Session()
        self.region = self.session.region_name
        self.oss_client = self._initialize_opensearch_client(host)

    def _initialize_opensearch_client(self, host):
        """Initialize OpenSearch client with AWS Signature V4 authentication."""
        credentials = self.session.get_credentials()
        
        # Use AWSV4SignerAuth with the full credential object, the region, 
        # and service='aoss' if you're using OpenSearch Serverless
        awsauth = AWSV4SignerAuth(credentials, self.region, 'aoss')
        
        logger.info("Initializing OpenSearch client")
        return OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=600
        )

    def query(self, query):
        """Execute a query on OpenSearch."""
        try:
            logger.info("Executing query on OpenSearch")
            response = self.oss_client.search(index="ecom_shipping_index", body=query)
            logger.info(f"Query response: {response}")
            return response
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise

def populate_function_response(event, response_body):
    return {
        'response': {
            'actionGroup': event['actionGroup'],
            'function': event['function'],
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
    host = "5g240wxidf1c8njrw17l.us-west-2.aoss.amazonaws.com"
    logger.info("Received event: " + json.dumps(event))
    
    # Extract DSL query from event parameters
    dsl_query = next(
        (param["value"] for param in event.get('parameters', []) 
         if param.get('name') == 'dsl_query'), None
    )
    
    # Initialize Bedrock response structure
    response_body = {'TEXT': {'body': ""}}
    session_attributes = event.get('sessionAttributes', {})
    prompt_session_attributes = event.get('promptSessionAttributes', {})

    try:
        if not dsl_query:
            raise ValueError("Missing dsl_query parameter")

        logger.info("Executing DSL query: " + json.dumps(json.loads(dsl_query), indent=2))
        
        # Execute OpenSearch query
        oss_response = OpenSearchManager(host).query(json.loads(dsl_query))
        logger.info(f"Response from OSS: {oss_response}")
        
        # Extract the count of orders in transit
        count = oss_response["hits"]["total"]["value"]
        
        # Format successful response for Bedrock
        response_body['TEXT']['body'] = oss_response

        
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        response_body['TEXT']['body'] = f"Error: {str(e)}"

    # Construct final Bedrock-compatible response
    final_response = {
        'messageVersion': '1.0',
        **populate_function_response(event, response_body),
        'sessionAttributes': session_attributes,
        'promptSessionAttributes': prompt_session_attributes
    }
    
    # Print the response before returning
    print("Final response:", json.dumps(final_response, indent=2))
    
    return final_response