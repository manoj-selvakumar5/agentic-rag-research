import json
import boto3
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

# Configure Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
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
    
    # Extract parameters from both possible sources
    parameters = {p['name']: p['value'] for p in event.get('parameters', [])}
    modified_query = parameters.get('modified_dsl_query') or parameters.get('dsl_query')
    revision_notes = parameters.get('revision_notes', 'No revision notes provided')

    response_body = {'TEXT': {'body': ""}}
    session_attrs = event.get('sessionAttributes', {})
    prompt_attrs = event.get('promptSessionAttributes', {})

    try:
        if not modified_query:
            raise ValueError("Missing required query parameter (modified_dsl_query/dsl_query)")

        logger.info("Executing modified query:\n%s", json.dumps(json.loads(modified_query), indent=2))
        logger.info("Revision notes: %s", revision_notes)
        
        oss_response = OpenSearchManager(host).query(json.loads(modified_query))
        
        # Process response for better readability
        result = {
            'hits': oss_response["hits"]["total"]["value"],
            'aggregations': oss_response.get('aggregations', {}),
            'revision_notes': revision_notes,
            'success': True
        }
        
        response_body['TEXT']['body'] = json.dumps(result, indent=2)

    except Exception as e:
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'modified_query': modified_query,
            'revision_notes': revision_notes,
            'success': False
        }
        response_body['TEXT']['body'] = json.dumps(error_info, indent=2)
        logger.error("Query execution failed: %s", error_info)

    # Construct final Bedrock-compatible response
    final_response = {
        'messageVersion': '1.0',
        **populate_function_response(event, response_body),
        'sessionAttributes': session_attrs,
        'promptSessionAttributes': prompt_attrs
    }

    # Print the response before returning
    print("Final response:", json.dumps(final_response, indent=2))
    
    return final_response