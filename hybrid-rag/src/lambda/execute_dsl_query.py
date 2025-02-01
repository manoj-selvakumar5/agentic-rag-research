import json
import boto3
from opensearchpy import OpenSearch

def initialize_opensearch_client(host):
    return OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        use_ssl=True,
        verify_certs=True
    )

def query_oss(client, query, verbose=False):
    """
    Queries OpenSearch with the provided query dictionary.

    Args:
        client (OpenSearch): The OpenSearch client.
        query (dict): The query dictionary.
        verbose (bool): If True, prints additional information.

    Returns:
        dict: The response from the OpenSearch search operation.
    """
    if verbose:
        print(f"Querying OpenSearch with: {json.dumps(query, indent=2)}")

    response = client.search(
        index="_all",
        body=query
    )

    return response

def lambda_handler(event, context):
    # Extract the DSL query from the event
    dsl_query = event.get('dsl_query')
    if not dsl_query:
        return {
            'statusCode': 400,
            'body': json.dumps('Missing dsl_query parameter')
        }

    # Initialize the OpenSearch client
    host = "5g240wxidf1c8njrw17l.us-west-2.aoss.amazonaws.com"
    client = initialize_opensearch_client(host)

    # Execute the DSL query
    try:
        response = query_oss(client, dsl_query)
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }