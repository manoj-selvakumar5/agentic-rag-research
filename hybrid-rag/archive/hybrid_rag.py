#!/usr/bin/env python3
"""
Description:
    This script demonstrates how to create Bedrock Agents (using Anthropics Claude-3),
    IAM Roles, and AWS Lambda functions to execute DSL (Domain-Specific Language) queries
    against Amazon OpenSearch or Amazon AOSS. It also shows how to add resource-based
    permissions to allow the Bedrock Agents to invoke the created Lambda functions.

    High-level steps:
      1. Create an IAM Role for Lambdas.
      2. Create/Update two Lambda functions:
         - execute-dsl-query
         - execute-modified-dsl-query
      3. Create DSL Query Agent and Query Fixer Agent, referencing the above Lambdas.
      4. Retrieve the newly created agent IDs.
      5. Add resource-based policy to each Lambda to allow invocation by those agent IDs.
      6. Create a Supervisor Agent orchestrating both DSL Query Agent and Query Fixer Agent.
      7. Invoke the Supervisor Agent with a sample query.
      8. Clean up by deleting the created agents.

Author:
    Your Name

Requirements:
    - Python 3.7 or higher
    - AWS SDK for Python (boto3)
    - Access to the Amazon Bedrock service (currently in preview)
    - Access to IAM and Lambda services

Usage:
    python create_bedrock_agents_and_lambdas.py
"""

import logging
import boto3
import os
import json
import time
import zipfile
import subprocess
from textwrap import dedent

# -----------------------------------------------------------------------------
# Configure Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Try importing the Bedrock utilities. If you have them locally, uncomment:
from src.utils.bedrock_agent import Agent, SupervisorAgent, agents_helper, region, account_id
# # -----------------------------------------------------------------------------
# try:
#     from src.utils.bedrock_agent import Agent, SupervisorAgent, agents_helper, region, account_id
# except ImportError:
#     logger.warning("Could not import bedrock_agent from src.utils; define or modify for your environment.")
#     # Mock classes for demonstration. Remove these mocks if you have the real module.
#     class Agent:
#         @staticmethod
#         def set_force_recreate_default(val: bool):
#             pass

#         @staticmethod
#         def direct_create(name, role, goal, instructions, tool_code, tool_defs):
#             return Agent()

#         def __getattr__(self, item):
#             return None

#         name = "demo"

#     class SupervisorAgent:
#         @staticmethod
#         def direct_create(name, role, collaboration_type, collaborator_objects, collaborator_agents, instructions):
#             return SupervisorAgent()

#         def invoke(self, input_text, session_id, enable_trace, trace_level):
#             return {"message": "Supervisor agent invoked (mock)."}

#     class agents_helper:
#         @staticmethod
#         def get_agent_id_by_name(name):
#             return f"mock-agent-id-for-{name}"

#         @staticmethod
#         def delete_agent(name, verbose=False):
#             if verbose:
#                 logger.info(f"Mock delete agent called for {name}")



# -----------------------------------------------------------------------------
# Global AWS Clients
# -----------------------------------------------------------------------------
sts_client = boto3.client('sts')
session = boto3.session.Session()

account_id = sts_client.get_caller_identity()["Account"]
region = session.region_name
account_id_suffix = account_id[:3]
agent_suffix = f"{region}-{account_id_suffix}"

s3_client = boto3.client('s3', region_name=region)
bedrock_client = boto3.client('bedrock-runtime', region_name=region)
iam_client = boto3.client('iam', region_name=region)
lambda_client = boto3.client('lambda', region_name=region)

# -----------------------------------------------------------------------------
# Load Shipping Schema (Example)
# -----------------------------------------------------------------------------
with open('schemas/ecom_shipping_schema.json', 'r') as file:
    ecom_shipping_schema = json.load(file)
ecom_shipping_schema_string = json.dumps(ecom_shipping_schema, indent=2)

# -----------------------------------------------------------------------------
# Agent foundation model (optional usage)
# -----------------------------------------------------------------------------
agent_foundation_model = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0"
]

# Force re-create default setting for Agent objects
Agent.set_force_recreate_default(True)

# -----------------------------------------------------------------------------
# IAM / Lambda helper functions
# -----------------------------------------------------------------------------
def create_iam_role(role_name: str) -> str:
    """
    Creates or retrieves an IAM Role with the necessary trust policy for Lambda.
    Attaches AWSLambdaBasicExecutionRole, and adds inline policies for OpenSearch 
    and AOSS access.

    :param role_name: Name of the IAM Role to create or retrieve.
    :return: ARN of the created or retrieved IAM Role.
    """
    logger.info(f"Creating or retrieving IAM Role: {role_name}")
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    try:
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
        )
        logger.info(f"IAM Role {role_name} created.")
    except iam_client.exceptions.EntityAlreadyExistsException:
        logger.info(f"IAM Role {role_name} already exists. Retrieving existing role.")
        role = iam_client.get_role(RoleName=role_name)

    # Attach AWS Lambda execution policy
    iam_client.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    )
    logger.info(f"Attached AWSLambdaBasicExecutionRole to {role_name}.")

    # Attach additional policies for OpenSearch access
    opensearch_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "es:Describe*",
                    "es:List*",
                    "es:Get*"
                ],
                "Resource": "*"
            }
        ]
    }
    opensearch_policy_name = f"{role_name}-OpenSearchPolicy"
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=opensearch_policy_name,
            PolicyDocument=json.dumps(opensearch_policy_document)
        )
        logger.info(f"Attached OpenSearch policy to IAM Role {role_name}.")
    except Exception as e:
        logger.error(f"Failed to attach OpenSearch policy to IAM Role {role_name}: {str(e)}")

    # Attach the new policy for aoss:APICall
    aoss_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "aoss:*"
                ],
                "Resource": "*"
            }
        ]
    }
    aoss_policy_name = f"{role_name}-AOSSPolicy"
    try:
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=aoss_policy_name,
            PolicyDocument=json.dumps(aoss_policy_document)
        )
        logger.info(f"Attached AOSS policy to IAM Role {role_name}.")
    except Exception as e:
        logger.error(f"Failed to attach AOSS policy to IAM Role {role_name}: {str(e)}")

    role_arn = role['Role']['Arn']

    # Wait for IAM role to propagate
    logger.info("Waiting 10 seconds for IAM role to propagate...")
    time.sleep(10)

    return role_arn


def create_lambda_package(source_file: str, zip_file_path: str, dependencies: list):
    """
    Packages a Lambda function and its dependencies into a single ZIP file.

    :param source_file: Path to the Lambda function source code.
    :param zip_file_path: Path to the ZIP file that will be created.
    :param dependencies: A list of Python packages required by the Lambda.
    """
    logger.info(f"Packaging Lambda function from {source_file} with dependencies {dependencies}")
    package_dir = "package"

    # Install dependencies to a local folder
    if not os.path.exists(package_dir):
        os.makedirs(package_dir)
    logger.info("Installing dependencies locally...")
    subprocess.run(
        f"pip install {' '.join(dependencies)} -t {package_dir}",
        shell=True,
        check=True
    )

    # Create ZIP file with dependencies and function
    logger.info(f"Creating Lambda deployment package: {zip_file_path}")
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        # Add dependencies
        for root, _, files in os.walk(package_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, package_dir)
                zipf.write(file_path, arcname)

        # Add the Lambda function code
        zipf.write(source_file, os.path.basename(source_file))

    # Cleanup temporary package directory
    logger.info("Cleaning up temporary package directory...")
    subprocess.run(f"rm -rf {package_dir}", shell=True)
    logger.info("Lambda package created successfully.")


def create_lambda_function(function_name: str,
                           role_arn: str,
                           handler: str,
                           runtime: str,
                           zip_file_path: str,
                           region_name: str = region) -> dict:
    """
    Creates or updates an AWS Lambda function.

    :param function_name: Name of the Lambda function to create or update.
    :param role_arn: ARN of the IAM Role that Lambda will assume.
    :param handler: The function handler (e.g., 'index.lambda_handler').
    :param runtime: The Lambda runtime (e.g., 'python3.12').
    :param zip_file_path: Path to the ZIP file containing the Lambda code.
    :param region_name: AWS region where the Lambda will be created.
    :return: The response from the create_function or update_function_code API call.
    """
    logger.info(f"Creating/updating Lambda function: {function_name}")
    lambda_client = boto3.client('lambda', region_name=region_name)

    with open(zip_file_path, 'rb') as f:
        zip_content = f.read()

    try:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role_arn,
            Handler=handler,
            Code={'ZipFile': zip_content},
            Description='Lambda function to execute DSL queries',
            Timeout=15,
            MemorySize=128,
            Publish=True
        )
        logger.info(f"Lambda function {function_name} created successfully.")
    except lambda_client.exceptions.ResourceConflictException:
        logger.info(f"Lambda function {function_name} already exists. Updating its code...")
        response = lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content
        )
        logger.info(f"Lambda function {function_name} updated successfully.")

    return response


def add_resource_based_policy(function_name: str,
                              agent_ids: list,
                              region_name: str,
                              account_id: str):
    """
    Adds a resource-based policy to the specified Lambda function to allow invocation
    from one or more Bedrock agents.

    :param function_name: Name of the Lambda function.
    :param agent_ids: List of agent IDs permitted to invoke this Lambda.
    :param region_name: AWS region.
    :param account_id: AWS account ID.
    """
    logger.info(f"Adding resource-based policy to Lambda function {function_name} for agents: {agent_ids}")
    statement_id_prefix = "AllowExecutionFromBedrockAgent"
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": []
    }

    for agent_id in agent_ids:
        sid = f"{statement_id_prefix}_{agent_id}"
        policy_doc['Statement'].append({
            "Sid": sid,
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "lambda:InvokeFunction",
            "Resource": f"arn:aws:lambda:{region_name}:{account_id}:function:{function_name}",
            "Condition": {
                "ArnLike": {
                    "AWS:SourceArn": f"arn:aws:bedrock:{region_name}:{account_id}:agent/{agent_id}"
                }
            }
        })

    # Retrieve existing policy and remove any existing statements with the same prefix
    try:
        existing_policy = lambda_client.get_policy(FunctionName=function_name)
        existing_policy_doc = json.loads(existing_policy['Policy'])
        for stmt in existing_policy_doc['Statement']:
            if stmt['Sid'].startswith(statement_id_prefix):
                sid_to_remove = stmt['Sid']
                logger.info(f"Removing existing statement: {sid_to_remove}")
                lambda_client.remove_permission(
                    FunctionName=function_name,
                    StatementId=sid_to_remove
                )
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.info(f"No existing policy found for Lambda function {function_name}.")
    except Exception as e:
        logger.error(f"Error retrieving/removing existing policy for {function_name}: {str(e)}")

    # Add new permissions
    for stmt in policy_doc['Statement']:
        sid_val = stmt['Sid']
        try:
            lambda_client.add_permission(
                FunctionName=function_name,
                StatementId=sid_val,
                Action=stmt['Action'],
                Principal=stmt['Principal']['Service'],
                SourceArn=stmt['Condition']['ArnLike']['AWS:SourceArn']
            )
            logger.info(f"Added permission for statement: {sid_val}")
        except Exception as e:
            logger.error(f"Failed to add resource-based policy for {function_name}, statement {sid_val}: {str(e)}")


def main():
    """
    Main execution flow:
      1. Create an IAM Role for Lambda.
      2. Create/Update two Lambda functions (execute-dsl-query, execute-modified-dsl-query).
      3. Create DSL Query Agent & Query Fixer Agent referencing those Lambda functions.
      4. Retrieve the newly created agent IDs.
      5. Add resource-based policies to each Lambda function for those agent IDs.
      6. Create the Supervisor Agent to orchestrate both DSL Query and Query Fixer agents.
      7. Invoke the Supervisor Agent with a sample query.
      8. Delete the agents (cleanup).
    """
    # -------------------------------------------------------------------------
    # 1. Create (or retrieve) IAM Role for Lambda
    # -------------------------------------------------------------------------
    IAM_ROLE_NAME = f"LambdaExecutionRole-{agent_suffix}"
    role_arn = create_iam_role(IAM_ROLE_NAME)

    # -------------------------------------------------------------------------
    # 2. Create the first Lambda (execute-dsl-query)
    # -------------------------------------------------------------------------
    DSL_QUERY_LAMBDA_NAME = f"execute-dsl-query-{agent_suffix}"
    DSL_QUERY_LAMBDA_PATH = "src/lambda/execute_dsl_query.py"
    DSL_QUERY_ZIP_PATH = "dsl_query_function.zip"

    if not os.path.exists(DSL_QUERY_LAMBDA_PATH):
        logger.error(f"Error: {DSL_QUERY_LAMBDA_PATH} does not exist.")
        return

    DEPENDENCIES = ["opensearch-py", "requests", "urllib3"]

    # Package & create the Lambda
    create_lambda_package(DSL_QUERY_LAMBDA_PATH, DSL_QUERY_ZIP_PATH, DEPENDENCIES)
    create_lambda_function(
        function_name=DSL_QUERY_LAMBDA_NAME,
        role_arn=role_arn,
        handler="execute_dsl_query.lambda_handler",
        runtime="python3.12",
        zip_file_path=DSL_QUERY_ZIP_PATH
    )
    os.remove(DSL_QUERY_ZIP_PATH)

    # -------------------------------------------------------------------------
    # 2(b). Create the second Lambda (execute-modified-dsl-query)
    # -------------------------------------------------------------------------
    MODIFIED_QUERY_LAMBDA_NAME = f"execute-modified-dsl-query-{agent_suffix}"
    MODIFIED_QUERY_LAMBDA_PATH = "src/lambda/execute_modified_dsl_query.py"
    MODIFIED_QUERY_ZIP_PATH = "modified_query_function.zip"

    if not os.path.exists(MODIFIED_QUERY_LAMBDA_PATH):
        logger.error(f"Error: {MODIFIED_QUERY_LAMBDA_PATH} does not exist.")
        return

    create_lambda_package(MODIFIED_QUERY_LAMBDA_PATH, MODIFIED_QUERY_ZIP_PATH, DEPENDENCIES)
    create_lambda_function(
        function_name=MODIFIED_QUERY_LAMBDA_NAME,
        role_arn=role_arn,
        handler="execute_modified_dsl_query.lambda_handler",
        runtime="python3.12",
        zip_file_path=MODIFIED_QUERY_ZIP_PATH
    )
    os.remove(MODIFIED_QUERY_ZIP_PATH)

    # -------------------------------------------------------------------------
    # 3. Create the DSL Query Agent & Query Fixer Agent
    #
    #    Important: reference the just-created Lambda ARNs in "tool_code"
    #    The actual ARN is "arn:aws:lambda:<REGION>:<ACCOUNT>:function:<FUNCTION_NAME>"
    # -------------------------------------------------------------------------
    dsl_query_lambda_arn = f"arn:aws:lambda:{region}:{account_id}:function:{DSL_QUERY_LAMBDA_NAME}"
    modified_query_lambda_arn = f"arn:aws:lambda:{region}:{account_id}:function:{MODIFIED_QUERY_LAMBDA_NAME}"

    logger.info("Creating DSL Query Agent...")
    dsl_query_agent = Agent.direct_create(
        name=f"dsl-query-agent-{agent_suffix}",
        role="DSL Query Creator",
        goal="Create DSL queries for a given user query",
        instructions=f"""
        You are an expert in generating Query DSL for Elasticsearch-style queries. Your task is to convert a 
        given natural language user question into a well-structured Query DSL.
        
        ## Instructions:
        - Use the provided e-commerce shipping schema to construct the query.
        - Encapsulate the output in <json>...</json> tags.
        - Follow the syntax of the Query DSL strictly; do not introduce elements outside the provided schema.
        
        ## Query Construction Rules:
        - **Keyword fields** (carrier, status, country): Use `term` for exact matches or `prefix`/`wildcard` for partial matches.
        - **Text fields** (description, address): Use `match` queries to account for analyzed terms.
        - **Nested fields** (tracking): Always use `nested` queries.
        - **Date fields**: Use `range` queries with date math for filtering by date ranges.
        - Break down complex queries into smaller parts for accuracy.
        - Think step-by-step before constructing the query.

        ## Schema:
        {ecom_shipping_schema_string}

        ## Output Format:
        - Return only the generated Query DSL within <json>...</json> tags.
        - Do not include explanations, comments, or additional text.
        """,
        tool_code=dsl_query_lambda_arn,
        tool_defs=[
            {
                "name": "execute_dsl_query",
                "description": "Executes a given DSL query and returns the results",
                "parameters": {
                    "dsl_query": {
                        "description": "The DSL query to execute",
                        "type": "string",
                        "required": True,
                    }
                }
            }
        ]
    )

    logger.info("Creating Query Fixer Agent...")
    query_fixer_agent = Agent.direct_create(
        name=f"query-fixer-agent-{agent_suffix}",
        role="Query Repair Specialist",
        goal="Fix and optimize failed DSL queries",
        instructions=f"""
        You are an expert query debugger and optimizer. Your tasks are:
        1. Analyze failed DSL queries from the query generator
        2. Diagnose errors using OpenSearch error messages
        3. Apply targeted fixes while maintaining original intent
        4. Optimize queries for better recall when results are empty

        ## Repair Strategies:
        - SYNTAX ERRORS: Fix formatting issues in nested queries/aggregations
        - FIELD ERRORS: Map invalid fields to valid schema equivalents
        - ZERO HITS: Apply query relaxation techniques:
          * Add wildcards to keyword searches
          * Expand date ranges
          * Reduce strictness of term matches
          * Add synonym expansion

        ## Optimization Rules:
        - Maintain original query structure where possible
        - Prefer query-time fixes over rearchitecting
        - Document all modifications in revision notes
        - Limit query relaxation to 3 iterations

        ## Schema:
        {ecom_shipping_schema_string}

        ## Output Format:
        - Return modified query in <json> tags
        - Include revision notes in <notes> tags
        """,
        tool_code=modified_query_lambda_arn,
        tool_defs=[
            {
                "name": "retry_query",
                "description": "Retries a modified version of the failed query",
                "parameters": {
                    "modified_dsl_query": {
                        "description": "The corrected DSL query",
                        "type": "string",
                        "required": True
                    },
                    "revision_notes": {
                        "description": "Description of modifications made",
                        "type": "string",
                    },
                }
            }
        ]
    )

    # -------------------------------------------------------------------------
    # 4. Retrieve the newly created Agent IDs
    # -------------------------------------------------------------------------
    logger.info("Retrieving DSL Query Agent ID...")
    dsl_query_agent_id = agents_helper.get_agent_id_by_name(dsl_query_agent.name)
    logger.info(f"DSL Query Agent ID: {dsl_query_agent_id}")

    logger.info("Retrieving Query Fixer Agent ID...")
    query_fixer_agent_id = agents_helper.get_agent_id_by_name(query_fixer_agent.name)
    logger.info(f"Query Fixer Agent ID: {query_fixer_agent_id}")

    # -------------------------------------------------------------------------
    # 5. Add resource-based policy to each Lambda so the Agents can invoke them
    # -------------------------------------------------------------------------
    add_resource_based_policy(DSL_QUERY_LAMBDA_NAME, [dsl_query_agent_id], region, account_id)
    add_resource_based_policy(MODIFIED_QUERY_LAMBDA_NAME, [query_fixer_agent_id], region, account_id)

    # -------------------------------------------------------------------------
    # 6. Create the Supervisor Agent orchestrating DSL & Query Fixer
    # -------------------------------------------------------------------------
    logger.info("Creating Supervisor Agent...")
    supervisor_agent = SupervisorAgent.direct_create(
        name=f"supervisor-agent-{agent_suffix}",
        role="Query Pipeline Orchestrator",
        collaboration_type="SUPERVISOR",
        collaborator_objects=[dsl_query_agent, query_fixer_agent],
        collaborator_agents=[
            {
                "agent": dsl_query_agent.name,
                "instructions": "Primary DSL query generation using the schema",
                "relay_conversation_history": "DISABLED"
            },
            {
                "agent": query_fixer_agent.name,
                "instructions": dedent("""
                    Engage when:
                    1. DSL query returns errors (parsing/validation)
                    2. Search results are empty (zero hits)
                    3. Query needs optimization for better recall

                    Responsibilities:
                    - Analyze error messages and query structure
                    - Apply targeted fixes while preserving intent
                    - Implement query relaxation strategies
                    - Document modifications made
                """),
                "relay_conversation_history": "TO_COLLABORATOR"
            }
        ],
        instructions=dedent(f"""
            Orchestrate the end-to-end query generation and validation workflow:

            1. Initial Query Generation:
            - Receive natural language query from user
            - Route to DSL Query Agent for initial construction
            - Validate query structure against schema:
              {ecom_shipping_schema_string}

            2. Error Handling & Retry:
            - Monitor for query execution errors
            - On error/zero results:
              a. Capture error context and original query
              b. Route to Query Fixer Agent with full diagnostics
              c. Validate fixer's modified query
              d. Approve max 3 retry attempts

            3. Quality Assurance:
            - Ensure final query meets quality standards:
              - Proper use of nested queries
              - Correct field types and mappings
              - Appropriate query strictness level
            - Maintain audit trail of all query versions
            - Provide user with cleaned error explanations

            4. Final Approval:
            - Sign off on valid queries
            - Block invalid queries with detailed feedback
            - Generate execution summary with:
              - Query versions attempted
              - Modification reasons
              - Performance metrics
        """)
    )

    # -------------------------------------------------------------------------
    # 7. Invoke the Supervisor Agent with a sample query
    # -------------------------------------------------------------------------
    logger.info("Invoking Supervisor Agent with a sample query...")
    response = supervisor_agent.invoke(
        input_text="How many orders have been shipped by DHL?",
        session_id="12345",
        enable_trace=True,
        trace_level="core"
    )
    logger.info(f"Supervisor agent response: {response}")

    # # -------------------------------------------------------------------------
    # # 8. Cleanup: Delete the created agents
    # # -------------------------------------------------------------------------
    # logger.info("Deleting Supervisor Agent...")
    # agents_helper.delete_agent(supervisor_agent.name, verbose=True)

    # logger.info("Deleting DSL Query Agent...")
    # agents_helper.delete_agent(dsl_query_agent.name, verbose=True)

    # logger.info("Deleting Query Fixer Agent...")
    # agents_helper.delete_agent(query_fixer_agent.name, verbose=True)

    # logger.info("All agents deleted. Script completed.")


if __name__ == "__main__":
    main()
