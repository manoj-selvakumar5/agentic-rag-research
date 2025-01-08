#!/usr/bin/env python3
"""
===============================================================================
Improved Bedrock Agents Script (with Extended Tracing)
===============================================================================
This script provides classes and methods to create, update, and delete
Bedrock Agents, along with optional AWS resources (e.g., Lambda, IAM, DynamoDB).
It includes functionality to configure an agent, attach code interpreters,
create action groups, and invoke the agent for inference or retrieval tasks.

Logging is set up to print directly to stdout (which Jupyter notebooks can
display in real time). Classes and methods are extensively commented for clarity.
"""

import logging
import sys  # <-- For attaching the stream handler to sys.stdout
import boto3
import json
import time
import zipfile
from io import BytesIO
import uuid
from typing import List, Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from datetime import datetime

# For colored printing in the trace (like the reference code)
try:
    from termcolor import colored
except ImportError:
    def colored(text, _color):
        return text  # fallback: no coloring

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

# Default constants for agent alias/policies
DEFAULT_ALIAS = "TSTALIASID"
DEFAULT_AGENT_IAM_ASSUME_ROLE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowBedrock",
            "Effect": "Allow",
            "Principal": {
                "Service": "bedrock.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ],
}
DEFAULT_AGENT_IAM_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AmazonBedrockAgentInferencProfilePolicy1",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel*",
                "bedrock:CreateInferenceProfile"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/*",
                "arn:aws:bedrock:*:*:inference-profile/*",
                "arn:aws:bedrock:*:*:application-inference-profile/*",
            ],
        },
        {
            "Sid": "AmazonBedrockAgentInferencProfilePolicy2",
            "Effect": "Allow",
            "Action": [
                "bedrock:GetInferenceProfile",
                "bedrock:ListInferenceProfiles",
                "bedrock:DeleteInferenceProfile",
                "bedrock:TagResource",
                "bedrock:UntagResource",
                "bedrock:ListTagsForResource"
            ],
            "Resource": [
                "arn:aws:bedrock:*:*:inference-profile/*",
                "arn:aws:bedrock:*:*:application-inference-profile/*"
            ]
        },
        {
            "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicy",
            "Effect": "Allow",
            "Action": [
                "bedrock:GetAgentAlias",
                "bedrock:InvokeAgent"
            ],
            "Resource": [
                "arn:aws:bedrock:*:*:agent/*",
                "arn:aws:bedrock:*:*:agent-alias/*"
            ]
        },
        {
            "Sid": "AmazonBedrockAgentBedrockInvokeGuardrailModelPolicy",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:GetGuardrail",
                "bedrock:ApplyGuardrail"
            ],
            "Resource": "arn:aws:bedrock:*:*:guardrail/*"
        },
        {
            "Sid": "QueryKB",
            "Effect": "Allow",
            "Action": [
                "bedrock:Retrieve",
                "bedrock:RetrieveAndGenerate"
            ],
            "Resource": "arn:aws:bedrock:*:*:knowledge-base/*"
        }
    ]
}

# For optional trace logic
TRACE_TRUNCATION_LENGTH = 1000000
UNDECIDABLE_CLASSIFICATION = 'undecidable'

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
# 3. BedrockAgents Class
# -----------------------------------------------------------------------------
class BedrockAgents:
    """
    Provides methods to create, update, and delete Bedrock Agents, along with 
    helper methods to manage agent action groups and optional code interpreter 
    integration. Agents can also be associated with knowledge bases.
    """

    def __init__(self):
        """
        Initializes BedrockAgents with:
            - A shared AppContext for region/account info
            - Clients for bedrock-agent, bedrock-agent-runtime, IAM, STS, and Lambda
        """
        self.context = AppContext()
        self.bedrock_agent_client = self.context.client('bedrock-agent')
        self.bedrock_agent_runtime_client = self.context.client('bedrock-agent-runtime')
        self.iam_client = self.context.client('iam')
        self.sts_client = self.context.client('sts')
        self.lambda_client = self.context.client('lambda')

    # -------------------------------------------------------------------------
    # 3.1: Create Agent Role
    # -------------------------------------------------------------------------
    def create_bedrock_agent_role(
        self,
        role_name: str,
        model_list: List[str],
        agent_name: str,
        verbose: bool = False
    ) -> str:
        """
        Creates an IAM role for the Bedrock Agent with specified policies.
        The role allows the Agent to invoke specified foundation models 
        or inference profiles.
    
        Args:
            role_name (str): Name of the IAM role to create.
            model_list (List[str]): List of model or inference profile IDs 
                                    the role should have access to.
            agent_name (str): Name of the agent to use in policy names.
            verbose (bool): If True, logs additional information.
    
        Returns:
            str: ARN of the created (or existing) role.
        """
        if verbose:
            logger.info(f"Creating IAM role: {role_name}")
    
        # Assume role policy document for Bedrock
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
    
        # Build resource ARNs (both inference profiles and foundation models)
        inference_profile_resources = [
            f"arn:aws:bedrock:us-west-2:533267284022:inference-profile/{model}" for model in model_list
        ]
        foundation_model_resources = [
            f"arn:aws:bedrock:*::foundation-model/{model.split('us.')[-1]}" 
            for model in model_list
        ]
    
        # Custom policy to allow this role to invoke the specified resources
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AmazonBedrockAgentModelAccessPolicy",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:GetInferenceProfile",
                        "bedrock:GetFoundationModel"
                    ],
                    "Resource": inference_profile_resources + foundation_model_resources
                }
            ]
        }
    
        try:
            # Create the IAM role
            role_response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
            )
            role_arn = role_response["Role"]["Arn"]
            if verbose:
                logger.info(f"Created IAM role: {role_arn}")
    
            # Wait for IAM to propagate
            time.sleep(5)
    
            # Policy details
            policy_name = f"{agent_name}-AmazonBedrockAgentModelAccessPolicy"
            policy_arn = f"arn:aws:iam::{self.context.account_number}:policy/{policy_name}"
    
            # Check if the policy already exists
            try:
                self.iam_client.get_policy(PolicyArn=policy_arn)
                if verbose:
                    logger.info(f"Policy '{policy_name}' already exists. Attaching to role.")
            except self.iam_client.exceptions.NoSuchEntityException:
                # Create the policy if it does not exist
                policy_response = self.iam_client.create_policy(
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(policy_document)
                )
                policy_arn = policy_response["Policy"]["Arn"]
                if verbose:
                    logger.info(f"Created managed policy: {policy_arn}")
    
            # Attach the policy to the role
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            if verbose:
                logger.info(f"Attached policy '{policy_name}' to role '{role_name}'")
    
            return role_arn
    
        except ClientError as e:
            logger.error(f"Error creating role '{role_name}': {e}")
            raise e

    # -------------------------------------------------------------------------
    # 3.2: Update Role Policy
    # -------------------------------------------------------------------------
    def update_role_policy(
        self, 
        role_arn: str, 
        policy_name: str, 
        new_statement: Dict[str, Any], 
        agent_name: str,
        verbose: bool = False
    ) -> None:
        """
        Updates (or creates) an IAM policy attached to a given role by ARN, 
        appending a new policy statement if the policy already exists.

        Args:
            role_arn (str): ARN of the IAM role to update.
            policy_name (str): Base policy name to create or update.
            new_statement (Dict[str, Any]): New policy statement to add.
            agent_name (str): Name of the agent, for constructing final policy name.
            verbose (bool, optional): If True, logs additional info.
        """
        policy_name = f"{agent_name}-{policy_name}"
        role_name = role_arn.split('/')[-1]
        
        if verbose:
            logger.info(f"Updating policy '{policy_name}' for role '{role_arn}'")
    
        policy_arn = f"arn:aws:iam::{self.context.account_number}:policy/{policy_name}"
    
        try:
            try:
                # Try to get existing policy version
                policy_version = self.iam_client.get_policy_version(
                    PolicyArn=policy_arn,
                    VersionId='v1'
                )
                policy_document = policy_version['PolicyVersion']['Document']
                policy_document['Statement'].append(new_statement)
            
            except self.iam_client.exceptions.NoSuchEntityException:
                # Create new policy if it doesn't exist
                policy_document = {
                    "Version": "2012-10-17",
                    "Statement": [new_statement]
                }
                self.iam_client.create_policy(
                    PolicyName=policy_name,
                    PolicyDocument=json.dumps(policy_document)
                )
                if verbose:
                    logger.info(f"Created new policy: {policy_name}")
                
                # Attach the new policy to the role
                self.iam_client.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
                if verbose:
                    logger.info(f"Attached policy '{policy_name}' to role '{role_name}'")
                return
    
            # Update existing policy by creating a new version
            self.iam_client.create_policy_version(
                PolicyArn=policy_arn,
                PolicyDocument=json.dumps(policy_document),
                SetAsDefault=True
            )
    
            # Delete older versions if there are more than 5
            versions = self.iam_client.list_policy_versions(PolicyArn=policy_arn)['Versions']
            if len(versions) > 5:
                versions_to_delete = sorted(versions, key=lambda x: x['CreateDate'])[:-5]
                for version in versions_to_delete:
                    self.iam_client.delete_policy_version(
                        PolicyArn=policy_arn,
                        VersionId=version['VersionId']
                    )
    
            if verbose:
                logger.info(f"Updated policy '{policy_name}' successfully")
    
        except ClientError as e:
            logger.error(f"Error managing policy '{policy_name}': {e}")
            raise

    # -------------------------------------------------------------------------
    # 3.3: Create a Bedrock Agent
    # -------------------------------------------------------------------------
    def create_bedrock_agent(
        self, 
        agent_name: str, 
        agent_description: str,
        agent_instructions: str,
        model_id: str,
        kb_id: str = None,
        associate_kb: bool=False,
        code_interpreter: bool=False,
        kb_usage_description: str=None,
        agent_collaboration: str='DISABLED',
        verbose: bool=False
    ) -> Tuple[str, str, str]:
        """
        Create a new Bedrock Agent with optional knowledge base association 
        and code interpreter action group.

        Args:
            agent_name (str): Name of the agent to create.
            agent_description (str): Agent description for reference.
            agent_instructions (str): Instruction text for the agent.
            model_id (str): The foundation model or inference profile ID the agent will use.
            kb_id (str, optional): ID of the knowledge base to associate, if needed.
            associate_kb (bool, optional): If True, associates the agent with the provided kb_id.
            code_interpreter (bool, optional): If True, adds a code interpreter action group.
            kb_usage_description (str, optional): Description of how the agent will use the KB.
            agent_collaboration (str, optional): 'DISABLED' or 'ENABLED' for agent collaboration.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            (agent_id, agent_alias_id, agent_alias_arn): Identifiers for the created agent.
        """
        # 1) Create or retrieve an IAM role for this agent
        agent_role_arn = self.create_bedrock_agent_role(
            role_name=f"{agent_name}-exec-role",
            model_list=[model_id],
            agent_name=agent_name,
            verbose=verbose
        )

        # 2) Create the actual agent
        create_agent_response = self.bedrock_agent_client.create_agent(
            agentName=agent_name,
            description=agent_description,
            agentResourceRoleArn=agent_role_arn,
            idleSessionTTLInSeconds=1800,
            foundationModel=model_id,
            instruction=agent_instructions,
            agentCollaboration=agent_collaboration
        )

        agent_id = create_agent_response["agent"]["agentId"]
        if verbose:
            logger.info(f"Created agent: {agent_id}")

        # Build the alias ARN from the returned agent ARN
        agent_alias_id = DEFAULT_ALIAS
        agent_alias_arn = create_agent_response["agent"]["agentArn"].replace("agent", "agent-alias") + f"/{agent_alias_id}"

        # Wait a bit for the agent to be fully created
        time.sleep(15)

        # 3) If requested, associate knowledge base
        if associate_kb and kb_id:
            # Add permission to query the KB
            kb_policy_statement = {
                "Sid": "QueryKB",
                "Effect": "Allow",
                "Action": ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                "Resource": "arn:aws:bedrock:*:*:knowledge-base/*",
            }
            # Update the agent's role policy
            self.update_role_policy(
                role_arn=agent_role_arn, 
                policy_name="AmazonBedrockAgentKBAccessPolicy", 
                new_statement=kb_policy_statement, 
                agent_name=agent_name,
                verbose=verbose
            )
            # Associate the agent with the knowledge base
            self.associate_agent_with_kb(agent_id, kb_id, kb_usage_description, verbose)

        # 4) If requested, add the code interpreter
        if code_interpreter:
            self.add_code_interpreter(agent_name, verbose)

        # 5) Update the agent's role policy to allow agent collaboration
        if agent_collaboration != 'DISABLED':
            collaboration_policy_statement = {
                "Sid": "CollaborationPolicy",
                "Effect": "Allow",
                "Action": [
                    "bedrock:GetAgentAlias",
                    "bedrock:InvokeAgent"
                ],
                "Resource": [
                    "arn:aws:bedrock:*:*:agent/*",
                    "arn:aws:bedrock:*:*:agent-alias/*"
                ]
            }
            self.update_role_policy(
                role_arn=agent_role_arn, 
                policy_name="AmazonBedrockMultiAgentAccessPolicy", 
                new_statement=collaboration_policy_statement, 
                agent_name=agent_name,
                verbose=verbose
            )
        
        return agent_id, agent_alias_id, agent_alias_arn

    # -------------------------------------------------------------------------
    # 3.4: Associate an Agent with a Knowledge Base
    # -------------------------------------------------------------------------
    def associate_agent_with_kb(self, agent_id: str, kb_id: str, kb_usage_description: str, verbose: bool=False) -> None:
        """
        Associates an existing Bedrock Agent with an existing Knowledge Base, 
        enabling the agent to retrieve or retrieve+generate from that KB.

        Args:
            agent_id (str): ID of the agent to associate.
            kb_id (str): ID of the knowledge base to associate.
            kb_usage_description (str): Description of how the agent will use the KB.
            verbose (bool, optional): If True, logs additional info.
        """
        if verbose:
            logger.info(f"Associating agent '{agent_id}' with knowledge base '{kb_id}'")

        associate_kb_with_agent_response = self.bedrock_agent_client.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion="DRAFT",
            description=kb_usage_description,
            knowledgeBaseId=kb_id,
            knowledgeBaseState="ENABLED",
        )

        if verbose:
            logger.info(f"Associated agent '{agent_id}' with KB '{kb_id}'")
            logger.info(f"Response: {associate_kb_with_agent_response}")

        # Prepare the agent to ensure it's ready for invocation
        prepare_agent_response = self.bedrock_agent_client.prepare_agent(agentId=agent_id)
        if verbose:
            logger.info(f"Agent '{agent_id}' is prepared for invocation: {prepare_agent_response}")

    # -------------------------------------------------------------------------
    # 3.5: Delete a Bedrock Agent
    # -------------------------------------------------------------------------
    def delete_bedrock_agent(self, agent_name: str, delete_role: bool=False, verbose: bool=False) -> None:
        """
        Delete a Bedrock Agent by name, including its aliases, action groups, 
        and optional associated IAM roles.

        Args:
            agent_name (str): Name of the agent to delete.
            delete_role (bool, optional): If True, deletes the agent IAM role.
            verbose (bool, optional): If True, logs additional information.
        """
        # 1) Find the agent by name
        agents_resp = self.bedrock_agent_client.list_agents(maxResults=100)
        agents_json = agents_resp["agentSummaries"]
        target_agent = next(
            (agent for agent in agents_json if agent["agentName"] == agent_name), None
        )
    
        if target_agent is None:
            logger.info(f"Agent '{agent_name}' not found. Skipping deletion.")
        else:
            if verbose:
                logger.info(f"Found agent '{agent_name}' with ID: {target_agent['agentId']}")
        
            agent_id = target_agent["agentId"]

            # 2) Delete agent aliases
            if verbose:
                logger.info(f"Deleting aliases for agent {agent_id}...")
            try:
                agent_aliases = self.bedrock_agent_client.list_agent_aliases(agentId=agent_id, maxResults=100)
                for alias in agent_aliases["agentAliasSummaries"]:
                    alias_id = alias["agentAliasId"]
                    logger.info(f"Deleting alias {alias_id} from agent {agent_id}")
                    self.bedrock_agent_client.delete_agent_alias(agentAliasId=alias_id, agentId=agent_id)
            except Exception as e:
                logger.error(f"Error deleting aliases: {e}")

            # 3) Delete action groups + associated Lambda functions
            if verbose:
                logger.info(f"Deleting action groups for agent {agent_id}...")
            try:
                action_groups = self.bedrock_agent_client.list_agent_action_groups(
                    agentId=agent_id, 
                    agentVersion="DRAFT", 
                    maxResults=100
                )
                for action_group in action_groups["actionGroupSummaries"]:
                    action_group_name = action_group["actionGroupName"]
                    action_group_id = action_group["actionGroupId"]
                    if verbose:
                        logger.info(f"Deleting action group '{action_group_name}' (ID: {action_group_id}) from agent {agent_id}")

                    # Get details to retrieve the Lambda ARN
                    action_group_details = self.bedrock_agent_client.get_agent_action_group(
                        agentId=agent_id, 
                        agentVersion="DRAFT", 
                        actionGroupId=action_group_id
                    )

                    if verbose:
                        logger.info(f"Action group details: {action_group_details}")

                    # Extract Lambda function name
                    lambda_arn = action_group_details["agentActionGroup"]["actionGroupExecutor"]["lambda"]
                    lambda_function_name = lambda_arn.split(":")[-1]

                    # Attempt to delete the Lambda function
                    if verbose:
                        logger.info(f"Deleting Lambda function: {lambda_function_name}")
                    try:
                        self.lambda_client.delete_function(FunctionName=lambda_function_name)
                        if verbose:
                            logger.info(f"Deleted Lambda function: {lambda_function_name}")

                        # Optionally delete the Lambda function’s IAM role
                        if delete_role:
                            function_conf = self.lambda_client.get_function(FunctionName=lambda_function_name)
                            lambda_role_arn = function_conf['Configuration']['Role']
                            lambda_role_name = lambda_role_arn.split('/')[-1]
                            logger.info(f"Deleting IAM role for Lambda: {lambda_role_name}")
                            try:
                                attached_policies = self.iam_client.list_attached_role_policies(RoleName=lambda_role_name)
                                for policy in attached_policies['AttachedPolicies']:
                                    self.iam_client.detach_role_policy(
                                        RoleName=lambda_role_name, 
                                        PolicyArn=policy['PolicyArn']
                                    )
                                    policy_versions = self.iam_client.list_policy_versions(PolicyArn=policy['PolicyArn'])
                                    for version in policy_versions['Versions']:
                                        if not version['IsDefaultVersion']:
                                            self.iam_client.delete_policy_version(
                                                PolicyArn=policy['PolicyArn'],
                                                VersionId=version['VersionId']
                                            )
                                    self.iam_client.delete_policy(PolicyArn=policy['PolicyArn'])
                                self.iam_client.delete_role(RoleName=lambda_role_name)
                                logger.info(f"Deleted role: {lambda_role_name}")
                            except ClientError as e:
                                if e.response['Error']['Code'] == 'NoSuchEntity':
                                    logger.warning(f"Role {lambda_role_name} does not exist.")
                                else:
                                    logger.error(f"Failed to delete role {lambda_role_name}: {e}", exc_info=True)
                    except ClientError as e:
                        logger.error(f"Error deleting Lambda function {lambda_function_name}: {e}")
        
                    # Finally, remove the action group
                    self.bedrock_agent_client.delete_agent_action_group(
                        agentId=agent_id, 
                        agentVersion="DRAFT", 
                        actionGroupId=action_group_id, 
                        skipResourceInUseCheck=True
                    )
                    if verbose:
                        logger.info(f"Deleted action group: {action_group_name}")
            except Exception as e:
                logger.error(f"Error deleting action groups: {e}")

            # 4) Delete the agent itself
            if verbose:
                logger.info(f"Deleting agent: {agent_id}")
            self.context.interactive_sleep(5)
            self.bedrock_agent_client.delete_agent(agentId=agent_id)
            self.context.interactive_sleep(5)
            logger.info(f"Deleted agent: {agent_id}")
        
        # 5) Optionally delete the agent's IAM role
        if delete_role:
            role_name = f"{agent_name}-exec-role"
            logger.info(f"Deleting IAM role: {role_name}")
            try:
                attached_policies = self.iam_client.list_attached_role_policies(RoleName=role_name)
                for policy in attached_policies['AttachedPolicies']:
                    self.iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
                    policy_versions = self.iam_client.list_policy_versions(PolicyArn=policy['PolicyArn'])
                    for version in policy_versions['Versions']:
                        if not version['IsDefaultVersion']:
                            self.iam_client.delete_policy_version(
                                PolicyArn=policy['PolicyArn'],
                                VersionId=version['VersionId']
                            )
                    self.iam_client.delete_policy(PolicyArn=policy['PolicyArn'])
                self.iam_client.delete_role(RoleName=role_name)
                logger.info(f"Deleted role: {role_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchEntity':
                    logger.warning(f"Role {role_name} does not exist.")
                else:
                    logger.error(f"Failed to delete role {role_name}: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 3.6: Get Agent ID by Name
    # -------------------------------------------------------------------------
    def get_agent_id_by_name(self, agent_name: str, verbose: bool=False) -> Optional[str]:
        """
        Retrieves the Agent ID for the specified Agent name, if it exists.

        Args:
            agent_name (str): The name of the agent to find.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            Optional[str]: The agent ID if found, otherwise None.
        """
        try:
            get_agents_resp = self.bedrock_agent_client.list_agents(maxResults=100)
            agents_json = get_agents_resp["agentSummaries"]
            target_agent = next(
                (agent for agent in agents_json if agent["agentName"] == agent_name), None
            )
            if target_agent:
                return target_agent["agentId"]
            else:
                if verbose:
                    logger.warning(f"Agent '{agent_name}' not found.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving agent ID for '{agent_name}': {e}")
            return None

    # -------------------------------------------------------------------------
    # 3.7: Prepare Bedrock Agent for Invocation
    # -------------------------------------------------------------------------
    def prepare_bedrock_agent(self, agent_name: str, verbose: bool=False) -> None:
        """
        Prepares an existing agent for invocation by calling `prepare_agent`.
    
        Args:
            agent_name (str): Name of the agent to prepare.
            verbose (bool, optional): If True, logs additional info.
        """
        agent_id = self.get_agent_id_by_name(agent_name, verbose=verbose)
        if agent_id is None:
            raise ValueError(f"Agent {agent_name} not found")
    
        self.bedrock_agent_client.prepare_agent(agentId=agent_id)
        
        # Check preparation status with retries
        max_retries = 12  # 1 minute total (5 seconds * 12)
        retry_count = 0
        
        while retry_count < max_retries:
            status = self.get_agent_status(agent_name, verbose)
            if status.get("agentStatus") == "PREPARED":
                if verbose:
                    logger.info(f"Agent {agent_name} is now prepared")
                return
                
            retry_count += 1
            if verbose:
                logger.info(f"Agent preparing... (attempt {retry_count}/{max_retries})")
            self.context.interactive_sleep(5)
        
        raise TimeoutError(f"Agent {agent_name} failed to prepare after {max_retries} attempts")

    # -------------------------------------------------------------------------
    # 3.8: Add Action Group to an Agent
    # -------------------------------------------------------------------------
    def add_action_group_to_agent(
            self, 
            agent_name: str, 
            action_group_name: str, 
            action_group_description: str,
            agent_functions: List[Dict]=None,
            action_group_invocation_method: str=None,
            lambda_function_name: str=None,
            lambda_env_variables: Dict=None,
            source_code_file: str=None,
            additional_function_iam_policy: Dict=None,
            verbose: bool=False
    ) -> None:
        """
        Adds an action group to a Bedrock Agent. This action group can be 
        either 'ROC' (return of control) or 'LAMBDA' (invoke a Lambda function).

        Args:
            agent_name (str): Name of the existing agent.
            action_group_name (str): Name of the action group.
            action_group_description (str): Description of the action group.
            agent_functions (List[Dict], optional): A list of function definitions 
                                                    for this group.
            action_group_invocation_method (str): One of ["ROC", "LAMBDA"].
            lambda_function_name (str, optional): The name of the Lambda function 
                                                  to create/invoke (if LAMBDA).
            lambda_env_variables (Dict, optional): Environment variables for the Lambda.
            source_code_file (str, optional): Python source code to package for Lambda.
            additional_function_iam_policy (Dict, optional): Additional policy 
                                                             for the Lambda's IAM role.
            verbose (bool, optional): If True, logs additional info.
        """
        if action_group_invocation_method not in ["ROC", "LAMBDA"]:
            logger.error("Invalid action group invocation method. Must be 'ROC' or 'LAMBDA'.")
            return
        
        # If we need a Lambda action group, create the Lambda function
        if action_group_invocation_method == "LAMBDA":
            lambda_arn = self.create_lambda(
                agent_name, 
                lambda_function_name=lambda_function_name, 
                source_code_file=source_code_file, 
                lambda_env_variables=lambda_env_variables, 
                additional_function_iam_policy=additional_function_iam_policy,
                verbose=verbose
            )
            action_group_executor = {"lambda": lambda_arn}
        else:
            action_group_executor = {"customControl": "RETURN_CONTROL"}

        # Retrieve the agent ID
        agent_id = self.get_agent_id_by_name(agent_name, verbose=verbose)
        if agent_id is None:
            if verbose:
                logger.error(f"Agent '{agent_name}' not found")
            return
        
        if verbose:
            logger.info(f"Adding functions to action group '{action_group_name}'")
            logger.info(f"Functions: {agent_functions}")

        # Create the action group
        create_action_group_response = self.bedrock_agent_client.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupName=action_group_name,
            description=action_group_description,
            actionGroupExecutor=action_group_executor,
            functionSchema={"functions": agent_functions}
        )

        if verbose:
            logger.info(f"Created action group '{action_group_name}' for agent '{agent_name}'")
            logger.info(f"Response: {create_action_group_response}")

    # -------------------------------------------------------------------------
    # 3.9: Create a Lambda for an Agent Action Group
    # -------------------------------------------------------------------------
    def create_lambda(
        self,
        agent_name: str,
        lambda_function_name: str,
        source_code_file: str,
        additional_function_iam_policy: Dict=None,
        lambda_env_variables: Dict=None,
        verbose: bool=False
    ) -> str:
        """
        Creates a new Lambda function for an Agent Action Group. The Lambda 
        code is packaged from the specified Python source file.

        Args:
            agent_name (str): Name of the existing agent that this Lambda supports.
            lambda_function_name (str): Name of the Lambda function to create.
            source_code_file (str): Python file containing the Lambda handler code.
            additional_function_iam_policy (Dict, optional): Additional IAM policy 
                                                             for the Lambda.
            lambda_env_variables (Dict, optional): Environment variables for the Lambda.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            str: ARN of the newly created Lambda function.
        """
        agent_id = self.get_agent_id_by_name(agent_name, verbose=verbose)
        if agent_id is None:
            if verbose:
                logger.error(f"Agent '{agent_name}' not found.")
            return "Agent not found"

        base_filename = source_code_file.split(".py")[0]

        # Zip up the Lambda code
        s = BytesIO()
        with zipfile.ZipFile(s, "w") as z:
            z.write(source_code_file)
        zip_content = s.getvalue()

        # Create the Lambda function
        lambda_function = self.lambda_client.create_function(
            FunctionName=lambda_function_name,
            Runtime='python3.12',
            Timeout=300,
            Role=self.create_lambda_iam_role(agent_name, additional_function_iam_policy, verbose),
            Code={"ZipFile": zip_content},
            Handler=f"{base_filename}.lambda_handler",
            Environment={"Variables": lambda_env_variables or {}}
        )

        if verbose:
            logger.info(f"Created Lambda function '{lambda_function_name}' for agent '{agent_name}'")
            logger.info(f"Lambda ARN: {lambda_function['FunctionArn']}")

        time.sleep(10)  # Wait for Lambda to be fully registered

        # Allow the agent to invoke this Lambda
        self.allow_agent_to_invoke_lambda(agent_id, lambda_function_name, verbose)

        return lambda_function["FunctionArn"]

    # -------------------------------------------------------------------------
    # 3.9.1: Create a Lambda IAM Role
    # -------------------------------------------------------------------------
    def create_lambda_iam_role(
        self,
        agent_name: str,
        additional_function_iam_policy: Dict = None,
        verbose: bool = False,
    ) -> str:
        """
        Creates (or retrieves if existing) an IAM role for a Lambda function 
        that supports an Agent Action Group.

        Args:
            agent_name (str): Name of the agent, used to name the role.
            additional_function_iam_policy (Dict, optional): Extra inline policy 
                                                             to attach to the role.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            str: ARN of the Lambda execution role.
        """
        lambda_function_role_name = f"{agent_name}-lambda-role"
    
        # Build the trust relationship
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        try:
            # Attempt to create the role
            lambda_iam_role = self.iam_client.create_role(
                RoleName=lambda_function_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            )
            # Sleep to allow IAM to fully propagate
            time.sleep(10)
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            # If it already exists, get it
            lambda_iam_role = self.iam_client.get_role(RoleName=lambda_function_role_name)

        # Attach the basic AWS Lambda execution policy
        self.iam_client.attach_role_policy(
            RoleName=lambda_function_role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        # Optionally attach an additional inline policy for this Lambda
        if additional_function_iam_policy is not None:
            if verbose:
                logger.info("Attaching additional IAM policy to Lambda role.")
            self.iam_client.put_role_policy(
                PolicyDocument=json.dumps(additional_function_iam_policy),
                PolicyName="additional_function_policy",
                RoleName=lambda_function_role_name,
            )
    
        return lambda_iam_role["Role"]["Arn"]

    # -------------------------------------------------------------------------
    # 3.9.2: Allow Agent to Invoke Lambda
    # -------------------------------------------------------------------------
    def allow_agent_to_invoke_lambda(self, agent_id: str, lambda_function_name: str, verbose: bool=False) -> None:
        """
        Grants a specific agent permission to invoke a particular Lambda function, 
        by adding a resource-based policy to the function.

        Args:
            agent_id (str): The ID of the Bedrock agent to allow.
            lambda_function_name (str): The name of the Lambda function to allow.
            verbose (bool, optional): If True, logs additional info.
        """
        # Build the agent ARN
        agent_arn = f"arn:aws:bedrock:{self.context.region}:{self.context.account_number}:agent/{agent_id}"
        lambda_arn = f"arn:aws:lambda:{self.context.region}:{self.context.account_number}:function:{lambda_function_name}"

        if verbose:
            logger.info(f"Allowing agent '{agent_id}' to invoke Lambda '{lambda_function_name}'")
            logger.info(f"Agent ARN: {agent_arn}")
            logger.info(f"Lambda ARN: {lambda_arn}")

        try:
            # Add permission for the agent to invoke the Lambda
            response = self.lambda_client.add_permission(
                FunctionName=lambda_function_name,
                StatementId=f"AllowBedrockAgent_{agent_id}",
                Action="lambda:InvokeFunction",
                Principal="bedrock.amazonaws.com",
                SourceArn=agent_arn
            )
            logger.info(f"Added resource-based policy to Lambda '{lambda_function_name}' for Agent '{agent_id}'.")
            logger.info(f"add_permission response: {response}")

        except self.lambda_client.exceptions.ResourceConflictException:
            # This typically means the statement is already in place
            logger.warning(f"Permission statement 'AllowBedrockAgent_{agent_id}' already exists for Lambda '{lambda_function_name}'.")
        except Exception as e:
            logger.error(f"Failed to add permission to lambda '{lambda_function_name}': {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 3.10: Add a Code Interpreter Action Group
    # -------------------------------------------------------------------------
    def add_code_interpreter(self, agent_name: str, verbose: bool=False) -> None:
        """
        Adds a Code Interpreter action group (AMAZON.CodeInterpreter) to an existing agent,
        enabling the agent to handle code interpretation tasks.

        Args:
            agent_name (str): Name of the existing agent.
            verbose (bool, optional): If True, logs additional info.
        """
        agent_id = self.get_agent_id_by_name(agent_name, verbose=verbose)
        if agent_id is None:
            if verbose:
                logger.error(f"Agent '{agent_name}' not found.")
            return

        try:
            # Wait for agent to be in valid state before modification
            if not self.wait_for_agent_state(agent_id, "AVAILABLE"):
                raise RuntimeError("Agent failed to reach AVAILABLE state")

            create_action_group_response = self.bedrock_agent_client.create_agent_action_group(
                agentId=agent_id,
                agentVersion="DRAFT",
                actionGroupName="CodeInterpreterAction",
                parentActionGroupSignature="AMAZON.CodeInterpreter",
                actionGroupState="ENABLED"
            )

            if create_action_group_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                if verbose:
                    logger.info(f"Added code interpreter to agent '{agent_name}'")
                
                # Wait for agent to be ready before preparing
                if self.wait_for_agent_state(agent_id, "AVAILABLE"):
                    self.prepare_bedrock_agent(agent_name, verbose)
                else:
                    logger.error("Timeout waiting for agent to be available")
            else:
                logger.error(f"Failed to add code interpreter. Response: {create_action_group_response}")

        except Exception as e:
            logger.error(f"Error adding code interpreter to agent '{agent_name}': {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 3.11: Invoke a Bedrock Agent (Extended Tracing)
    # -------------------------------------------------------------------------
    def invoke(
        self,
        agent_name: str,
        input_text: str,
        verbose: bool = False,
        session_id: str = str(uuid.uuid4()),
        session_state: dict = {},
        enable_trace: bool = True,
        trace_level: str = "core",
        end_session: bool = False,
        multi_agent_names: dict = None,
        save_trace_json_file: str = None  # <-- If provided, we'll save everything
    ) -> dict:
        """
        Invokes a Bedrock Agent with the given input_text. Displays color-coded output
        in the notebook at the chosen `trace_level` (e.g. 'core' or 'outline'), but also
        collects a fully verbose set of color-coded lines plus the raw JSON trace
        in the final saved file, so you can inspect everything offline.

        The final JSON contains:
        - "response": the final textual answer from the agent.
        - "session_id": the session ID used during invocation.
        - "chosen_trace_level": e.g. 'core', 'outline', or 'all'.
        - "chosen_trace_lines": color-coded lines at your chosen trace level.
        - "all_trace_lines": color-coded lines at forced 'all' level (fully verbose).
        - "all_trace_raw": a list of the raw 'trace' JSON objects (unmodified).
        """
        if multi_agent_names is None:
            multi_agent_names = {}

        # A list for the lines at the user-chosen trace level (printed on-screen).
        chosen_trace_lines = []
        # A separate list for the fully verbose "all" lines (not printed on-screen).
        all_trace_lines = []
        # A list for storing the raw JSON objects from each event.
        all_trace_raw = []

        # -------------------------------------------------------------------------
        # Helper to print + store lines at the user-chosen trace level
        # -------------------------------------------------------------------------
        def _record_chosen_line(line: str):
            print(line)  # so you see it live in the notebook
            chosen_trace_lines.append(line)

        # -------------------------------------------------------------------------
        # Helper to store lines for the forced "all" parse (no on-screen print)
        # -------------------------------------------------------------------------
        def _record_all_line(line: str):
            all_trace_lines.append(line)

        # -------------------------------------------------------------------------
        # 1) Actually invoke the agent
        # -------------------------------------------------------------------------
        try:
            agent_id = self.get_agent_id_by_name(agent_name, verbose=verbose)
            if agent_id is None:
                error_msg = f"Agent '{agent_name}' not found."
                if verbose:
                    logger.error(error_msg)
                return {"error": error_msg}

            if verbose:
                logger.info(f"Invoking agent '{agent_name}' with input: {input_text}")

            time_before_call = datetime.now()

            invocation_args = {
                "inputText": input_text,
                "agentId": agent_id,
                "agentAliasId": DEFAULT_ALIAS,
                "sessionId": session_id,
                "sessionState": session_state,
                "enableTrace": enable_trace,
                "endSession": end_session,
            }

            response = self.bedrock_agent_runtime_client.invoke_agent(**invocation_args)
            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                error_message = f"API Response was not 200: {response}"
                if enable_trace:
                    # Print an error in color
                    _record_chosen_line(colored(error_message, "red"))
                return {"error": error_message}

            event_stream = response["completion"]
            agent_answer = ""

            # Track usage for the chosen parse
            total_in_tokens = 0
            total_out_tokens = 0
            total_llm_calls = 0
            orch_step = 0
            sub_step = 0
            time_before_orchestration = datetime.now()
            sub_agent_name = "<collab-name-not-yet-provided>"

            # Track usage for the forced "all" parse
            total_in_tokens_all = 0
            total_out_tokens_all = 0
            total_llm_calls_all = 0
            orch_step_all = 0
            sub_step_all = 0
            time_before_orchestration_all = datetime.now()
            sub_agent_name_all = "<collab-name-not-yet-provided>"

            # ---------------------------------------------------------------------
            # 2) Stream events. For each trace event, we store the raw JSON, parse
            #    at the chosen level, and parse at the forced 'all' level.
            # ---------------------------------------------------------------------
            for event in event_stream:
                # 2a) If there's chunk data, accumulate in agent_answer
                if "chunk" in event:
                    chunk_bytes = event["chunk"]["bytes"]
                    agent_answer += chunk_bytes.decode("utf8")

                # 2b) If there's trace data
                if "trace" in event and enable_trace:
                    trace_obj = event["trace"]

                    # (i) Save the entire raw trace object for offline inspection
                    all_trace_raw.append(trace_obj)

                    # (ii) If user literally asked for "all" on-screen, print it
                    if trace_level == "all":
                        _record_chosen_line("---\n" + json.dumps(trace_obj, indent=2))

                    # Possibly handle sub-agent name if there's a callerChain
                    if "callerChain" in trace_obj and len(trace_obj["callerChain"]) > 1:
                        sub_agent_alias_arn = trace_obj["callerChain"][1]["agentAliasArn"]
                        sub_agent_alias_id = sub_agent_alias_arn.split("/", 1)[1]
                        if sub_agent_alias_id in multi_agent_names:
                            sub_agent_name = multi_agent_names[sub_agent_alias_id]
                        else:
                            _record_chosen_line(colored(
                                "You haven't provided sub-agent name in 'multi_agent_names'.",
                                "yellow"
                            ))
                            sub_agent_name = "<not-yet-provided>"

                    # (iii) If there's an internal 'trace' subfield, parse it
                    if "trace" in trace_obj:
                        # [1] Parse for the user-chosen trace level
                        self._process_trace_event_overridden(
                            trace_data=trace_obj["trace"],
                            trace_level=trace_level,
                            sub_agent_name=sub_agent_name,
                            multi_agent_names=multi_agent_names,
                            total_in_out_llm_usage={
                                "total_in_tokens": total_in_tokens,
                                "total_out_tokens": total_out_tokens,
                                "total_llm_calls": total_llm_calls
                            },
                            step_tracking={
                                "orch_step": orch_step,
                                "sub_step": sub_step,
                                "time_before_orchestration": time_before_orchestration
                            },
                            record_line_callback=_record_chosen_line
                        )
                        # Update chosen-level counters
                        total_in_tokens = self._last_in_tokens
                        total_out_tokens = self._last_out_tokens
                        total_llm_calls = self._last_llm_calls
                        orch_step = self._last_orch_step
                        sub_step = self._last_sub_step
                        time_before_orchestration = self._last_time_before_orchestration

                        # [2] Parse again for the forced "all" level (not printed)
                        # Possibly handle sub-agent name again
                        if "callerChain" in trace_obj and len(trace_obj["callerChain"]) > 1:
                            sub_agent_alias_arn_all = trace_obj["callerChain"][1]["agentAliasArn"]
                            sub_agent_alias_id_all = sub_agent_alias_arn_all.split("/", 1)[1]
                            if sub_agent_alias_id_all in multi_agent_names:
                                sub_agent_name_all = multi_agent_names[sub_agent_alias_id_all]
                            else:
                                sub_agent_name_all = "<not-yet-provided>"

                        self._process_trace_event_overridden(
                            trace_data=trace_obj["trace"],
                            trace_level="all",  # forced all
                            sub_agent_name=sub_agent_name_all,
                            multi_agent_names=multi_agent_names,
                            total_in_out_llm_usage={
                                "total_in_tokens": total_in_tokens_all,
                                "total_out_tokens": total_out_tokens_all,
                                "total_llm_calls": total_llm_calls_all
                            },
                            step_tracking={
                                "orch_step": orch_step_all,
                                "sub_step": sub_step_all,
                                "time_before_orchestration": time_before_orchestration_all
                            },
                            record_line_callback=_record_all_line  # no on-screen printing
                        )
                        # Update all-level counters
                        total_in_tokens_all = self._last_in_tokens
                        total_out_tokens_all = self._last_out_tokens
                        total_llm_calls_all = self._last_llm_calls
                        orch_step_all = self._last_orch_step
                        sub_step_all = self._last_sub_step
                        time_before_orchestration_all = self._last_time_before_orchestration

            # ---------------------------------------------------------------------
            # 3) Summarize usage for the on-screen chosen level
            # ---------------------------------------------------------------------
            if enable_trace:
                duration = datetime.now() - time_before_call
                # If user selected 'core' or 'outline', we do a final usage line
                if trace_level in ["core", "outline"]:
                    usage_msg = colored(
                        f"Agent made a total of {total_llm_calls} LLM calls, "
                        f"using {total_in_tokens + total_out_tokens} tokens "
                        f"(in: {total_in_tokens}, out: {total_out_tokens}), "
                        f"and took {duration.total_seconds():,.1f} total seconds.",
                        "yellow"
                    )
                    _record_chosen_line(usage_msg)
                elif trace_level == "all":
                    _record_chosen_line(f"Returning agent answer as: {agent_answer}")

            # ---------------------------------------------------------------------
            # 4) Optionally save to JSON
            # ---------------------------------------------------------------------
            if save_trace_json_file is not None:
                # Build the final structure
                output_dict = {
                    "response": agent_answer,
                    "session_id": session_id,
                    "chosen_trace_level": trace_level,
                    "chosen_trace_lines": chosen_trace_lines,
                    "all_trace_lines": all_trace_lines,
                    # This is the raw unmodified JSON for each trace event
                    # so you can see the entire text in full detail:
                    "all_trace_raw": all_trace_raw
                }
                # Write it out
                with open(save_trace_json_file, "w", encoding="utf-8") as f:
                    json.dump(output_dict, f, indent=2, ensure_ascii=False)

            # 5) Return final result
            return {
                "response": agent_answer,
                "trace": event_stream
            }

        except Exception as e:
            error_message = f"Error invoking agent '{agent_name}': {str(e)}"
            logger.error(error_message, exc_info=True)
            return {"error": error_message}




    # -------------------------------------------------------------------------
    # 3.11a: Helper to Process a Single Trace Event
    # -------------------------------------------------------------------------
    def _process_trace_event_overridden(
            self,
        trace_data: dict,
        trace_level: str,
        sub_agent_name: str,
        multi_agent_names: dict,
        total_in_out_llm_usage: dict,
        step_tracking: dict,
        record_line_callback
    ) -> None:
        """
        Processes a single trace event dictionary, but instead of printing directly,
        we funnel all colored output through 'record_line_callback'.

        This lets us capture the lines for both the on-screen (chosen) trace level 
        AND the fully verbose 'all' trace level in parallel.
        """
        self._last_in_tokens = total_in_out_llm_usage["total_in_tokens"]
        self._last_out_tokens = total_in_out_llm_usage["total_out_tokens"]
        self._last_llm_calls = total_in_out_llm_usage["total_llm_calls"]

        self._last_orch_step = step_tracking["orch_step"]
        self._last_sub_step = step_tracking["sub_step"]
        self._last_time_before_orchestration = step_tracking["time_before_orchestration"]

        # 1) routingClassifierTrace
        if 'routingClassifierTrace' in trace_data:
            route = trace_data['routingClassifierTrace']
            if 'modelInvocationInput' in route:
                self._last_orch_step += 1
                record_line_callback(colored(f"---- Step {self._last_orch_step} ----", "green"))
                record_line_callback(colored("Classifying request to route to a single collaborator if possible.", "blue"))

            if 'modelInvocationOutput' in route:
                usage = route['modelInvocationOutput']['metadata']['usage']
                in_tokens = usage['inputTokens']
                out_tokens = usage['outputTokens']
                self._last_in_tokens += in_tokens
                self._last_out_tokens += out_tokens
                self._last_llm_calls += 1

                raw_resp_str = route['modelInvocationOutput']['rawResponse']['content']
                raw_resp = json.loads(raw_resp_str)
                classification = raw_resp['content'][0]['text'].replace('<a>', '').replace('</a>', '')

                if classification == UNDECIDABLE_CLASSIFICATION:
                    record_line_callback(colored("Routing classifier did not find a matching collaborator. Using 'SUPERVISOR'.", "magenta"))
                elif classification == 'keep_previous_agent':
                    record_line_callback(colored("Continuing conversation with previous collaborator.", "magenta"))
                else:
                    record_line_callback(colored(f"Routing classifier chose collaborator: '{classification}'", "magenta"))

                record_line_callback(colored(
                    f"Routing classifier used {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens}).\n",
                    "yellow"
                ))

        # 2) failureTrace
        if 'failureTrace' in trace_data:
            fail_trace = trace_data['failureTrace']
            record_line_callback(colored(f"Agent error: {fail_trace['failureReason']}", "red"))

        # 3) orchestrationTrace
        if 'orchestrationTrace' in trace_data:
            orch = trace_data['orchestrationTrace']

            if trace_level in ["core", "outline"] and "rationale" in orch:
                rationale = orch['rationale']
                record_line_callback(colored(rationale['text'], "blue"))

            if "invocationInput" in orch:
                invocation_input = orch["invocationInput"]

                # Tools
                if 'actionGroupInvocationInput' in invocation_input:
                    fn_name = invocation_input['actionGroupInvocationInput']['function']
                    if trace_level == "outline":
                        record_line_callback(colored(f"Using tool: {fn_name}", "magenta"))
                    else:
                        record_line_callback(colored(f"Using tool: {fn_name} with inputs:", "magenta"))
                        record_line_callback(colored(
                            f"{invocation_input['actionGroupInvocationInput']['parameters']}\n",
                            "magenta"
                        ))

                # Sub-agent
                elif 'agentCollaboratorInvocationInput' in invocation_input:
                    collab_name = invocation_input['agentCollaboratorInvocationInput']['agentCollaboratorName']
                    collab_input_text = invocation_input['agentCollaboratorInvocationInput']['input']['text'][:TRACE_TRUNCATION_LENGTH]
                    collab_arn = invocation_input['agentCollaboratorInvocationInput']['agentCollaboratorAliasArn']
                    collab_ids = collab_arn.split('/', 1)[1]

                    if trace_level == "outline":
                        record_line_callback(colored(
                            f"Using sub-agent collaborator: '{collab_name} [{collab_ids}]'",
                            "magenta"
                        ))
                    else:
                        record_line_callback(colored(
                            f"Using sub-agent collaborator: '{collab_name} [{collab_ids}]' passing input text:",
                            "magenta"
                        ))
                        record_line_callback(colored(f"{collab_input_text}\n", "magenta"))

                # Code interpreter
                elif 'codeInterpreterInvocationInput' in invocation_input:
                    if trace_level == "outline":
                        record_line_callback(colored("Using code interpreter", "magenta"))
                    else:
                        record_line_callback(colored("Code interpreter usage details here.", "magenta"))

            # Observations
            if "observation" in orch and trace_level == "core":
                obs = orch["observation"]
                if 'actionGroupInvocationOutput' in obs:
                    text_out = obs['actionGroupInvocationOutput']['text'][:TRACE_TRUNCATION_LENGTH]
                    record_line_callback(colored(f"--tool outputs:\n{text_out}...\n", "magenta"))

                if 'agentCollaboratorInvocationOutput' in obs:
                    collab_name = obs['agentCollaboratorInvocationOutput']['agentCollaboratorName']
                    collab_output_text = obs['agentCollaboratorInvocationOutput']['output']['text'][:TRACE_TRUNCATION_LENGTH]
                    record_line_callback(colored(
                        f"\n----sub-agent {collab_name} output text:\n{collab_output_text}...\n",
                        "magenta"
                    ))

                if 'finalResponse' in obs:
                    final_text = obs['finalResponse']['text'][:TRACE_TRUNCATION_LENGTH]
                    record_line_callback(colored(f"Final response:\n{final_text}...", "cyan"))

            if 'modelInvocationOutput' in orch:
                usage = orch['modelInvocationOutput']['metadata']['usage']
                in_tokens = usage['inputTokens']
                out_tokens = usage['outputTokens']
                self._last_in_tokens += in_tokens
                self._last_out_tokens += out_tokens
                self._last_llm_calls += 1

                if sub_agent_name != "<collab-name-not-yet-provided>":
                    self._last_sub_step += 1
                    record_line_callback(colored(
                        f"---- Step {self._last_orch_step}.{self._last_sub_step} [using sub-agent: {sub_agent_name}] ----",
                        "green"
                    ))
                else:
                    self._last_orch_step += 1
                    self._last_sub_step = 0
                    record_line_callback(colored(f"---- Step {self._last_orch_step} ----", "green"))

                orch_duration = datetime.now() - self._last_time_before_orchestration
                record_line_callback(colored(
                    f"Took {orch_duration.total_seconds():,.1f}s, "
                    f"using {in_tokens + out_tokens} tokens (in: {in_tokens}, out: {out_tokens}) "
                    f"to complete action.",
                    "yellow"
                ))
                self._last_time_before_orchestration = datetime.now()

        # 4) preProcessingTrace
        if 'preProcessingTrace' in trace_data:
            _pre = trace_data['preProcessingTrace']
            if 'modelInvocationOutput' in _pre:
                usage = _pre['modelInvocationOutput']['metadata']['usage']
                in_tokens = usage['inputTokens']
                out_tokens = usage['outputTokens']
                self._last_in_tokens += in_tokens
                self._last_out_tokens += out_tokens
                self._last_llm_calls += 1
                record_line_callback(colored(
                    f"Pre-processing trace: agent formed an initial plan. "
                    f"Used LLM tokens in: {in_tokens}, out: {out_tokens}",
                    "yellow"
                ))

        # 5) postProcessingTrace
        if 'postProcessingTrace' in trace_data:
            _post = trace_data['postProcessingTrace']
            if 'modelInvocationOutput' in _post:
                usage = _post['modelInvocationOutput']['metadata']['usage']
                in_tokens = usage['inputTokens']
                out_tokens = usage['outputTokens']
                self._last_in_tokens += in_tokens
                self._last_out_tokens += out_tokens
                self._last_llm_calls += 1
                record_line_callback(colored(
                    f"Agent post-processing complete. "
                    f"Used LLM tokens in: {in_tokens}, out: {out_tokens}",
                    "yellow"
                ))

        # 6) Write updated usage/step counters back
        total_in_out_llm_usage["total_in_tokens"] = self._last_in_tokens
        total_in_out_llm_usage["total_out_tokens"] = self._last_out_tokens
        total_in_out_llm_usage["total_llm_calls"] = self._last_llm_calls

        step_tracking["orch_step"] = self._last_orch_step
        step_tracking["sub_step"] = self._last_sub_step
        step_tracking["time_before_orchestration"] = self._last_time_before_orchestration

    # -------------------------------------------------------------------------
    # 3.12: Get Agent Status
    # -------------------------------------------------------------------------
    def get_agent_status(self, agent_name: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Gets the current status of a Bedrock Agent.

        Args:
            agent_name (str): Name of the agent to check.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            Dict[str, Any]: Dictionary containing agent status information including:
                - agentId: The agent's ID
                - agentName: The agent's name  
                - agentStatus: Current status (e.g. PREPARED, PREPARING, FAILED)
                - createdAt: Timestamp when agent was created
                - lastUpdatedAt: Timestamp of last update
        """
        try:
            # Get agent ID from name
            agent_id = self.get_agent_id_by_name(agent_name)
            if agent_id is None:
                if verbose:
                    logger.error(f"Agent '{agent_name}' not found.")
                return {"error": "Agent not found"}

            # Get agent details
            response = self.bedrock_agent_client.get_agent(
                agentId=agent_id,
            )

            # if verbose:
            #     logger.info(f"Agent '{agent_name}' status response: {response}")

            # Extract relevant status information from AgentDetails
            status_info = {
                "agentId": response["agent"]["agentId"],
                "agentName": response["agent"]["agentName"],
                "agentStatus": response["agent"]["agentStatus"],
                "foundationModel":response["agent"]["foundationModel"],  # Mandatory
                "agentCollaboration" : response["agent"]["agentCollaboration"],  # Mandatory
                "orchestrationType": response["agent"]["orchestrationType"],  # Mandatory
                "createdAt": response["agent"].get("createdAt"),  # Optional
                "lastUpdatedAt": response["agent"].get("updatedAt")  # Optional, using updatedAt instead
            }


            if verbose:
                logger.info(f"Agent '{agent_name}' status: {status_info}")

            return status_info

        except Exception as e:
            error_message = f"Error getting status for agent '{agent_name}': {str(e)}"
            logger.error(error_message, exc_info=True)
            return {"error": error_message}
        
    # -------------------------------------------------------------------------
    # 3.13: Wait for Agent State
    # -------------------------------------------------------------------------
    def wait_for_agent_state(self, agent_id: str, target_state: str, 
                             max_attempts: int = 60, initial_delay: int = 2) -> bool:
        """
        Wait for agent to reach target state with exponential backoff.
        
        Args:
            agent_id (str): The ID of the agent to check
            target_state (str): Desired state to wait for
            max_attempts (int): Maximum number of retry attempts
            initial_delay (int): Starting delay in seconds
            
        Returns:
            bool: True if target state reached, False if timeout
        """
        attempt = 0
        delay = initial_delay
        
        while attempt < max_attempts:
            status_info = self.get_agent_status(agent_id)
            if isinstance(status_info, dict) and 'agentStatus' in status_info:
                current_state = status_info['agentStatus']
                if current_state == target_state:
                    return True
                    
                logger.info(f"Agent in {current_state} state, waiting for {target_state}... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(delay)
                delay *= 2
                attempt += 1
            else:
                logger.error(f"Failed to get agent status: {status_info}")
                return False
        
        logger.warning(f"Timeout waiting for agent state {target_state} after {max_attempts} attempts")
        return False
    
    # -------------------------------------------------------------------------
    # 3.14: Associate Sub-Agents
    # -------------------------------------------------------------------------
    def associate_sub_agents(
        self, 
        supervisor_agent_id: str, 
        sub_agents_list: List[dict]
    ) -> Tuple[str, str]:
        """
        Associates sub-agents with a supervisor agent and prepares (finalizes) 
        the supervisor agent after all associations. Finally, creates an alias 
        named "multi-agent-supervisor" for the supervisor agent.

        Args:
            supervisor_agent_id (str): The ID of the supervisor agent.
            sub_agents_list (List[dict]): Each dict must include:
                {
                    "sub_agent_alias_arn": str,
                    "sub_agent_association_name": str,
                    "sub_agent_instruction": str,
                    "relay_conversation_history": str,
                }

        Returns:
            Tuple[str, str]: (alias_id, alias_arn) for the newly created alias.

        Raises:
            ValueError: If sub_agents_list is empty.
            RuntimeError: If all sub-agent associations fail.
        """
        if not sub_agents_list:
            raise ValueError("The sub_agents_list cannot be empty.")

        successful_associations = 0

        for sub_agent in sub_agents_list:
            try:
                # 1) Wait for the supervisor agent to be in a stable state
                self.wait_agent_status_update(supervisor_agent_id)

                # 2) Associate the sub-agent
                self.bedrock_agent_client.associate_agent_collaborator(
                    agentId=supervisor_agent_id,
                    agentVersion="DRAFT",
                    agentDescriptor={"aliasArn": sub_agent["sub_agent_alias_arn"]},
                    collaboratorName=sub_agent["sub_agent_association_name"],
                    collaborationInstruction=sub_agent["sub_agent_instruction"],
                    relayConversationHistory=sub_agent["relay_conversation_history"],
                )
                successful_associations += 1

                # 3) Wait again so the agent is stable before moving on
                self.wait_agent_status_update(supervisor_agent_id)

            except Exception as e:
                print(
                    f"Failed to associate sub-agent: {sub_agent.get('sub_agent_association_name')}. "
                    f"Error: {e}"
                )

        if successful_associations == 0:
            raise RuntimeError(
                f"Supervisor agent {supervisor_agent_id} cannot be prepared "
                "as no sub-agents were successfully associated."
            )

        # 4) Now that all sub-agents are associated, finalize (prepare) the agent
        try:
            self.bedrock_agent_client.prepare_agent(agentId=supervisor_agent_id)
            self.wait_agent_status_update(supervisor_agent_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to prepare supervisor agent {supervisor_agent_id}: {e}"
            )

        # 5) Create an alias for the supervisor agent
        try:
            supervisor_agent_alias = self.bedrock_agent_client.create_agent_alias(
                agentAliasName="multi-agent-supervisor",
                agentId=supervisor_agent_id
            )
            alias_id = supervisor_agent_alias["agentAlias"]["agentAliasId"]
            alias_arn = supervisor_agent_alias["agentAlias"]["agentAliasArn"]
            return alias_id, alias_arn

        except Exception as e:
            raise RuntimeError(
                f"Failed to create alias for supervisor agent {supervisor_agent_id}: {e}"
            )

    # -------------------------------------------------------------------------
    # 3.15: Create Agent Alias
    # -------------------------------------------------------------------------
    def create_agent_alias(self, agent_name: str, alias_name: str) -> Tuple[str, str]:
        """
        Creates an agent alias. This is required to use the agent as a sub-agent for
        multi-agent collaboration.
    
        Args:
            agent_name (str): Name of the existing agent
            alias_name (str): Name of the alias to create
    
        Returns:
            Tuple[str, str]: The agent alias ID and ARN.
        """
        agent_id = self.get_agent_id_by_name(agent_name, verbose=True)
        if agent_id is None:
            raise ValueError(f"Agent with name '{agent_name}' not found.")
    
        agent_alias = self.bedrock_agent_client.create_agent_alias(
            agentAliasName=alias_name, agentId=agent_id
        )
        agent_alias_id = agent_alias["agentAlias"]["agentAliasId"]
        agent_alias_arn = agent_alias["agentAlias"]["agentAliasArn"]
        return agent_alias_id, agent_alias_arn

    # -------------------------------------------------------------------------
    # 3.16: Delete Agent Aliases
    # -------------------------------------------------------------------------
    def delete_agent_aliases(self, agent_name: str, verbose: bool=False) -> None:
        """
        Deletes all aliases for a given agent by its name.

        Args:
            agent_name (str): Name of the agent.
            verbose (bool, optional): If True, logs additional info.
        """
        agent_id = self.get_agent_id_by_name(agent_name, verbose=verbose)
        if agent_id is None:
            logger.error(f"Agent '{agent_name}' not found.")
            return

        if verbose:
            logger.info(f"Deleting all aliases for agent '{agent_name}' with ID '{agent_id}'")

        try:
            agent_aliases = self.bedrock_agent_client.list_agent_aliases(agentId=agent_id, maxResults=100)
            for alias in agent_aliases["agentAliasSummaries"]:
                alias_id = alias["agentAliasId"]
                if verbose:
                    logger.info(f"Deleting alias '{alias_id}' for agent '{agent_id}'")
                response = self.bedrock_agent_client.delete_agent_alias(
                    agentAliasId=alias_id,
                    agentId=agent_id
                )
                if verbose:
                    logger.info(f"Deleted alias '{alias_id}' for agent '{agent_id}': {response}")
        except Exception as e:
            logger.error(f"Failed to delete aliases for agent '{agent_id}': {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 3.17: Wait Agent Status Update (Helper)
    # -------------------------------------------------------------------------
    def wait_agent_status_update(self, agent_id: str, verbose: bool = False, timeout: int = 300) -> None:
        """
        Waits for the specified agent to reach a stable status (not in a transitional state).

        Args:
            agent_id (str): The ID of the agent to monitor.
            verbose (bool, optional): If True, logs status updates. Defaults to False.
            timeout (int, optional): Maximum time in seconds to wait for the status to stabilize. Defaults to 300.

        Raises:
            RuntimeError: If the agent remains in a transitional state beyond the timeout period.
        """
        start_time = time.time()
        agent_status = None

        if verbose:
            print(f"Checking status for agent {agent_id}...")

        while True:
            try:
                # Fetch agent status
                response = self.bedrock_agent_client.get_agent(agentId=agent_id)
                agent_status = response["agent"]["agentStatus"]

                # Exit if the agent is in a stable state
                if not agent_status.endswith("ING"):
                    if verbose:
                        print(f"Agent {agent_id} has reached stable status: {agent_status}")
                    break

                # Log transitional status
                if verbose:
                    print(f"Agent {agent_id} is in transitional status: {agent_status}. Waiting...")

                # Check for timeout
                if time.time() - start_time > timeout:
                    raise RuntimeError(
                        f"Agent {agent_id} did not stabilize within {timeout} seconds. "
                        f"Last known status: {agent_status}"
                    )

                # Wait before checking again
                time.sleep(5)

            except self.bedrock_agent_client.exceptions.ResourceNotFoundException:
                raise RuntimeError(f"Agent {agent_id} not found.")
            except Exception as e:
                raise RuntimeError(f"Error while monitoring agent {agent_id}: {e}")
                



# -----------------------------------------------------------------------------
# 4. AWSResourceManager Class
# -----------------------------------------------------------------------------
class AWSResourceManager:
    """
    Utility class to manage common AWS resources (S3, DynamoDB, etc.).
    Useful for testing or demonstration with a Bedrock Agent or other solutions.
    """

    def __init__(self):
        """
        Initialize AWSResourceManager with a shared AppContext, 
        and create S3, DynamoDB client/resource references.
        """
        self.context = AppContext()
        self.s3_client = self.context.client('s3')
        self.dynamodb_client = self.context.client("dynamodb")
        self.dynamodb_resource = self.context.resource("dynamodb")
        
    # -------------------------------------------------------------------------
    # 4.1: Create a DynamoDB Table
    # -------------------------------------------------------------------------
    def create_dynamodb_table(
        self, 
        dynamodb_table_name: str, 
        partition_key_name: str, 
        sort_key_name: str, 
        verbose: bool=False
    ) -> None:
        """
        Creates a DynamoDB table with specified partition (HASH) and sort (RANGE) keys.

        Args:
            dynamodb_table_name (str): Name of the DynamoDB table to create.
            partition_key_name (str): Attribute name for the partition key.
            sort_key_name (str): Attribute name for the sort key.
            verbose (bool, optional): If True, logs additional info.
        """
        if verbose:
            logger.info(f"Creating DynamoDB table '{dynamodb_table_name}' "
                        f"with partition key '{partition_key_name}' "
                        f"and sort key '{sort_key_name}'")
        try:
            dynamodb_table = self.dynamodb_resource.create_table(
                TableName=dynamodb_table_name,
                KeySchema=[
                    {"AttributeName": partition_key_name, "KeyType": "HASH"},
                    {"AttributeName": sort_key_name, "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": partition_key_name, "AttributeType": "S"},
                    {"AttributeName": sort_key_name,  "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",  # On-demand capacity
            )
            dynamodb_table.wait_until_exists()
            if verbose:
                logger.info(f"DynamoDB table '{dynamodb_table_name}' created successfully")
        except self.dynamodb_client.exceptions.ResourceInUseException:
            logger.info(f"DynamoDB table '{dynamodb_table_name}' already exists. Skipping creation.")

    # -------------------------------------------------------------------------
    # 4.2: Delete a DynamoDB Table
    # -------------------------------------------------------------------------
    def delete_dynamodb_table(self, table_name: str, verbose: bool = False) -> None:
        """
        Deletes a DynamoDB table and waits for its deletion to complete.

        Args:
            table_name (str): Name of the DynamoDB table to delete.
            verbose (bool, optional): If True, logs additional info.
        """
        if verbose:
            logger.info(f"Attempting to delete DynamoDB table '{table_name}'")

        try:
            table = self.dynamodb_resource.Table(table_name)
            table.delete()
            table.wait_until_not_exists()
            if verbose:
                logger.info(f"DynamoDB table '{table_name}' deleted successfully")
        except self.dynamodb_client.exceptions.ResourceNotFoundException:
            logger.warning(f"DynamoDB table '{table_name}' does not exist.")
        except self.dynamodb_client.exceptions.ResourceInUseException:
            logger.error(f"DynamoDB table '{table_name}' is currently in use or being deleted.")
        except Exception as e:
            logger.error(f"Error deleting DynamoDB table '{table_name}': {str(e)}")

    # -------------------------------------------------------------------------
    # 4.3: Load Items into a DynamoDB Table
    # -------------------------------------------------------------------------
    def load_dynamodb_items(self, table_name: str, items: List[Dict], verbose: bool = False) -> None:
        """
        Loads multiple items into a DynamoDB table.

        Args:
            table_name (str): Name of the target DynamoDB table.
            items (List[Dict]): List of item dictionaries to load.
            verbose (bool, optional): If True, logs additional info.
        """
        if verbose:
            logger.info(f"Loading {len(items)} items into DynamoDB table '{table_name}'")

        try:
            table = self.dynamodb_resource.Table(table_name)
            for item in items:
                table.put_item(Item=item)
            if verbose:
                logger.info(f"Successfully loaded items into DynamoDB table '{table_name}'")
        except self.dynamodb_client.exceptions.ResourceNotFoundException:
            logger.error(f"DynamoDB table '{table_name}' does not exist.")
        except Exception as e:
            logger.error(f"Error loading items into DynamoDB table '{table_name}': {str(e)}")

    # -------------------------------------------------------------------------
    # 4.4: Query Items from a DynamoDB Table
    # -------------------------------------------------------------------------
    def query_dynamodb_items(
        self,
        table_name: str, 
        partition_key_name: str,
        partition_key_value: str,
        sort_key_name: str = None,
        sort_key_value: str = None,
        verbose: bool = False
    ) -> List[Dict]:
        """
        Queries items from a DynamoDB table based on partition key and optional 
        sort key prefix.

        Args:
            table_name (str): Name of the DynamoDB table.
            partition_key_name (str): Partition key field name.
            partition_key_value (str): Partition key value for the query.
            sort_key_name (str, optional): Sort key field name.
            sort_key_value (str, optional): If provided, a "begins_with" condition is used.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            List[Dict]: A list of matching items.
        """
        if verbose:
            logger.info(f"Querying DynamoDB table '{table_name}'")

        try:
            table = self.dynamodb_resource.Table(table_name)
            
            # Construct a KeyConditionExpression
            if sort_key_name and sort_key_value:
                key_condition = Key(partition_key_name).eq(partition_key_value) & Key(sort_key_name).begins_with(sort_key_value)
            else:
                key_condition = Key(partition_key_name).eq(partition_key_value)
    
            response = table.query(KeyConditionExpression=key_condition)
            items = response.get('Items', [])

            if verbose:
                logger.info(f"Retrieved {len(items)} items from DynamoDB table '{table_name}'")
            return items

        except self.dynamodb_client.exceptions.ResourceNotFoundException:
            logger.error(f"DynamoDB table '{table_name}' does not exist.")
            return []
        except Exception as e:
            logger.error(f"Error querying DynamoDB table '{table_name}': {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # 4.5: Load JSON Data from File
    # -------------------------------------------------------------------------
    def load_json_data(self, file_path: str, verbose: bool = False) -> List[Dict]:
        """
        Loads JSON data from a file, handling both single and multi-line JSON.

        Args:
            file_path (str): Path to the JSON file.
            verbose (bool, optional): If True, logs additional info.

        Returns:
            List[Dict]: A list of dictionaries containing the JSON data.
        """
        if verbose:
            logger.info(f"Loading JSON data from {file_path}")

        try:
            with open(file_path) as f:
                content = f.read()
                try:
                    # Attempt to parse as a single JSON object or array
                    data = json.loads(content)
                    return [data] if isinstance(data, dict) else data
                except json.JSONDecodeError:
                    # Fallback: parse line-by-line for NDJSON
                    data = []
                    for line in content.splitlines():
                        if line.strip():
                            data.append(json.loads(line))
                    return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            return []
