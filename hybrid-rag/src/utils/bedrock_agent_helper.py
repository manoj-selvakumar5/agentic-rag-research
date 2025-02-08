# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission

"""
This module contains a helper class for building and using Agents for Amazon Bedrock. 
The AgentsForAmazonBedrock class provides a convenient interface for working with Agents.
It includes methods for creating, updating, and invoking Agents, as well as managing 
IAM roles and Lambda functions for action groups.
"""

import boto3
import json
import time
import uuid
import zipfile
from dateutil.tz import tzutc
import os
import datetime
from io import BytesIO
from typing import List, Dict, Tuple
import re
from boto3.session import Session
from botocore.config import Config
from boto3.dynamodb.conditions import Key

# If you want to display images or charts, un-comment these:
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from IPython.display import display, Markdown

from termcolor import colored
from rich.console import Console
from rich.markdown import Markdown
import base64  # needed to base64-encode the chunk/file bytes for JSON trace

PYTHON_TIMEOUT = 180
PYTHON_RUNTIME = "python3.12"
DEFAULT_ALIAS = "TSTALIASID"
DEFAULT_CI_ACTION_GROUP_NAME = "CodeInterpreterAction"
UNDECIDABLE_CLASSIFICATION = "undecidable"
ROUTER_MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"
TRACE_TRUNCATION_LENGTH = 300

# TODO: Take advantage of a default execution role so that we do not need to have lengthy
# waiting times when creating a new Agent or new Lambda to give time for the IAM role to
# take effect. When this is supported, need to change the default "delete_role_flag" to False
# to avoid deleting the default. And add logic to create the default role if it is not found.
# That way it should only need to be created once.
DEFAULT_AGENT_IAM_ROLE_NAME = "DEFAULT_AgentExecutionRole"
DEFAULT_AGENT_IAM_ASSUME_ROLE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowBedrock",
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}

DEFAULT_AGENT_IAM_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AmazonBedrockAgentInferencProfilePolicy1",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel*", "bedrock:CreateInferenceProfile"],
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
                "bedrock:ListTagsForResource",
            ],
            "Resource": [
                "arn:aws:bedrock:*:*:inference-profile/*",
                "arn:aws:bedrock:*:*:application-inference-profile/*",
            ],
        },
        {
            "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicy",
            "Effect": "Allow",
            "Action": ["bedrock:GetAgentAlias", "bedrock:InvokeAgent"],
            "Resource": [
                "arn:aws:bedrock:*:*:agent/*",
                "arn:aws:bedrock:*:*:agent-alias/*",
            ],
        },
        {
            "Sid": "AmazonBedrockAgentBedrockInvokeGuardrailModelPolicy",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:GetGuardrail",
                "bedrock:ApplyGuardrail",
            ],
            "Resource": "arn:aws:bedrock:*:*:guardrail/*",
        },
        {
            "Sid": "QueryKB",
            "Effect": "Allow",
            "Action": ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
            "Resource": "arn:aws:bedrock:*:*:knowledge-base/*",
        },
    ],
}


class AgentsForAmazonBedrock:
    """Provides an easy to use wrapper for Agents for Amazon Bedrock."""

    def __init__(self):
        """Constructs an instance."""
        self._boto_session = Session()
        self._region = self._boto_session.region_name
        self._account_id = boto3.client("sts").get_caller_identity()["Account"]

        self._bedrock_agent_client = boto3.client("bedrock-agent")

        long_invoke_time_config = Config(read_timeout=600)
        self._bedrock_agent_runtime_client = boto3.client(
            "bedrock-agent-runtime", config=long_invoke_time_config
        )

        self._sts_client = boto3.client("sts")
        self._iam_client = boto3.client("iam")
        self._lambda_client = boto3.client("lambda")
        self._s3_client = boto3.client("s3", region_name=self._region)
        self._dynamodb_client = boto3.client("dynamodb", region_name=self._region)
        self._dynamodb_resource = boto3.resource("dynamodb", region_name=self._region)

        self._suffix = f"{self._region}-{self._account_id}"

    def get_region(self) -> str:
        """Returns the region for this instance."""
        return self._region

    def _create_lambda_iam_role(
        self,
        agent_name: str,
        additional_function_iam_policy: Dict = None,
        sub_agent_arns: List[str] = None,
        dynamodb_table_name: str = None,
        enable_trace: bool = False,
    ) -> object:
        """Creates an IAM role for a Lambda function built to implement an Action Group for an Agent."""
        _lambda_function_role_name = f"{agent_name}-lambda-role-{self._suffix}"
        _dynamodb_access_policy_name = f"{agent_name}-dynamodb-policy"

        try:
            _assume_role_policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }
            _assume_role_policy_document_json = json.dumps(_assume_role_policy_document)

            _lambda_iam_role = self._iam_client.create_role(
                RoleName=_lambda_function_role_name,
                AssumeRolePolicyDocument=_assume_role_policy_document_json,
            )
            # Pause to make sure role is created
            time.sleep(10)
        except:
            _lambda_iam_role = self._iam_client.get_role(
                RoleName=_lambda_function_role_name
            )

        # attach Lambda basic execution policy
        self._iam_client.attach_role_policy(
            RoleName=_lambda_function_role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        # If extra IAM policy is provided:
        if additional_function_iam_policy is not None:
            if enable_trace:
                print(
                    f"Attaching additional IAM policy to Lambda role:\n{additional_function_iam_policy}"
                )
            self._iam_client.put_role_policy(
                PolicyDocument=additional_function_iam_policy,
                PolicyName="additional_function_policy",
                RoleName=_lambda_function_role_name,
            )

        # Policy to allow sub-agent invocation
        if sub_agent_arns is not None:
            _tmp_resources = [
                _sub_agent_arn.replace(":agent/", ":agent*/") + "*"
                for _sub_agent_arn in sub_agent_arns
            ]
            _sub_agent_policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AmazonBedrockAgentInvokeSubAgentPolicy",
                        "Effect": "Allow",
                        "Action": ["bedrock:InvokeAgent", "bedrock:GetAgentAlias"],
                        "Resource": _tmp_resources,
                    },
                    {
                        "Sid": "AmazonBedrockAgentGetAgentPolicy",
                        "Effect": "Allow",
                        "Action": "bedrock:GetAgent",
                        "Resource": [_sub_agent_arn for _sub_agent_arn in sub_agent_arns],
                    },
                ],
            }
            sub_agent_policy_json = json.dumps(_sub_agent_policy_document)
            self._iam_client.put_role_policy(
                PolicyDocument=sub_agent_policy_json,
                PolicyName="sub_agent_policy",
                RoleName=_lambda_function_role_name,
            )

        # DynamoDB policy if needed
        if dynamodb_table_name:
            _dynamodb_access_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:GetItem",
                            "dynamodb:PutItem",
                            "dynamodb:DeleteItem",
                            "dynamodb:Query",
                            "dynamodb:UpdateItem",
                        ],
                        "Resource": f"arn:aws:dynamodb:{self._region}:{self._account_id}:table/{dynamodb_table_name}",
                    }
                ],
            }
            _dynamodb_access_policy_json = json.dumps(_dynamodb_access_policy)
            self._iam_client.put_role_policy(
                PolicyDocument=_dynamodb_access_policy_json,
                PolicyName=_dynamodb_access_policy_name,
                RoleName=_lambda_function_role_name,
            )

        return _lambda_iam_role["Role"]["Arn"]

    def get_agent_latest_alias_id(self, agent_id: str, verbose: bool = False) -> str:
        """Gets the latest alias ID for the specified Agent."""
        _agent_aliases = self._bedrock_agent_client.list_agent_aliases(
            agentId=agent_id, maxResults=100
        )

        _latest_alias_id = ""
        _latest_update = datetime.datetime(1970, 1, 1, 0, 0, 0, tzinfo=tzutc())

        for _summary in _agent_aliases["agentAliasSummaries"]:
            _curr_update = _summary["updatedAt"]
            if _curr_update > _latest_update:
                _latest_alias_id = _summary["agentAliasId"]
                self.wait_agent_alias_status_update(agent_id, _latest_alias_id, False)
                _latest_update = _curr_update
                _alias_name = _summary["agentAliasName"]

        if verbose:
            print(f"for id: {agent_id}, picked latest alias: {_latest_alias_id}")
            print(f"  updated at: {_latest_update}")
            print(f"  alias name: {_alias_name}\n")

        return _latest_alias_id

    def get_agent_alias_arn(
        self, agent_id: str, agent_alias_id: str, verbose: bool = False
    ) -> str:
        """Gets the ARN of the specified Agent Alias."""
        _agent_alias = self._bedrock_agent_client.get_agent_alias(
            agentId=agent_id, agentAliasId=agent_alias_id
        )
        _alias_arn = _agent_alias["agentAlias"]["agentAliasArn"]
        return _alias_arn

    def get_agent_id_by_name(self, agent_name: str) -> str:
        """Gets the Agent ID for the specified Agent name."""
        _get_agents_resp = self._bedrock_agent_client.list_agents(maxResults=100)
        _agents_json = _get_agents_resp["agentSummaries"]
        _target_agent = next(
            (agent for agent in _agents_json if agent["agentName"] == agent_name), None
        )
        if _target_agent is None:
            return None
        else:
            return _target_agent["agentId"]

    def associate_kb_with_agent(self, agent_id, description, kb_id):
        """Associates a Knowledge Base with an Agent, and prepares the agent."""
        _resp = self._bedrock_agent_client.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion="DRAFT",
            description=description,
            knowledgeBaseId=kb_id,
            knowledgeBaseState="ENABLED",
        )
        _resp = self._bedrock_agent_client.prepare_agent(agentId=agent_id)

    def get_agent_arn_by_name(self, agent_name: str) -> str:
        """Gets the Agent ARN for the specified Agent name."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            raise ValueError(f"Agent {agent_name} not found")
        _get_agent_resp = self._bedrock_agent_client.get_agent(agentId=_agent_id)
        return _get_agent_resp["agent"]["agentArn"]

    def get_agent_instructions_by_name(self, agent_name: str) -> str:
        """Gets the current Agent Instructions for the specified Agent."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            raise ValueError(f"Agent {agent_name} not found")
        _get_agent_resp = self._bedrock_agent_client.get_agent(agentId=_agent_id)
        return _get_agent_resp["agent"]["instruction"]

    def _allow_agent_lambda(self, agent_id: str, lambda_function_name: str) -> None:
        """Allows the specified Agent to invoke the specified Lambda function."""
        _permission_resp = self._lambda_client.add_permission(
            FunctionName=lambda_function_name,
            StatementId=f"allow_bedrock_{agent_id}",
            Action="lambda:InvokeFunction",
            Principal="bedrock.amazonaws.com",
            SourceArn=f"arn:aws:bedrock:{self._region}:{self._account_id}:agent/{agent_id}",
        )

    def _make_agent_string(self, agent_arns: List[str] = None) -> str:
        """Makes a comma-separated string of agent IDs from a list of agent ARNs."""
        if agent_arns is None:
            return ""
        else:
            _agent_string = ""
            for _agent_arn in agent_arns:
                _agent_string += _agent_arn.split("/")[1] + ","
            return _agent_string.strip()[:-1]

    def create_lambda(
        self,
        agent_name: str,
        lambda_function_name: str,
        source_code_file: str,
        additional_function_iam_policy: Dict = None,
        sub_agent_arns: List[str] = None,
        dynamo_args: List[str] = None,
    ) -> str:
        """Creates a new Lambda function that implements a set of actions for an Agent Action Group."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            return "Agent not found"

        _base_filename = source_code_file.split(".py")[0]

        # Package the lambda code
        s = BytesIO()
        z = zipfile.ZipFile(s, "w")
        z.write(f"{source_code_file}")
        z.close()
        zip_content = s.getvalue()

        # Environment variables if sub-agents or Dynamo
        if sub_agent_arns:
            env_variables = {
                "Variables": {"SUB_AGENT_IDS": self._make_agent_string(sub_agent_arns)}
            }
        else:
            env_variables = {"Variables": {}}

        if dynamo_args:
            lambda_role = self._create_lambda_iam_role(
                agent_name, sub_agent_arns, dynamodb_table_name=dynamo_args[0]
            )
            self.create_dynamodb(dynamo_args[0], dynamo_args[1], dynamo_args[2])
            env_variables["Variables"]["dynamodb_table"] = dynamo_args[0]
            env_variables["Variables"]["dynamodb_pk"] = dynamo_args[1]
            env_variables["Variables"]["dynamodb_sk"] = dynamo_args[2]
        else:
            lambda_role = self._create_lambda_iam_role(agent_name, sub_agent_arns)

        _lambda_function = self._lambda_client.create_function(
            FunctionName=lambda_function_name,
            Runtime=PYTHON_RUNTIME,
            Timeout=PYTHON_TIMEOUT,
            Role=lambda_role,
            Code={"ZipFile": zip_content},
            Handler=f"{_base_filename}.lambda_handler",
            Environment=env_variables,
        )

        self._allow_agent_lambda(_agent_id, lambda_function_name)

        return _lambda_function["FunctionArn"]

    def delete_lambda(
        self, lambda_function_name: str, delete_role_flag: bool = True
    ) -> None:
        """Deletes the specified Lambda function, optionally its IAM role."""
        if delete_role_flag:
            try:
                _function_resp = self._lambda_client.get_function(
                    FunctionName=lambda_function_name
                )
                _role_arn = _function_resp["Configuration"]["Role"]
                _role_name = _role_arn.split("/")[1]
                self._iam_client.detach_role_policy(
                    RoleName=_role_name,
                    PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                )
                self._iam_client.delete_role(RoleName=_role_name)
            except:
                pass

        try:
            self._lambda_client.delete_function(FunctionName=lambda_function_name)
        except:
            pass

    def get_agent_role(self, agent_name: str) -> str:
        """Gets the ARN of the IAM role associated with the specified Agent."""
        _get_agents_resp = self._bedrock_agent_client.list_agents(maxResults=100)
        _agents_json = _get_agents_resp["agentSummaries"]
        _target_agent = next(
            (agent for agent in _agents_json if agent["agentName"] == agent_name), None
        )
        if _target_agent is not None:
            _agent_id = _target_agent["agentId"]
            _get_agent_resp = self._bedrock_agent_client.get_agent(agentId=_agent_id)
            return _get_agent_resp["agent"]["agentResourceRoleArn"]
        else:
            return "Agent not found"

    def delete_agent(
        self, agent_name: str, delete_role_flag: bool = True, verbose: bool = False
    ) -> None:
        """Deletes an existing agent, optionally its IAM role."""
        _get_agents_resp = self._bedrock_agent_client.list_agents(maxResults=100)
        _agents_json = _get_agents_resp["agentSummaries"]
        _target_agent = next(
            (agent for agent in _agents_json if agent["agentName"] == agent_name), None
        )

        if _target_agent is None:
            print(f"Agent {agent_name} not found")
            return

        if _target_agent is not None and verbose:
            print(
                f"Found target agent, name: {agent_name}, id: {_target_agent['agentId']}"
            )

        # Delete aliases
        if _target_agent is not None:
            _agent_id = _target_agent["agentId"]
            if verbose:
                print(f"Deleting aliases for agent {_agent_id}...")
            try:
                _agent_aliases = self._bedrock_agent_client.list_agent_aliases(
                    agentId=_agent_id, maxResults=100
                )
                for alias in _agent_aliases["agentAliasSummaries"]:
                    alias_id = alias["agentAliasId"]
                    print(f"Deleting alias {alias_id} from agent {_agent_id}")
                    response = self._bedrock_agent_client.delete_agent_alias(
                        agentAliasId=alias_id, agentId=_agent_id
                    )
            except Exception as e:
                print(f"Error deleting aliases: {e}")
                pass

        # Delete the agent itself
        if _target_agent is not None:
            _agent_id = _target_agent["agentId"]
            if verbose:
                print(f"Deleting agent: {_agent_id}...")
            time.sleep(5)
            self._bedrock_agent_client.delete_agent(agentId=_agent_id)
            time.sleep(5)

        # Delete Agent IAM role if desired
        if delete_role_flag:
            _agent_role_name = f"AmazonBedrockExecutionRoleForAgents_{agent_name}"
            if verbose:
                print(f"Deleting IAM role: {_agent_role_name}...")

            try:
                self._iam_client.delete_role_policy(
                    PolicyName="bedrock_gr_allow_policy", RoleName=_agent_role_name
                )
            except:
                pass
            try:
                self._iam_client.delete_role_policy(
                    PolicyName="bedrock_allow_policy", RoleName=_agent_role_name
                )
            except:
                pass
            try:
                self._iam_client.delete_role_policy(
                    PolicyName="bedrock_kb_allow_policy", RoleName=_agent_role_name
                )
            except:
                pass
            try:
                self._iam_client.delete_role(RoleName=_agent_role_name)
            except:
                pass

    def _create_agent_role(
        self,
        agent_name: str,
        agent_foundation_models: List[str],
        kb_arns: List[str] = None,
        reuse_default: bool = True,
        verbose: bool = True,
    ) -> str:
        """Creates an IAM role for an agent."""
        if verbose:
            print(f"Creating IAM role for agent: {agent_name}")

        if reuse_default:
            _agent_role_name = DEFAULT_AGENT_IAM_ROLE_NAME
            try:
                _assume_role_policy_document_json = DEFAULT_AGENT_IAM_ASSUME_ROLE_POLICY
                _assume_role_policy_document = json.dumps(
                    _assume_role_policy_document_json
                )

                _bedrock_agent_bedrock_allow_policy_document_json = (
                    DEFAULT_AGENT_IAM_POLICY
                )
                _bedrock_agent_bedrock_allow_policy_document = json.dumps(
                    _bedrock_agent_bedrock_allow_policy_document_json
                )

                _agent_role = self._iam_client.create_role(
                    RoleName=_agent_role_name,
                    AssumeRolePolicyDocument=_assume_role_policy_document,
                )
            except Exception as e:
                if verbose:
                    print(
                        f"Caught exc when creating default role for role: {_agent_role_name}: {e}"
                    )
                    print(f"using assume role json: {_assume_role_policy_document}")
            else:
                self._iam_client.put_role_policy(
                    PolicyDocument=_bedrock_agent_bedrock_allow_policy_document,
                    PolicyName="bedrock_allow_policy",
                    RoleName=_agent_role_name,
                )

            return f"arn:aws:iam::{self._account_id}:role/{DEFAULT_AGENT_IAM_ROLE_NAME}"

        else:
            _agent_role_name = f"AmazonBedrockExecutionRoleForAgents_{agent_name}"
            _assume_role_policy_document = DEFAULT_AGENT_IAM_ASSUME_ROLE_POLICY
            _assume_role_policy_document_json = json.dumps(_assume_role_policy_document)

            _agent_role = self._iam_client.create_role(
                RoleName=_agent_role_name,
                AssumeRolePolicyDocument=_assume_role_policy_document_json,
            )

            time.sleep(10)

            if verbose:
                print(
                    f"Role {_agent_role_name} created. ARN: {_agent_role['Role']['Arn']}"
                )

            _bedrock_agent_bedrock_allow_policy_statement = DEFAULT_AGENT_IAM_POLICY
            _bedrock_policy_json = json.dumps(
                _bedrock_agent_bedrock_allow_policy_statement
            )

            if verbose:
                print(
                    f"Adding bedrock_allow_policy to role {_agent_role_name}\n{_bedrock_policy_json}..."
                )

            self._iam_client.put_role_policy(
                PolicyDocument=_bedrock_policy_json,
                PolicyName="bedrock_allow_policy",
                RoleName=_agent_role_name,
            )

            # If KB ARNs exist, add them
            if kb_arns is not None:
                _kb_policy_doc = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "QueryKB",
                            "Effect": "Allow",
                            "Action": ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                            "Resource": kb_arns,
                        }
                    ],
                }
                _kb_policy_json = json.dumps(_kb_policy_doc)
                self._iam_client.put_role_policy(
                    PolicyDocument=_kb_policy_json,
                    PolicyName="bedrock_kb_allow_policy",
                    RoleName=_agent_role_name,
                )
                time.sleep(10)

            return _agent_role["Role"]["Arn"]

    def wait_agent_status_update(self, agent_id):
        """Waits for the agent to finish 'ING' states."""
        response = self._bedrock_agent_client.get_agent(agentId=agent_id)
        agent_status = response["agent"]["agentStatus"]
        _waited_at_least_once = False
        while agent_status.endswith("ING"):
            print(f"Waiting for agent status to change. Current status {agent_status}")
            time.sleep(5)
            _waited_at_least_once = True
            try:
                response = self._bedrock_agent_client.get_agent(agentId=agent_id)
                agent_status = response["agent"]["agentStatus"]
            except self._bedrock_agent_client.exceptions.ResourceNotFoundException:
                agent_status = "DELETED"
        if _waited_at_least_once:
            print(f"Agent id {agent_id} current status: {agent_status}")

    def wait_agent_alias_status_update(self, agent_id, agent_alias_id, verbose=False):
        """Waits for the agent alias to finish 'ING' states."""
        response = self._bedrock_agent_client.get_agent_alias(
            agentId=agent_id, agentAliasId=agent_alias_id
        )
        agent_alias_status = response["agentAlias"]["agentAliasStatus"]
        while agent_alias_status.endswith("ING"):
            if verbose:
                print(
                    f"Waiting for agent ALIAS status to change. Current status {agent_alias_status}"
                )
            time.sleep(5)
            try:
                response = self._bedrock_agent_client.get_agent_alias(
                    agentId=agent_id, agentAliasId=agent_alias_id
                )
                agent_alias_status = response["agentAlias"]["agentAliasStatus"]
            except self._bedrock_agent_client.exceptions.ResourceNotFoundException:
                agent_alias_status = "DELETED"
        if verbose:
            print(
                f"Agent id {agent_id}, Alias {agent_alias_id} current status: {agent_alias_status}"
            )

    def associate_sub_agents(self, supervisor_agent_id, sub_agents_list):
        """Associates sub-agents with a supervisor agent."""
        for sub_agent in sub_agents_list:
            self.wait_agent_status_update(supervisor_agent_id)
            self._bedrock_agent_client.associate_agent_collaborator(
                agentId=supervisor_agent_id,
                agentVersion="DRAFT",
                agentDescriptor={"aliasArn": sub_agent["sub_agent_alias_arn"]},
                collaboratorName=sub_agent["sub_agent_association_name"],
                collaborationInstruction=sub_agent["sub_agent_instruction"],
                relay_conversation_history=sub_agent["relay_conversation_history"],
            )
            self.wait_agent_status_update(supervisor_agent_id)
            self._bedrock_agent_client.prepare_agent(agentId=supervisor_agent_id)
            self.wait_agent_status_update(supervisor_agent_id)

        supervisor_agent_alias = self._bedrock_agent_client.create_agent_alias(
            agentAliasName="multi-agent", agentId=supervisor_agent_id
        )
        supervisor_agent_alias_id = supervisor_agent_alias["agentAlias"]["agentAliasId"]
        supervisor_agent_alias_arn = supervisor_agent_alias["agentAlias"][
            "agentAliasArn"
        ]
        return supervisor_agent_alias_id, supervisor_agent_alias_arn

    def build_sub_agent_list(self, sub_agent_names: List[str]) -> List:
        """Builds the sub-agent list data structure for multi-agent association."""
        _sub_agent_list = []
        for _agent_name in sub_agent_names:
            _agent_id = self.get_agent_id_by_name(_agent_name)
            _agent_details = self._bedrock_agent_client.get_agent(agentId=_agent_id)[
                "agent"
            ]
            _sub_agent_list.append(
                {
                    "sub_agent_alias_arn": _agent_details["agentArn"],
                    "sub_agent_instruction": _agent_details["instruction"],
                    "sub_agent_association_name": _agent_details["agentName"],
                    "relay_conversation_history": "DISABLED",
                }
            )
        return _sub_agent_list

    def create_agent(
        self,
        agent_name: str,
        agent_description: str,
        agent_instructions: str,
        model_ids: List[str],
        kb_arns: List[str] = None,
        agent_collaboration: str = "DISABLED",
        routing_classifier_model: str = None,
        code_interpretation: bool = False,
        guardrail_id: str = None,
        kb_id: str = None,
        verbose: bool = False,
    ) -> Tuple[str, str, str]:
        """Creates an agent given a name, instructions, model, description, etc."""
        if verbose:
            print(f"Creating agent: {agent_name}...")

        _role_arn = self._create_agent_role(
            agent_name, model_ids, kb_arns, reuse_default=True, verbose=False
        )
        _model_id = model_ids[0]

        if verbose:
            print(f"Created agent IAM role: {_role_arn}...")
            print(f"Creating agent: {agent_name} with model: {_model_id}...")

        _num_tries = 0
        _agent_created = False
        _create_agent_response = None
        _agent_id = None

        _kwargs = {}
        if routing_classifier_model is not None:
            _kwargs["promptOverrideConfiguration"] = {
                "promptConfigurations": [
                    {
                        "promptType": "ROUTING_CLASSIFIER",
                        "promptCreationMode": "DEFAULT",
                        "foundationModel": routing_classifier_model,
                        "parserMode": "DEFAULT",
                        "promptState": "ENABLED",
                    }
                ]
            }
        if guardrail_id is not None:
            _kwargs["guardrailConfiguration"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
            }

        while not _agent_created and _num_tries <= 2:
            try:
                if verbose:
                    print(f"kwargs: {_kwargs}")
                _create_agent_response = self._bedrock_agent_client.create_agent(
                    agentName=agent_name,
                    agentResourceRoleArn=_role_arn,
                    description=agent_description.replace("\n", ""),
                    idleSessionTTLInSeconds=1800,
                    foundationModel=_model_id,
                    instruction=agent_instructions,
                    agentCollaboration=agent_collaboration,
                    **_kwargs,
                )
                _agent_id = _create_agent_response["agent"]["agentId"]
                if verbose:
                    print(f"Created agent, resulting id: {_agent_id}")
                _agent_created = True
            except Exception as e:
                if verbose:
                    print(
                        f"Error creating agent: {e}\n. Retrying in case it was just waiting to be deleted."
                    )
                _num_tries += 1
                if _num_tries <= 2:
                    time.sleep(4)
                else:
                    if verbose:
                        print(f"Giving up on agent creation after 2 tries.")
                    raise e

        if code_interpretation:
            self.add_code_interpreter(agent_name)

        _agent_alias_id = DEFAULT_ALIAS
        _agent_alias_arn = (
            _create_agent_response["agent"]["agentArn"].replace("agent", "agent-alias")
            + f"/{_agent_alias_id}"
        )
        return _agent_id, _agent_alias_id, _agent_alias_arn

    def prepare(self, agent_name: str) -> None:
        """Prepares an agent for invocation."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            return "Agent not found"

        _resp = self._bedrock_agent_client.prepare_agent(agentId=_agent_id)
        time.sleep(5)
        return

    def create_agent_alias(self, agent_id: str, alias_name: str) -> Tuple[str, str]:
        """Creates an agent alias (for multi-agent usage, etc)."""
        agent_alias = self._bedrock_agent_client.create_agent_alias(
            agentAliasName=alias_name, agentId=agent_id
        )
        agent_alias_id = agent_alias["agentAlias"]["agentAliasId"]
        agent_alias_arn = agent_alias["agentAlias"]["agentAliasArn"]
        return agent_alias_id, agent_alias_arn

    def add_code_interpreter(self, agent_name: str) -> None:
        """Adds a code interpreter action group to an existing agent."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            return "Agent not found"

        self.wait_agent_status_update(_agent_id)
        _agent_action_group_resp = self._bedrock_agent_client.create_agent_action_group(
            agentId=_agent_id,
            agentVersion="DRAFT",
            actionGroupName=DEFAULT_CI_ACTION_GROUP_NAME,
            parentActionGroupSignature="AMAZON.CodeInterpreter",
            actionGroupState="ENABLED",
        )
        if _agent_action_group_resp["ResponseMetadata"]["HTTPStatusCode"] == 200:
            self._bedrock_agent_client.prepare_agent(agentId=_agent_id)
            time.sleep(5)
        else:
            print(f"Error adding code interpreter: {_agent_action_group_resp}")

    def add_action_group_with_lambda(
        self,
        agent_name: str,
        lambda_function_name: str,
        source_code_file: str,
        agent_functions: List[Dict],
        agent_action_group_name: str,
        agent_action_group_description: str,
        additional_function_iam_policy: Dict = None,
        sub_agent_arns: List[str] = None,
        dynamo_args: List[str] = None,
        verbose: bool = False,
    ) -> None:
        """Adds an action group that uses a Lambda function to an existing agent."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            return "Agent not found"

        if "arn:" in source_code_file:
            _lambda_arn = source_code_file
        else:
            _lambda_arn = self.create_lambda(
                agent_name,
                lambda_function_name,
                source_code_file,
                additional_function_iam_policy=additional_function_iam_policy,
                sub_agent_arns=sub_agent_arns,
                dynamo_args=dynamo_args,
            )

        self.wait_agent_status_update(_agent_id)

        if verbose:
            print(f"Creating action group: {agent_action_group_name}...")
            print(f"Lambda ARN: {_lambda_arn}")
            print(f"Agent functions: {agent_functions}")

        self._bedrock_agent_client.create_agent_action_group(
            agentId=_agent_id,
            agentVersion="DRAFT",
            actionGroupExecutor={"lambda": _lambda_arn},
            actionGroupName=agent_action_group_name,
            functionSchema={"functions": agent_functions},
            description=agent_action_group_description,
        )

    def add_action_group_with_roc(
        self,
        agent_id: str,
        agent_functions: List[Dict],
        agent_action_group_name: str,
        agent_action_group_description: str = None,
    ) -> None:
        """Adds a Return-of-Control (ROC) action group to an existing agent."""
        self._bedrock_agent_client.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupExecutor={"customControl": "RETURN_CONTROL"},
            actionGroupName=agent_action_group_name,
            functionSchema={"functions": agent_functions},
            description=agent_action_group_description,
        )
        self._bedrock_agent_client.prepare_agent(agentId=agent_id)
        time.sleep(5)

    def get_function_defs(self, agent_name: str) -> List[dict]:
        """Returns the function definitions for an agent's DRAFT version action group."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        if _agent_id is None:
            raise ValueError(f"Agent {agent_name} not found")
        _list_resp = self._bedrock_agent_client.list_agent_action_groups(
            agentId=_agent_id, agentVersion="DRAFT"
        )
        _action_group_id = _list_resp["actionGroupSummaries"][0]["actionGroupId"]
        _get_ag_resp = self._bedrock_agent_client.get_agent_action_group(
            agentId=_agent_id, actionGroupId=_action_group_id, agentVersion="DRAFT"
        )
        return _get_ag_resp["agentActionGroup"]["functionSchema"]

    def create_supervisor_agent(
        self,
        supervisor_agent_name: str,
        sub_agent_names: List[str],
        model_ids: List[str],
        kb_arn: str = None,
        kb_descr: str = None,
    ) -> tuple[List[dict], str]:
        """(Optional) Creates a supervisor agent that delegates tasks to sub-agents."""
        # Omitted for brevity or left as is. Not central to single-pass event usage.
        pass

    def _make_fully_cited_answer(
        self, orig_agent_answer, event, enable_trace=False, trace_level="none"
    ):
        """Adds knowledge-base citations to the chunk text if present."""
        _citations = event.get("chunk", {}).get("attribution", {}).get("citations", [])
        if not _citations:
            return orig_agent_answer

        # Remove <sources> placeholders
        _pattern = r"\n\n<sources>\n\d+\n</sources>\n\n"
        _cleaned_text = re.sub(_pattern, "", orig_agent_answer)
        _pattern = "<sources><REDACTED></sources>"
        _cleaned_text = re.sub(_pattern, "", _cleaned_text)
        _pattern = "<sources></sources>"
        _cleaned_text = re.sub(_pattern, "", _cleaned_text)

        _fully_cited_answer = ""
        _curr_citation_idx = 0

        for _citation in _citations:
            if enable_trace and trace_level == "all":
                print(f"full citation: {_citation}")

            _start = _citation["generatedResponsePart"]["textResponsePart"]["span"][
                "start"
            ] - (_curr_citation_idx + 1)
            _end = (
                _citation["generatedResponsePart"]["textResponsePart"]["span"]["end"]
                - (_curr_citation_idx + 2)
                + 4
            )
            _refs = _citation.get("retrievedReferences", [])
            if len(_refs) > 0:
                _ref_url = _refs[0].get("location", {}).get("s3Location", {}).get(
                    "uri", ""
                )
            else:
                _ref_url = ""
                _fully_cited_answer = _cleaned_text
                break

            _fully_cited_answer += _cleaned_text[_start:_end] + " [" + _ref_url + "] "

            if _curr_citation_idx == 0:
                _answer_prefix = _cleaned_text[:_start]
                _fully_cited_answer = _answer_prefix + _fully_cited_answer

            _curr_citation_idx += 1

            if enable_trace and trace_level == "all":
                print(f"\n\ncitation {_curr_citation_idx}:")
                print(
                    f"got {len(_citation['retrievedReferences'])} retrieved references for this citation\n"
                )
                print(f"citation span... start: {_start}, end: {_end}")
                print(
                    f"citation based on span:====\n{_cleaned_text[_start:_end]}\n===="
                )
                print(f"citation url: {_ref_url}\n============")

        if enable_trace and trace_level == "all":
            print(
                f"\nfullly cited answer:*************\n{_fully_cited_answer}\n*************"
            )

        return _fully_cited_answer

    def invoke(
        self,
        input_text: str,
        agent_id: str,
        agent_alias_id: str = "TSTALIASID",
        session_id: str = str(uuid.uuid1()),
        session_state: dict = {},
        enable_trace: bool = False,
        end_session: bool = False,
        trace_level: str = "core",
        multi_agent_names: dict = {},
    ):
        """
        Invokes an agent with a given input text, optional session info, 
        and optional trace settings.

        We do a single-pass read of the event stream to get:
          1) The final agent answer.
          2) A JSON-serializable structure (agent_resp_for_json) for dumping to 'trace.json' 
             if trace_level='all'.
        """

        _time_before_call = datetime.datetime.now()

        _agent_resp = self._bedrock_agent_runtime_client.invoke_agent(
            inputText=input_text,
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            sessionState=session_state,
            enableTrace=enable_trace,
            endSession=end_session,
        )

        # Print partial info or full if trace=all
        if enable_trace:
            if trace_level == "all":
                print(f"invokeAgent API response object: {_agent_resp}")
            else:
                print(
                    f"invokeAgent API request ID: {_agent_resp['ResponseMetadata']['RequestId']}"
                )
                print(f"invokeAgent API session ID: {session_id}")
                print(f"  agent id: {agent_id}, agent alias id: {agent_alias_id}")

        # If not 200, return error
        if _agent_resp["ResponseMetadata"]["HTTPStatusCode"] != 200:
            _error_message = f"API Response was not 200: {_agent_resp}"
            if enable_trace and trace_level == "all":
                print(_error_message)
            return _error_message

        # Build a dict for JSON-based trace output
        agent_resp_for_json = {
            "ResponseMetadata": _agent_resp["ResponseMetadata"],
            "contentType": _agent_resp.get("contentType", ""),
            "sessionId": _agent_resp.get("sessionId", session_id),
            "completion": [],
        }

        _total_in_tokens = 0
        _total_out_tokens = 0
        _total_llm_calls = 0
        _orch_step = 0
        _sub_step = 0
        _time_before_orchestration = datetime.datetime.now()

        _agent_answer = ""
        _event_stream = _agent_resp["completion"]

        try:
            _sub_agent_name = "<collab-name-not-yet-provided>"
            _sub_agent_alias_id = None

            for _event in _event_stream:
                # Build an event_copy to store in agent_resp_for_json
                event_copy = {}

                # Copy event fields, base64-encoding chunk/file bytes
                for ek, ev in _event.items():
                    event_copy[ek] = ev

                # 1) If chunk is present, decode it for final agent answer
                if "chunk" in _event and "bytes" in _event["chunk"]:
                    chunk_bytes = _event["chunk"]["bytes"]
                    chunk_str = chunk_bytes.decode("utf8")
                    _agent_answer = chunk_str
                    # Possibly add citations
                    _agent_answer = self._make_fully_cited_answer(
                        _agent_answer, _event, enable_trace, trace_level
                    )

                    # Base64 for JSON
                    b64_chunk = base64.b64encode(chunk_bytes).decode("utf-8")
                    # Rebuild chunk object
                    new_chunk = {"bytes": b64_chunk}
                    if "attribution" in _event["chunk"]:
                        new_chunk["attribution"] = _event["chunk"]["attribution"]
                    event_copy["chunk"] = new_chunk

                # 2) If "files" present, also base64
                if "files" in _event:
                    files_section = {"files": []}
                    for file_obj in _event["files"].get("files", []):
                        file_copy = dict(file_obj)
                        if "bytes" in file_copy:
                            file_copy["bytes"] = base64.b64encode(
                                file_copy["bytes"]
                            ).decode("utf-8")
                        files_section["files"].append(file_copy)
                    event_copy["files"] = files_section

                # 3) If trace is present, do the printing logic EXACTLY as in original code
                if "trace" in _event and enable_trace:
                    if trace_level == "all":
                        print("---")
                    else:
                        # original snippet for sub-agent name
                        if "callerChain" in _event["trace"]:
                            if len(_event["trace"]["callerChain"]) > 1:
                                _sub_agent_alias_arn = _event["trace"]["callerChain"][
                                    1
                                ]["agentAliasArn"]
                                _sub_agent_alias_id = _sub_agent_alias_arn.split("/", 1)[
                                    1
                                ]
                                if _sub_agent_alias_id in multi_agent_names:
                                    _sub_agent_name = multi_agent_names[_sub_agent_alias_id]

                    if "routingClassifierTrace" in _event["trace"]["trace"]:
                        _route = _event["trace"]["trace"]["routingClassifierTrace"]

                        if "modelInvocationInput" in _route:
                            _orch_step += 1
                            print(colored(f"---- Step {_orch_step} ----", "green"))
                            _time_before_routing = datetime.datetime.now()
                            print(
                                colored(
                                    "Classifying request to immediately route to one collaborator if possible.",
                                    "blue",
                                )
                            )

                        if "modelInvocationOutput" in _route:
                            _llm_usage = _route["modelInvocationOutput"]["metadata"][
                                "usage"
                            ]
                            _in_tokens = _llm_usage["inputTokens"]
                            _total_in_tokens += _in_tokens
                            _out_tokens = _llm_usage["outputTokens"]
                            _total_out_tokens += _out_tokens
                            _total_llm_calls += 1

                            _route_duration = (
                                datetime.datetime.now() - _time_before_routing
                            )

                            _raw_resp_str = _route["modelInvocationOutput"][
                                "rawResponse"
                            ]["content"]
                            _raw_resp = json.loads(_raw_resp_str)
                            _classification = (
                                _raw_resp["content"][0]["text"]
                                .replace("<a>", "")
                                .replace("</a>", "")
                            )

                            if _classification == UNDECIDABLE_CLASSIFICATION:
                                print(
                                    colored(
                                        f"Routing classifier did not find a matching collaborator. Reverting to 'SUPERVISOR' mode.",
                                        "magenta",
                                    )
                                )
                            elif _classification == "keep_previous_agent":
                                print(
                                    colored(
                                        f"Continuing conversation with previous collaborator.",
                                        "magenta",
                                    )
                                )
                            else:
                                _sub_agent_name = _classification
                                print(
                                    colored(
                                        f"Routing classifier chose collaborator: '{_classification}'",
                                        "magenta",
                                    )
                                )
                            print(
                                colored(
                                    f"Routing classifier took {_route_duration.total_seconds():,.1f}s, using {_in_tokens+_out_tokens} tokens (in: {_in_tokens}, out: {_out_tokens}).\n",
                                    "yellow",
                                )
                            )

                    if "failureTrace" in _event["trace"]["trace"]:
                        print(
                            colored(
                                f"Agent error: {_event['trace']['trace']['failureTrace']['failureReason']}",
                                "red",
                            )
                        )

                    if "orchestrationTrace" in _event["trace"]["trace"]:
                        _orch = _event["trace"]["trace"]["orchestrationTrace"]

                        if trace_level in ["core", "outline"]:
                            if "rationale" in _orch:
                                _rationale = _orch["rationale"]
                                print(colored(f"{_rationale['text']}", "blue"))

                            if "invocationInput" in _orch:
                                _input = _orch["invocationInput"]

                                if "actionGroupInvocationInput" in _input:
                                    if trace_level == "outline":
                                        print(
                                            colored(
                                                f"Using tool: {_input['actionGroupInvocationInput']['function']}",
                                                "magenta",
                                            )
                                        )
                                    else:
                                        if (
                                            "function"
                                            not in _input["actionGroupInvocationInput"]
                                        ):
                                            print(
                                                colored(
                                                    f"EXPECTING to capture 'Using tool', but 'function' not found\n{_input['actionGroupInvocationInput']}",
                                                    "red",
                                                )
                                            )
                                        else:
                                            print(
                                                colored(
                                                    f"Using tool: {_input['actionGroupInvocationInput']['function']} with these inputs:",
                                                    "magenta",
                                                )
                                            )
                                            if (
                                                "parameters"
                                                in _input["actionGroupInvocationInput"]
                                            ):
                                                if (
                                                    len(
                                                        _input[
                                                            "actionGroupInvocationInput"
                                                        ]["parameters"]
                                                    )
                                                    == 1
                                                ) and (
                                                    _input[
                                                        "actionGroupInvocationInput"
                                                    ]["parameters"][0]["name"]
                                                    == "input_text"
                                                ):
                                                    print(
                                                        colored(
                                                            f"{_input['actionGroupInvocationInput']['parameters'][0]['value']}",
                                                            "magenta",
                                                        )
                                                    )
                                                else:
                                                    print(
                                                        colored(
                                                            f"{_input['actionGroupInvocationInput']['parameters']}\n",
                                                            "magenta",
                                                        )
                                                    )
                                            else:
                                                print(
                                                    colored(
                                                        f"    no input parameters being sent\n",
                                                        "magenta",
                                                    )
                                                )

                                elif "agentCollaboratorInvocationInput" in _input:
                                    _collab_name = _input[
                                        "agentCollaboratorInvocationInput"
                                    ]["agentCollaboratorName"]
                                    _sub_agent_name = _collab_name
                                    _collab_input_text = _input[
                                        "agentCollaboratorInvocationInput"
                                    ]["input"]["text"]
                                    _collab_arn = _input[
                                        "agentCollaboratorInvocationInput"
                                    ]["agentCollaboratorAliasArn"]
                                    _collab_ids = _collab_arn.split("/", 1)[1]

                                    if trace_level == "outline":
                                        print(
                                            colored(
                                                f"Using sub-agent collaborator: '{_collab_name} [{_collab_ids}]'",
                                                "magenta",
                                            )
                                        )
                                    else:
                                        print(
                                            colored(
                                                f"Using sub-agent collaborator: '{_collab_name} [{_collab_ids}]' passing input text:",
                                                "magenta",
                                            )
                                        )
                                        print(
                                            colored(
                                                f"{_collab_input_text[0:TRACE_TRUNCATION_LENGTH]}\n",
                                                "magenta",
                                            )
                                        )

                                elif "codeInterpreterInvocationInput" in _input:
                                    if trace_level == "outline":
                                        print(
                                            colored(
                                                f"Using code interpreter", "magenta"
                                            )
                                        )
                                    else:
                                        console = Console()
                                        _gen_code = _input[
                                            "codeInterpreterInvocationInput"
                                        ]["code"]
                                        _code = f"```python\n{_gen_code}\n```"
                                        console.print(
                                            Markdown(f"**Generated code**\n{_code}")
                                        )

                                elif "knowledgeBaseLookupInput" in _input:
                                    if trace_level == "outline":
                                        print(
                                            colored(f"Using knowledge base", "magenta")
                                        )
                                    else:
                                        _kb_id = _input["knowledgeBaseLookupInput"][
                                            "knowledgeBaseId"
                                        ]
                                        _kb_query = _input["knowledgeBaseLookupInput"][
                                            "text"
                                        ]
                                        print(
                                            colored(
                                                f"Using knowledge base id: {_kb_id} to search for:",
                                                "magenta",
                                            )
                                        )
                                        print(colored(f"  {_kb_query}\n", "magenta"))

                            if "observation" in _orch:
                                if trace_level == "core":
                                    _output = _orch["observation"]
                                    if "actionGroupInvocationOutput" in _output:
                                        print(
                                            colored(
                                                f"--tool outputs:\n{_output['actionGroupInvocationOutput']['text'][0:TRACE_TRUNCATION_LENGTH]}...\n",
                                                "magenta",
                                            )
                                        )

                                    if "agentCollaboratorInvocationOutput" in _output:
                                        _collab_name = _output[
                                            "agentCollaboratorInvocationOutput"
                                        ]["agentCollaboratorName"]
                                        _collab_output_text = _output[
                                            "agentCollaboratorInvocationOutput"
                                        ]["output"]["text"][
                                            0:TRACE_TRUNCATION_LENGTH
                                        ]
                                        print(
                                            colored(
                                                f"\n----sub-agent {_collab_name} output text:\n{_collab_output_text}...\n",
                                                "magenta",
                                            )
                                        )

                                    if "knowledgeBaseLookupOutput" in _output:
                                        _refs = _output["knowledgeBaseLookupOutput"][
                                            "retrievedReferences"
                                        ]
                                        _ref_count = len(_refs)
                                        print(
                                            colored(
                                                f"Knowledge base lookup output, {_ref_count} references:\n",
                                                "magenta",
                                            )
                                        )
                                        _curr = 1
                                        for _ref in _refs:
                                            print(
                                                colored(
                                                    f"  ({_curr}) {_ref['content']['text'][0:TRACE_TRUNCATION_LENGTH]}...\n",
                                                    "magenta",
                                                )
                                            )
                                            _curr += 1

                                    if "finalResponse" in _output:
                                        print(
                                            colored(
                                                f"Final response:\n{_output['finalResponse']['text'][0:TRACE_TRUNCATION_LENGTH]}...",
                                                "cyan",
                                            )
                                        )

                        if "modelInvocationOutput" in _orch:
                            if _sub_agent_alias_id is not None:
                                _sub_step += 1
                                print(
                                    colored(
                                        f"---- Step {_orch_step}.{_sub_step} [using sub-agent name:{_sub_agent_name}, id:{_sub_agent_alias_id}] ----",
                                        "green",
                                    )
                                )
                            else:
                                _orch_step += 1
                                _sub_step = 0
                                print(colored(f"---- Step {_orch_step} ----", "green"))

                            _llm_usage = _orch["modelInvocationOutput"]["metadata"][
                                "usage"
                            ]
                            _in_tokens = _llm_usage["inputTokens"]
                            _out_tokens = _llm_usage["outputTokens"]
                            _total_in_tokens += _in_tokens
                            _total_out_tokens += _out_tokens
                            _total_llm_calls += 1
                            _orch_duration = (
                                datetime.datetime.now() - _time_before_orchestration
                            )
                            print(
                                colored(
                                    f"Took {_orch_duration.total_seconds():,.1f}s, using {_in_tokens+_out_tokens} tokens (in: {_in_tokens}, out: {_out_tokens}) to complete prior action, observe, orchestrate.",
                                    "yellow",
                                )
                            )
                            _time_before_orchestration = datetime.datetime.now()

                    elif "preProcessingTrace" in _event["trace"]["trace"]:
                        _pre = _event["trace"]["trace"]["preProcessingTrace"]
                        if "modelInvocationOutput" in _pre:
                            _llm_usage = _pre["modelInvocationOutput"]["metadata"][
                                "usage"
                            ]
                            _in_tokens = _llm_usage["inputTokens"]
                            _total_in_tokens += _in_tokens
                            _out_tokens = _llm_usage["outputTokens"]
                            _total_out_tokens += _out_tokens
                            _total_llm_calls += 1
                            print(
                                colored(
                                    "Pre-processing trace, agent came up with an initial plan.",
                                    "yellow",
                                )
                            )
                            print(
                                colored(
                                    f"Used LLM tokens, in: {_in_tokens}, out: {_out_tokens}",
                                    "yellow",
                                )
                            )

                    elif "postProcessingTrace" in _event["trace"]["trace"]:
                        _post = _event["trace"]["trace"]["postProcessingTrace"]
                        if "modelInvocationOutput" in _post:
                            _llm_usage = _post["modelInvocationOutput"]["metadata"][
                                "usage"
                            ]
                            _in_tokens = _llm_usage["inputTokens"]
                            _out_tokens = _llm_usage["outputTokens"]
                            _total_in_tokens += _in_tokens
                            _total_out_tokens += _out_tokens
                            _total_llm_calls += 1
                            print(colored("Agent post-processing complete.", "yellow"))
                            print(
                                colored(
                                    f"Used LLM tokens, in: {_in_tokens}, out: {_out_tokens}",
                                    "yellow",
                                )
                            )

                    if trace_level == "all":
                        print(json.dumps(_event["trace"], indent=2))

                # 4) If "files" appear and trace is enabled, we do the same file-saving logic
                if "files" in _event.keys() and enable_trace:
                    console = Console()
                    files_event = _event["files"]
                    console.print(Markdown("**Files**"))

                    files_list = files_event["files"]
                    for this_file in files_list:
                        print(f"{this_file['name']} ({this_file['type']})")
                        file_bytes = this_file["bytes"]

                        file_name = os.path.join("output", this_file["name"])
                        with open(file_name, "wb") as f:
                            f.write(file_bytes)
                        # If you'd like to visualize images, uncomment code for e.g. matplotlib.

                # Finally, append event_copy to the JSON structure
                agent_resp_for_json["completion"].append(event_copy)

            # after reading the entire stream
            if enable_trace:
                duration = datetime.datetime.now() - _time_before_call

                if trace_level in ["core", "outline"]:
                    print(
                        colored(
                            f"Agent made a total of {_total_llm_calls} LLM calls, "
                            + f"using {_total_in_tokens+_total_out_tokens} tokens "
                            + f"(in: {_total_in_tokens}, out: {_total_out_tokens})"
                            + f", and took {duration.total_seconds():,.1f} total seconds",
                            "yellow",
                        )
                    )

                if trace_level == "all":
                    # same final "Returning agent answer as: " line:
                    print(f"Returning agent answer as: {_agent_answer}")

                    # Dump to trace.json
                    with open("trace.json", "w", encoding="utf-8") as f:
                        json.dump(agent_resp_for_json, f, indent=2)

            return _agent_answer

        except Exception as e:
            print(f"Caught exception while processing input to invokeAgent:\n")
            print(f"  for input text:\n{input_text}\n")
            print(f"  on agent: {agent_id}, alias: {agent_alias_id}")
            print(
                f"  request ID: {_agent_resp['ResponseMetadata']['RequestId']}, "
                f"retries: {_agent_resp['ResponseMetadata']['RetryAttempts']}\n"
            )
            print(f"Error: {e}")
            raise Exception("Unexpected exception: ", e)

    def invoke_roc(
        self,
        input_text: str,
        agent_id: str,
        agent_alias_id: str = DEFAULT_ALIAS,
        session_id: str = str(uuid.uuid1()),
        function_call: str = None,
        function_call_result: str = None,
        enable_trace: bool = False,
        end_session: bool = False,
    ):
        """Invokes an agent with Return-of-Control, optionally providing function_call_result from a prior step."""
        if function_call is not None:
            _agent_resp = self._bedrock_agent_runtime_client.invoke_agent(
                inputText=input_text,
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                sessionState={
                    "invocationId": function_call["invocationId"],
                    "returnControlInvocationResults": [
                        {
                            "functionResult": {
                                "actionGroup": function_call["invocationInputs"][0][
                                    "functionInvocationInput"
                                ]["actionGroup"],
                                "function": function_call["invocationInputs"][0][
                                    "functionInvocationInput"
                                ]["function"],
                                "responseBody": {
                                    "TEXT": {"body": function_call_result}
                                },
                            }
                        }
                    ],
                },
                enableTrace=enable_trace,
                endSession=end_session,
            )
        else:
            _agent_resp = self._bedrock_agent_runtime_client.invoke_agent(
                inputText=input_text,
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                enableTrace=enable_trace,
                endSession=end_session,
            )

        _agent_answer = ""
        _event_stream = _agent_resp["completion"]

        try:
            for _event in _event_stream:
                if "chunk" in _event:
                    _data = _event["chunk"]["bytes"]
                    _agent_answer = _data.decode("utf8")
                elif "returnControl" in _event:
                    _agent_answer = _event["returnControl"]
                elif "trace" in _event:
                    # For ROC usage, optional debug prints. 
                    print(json.dumps(_event["trace"], indent=2))
                else:
                    raise Exception("unexpected event.", _event)
            return _agent_answer
        except Exception as e:
            raise Exception("unexpected event.", e)

    def update_agent(
        self,
        agent_name: str,
        new_model_id: str = None,
        new_instructions: str = None,
        guardrail_id: str = None,
    ):
        """Updates an agent's model, instructions, or guardrail."""
        _agent_id = self.get_agent_id_by_name(agent_name)
        _get_agent_response = self._bedrock_agent_client.get_agent(agentId=_agent_id)
        _agent_details = _get_agent_response.get("agent")

        if new_model_id is not None:
            _agent_details["foundationModel"] = new_model_id
        if new_instructions is not None:
            _agent_details["instruction"] = new_instructions

        if guardrail_id is not None:
            _agent_details["guardrailConfiguration"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": "DRAFT",
            }
        else:
            if "guardrailConfiguration" in _agent_details:
                del _agent_details["guardrailConfiguration"]

        # Preserve prompt override configs
        _promptOverrideConfigsList = _agent_details["promptOverrideConfiguration"].get(
            "promptConfigurations"
        )
        _filteredPromptOverrideConfigsList = list(
            filter(
                lambda x: (x["promptCreationMode"] == "OVERRIDDEN"),
                _promptOverrideConfigsList,
            )
        )
        _agent_details["promptOverrideConfiguration"][
            "promptConfigurations"
        ] = _filteredPromptOverrideConfigsList

        # Remove fields that the UpdateAgent call doesn't accept
        for key_to_remove in [
            "clientToken",
            "createdAt",
            "updatedAt",
            "preparedAt",
            "agentStatus",
            "agentArn",
        ]:
            if key_to_remove in _agent_details:
                del _agent_details[key_to_remove]

        _update_agent_response = self._bedrock_agent_client.update_agent(
            **_agent_details
        )
        time.sleep(3)
        self._bedrock_agent_client.prepare_agent(agentId=_agent_id)
        return _update_agent_response

    def create_dynamodb(self, table_name, pk_item, sk_item):
        """Creates a DynamoDB table (if not exists)."""
        try:
            table = self._dynamodb_resource.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": pk_item, "KeyType": "HASH"},
                    {"AttributeName": sk_item, "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": pk_item, "AttributeType": "S"},
                    {"AttributeName": sk_item, "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
        except self._dynamodb_client.exceptions.ResourceInUseException:
            print(f"Table {table_name} already exists, skipping table creation step")

    def load_dynamodb(self, table_name: str, items: List):
        """Loads items into a DynamoDB table."""
        try:
            table = self._dynamodb_resource.Table(table_name)
            for item in items:
                table.put_item(Item=item)
        except self._dynamodb_client.exceptions.ResourceInUseException:
            print(f"Error on loading process for table: {table_name}.")

    def query_dynamodb(
        self,
        table_name: str,
        pk_field: str,
        pk_value: str,
        sk_field: str = None,
        sk_value: str = None,
    ):
        """Queries DynamoDB by partition key (and optional begins_with sort key)."""
        try:
            table = self._dynamodb_resource.Table(table_name)
            if sk_field:
                key_expression = Key(pk_field).eq(pk_value) & Key(sk_field).begins_with(
                    sk_value
                )
            else:
                key_expression = Key(pk_field).eq(pk_value)
            query_data = table.query(KeyConditionExpression=key_expression)
            return query_data["Items"]
        except self._dynamodb_client.exceptions.ResourceInUseException:
            print(f"Error querying table: {table_name}.")

    def get_agent_by_name(self, agent_name: str) -> dict:
        """Returns the agent record by name, or None if not found."""
        agent_id = self.get_agent_id_by_name(agent_name)
        if agent_id is None:
            return None
        return self._bedrock_agent_client.get_agent(agentId=agent_id)["agent"]

    def get_agent_by_id(self, agent_id: str) -> dict:
        """Returns the agent record by ID, or None if not found."""
        try:
            return self._bedrock_agent_client.get_agent(agentId=agent_id)["agent"]
        except self._bedrock_agent_client.exceptions.ResourceNotFoundException:
            return None
