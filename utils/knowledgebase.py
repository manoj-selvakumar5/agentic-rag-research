#!/usr/bin/env python3
"""
===============================================================================
Improved Bedrock Knowledge Base Script (with integrated NameGenerator)
===============================================================================
This script provides classes and methods to create, update, and delete
Bedrock Knowledge Bases and associated AWS resources, printing logs
directly to the Jupyter Notebook output using a StreamHandler.
"""

import logging
import sys  # <-- For attaching the stream handler to sys.stdout
import boto3
import json
import time
from datetime import datetime

from typing import List, Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError
from opensearchpy import (
    AuthenticationException,
    OpenSearch,
    RequestsHttpConnection,
    AWSV4SignerAuth,
    RequestError
)

import re

# -----------------------------------------------------------------------------
# 0. Name Generator Class
# -----------------------------------------------------------------------------
class NameGenerator:
    """
    Dynamically apply length constraints based on resource type.
    """

    RESOURCE_LENGTH_LIMITS = {
        "bucket": 60,         # S3
        "execution-role": 60, # IAM
        "vector-store": 32,   # e.g., OpenSearch
        "index": 64,          # e.g., OpenSearch index
        "policy": 32,         # Example policy name limit
        # fallback is 30 if resource type not found
    }

    @staticmethod
    def normalize_string(s: str) -> str:
        """Lowercase, replace underscores with hyphens."""
        return s.lower().replace('_', '-')

    @staticmethod
    def build_prefix(kb_name: Any, suffix: Any, max_length: int) -> str:
        """
        Compose "{kb_name}-{suffix}" and abbreviate from the right of kb_name 
        if it exceeds max_length (minus the length needed for further expansions).
        """
        # Ensure both are strings before normalization
        kb_name_str = NameGenerator.normalize_string(str(kb_name))
        suffix_str = NameGenerator.normalize_string(str(suffix))

        prefix = f"{kb_name_str}-{suffix_str}"
        if len(prefix) <= max_length:
            return prefix

        # Need to shorten from the kb_name portion, preserving the suffix if possible
        allowed_for_kb_name = max_length - (len(suffix_str) + 1)  # +1 for '-'
        if allowed_for_kb_name < 1:
            # If even the suffix alone doesn't fit fully, fallback to suffix[:max_length].
            return suffix_str[:max_length]

        kb_name_abbrev = kb_name_str[:allowed_for_kb_name]
        return f"{kb_name_abbrev}-{suffix_str}"

    @staticmethod
    def finalize_name(prefix: str, resource_type: Any, max_length: int) -> str:
        """
        Add '-{resource_type}' to prefix. If it's still too long,
        trim from the left of prefix.
        """
        resource_str = NameGenerator.normalize_string(str(resource_type))
        full_name = f"{prefix}-{resource_str}"

        if len(full_name) <= max_length:
            return full_name

        surplus = len(full_name) - max_length
        clipped_prefix = prefix[surplus:]  # remove surplus from left
        clipped_prefix = re.sub(r'^-+', '', clipped_prefix)  # remove leading dashes
        return f"{clipped_prefix}-{resource_str}"

    @staticmethod
    def generate_resource_name(kb_name: str, suffix: Any, resource_type: str) -> str:
        """
        Generic method that reads from RESOURCE_LENGTH_LIMITS 
        and returns a safely-limited name.
        """
        max_len = NameGenerator.RESOURCE_LENGTH_LIMITS.get(resource_type, 30)
        prefix = NameGenerator.build_prefix(kb_name, suffix, max_len)
        return NameGenerator.finalize_name(prefix, resource_type, max_len)

    @staticmethod
    def generate_bucket_name(kb_name: str, suffix: Any) -> str:
        return NameGenerator.generate_resource_name(kb_name, suffix, "bucket")

    @staticmethod
    def generate_execution_role_name(kb_name: str, suffix: Any) -> str:
        return NameGenerator.generate_resource_name(kb_name, suffix, "execution-role")

    @staticmethod
    def generate_vector_store_name(kb_name: str, suffix: Any) -> str:
        return NameGenerator.generate_resource_name(kb_name, suffix, "vector-store")

    @staticmethod
    def generate_index_name(vector_store_name: str) -> str:
        # For an index, pass the "vector_store_name" as "kb_name"
        max_len = NameGenerator.RESOURCE_LENGTH_LIMITS.get("index", 30)
        return NameGenerator.finalize_name(vector_store_name, "index", max_len)

    @staticmethod
    def generate_policy_name(base_name: str, policy_type: str) -> str:
        resource_str = f"{policy_type}-policy"
        max_len = NameGenerator.RESOURCE_LENGTH_LIMITS.get("policy", 30)
        return NameGenerator.finalize_name(base_name, resource_str, max_len)


# -----------------------------------------------------------------------------
# 1. Configure Logging to Print in Jupyter
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
stream_handler.setFormatter(formatter)

if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(stream_handler)


# -----------------------------------------------------------------------------
# 2. Centralized AppContext
# -----------------------------------------------------------------------------
class AppContext:
    """
    Holds a shared boto3 Session, region, and account info. Also provides 
    common utility methods, such as interactive_sleep.
    """

    def __init__(self):
        self.session = boto3.Session()
        self.region: str = self.session.region_name
        self.account_number: str = self.session.client('sts').get_caller_identity()['Account']

    def client(self, service_name: str):
        return self.session.client(service_name, region_name=self.region)

    def resource(self, service_name: str):
        return self.session.resource(service_name, region_name=self.region)

    def interactive_sleep(self, seconds: int) -> None:
        dots = ''
        for _ in range(seconds):
            dots += '.'
            print(dots, end='\r')
            time.sleep(1)
        print("")

    def default_serializer(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# -----------------------------------------------------------------------------
# 3. OpenSearch Serverless Configuration
# -----------------------------------------------------------------------------
class OpenSearchServerlessConfiguration:
    """
    Manages operations related to Amazon OpenSearch Serverless.
    """

    def __init__(self, context: AppContext):
        self.context = context
        self.aoss_client = context.client('opensearchserverless')
        self.iam_client = context.client('iam')

        # Create AWS credentials for signing AOSS requests
        credentials = boto3.Session().get_credentials()
        self.awsauth = AWSV4SignerAuth(credentials, self.context.region, 'aoss')

        # Standard OS client (initialized later)
        self.oss_client: Optional[OpenSearch] = None

        # Identity used for principal in access policies
        sts_client = context.client('sts')
        self.identity = sts_client.get_caller_identity()['Arn']

    def create_aoss_configuration(self, collection_arn: str, index_name: str) -> Dict[str, Any]:
        return {
            "collectionArn": collection_arn,
            "vectorIndexName": index_name,
            "fieldMapping": {
                "vectorField": "vector",
                "textField": "text",
                "metadataField": "text-metadata"
            }
        }

    def create_policies_for_aoss(
        self, 
        vector_store_name: str, 
        bedrock_kb_execution_role: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        logger.info(f"Creating encryption, network, and access policies for {vector_store_name}")

        encryption_policy_name = NameGenerator.generate_policy_name(vector_store_name, "en")
        network_policy_name = NameGenerator.generate_policy_name(vector_store_name, "ne")
        access_policy_name = NameGenerator.generate_policy_name(vector_store_name, "ac")

        # Encryption policy
        encryption_policy = self.aoss_client.create_security_policy(
            name=encryption_policy_name,
            policy=json.dumps({
                'Rules': [{
                    'Resource': [f'collection/{vector_store_name}'],
                    'ResourceType': 'collection'
                }],
                'AWSOwnedKey': True
            }),
            type='encryption'
        )

        # Network policy
        network_policy = self.aoss_client.create_security_policy(
            name=network_policy_name,
            policy=json.dumps([
                {
                    'Rules': [{
                        'Resource': [f'collection/{vector_store_name}'],
                        'ResourceType': 'collection'
                    }],
                    'AllowFromPublic': True
                }
            ]),
            type='network'
        )

        # Access policy
        access_policy = self.aoss_client.create_access_policy(
            name=access_policy_name,
            policy=json.dumps([
                {
                    'Rules': [
                        {
                            'Resource': [f'collection/{vector_store_name}'],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'
                            ],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': [f'index/{vector_store_name}/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'
                            ],
                            'ResourceType': 'index'
                        }
                    ],
                    'Principal': [self.identity, bedrock_kb_execution_role['Role']['Arn']],
                    'Description': 'Easy data policy'
                }
            ]),
            type='data'
        )

        return encryption_policy, network_policy, access_policy

    def create_aoss_policy_attach_kb_execution_role(
        self, 
        kb_execution_role_name: str, 
        collection_id: str
    ) -> bool:
        aoss_policy_attached_to_kb_role = False
        try:
            aoss_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": ["aoss:APIAccessAll"],
                        "Resource": [
                            f"arn:aws:aoss:{self.context.region}:{self.context.account_number}:collection/{collection_id}"
                        ]
                    }
                ]
            }

            policy_name = f'{kb_execution_role_name}-aoss-policy'
            self.iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(aoss_policy),
                Description='Policy to access OpenSearch Serverless'
            )

            self.iam_client.attach_role_policy(
                RoleName=kb_execution_role_name,
                PolicyArn=f"arn:aws:iam::{self.context.account_number}:policy/{policy_name}"
            )
            aoss_policy_attached_to_kb_role = True
            logger.info(f"Successfully attached AOSS policy to {kb_execution_role_name}")
        except ClientError as e:
            logger.error(f"Failed to attach AOSS policy: {e}", exc_info=True)

        return aoss_policy_attached_to_kb_role

    def create_aoss_collection(
        self, 
        kb_execution_role_name: str, 
        vector_store_name: str
    ) -> Tuple[str, Dict[str, Any], str, str]:
        logger.info(f"Creating AOSS collection: {vector_store_name}")
        collection = self.aoss_client.create_collection(
            name=vector_store_name,
            type='VECTORSEARCH'
        )

        collection_id = collection['createCollectionDetail']['id']
        collection_arn = collection['createCollectionDetail']['arn']
        host = f"{collection_id}.{self.context.region}.aoss.amazonaws.com"

        # Wait for the collection to become active
        while True:
            response = self.aoss_client.batch_get_collection(names=[vector_store_name])
            details = response['collectionDetails'][0]
            if details['status'] == 'CREATING':
                logger.info("Collection is still creating, waiting 60 seconds...")
                self.context.interactive_sleep(60)
            else:
                logger.info("Collection is active")
                break

        # Attach the AOSS policy to the KB role
        attached = self.create_aoss_policy_attach_kb_execution_role(kb_execution_role_name, collection_id)
        if attached:
            logger.info("Sleeping 60 seconds to allow AOSS policy to propagate")
            self.context.interactive_sleep(60)

        return host, collection, collection_id, collection_arn

    def initialize_opensearch_client(self, host: str) -> None:
        try:
            self.oss_client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=self.awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=300
            )
            logger.info("Initialized OpenSearch client")
        except AuthenticationException as e:
            logger.error(f"Authentication failed: {e}", exc_info=True)
            self.oss_client = None

    def create_vector_index(self, index_name: str) -> None:
        if not self.oss_client:
            logger.warning("OSS client not initialized. Cannot create index.")
            return

        body_json = {
            "settings": {
                "index.knn": "true",
                "number_of_shards": 1,
                "knn.algo_param.ef_search": 512,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2"
                        }
                    },
                    "text": {"type": "text"},
                    "text-metadata": {"type": "text"}
                }
            }
        }

        logger.info(f"Creating index: {index_name}")
        try:
            response = self.oss_client.indices.create(index=index_name, body=json.dumps(body_json))
            logger.info(f"Create index response: {response}")
            self.context.interactive_sleep(60)
        except RequestError as e:
            logger.error(f"Failed to create index: {e}", exc_info=True)

    def delete_aoss_collection(self, collection_id: str) -> None:
        try:
            self.aoss_client.delete_collection(id=collection_id)
            logger.info(f"Deleted AOSS collection: {collection_id}")
        except ClientError as e:
            logger.error(f"Failed to delete AOSS collection: {e}", exc_info=True)

    def delete_aoss_index(self, index_name: str) -> None:
        if not self.oss_client:
            logger.warning("OSS client not initialized. Cannot delete index.")
            return

        try:
            self.oss_client.indices.delete(index=index_name)
            logger.info(f"Deleted index: {index_name}")
        except RequestError as e:
            logger.error(f"Failed to delete index: {e}", exc_info=True)

    def delete_aoss_policies(
        self, 
        encryption_policy_name: str, 
        network_policy_name: str, 
        access_policy_name: str
    ) -> None:
        logger.info(
            f"Deleting AOSS policies: {encryption_policy_name}, "
            f"{network_policy_name}, {access_policy_name}"
        )

        # Encryption
        try:
            self.aoss_client.delete_security_policy(
                name=encryption_policy_name, 
                type='encryption'
            )
            logger.info(f"Deleted encryption policy: {encryption_policy_name}")
        except ClientError as e:
            logger.error(f"Failed to delete encryption policy: {e}", exc_info=True)

        # Network
        try:
            self.aoss_client.delete_security_policy(
                name=network_policy_name, 
                type='network'
            )
            logger.info(f"Deleted network policy: {network_policy_name}")
        except ClientError as e:
            logger.error(f"Failed to delete network policy: {e}", exc_info=True)

        # Access
        try:
            self.aoss_client.delete_access_policy(
                name=access_policy_name, 
                type='data'
            )
            logger.info(f"Deleted access policy: {access_policy_name}")
        except ClientError as e:
            logger.error(f"Failed to delete access policy: {e}", exc_info=True)


# -----------------------------------------------------------------------------
# 4. Vector Store Configuration
# -----------------------------------------------------------------------------
class VectorStoreConfiguration:
    def __init__(self, context: AppContext):
        self.context = context
        self.open_search_serverless_configuration = OpenSearchServerlessConfiguration(context)

    def get_configuration(
        self, 
        vector_store_option: str, 
        collection_arn: str = "", 
        index_name: str = ""
    ) -> Dict[str, Any]:
        if vector_store_option.upper() == 'OPENSEARCH_SERVERLESS':
            logger.info(f"Building config for OpenSearch Serverless with {collection_arn}, {index_name}")
            return {
                "type": "OPENSEARCH_SERVERLESS",
                "opensearchServerlessConfiguration": 
                    self.open_search_serverless_configuration.create_aoss_configuration(
                        collection_arn,
                        index_name
                    )
            }
        else:
            logger.warning(f"Unsupported vector_store_option: {vector_store_option}")
            return {}


# -----------------------------------------------------------------------------
# 5. Main Knowledge Base Management Class
# -----------------------------------------------------------------------------
class BedrockKnowledgeBases:
    """
    Main class to create, update, and delete Bedrock knowledge bases and 
    associated AWS resources (S3, IAM, AOSS).
    """

    def __init__(self):
        self.context = AppContext()
        self.s3_client = self.context.client('s3')
        self.iam_client = self.context.client('iam')
        self.bedrock_agent_client = self.context.client('bedrock-agent')
        self.vector_store_configuration = VectorStoreConfiguration(self.context)

    def build_chunking_strategy(
        self, 
        strategy_type: str = "FIXED_SIZE", 
        max_tokens: int = 512, 
        overlap_percentage: int = 20
    ) -> Dict[str, Any]:
        if strategy_type.upper() == "FIXED_SIZE":
            return {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": max_tokens,
                    "overlapPercentage": overlap_percentage
                }
            }
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy_type}")

    # -------------------------------------------------------------------------
    # 5.2: S3 Bucket
    # -------------------------------------------------------------------------
    def create_s3_bucket(self, bucket_name: str) -> None:
        try:
            logger.info(f"Creating S3 bucket: {bucket_name}")
            if self.context.region == "us-east-1":
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                )
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.context.region}
                )
            logger.info(f"Created S3 bucket: {bucket_name}")
        except ClientError as e:
            logger.error(f"Failed to create S3 bucket {bucket_name}: {e}", exc_info=True)

    def delete_s3_bucket(self, bucket_name: str) -> None:
        logger.info(f"Deleting S3 bucket: {bucket_name}")
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                contents = page.get('Contents', [])
                for obj in contents:
                    self.s3_client.delete_object(Bucket=bucket_name, Key=obj['Key'])
            self.s3_client.delete_bucket(Bucket=bucket_name)
            logger.info(f"Deleted bucket: {bucket_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.warning(f"Bucket {bucket_name} does not exist.")
            else:
                logger.error(f"Failed to delete bucket {bucket_name}: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 5.3: IAM Role for Bedrock KB
    # -------------------------------------------------------------------------
    def create_bedrock_kb_execution_role(
        self,
        kb_execution_role_name: str,
        embedding_model: str,
        bucket_name: str
    ) -> Dict[str, Any]:
        logger.info(f"Creating Bedrock KB execution role: {kb_execution_role_name}")

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

        fm_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": [
                        f"arn:aws:bedrock:{self.context.region}::foundation-model/{embedding_model}"
                    ]
                }
            ]
        }

        s3_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*"
                    ],
                    "Condition": {
                        "StringEquals": {
                            "aws:ResourceAccount": f"{self.context.account_number}"
                        }
                    }
                }
            ]
        }

        try:
            bedrock_kb_execution_role = self.iam_client.create_role(
                RoleName=kb_execution_role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
                Description="Amazon Bedrock KB execution role for embedding + S3",
                MaxSessionDuration=3600
            )

            fm_policy = self.iam_client.create_policy(
                PolicyName=f'{kb_execution_role_name}-foundation-model-policy',
                PolicyDocument=json.dumps(fm_policy_document)
            )
            s3_policy = self.iam_client.create_policy(
                PolicyName=f'{kb_execution_role_name}-s3-policy',
                PolicyDocument=json.dumps(s3_policy_document)
            )

            self.iam_client.attach_role_policy(
                RoleName=kb_execution_role_name,
                PolicyArn=fm_policy['Policy']['Arn']
            )
            self.iam_client.attach_role_policy(
                RoleName=kb_execution_role_name,
                PolicyArn=s3_policy['Policy']['Arn']
            )

            return bedrock_kb_execution_role
        except ClientError as e:
            logger.error(f"Failed to create Bedrock KB execution role {kb_execution_role_name}: {e}", exc_info=True)
            raise

    def delete_bedrock_kb_execution_role(self, kb_execution_role_name: str) -> None:
        logger.info(f"Deleting Bedrock KB execution role: {kb_execution_role_name}")
        try:
            attached_policies = self.iam_client.list_attached_role_policies(RoleName=kb_execution_role_name)
            for policy in attached_policies['AttachedPolicies']:
                self.iam_client.detach_role_policy(
                    RoleName=kb_execution_role_name, 
                    PolicyArn=policy['PolicyArn']
                )
                self.iam_client.delete_policy(PolicyArn=policy['PolicyArn'])

            self.iam_client.delete_role(RoleName=kb_execution_role_name)
            logger.info(f"Deleted role: {kb_execution_role_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                logger.warning(f"Role {kb_execution_role_name} does not exist.")
            else:
                logger.error(f"Failed to delete role {kb_execution_role_name}: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 5.4: Create and Configure Knowledge Base
    # -------------------------------------------------------------------------
    def create_and_configure_knowledge_base(
        self,
        vector_store: str,
        collection_arn: str,
        index_name: str,
        kb_name: str,
        kb_description: str,
        bedrock_kb_execution_role: Dict[str, Any],
        embedding_model: str,
        bucket_name: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logger.info(f"Creating Bedrock Knowledge Base: {kb_name}")
        try:
            create_kb_response = self.bedrock_agent_client.create_knowledge_base(
                name=kb_name,
                description=kb_description,
                roleArn=bedrock_kb_execution_role['Role']['Arn'],
                knowledgeBaseConfiguration={
                    "type": "VECTOR",
                    "vectorKnowledgeBaseConfiguration": {
                        "embeddingModelArn": f"arn:aws:bedrock:{self.context.region}::foundation-model/{embedding_model}"
                    }
                },
                storageConfiguration=self.vector_store_configuration.get_configuration(
                    vector_store, 
                    collection_arn, 
                    index_name
                )
            )
            kb = create_kb_response["knowledgeBase"]
            logger.info(f"Created KB: {kb}")

            chunking_config = self.build_chunking_strategy()

            s3_configuration = {"bucketArn": f"arn:aws:s3:::{bucket_name}"}

            create_ds_response = self.bedrock_agent_client.create_data_source(
                name=kb_name,
                description=kb_description,
                knowledgeBaseId=kb['knowledgeBaseId'],
                dataDeletionPolicy='RETAIN',
                dataSourceConfiguration={
                    "type": "S3",
                    "s3Configuration": s3_configuration
                },
                vectorIngestionConfiguration={
                    "chunkingConfiguration": chunking_config
                }
            )
            ds = create_ds_response["dataSource"]
            logger.info(f"Created data source: {ds}")

            return kb, ds
        except ClientError as e:
            logger.error(f"Failed to create/configure knowledge base: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # 5.5: Listing and Deleting Knowledge Base + Data Sources
    # -------------------------------------------------------------------------
    def list_all_knowledge_bases(self) -> List[Dict[str, Any]]:
        logger.info("Listing all knowledge bases via pagination...")
        kb_summaries = []
        paginator = self.bedrock_agent_client.get_paginator('list_knowledge_bases')
        for page in paginator.paginate():
            kb_summaries.extend(page.get('knowledgeBaseSummaries', []))
        return kb_summaries

    def find_kb_id_by_name(self, kb_name: str) -> Optional[str]:
        for kb_summary in self.list_all_knowledge_bases():
            if kb_summary['name'] == kb_name:
                return kb_summary['knowledgeBaseId']
        return None

    def list_data_sources_for_kb(self, kb_id: str) -> List[Dict[str, Any]]:
        logger.info(f"Listing data sources for KB: {kb_id}")
        ds_summaries = []
        paginator = self.bedrock_agent_client.get_paginator('list_data_sources')
        for page in paginator.paginate(knowledgeBaseId=kb_id):
            ds_summaries.extend(page.get('dataSourceSummaries', []))
        return ds_summaries

    def delete_knowledge_base_and_data_sources(self, kb_id: str) -> None:
        logger.info(f"Deleting data sources and KB for ID: {kb_id}")
        try:
            ds_summaries = self.list_data_sources_for_kb(kb_id)
            for ds in ds_summaries:
                ds_id = ds['dataSourceId']
                try:
                    self.bedrock_agent_client.delete_data_source(
                        dataSourceId=ds_id, 
                        knowledgeBaseId=kb_id
                    )
                    logger.info(f"Deleted data source: {ds_id}")
                except ClientError as e:
                    logger.error(f"Failed to delete data source {ds_id}: {e}", exc_info=True)

            self.bedrock_agent_client.delete_knowledge_base(knowledgeBaseId=kb_id)
            logger.info(f"Deleted knowledge base: {kb_id}")

        except ClientError as e:
            logger.error(f"Failed to delete knowledge base/data sources for {kb_id}: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # 5.6: CREATE Knowledge Base (Orchestrator)
    # -------------------------------------------------------------------------
    def create_knowledge_base(
        self,
        kb_name: str,
        kb_description: str,
        vector_store: str,
        embedding_model: str,
        suffix: Any
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Orchestrator method to create all required resources.
        Returns: (kb_id, ds_id)
        """
        logger.info("================================================================")
        logger.info("Step 1: Create S3 bucket")
        bucket_name = NameGenerator.generate_bucket_name(kb_name, suffix)
        self.create_s3_bucket(bucket_name)

        logger.info("================================================================")
        logger.info("Step 2: Create KB execution role")
        kb_execution_role_name = NameGenerator.generate_execution_role_name(kb_name, suffix)
        bedrock_kb_execution_role = self.create_bedrock_kb_execution_role(
            kb_execution_role_name,
            embedding_model,
            bucket_name
        )

        logger.info("================================================================")
        logger.info("Step 3: Create AOSS encryption/network/access policies")
        vector_store_name = NameGenerator.generate_vector_store_name(kb_name, suffix)
        aoss = self.vector_store_configuration.open_search_serverless_configuration
        encryption_policy, network_policy, access_policy = aoss.create_policies_for_aoss(
            vector_store_name, 
            bedrock_kb_execution_role
        )
        encryption_policy_name = encryption_policy['securityPolicyDetail']['name']
        network_policy_name = network_policy['securityPolicyDetail']['name']
        access_policy_name = access_policy['accessPolicyDetail']['name']
        logger.info(f"Created AOSS policies: {encryption_policy_name}, {network_policy_name}, {access_policy_name}")

        logger.info("================================================================")
        logger.info("Step 4: Create AOSS collection")
        host, collection, collection_id, collection_arn = aoss.create_aoss_collection(
            kb_execution_role_name,
            vector_store_name
        )

        logger.info("================================================================")
        logger.info("Step 5: Initialize OpenSearch client and create index")
        aoss.initialize_opensearch_client(host)
        index_name = NameGenerator.generate_index_name(vector_store_name)
        aoss.create_vector_index(index_name)

        logger.info("================================================================")
        logger.info("Step 6: Create Bedrock KB and data source")
        kb, ds = self.create_and_configure_knowledge_base(
            vector_store,
            collection_arn,
            index_name,
            kb_name,
            kb_description,
            bedrock_kb_execution_role,
            embedding_model,
            bucket_name
        )

        kb_id = kb.get('knowledgeBaseId') if kb else None
        ds_id = ds.get('dataSourceId') if ds else None

        logger.info("================================================================")
        logger.info(f"KB created successfully! KB ID: {kb_id}, DS ID: {ds_id}")
        return kb_id, ds_id, bucket_name

    # -------------------------------------------------------------------------
    # 5.7: DELETE Knowledge Base by Name+Suffix
    # -------------------------------------------------------------------------
    def find_collection_id_by_name(self, collection_name: str) -> Optional[str]:
        logger.info(f"Finding AOSS collection for name: {collection_name}")
        aoss = self.vector_store_configuration.open_search_serverless_configuration

        response = aoss.aoss_client.list_collections(maxResults=50)
        for coll in response.get('collectionSummaries', []):
            if coll['name'] == collection_name:
                return coll['id']
        return None

    def delete_knowledge_base_resources_by_name(self, kb_name: str, suffix: Any) -> None:
        """
        Delete all resources for a KB, identified by kb_name+suffix.
        """
        logger.info("================================================================")
        logger.info(f"Deleting resources for kb_name='{kb_name}' and suffix='{suffix}'")

        bucket_name = NameGenerator.generate_bucket_name(kb_name, suffix)
        kb_execution_role_name = NameGenerator.generate_execution_role_name(kb_name, suffix)
        vector_store_name = NameGenerator.generate_vector_store_name(kb_name, suffix)

        # 1) Find & delete Knowledge Base + Data Sources
        kb_id = self.find_kb_id_by_name(kb_name)
        if kb_id:
            self.delete_knowledge_base_and_data_sources(kb_id)
        else:
            logger.warning(f"No KB found for name '{kb_name}'. Skipping KB deletion.")

        # 2) Delete S3 bucket
        self.delete_s3_bucket(bucket_name)

        # 3) Delete IAM Role
        self.delete_bedrock_kb_execution_role(kb_execution_role_name)

        # 4) Find the AOSS collection ID by name
        collection_id = self.find_collection_id_by_name(vector_store_name)
        if collection_id:
            aoss = self.vector_store_configuration.open_search_serverless_configuration
            host = f"{collection_id}.{self.context.region}.aoss.amazonaws.com"
            aoss.initialize_opensearch_client(host)

            # Delete the index
            index_name = NameGenerator.generate_index_name(vector_store_name)
            aoss.delete_aoss_index(index_name)

            # Delete the collection
            aoss.delete_aoss_collection(collection_id)
        else:
            logger.warning(f"No AOSS collection found for name '{vector_store_name}'. Skipping collection deletion.")

        # 5) Delete AOSS policies
        encryption_policy_name = NameGenerator.generate_policy_name(vector_store_name, "en")
        network_policy_name = NameGenerator.generate_policy_name(vector_store_name, "ne")
        access_policy_name = NameGenerator.generate_policy_name(vector_store_name, "ac")
        aoss.delete_aoss_policies(encryption_policy_name, network_policy_name, access_policy_name)
        logger.info("All resources deleted successfully.")

    # -------------------------------------------------------------------------
    # 5.8: Synchronize Data from S3 to Knowledge Base
    # -------------------------------------------------------------------------
    def synchronize_data(self, kb_id: str, ds_id: str) -> None:
        logger.info(f"Starting data synchronization for KB ID: {kb_id} and DS ID: {ds_id}")

        i_status = ['CREATING', 'DELETING', 'UPDATING']
        while self.bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb_id)['knowledgeBase']['status'] in i_status:
            logger.info("Knowledge base is not available yet. Waiting 10 seconds...")
            self.context.interactive_sleep(10)

        start_job_response = self.bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id
        )
        job = start_job_response["ingestionJob"]
        logger.info(f"Started ingestion job: {job}")

        while job['status'] not in ['COMPLETE', 'FAILED']:
            get_job_response = self.bedrock_agent_client.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                ingestionJobId=job["ingestionJobId"]
            )
            job = get_job_response["ingestionJob"]
            logger.info(f"Ingestion job status: {job['status']}")
            self.context.interactive_sleep(5)

        logger.info(f"Final ingestion job status: {job['status']}")
        logger.info("Here are the job details:" + json.dumps(job, indent=2, default=self.context.default_serializer))
