import os
from constructs import Construct
from aws_cdk import (
    Stack, Duration, RemovalPolicy,
    aws_iam as iam,
    aws_ssm as ssm,
    aws_s3 as s3,
    aws_logs as logs,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
)
from aws_cdk import aws_lambda_python_alpha as py_lambda

PROJECT = "ecg-app"

class ECGStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # (Opcional) bucket de datos
        bucket = s3.Bucket(self, "DataBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            auto_delete_objects=True,
            removal_policy=RemovalPolicy.DESTROY,
            versioned=True,
        )

        # Parámetros SSM (poned valores reales tras desplegar o antes)
        sm_endpoint_param = ssm.StringParameter(
            self, "SmEndpointName",
            parameter_name=f"/{PROJECT}/SM_ENDPOINT_NAME",
            string_value="REPLACE_ME"  # <- CAMBIAR tras crear el endpoint
        )
        bedrock_region = ssm.StringParameter(
            self, "BedrockRegion",
            parameter_name=f"/{PROJECT}/BEDROCK_REGION",
            string_value="us-east-1"
        )
        bedrock_model_id = ssm.StringParameter(
            self, "BedrockModelId",
            parameter_name=f"/{PROJECT}/BEDROCK_MODEL_ID",
            string_value="anthropic.claude-3-haiku-20240307-v1:0"
        )

        # Rol Lambda base
        role = iam.Role(self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ]
        )
        bucket.grant_read_write(role)
        sm_endpoint_param.grant_read(role)
        bedrock_region.grant_read(role)
        bedrock_model_id.grant_read(role)

        # Permisos
        role.add_to_policy(iam.PolicyStatement(
            actions=["sagemaker:InvokeEndpoint","sagemaker:InvokeEndpointAsync"],
            resources=["*"]  # limitar al ARN de tu endpoint en prod
        ))
        role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
            resources=["*"]
        ))

        here = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(here, ".."))

        # Lambda infer (→ SageMaker)
        fn_infer = py_lambda.PythonFunction(self, "LambdaInfer",
            entry=os.path.join(repo_root, "lambdas", "infer_sm"),
            index="app.py", handler="handler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            role=role,
            timeout=Duration.seconds(20),
            memory_size=512,
            environment={"SM_ENDPOINT_PARAM": sm_endpoint_param.parameter_name},
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # Lambda explain (→ Bedrock)
        fn_explain = py_lambda.PythonFunction(self, "LambdaExplain",
            entry=os.path.join(repo_root, "lambdas", "explain_bedrock"),
            index="app.py", handler="handler",
            runtime=_lambda.Runtime.PYTHON_3_12,
            role=role,
            timeout=Duration.seconds(25),
            memory_size=1024,
            environment={
                "BEDROCK_REGION_PARAM": bedrock_region.parameter_name,
                "BEDROCK_MODEL_ID_PARAM": bedrock_model_id.parameter_name,
            },
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # API Gateway
        api = apigw.RestApi(self, "PublicApi",
            rest_api_name=f"{PROJECT}-api",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=["POST","OPTIONS"],
            ),
        )
        r_infer = api.root.add_resource("infer")
        r_infer.add_method("POST", apigw.LambdaIntegration(fn_infer))
        r_explain = api.root.add_resource("explain")
        r_explain.add_method("POST", apigw.LambdaIntegration(fn_explain))

        self.export_value(api.url, name="ApiUrl")
        self.export_value(bucket.bucket_name, name="DataBucketName")
