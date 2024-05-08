import boto3
from botocore.config import Config
import sagemaker
import wandb
from datasets import load_dataset
from sagemaker.huggingface import HuggingFace, HuggingFaceProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import time

from src.utils import FINETUNE_PARAMETERS
from src.upload_dataset_to_s3 import DataProcessor

### Init Sagemaker session ###

boto_config = Config(
    region_name = "eu-north-1"
)
boto_client = boto3.client("sagemaker", config=boto_config)
sess = sagemaker.Session(sagemaker_client=boto_client)
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket="sagemaker-eu-north-1-252778267474"

if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

# try:
#     role = sagemaker.get_execution_role()
# except ValueError:
#     iam = boto3.client('iam')
#     role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

role = "arn:aws:iam::252778267474:role/MLS_SagemakerExecutionRole"

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")

### Init W&B logging ###
wandb.sagemaker_auth(path=".")


### Load dataset and upload to Sagemaker S3 ###

processor = HuggingFaceProcessor(
    role                    = role,
    instance_count          = 1,
    instance_type           = 'ml.g4dn.xlarge', # 'ml.g4dn.xlarge',
    transformers_version    = '4.26',            # the transformers version used in the training job
    pytorch_version         = '1.13',            # the pytorch_version version used in the training job
    py_version              = 'py39',            # the python version used in the training job
    base_job_name           = 'datasetprocessing'
)
processor.run(
    code='upload_dataset_to_s3.py',
    source_dir='src',
    inputs=[
        ProcessingInput(
            input_name='data',
            source=f"s3://{sagemaker_session_bucket}/raw/{FINETUNE_PARAMETERS['dataset_name']}",
            destination='/opt/ml/processing/input/data/'
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name='train',
            source='/opt/ml/processing/output/train/',
            destination=f's3://{sagemaker_session_bucket}/processed'
        )
    ],
    arguments=["--datasetname", FINETUNE_PARAMETERS['dataset_name']]
)

# ### Run training job on Sagemaker ###

# job_name = f'finetune_job-{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}'

# # create the Estimator
# huggingface_estimator = HuggingFace(
#     entry_point          = 'run_clm.py',      # train script
#     source_dir           = 'scripts',         # directory which includes all the files needed for training
#     instance_type        = 'ml.g5.2xlarge',   # instances type used for the training job
#     instance_count       = 1,                 # the number of instances used for training
#     base_job_name        = job_name,          # the name of the training job
#     role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
#     volume_size          = 300,               # the size of the EBS volume in GB
#     transformers_version = '4.26',            # the transformers version used in the training job
#     pytorch_version      = '1.13',            # the pytorch_version version used in the training job
#     py_version           = 'py39',            # the python version used in the training job
#     hyperparameters      =  FINETUNE_PARAMETERS
# )

# # define a data input dictonary with our uploaded s3 uris
# data = {'training': f's3://{sagemaker_session_bucket}/processed'}

# # starting the train job with our uploaded datasets as input
# huggingface_estimator.fit(data, wait=True)