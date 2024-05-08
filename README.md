#

## Setup

```
export AWS_DEFAULT_REGION="eu-north-1"
```

And set credentials through the environment variables found in AWS. 


## Run processing and training job on AWS Sagemaker

```
poetry run python run deploy_sagemaker.py
```

## TODO

 - Processing job fails because it cannot import src.utils - instead of using FINETUNEPARAMETERS in the called scripts, we can only use it in upload_dataset_to_s3.py and then pass the parameters as arguments to the sagemaker functions (like datasetname on line 74). 