import boto3

# Create a SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Define the training job parameters
job_name = 'my-training-job'
role_arn = 'arn:aws:iam::123456789012:role/SageMakerRole'  # Replace with your IAM role ARN
image_uri = '123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image'  # Replace with the appropriate image URI
instance_type = 'ml.p3.2xlarge'
instance_count = 1
s3_input_data = 's3://my-bucket-name/my-training-data/'
s3_output_data = 's3://my-bucket-name/my-output-data/'

# Define the input and output channels (S3 paths)
input_data = {
    'ChannelName': 'train',
    'DataSource': {
        'S3DataSource': {'S3Uri': s3_input_data, 'S3DataType': 'S3Prefix', 'S3DataDistributionType': 'FullyReplicated'}
    },
}

output_data = {'S3OutputPath': s3_output_data}

# Define the training job parameters
training_params = {
    'TrainingJobName': job_name,
    'AlgorithmSpecification': {'TrainingImage': image_uri, 'TrainingInputMode': 'File'},
    'RoleArn': role_arn,
    'InputDataConfig': [input_data],
    'OutputDataConfig': output_data,
    'ResourceConfig': {'InstanceType': instance_type, 'InstanceCount': instance_count, 'VolumeSizeInGB': 50},
    'StoppingCondition': {'MaxRuntimeInSeconds': 3600},  # Maximum runtime for the job (in seconds)
}

# Start the training job
response = sagemaker_client.create_training_job(**training_params)

# Print the response (for debugging)
print(response)
