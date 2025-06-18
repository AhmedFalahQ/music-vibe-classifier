from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import get_execution_role

# the S3 paths, and the role
bucket_name = 'proj-bucket-01'
dataset_prefix = 'data/'  # The prefix where your dataset is stored in S3
role = get_execution_role()

# Define the SageMaker training job
estimator = PyTorch(
    entry_point='train.py',  # Your training script
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge',  # GPU instance
    framework_version='1.12',
    py_version='py38',
    output_path=f's3://{bucket_name}/output/',  # Where to save model artifacts
    code_location=f's3://{bucket_name}/code/',  # Where to store the training script
)

# Define the input data using TrainingInput
train_data = TrainingInput(
    s3_data=f's3://{bucket_name}/{dataset_prefix}',
    content_type='csv',  # Specify the content type of the input data
    input_mode='File',  # Use 'File' mode to download the data to the training instance
    distribution='FullyReplicated',  # Distribute the data to all instances
)

# Start the training job
estimator.fit({'training': train_data})