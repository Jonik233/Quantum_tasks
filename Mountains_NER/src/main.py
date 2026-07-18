import sagemaker
from sagemaker.huggingface import HuggingFace

sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::519763207317:role/NERunner"


huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./src",                 # Directory containing train.py, model.py, config.py, etc.
    role=role,
    instance_count=1,
    instance_type="ml.g6.2xlarge",
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    output_path="s3://qunatum-tasks-bucket/Mountains_NER/output/",
    hyperparameters={
        "epochs": 3,
        "train_batch_size": 16,
        "valid_batch_size": 16,
        "learning_rate": 2e-5,
        "mlflow_tracking_arn": "arn:aws:sagemaker:eu-central-1:519763207317:mlflow-app/app-WHR2A5CWRLEK",
        "mlflow_experiment_name": "DistilBert_FineTuning",
    },
    enable_sagemaker_metrics=True
)

# Trigger the train job
# SageMaker will map the "train" key to the SM_CHANNEL_TRAIN env variable inside train.py
huggingface_estimator.fit({
    "train": "s3://qunatum-tasks-bucket/Mountains_NER/data/",
    "valid": "s3://qunatum-tasks-bucket/Mountains_NER/data/"
})