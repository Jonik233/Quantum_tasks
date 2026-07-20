### Download the compressed model from S3 bucket ###

aws s3 cp s3://qunatum-tasks-bucket/Mountains_NER/output/huggingface-pytorch-training-2026-07-20-12-29-44-225/output/model.tar.gz .

mkdir model_artifacts

tar -xvf model.tar.gz -C model_artifacts

rm -r model.tar.gz