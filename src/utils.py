FINETUNE_PARAMETERS = {
    "model_id": "microsoft/phi-2",                  # basemodel to finetune
    "dataset_name": "test__train_data.json",     # name of training dataset  
    "hf_username": "madslang",
    "dataset_path": "/opt/ml/input/data/training",  # dataset path on sagemaker

    "lr": 0.00002,      # learning rate
    "bs": 1 ,           # batch size
    "bs_eval": 16,      # batch size for evals
    "ga_steps": 16,     # gradient acc. steps
    "epochs": 20,       # n epochs
    "max_length": 1024, # max_lenght in tokenizer

    "qlora": True, # where to use QLORA (if False use LORA)
}