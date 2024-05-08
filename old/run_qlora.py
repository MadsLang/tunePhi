import argparse
import os
import shutil
import uuid

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_from_disk
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          GenerationConfig, Trainer, TrainingArguments)

from utils import CHATML_TEMPLATES, FINETUNE_PARAMETERS, IGNORE_INDEX

from upload_dataset

### Parse args ### 

def parse_arge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_id",
        type=str,
        default="microsoft/phi-2",
        help="Model id to use for finetuning"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str,
        default=""
    )
    args = parser.parse_known_args()
    return args


def training_fnc(args):
    set_seed(666)
    accelerator = Accelerator()
    run_id = str(uuid.uuid4())
    output_dir=f"out_{run_id}"

    project_name = "phi2-finetune"
    # adapters: path to folder with adapter_model.safetensors
    adapter_path="out/checkpoint-1880" 

    ### Load base model ###

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,    
        device_map={"": accelerator.process_index},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16,
        # does not work yet
        # attn_implementation="flash_attention_2",          
    )

    tokenizer = 

    model.config.eos_token_id = tokenizer.eos_token_id


    ### Add adapters to model ###
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) 

    lora_config = LoraConfig(
        r=32, 
        lora_alpha=32, 
        target_modules = [ "q_proj", "k_proj", "v_proj", "dense" ],
        modules_to_save = ["lm_head", "embed_tokens"],
        lora_dropout=0.1, 
        bias="none", 
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    model.config.use_cache = False

    ### Print stats ###
    if accelerator.is_main_process:
        model.print_trainable_parameters()



    ### Load dataset ###
        
    dataset = load_from_disk(FINETUNE_PARAMETERS["dataset_path"])


    ### Run training
    
    steps_per_epoch=len(dataset_tokenized["train"])//(accelerator.num_processes*FINETUNE_PARAMETERS["bs"]*FINETUNE_PARAMETERS["ga_steps"])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=FINETUNE_PARAMETERS["bs"],
        per_device_eval_batch_size=FINETUNE_PARAMETERS["bs_eval"],
        evaluation_strategy="steps",
        logging_steps=1,
        eval_steps=steps_per_epoch//2,    # 2 evals per epoch
        save_steps=steps_per_epoch,     # save once per epoch
        gradient_accumulation_steps=FINETUNE_PARAMETERS["ga_steps"],
        num_train_epochs=FINETUNE_PARAMETERS["epochs"],
        lr_scheduler_type="constant",
        optim="paged_adamw_32bit",      # val_loss will go nan with paged_adamw_8bit
        learning_rate=FINETUNE_PARAMETERS["lr"],
        group_by_length=False,
        bf16=True,        
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=collate,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["test"],
    )

    if accelerator.is_main_process:
        wandb_experiment_name = FINETUNE_PARAMETERS["modelpath"]+"_"+run_id
        config_for_logging = FINETUNE_PARAMETERS.update({
            "run_id": run_id,
            "output_dir": output_dir,
            "lora_config": lora_config, 
            "training_args": args,
            "GPUs": accelerator.num_processes,
        })

        run = wandb.init(
            project=project_name,
            name=wandb_experiment_name,
            config=config_for_logging,
        )
        run.log_code()

    trainer.train()
    trainer.model.save_pretrained(output_dir)

    # clear memory
    del model
    del trainer


    ### Merge Qlora with model and save ###

    base_model = AutoModelForCausalLM.from_pretrained(
        FINETUNE_PARAMETERS["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Add ChatML template to tokenizer
    tokenizer.chat_template="{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    base_model.config.eos_token_id = tokenizer.eos_token_id

    # Set a default Generation configuration: Llama precise
    generation_config = GenerationConfig(
        max_new_tokens=100, 
        temperature=0.7,
        top_p=0.1,
        top_k=40,
        repetition_penalty=1.18,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Load LoRA and merge
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model = model.merge_and_unload()

    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(output_dir)
    generation_config.save_pretrained(output_dir)

    # copy inference script
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "/opt/ml/model/code/requirements.txt",
    )