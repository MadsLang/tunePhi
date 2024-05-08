import os
from itertools import chain
from functools import partial
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import FINETUNE_PARAMETERS

class DataProcessor:

    def __init__(self): 
        # empty list to save remainder from batches to use in next batch
        remainder = {"input_ids": [], "attention_mask": []}

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(FINETUNE_PARAMETERS['model_id'])
        self.tokenizer = AutoTokenizer.from_pretrained(FINETUNE_PARAMETERS['model_id'])
        self.tokenizer.model_max_length = FINETUNE_PARAMETERS['max_length']

        # Add ChatML format
        self.tokenizer.add_tokens(["<|im_start|>", "<PAD>"])
        self.tokenizer.pad_token = "<PAD>"
        self.tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.prompt_template = "<|im_start|>user\n{prompt}<|im_end|>\n <|im_start|>assistant\n{completion}<|im_end|>"

    def add_template(self, sample): 
        sample["text"] = self.prompt_template.format(
            prompt=sample["prompt"],
            completion=sample["completion"],
            eos_token=self.tokenizer.eos_token
        )
        return sample

    def chunk(self, sample, chunk_length=2048):
        # define global remainder variable to save remainder from batches to use in next batch
        global remainder
        # Concatenate all texts and add remainder from previous batch
        concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
        concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
        # get total number of tokens for batch
        batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

        # get max number of chunks for batch
        if batch_total_length >= chunk_length:
            batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
            for k, t in concatenated_examples.items()
        }
        # add remainder to global variable for next batch
        remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
        # prepare labels
        result["labels"] = result["input_ids"].copy()
        return result


    def prepare_dataset(self, dataset_name: str) -> str:
        dataset = load_dataset("json", data_files=f"/opt/ml/processing/input/data/{dataset_name}", split="train") # format: {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

        print(f"Adding template to {len(dataset)} samples")
        dataset = dataset.map(self.add_template, remove_columns=list(dataset.features))

        # tokenize and chunk dataset
        print("Tokenizing samples")
        lm_dataset = dataset.map(
            lambda sample: self.tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
        ).map(
            partial(self.chunk, chunk_length=5),
            batched=True,
        )

        # Print total number of samples
        print(f"Total number of samples: {len(lm_dataset)}")

        # Save data
        lm_dataset.save_to_disk(f'/opt/ml/processing/output/train/{dataset_name}')
        print("Succesfully processed and save dataset!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetname", type=str)
    args, _ = parser.parse_known_args()
    print(f"Received arguments: {args}")


    dp = DataProcessor()
    dp.prepare_dataset(dataset_name=args['datasetname'])


