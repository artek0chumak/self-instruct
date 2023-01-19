import json
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import transformers
from tqdm.auto import tqdm


class SelfInstructDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: Path, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
        
        self.text_data = [json.loads(line) for line in open(data_file)]
        tokenized_prompt = tokenizer([d["prompt"] for d in self.text_data])["input_ids"]
        tokenized_completion = tokenizer([d["completion"] for d in self.text_data])["input_ids"]

        self.data = []
        data_iters = zip(tokenized_prompt, tokenized_completion)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        pbar = tqdm(data_iters, total=len(tokenized_prompt)) if local_rank == 0 else data_iters
        for prompt, completion in pbar:
            if len(prompt) + len(completion) < model.config.n_positions:
                rest = [model.config.eos_token_id for _ in range(model.config.n_positions - len(prompt) - len(completion))]
                input_ids = prompt + completion + rest
                attention_mask = [1 for _ in range(len(prompt) + len(completion))] + [0 for _ in range(len(rest))]
                labels = [-100 for _ in range(len(prompt) - 1)] + completion + [-100 for _ in range(len(rest) + 1)]
            else:
                prompt = prompt[-(model.config.n_positions - len(completion)):]
                input_ids = prompt + completion
                attention_mask = [1 for _ in range(len(prompt) + len(completion))]
                labels = [-100 for _ in range(len(prompt) - 1)] + completion + [-100]
            self.data.append(
                {
                    "input_ids": torch.Tensor(input_ids).to(torch.long),
                    "attention_mask": torch.Tensor(attention_mask).to(torch.long),
                    "labels": torch.Tensor(labels).to(torch.long),
                }
            )
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_file", default="data/finetuning/self_instruct_221203/gpt3_finetuning_data.jsonl")
    parser.add_argument("--config_file", default="training_config.json")
    parser.add_argument("--model_name", default="EleutherAI/gpt-j-6B")
    return parser.parse_args()


def main(args: Namespace):
    training_args = transformers.HfArgumentParser(transformers.Seq2SeqTrainingArguments).parse_json_file(args.config_file)[0]
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SelfInstructDataset(args.data_file, tokenizer, model)
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
