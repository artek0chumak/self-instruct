import dataclasses
import json
import os
import random
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path

import deepspeed
import torch
import transformers
import wandb
from tqdm.auto import tqdm
import bitsandbytes as bnb


class SelfInstructDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: Path, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel, max_length=2048, number_of_repeates=6):
        
        self.text_data = [json.loads(line) for line in open(data_file)]
        tokenized_prompt = tokenizer([d["prompt"] for d in self.text_data])["input_ids"]
        tokenized_completion = tokenizer(
            [d["completion"].replace("<|endoftext|>", "\n\n") for d in self.text_data]
        )["input_ids"]
        data_iters = [(p, c) for p, c in zip(tokenized_prompt, tokenized_completion)]

        self.data = []
        for _ in range(number_of_repeates):
            tmp_input_ids = []
            tmp_labels = []
            tmp_attention_mask = []
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            pbar = tqdm(data_iters, total=len(tokenized_prompt)) if local_rank == 0 else data_iters
            for prompt, completion in pbar:
                if len(tmp_input_ids) + len(prompt) + len(completion) > max_length:
                    tmp_input_ids.extend(
                        model.config.eos_token_id for _ in range(max_length - len(tmp_input_ids))
                    )
                    tmp_attention_mask.extend(0 for _ in range(max_length - len(tmp_attention_mask)))
                    tmp_labels.extend(-100 for _ in range(max_length - len(tmp_labels)))
                    self.data.append(
                        {
                            "input_ids": torch.Tensor(tmp_input_ids).to(torch.long),
                            "attention_mask": torch.Tensor(tmp_attention_mask).to(torch.long),
                            "labels": torch.Tensor(tmp_labels).to(torch.long),
                        }
                    )
                    tmp_input_ids = []
                    tmp_attention_mask = []
                    tmp_labels = []
                    
                tmp_input_ids.extend(prompt)
                tmp_input_ids.extend(completion)

                tmp_attention_mask.extend(1 for _ in prompt)
                tmp_attention_mask.extend(1 for _ in completion)

                tmp_labels.extend(-100 for _ in prompt)
                tmp_labels.extend(completion)
                
                assert len(tmp_input_ids) == len(tmp_attention_mask), f"len(tmp_input_ids): {len(tmp_input_ids)}, len(tmp_attention_mask): {len(tmp_attention_mask)}"
                assert len(tmp_input_ids) == len(tmp_labels), f"len(tmp_input_ids): {len(tmp_input_ids)}, len(tmp_labels): {len(tmp_labels)}"
            
            if len(tmp_input_ids) > 0:
                tmp_input_ids.extend(
                    model.config.eos_token_id for _ in range(max_length - len(tmp_input_ids))
                )
                tmp_attention_mask.extend(0 for _ in range(max_length - len(tmp_attention_mask)))
                tmp_labels.extend(-100 for _ in range(max_length - len(tmp_labels)))
                self.data.append(
                    {
                        "input_ids": torch.Tensor(tmp_input_ids).to(torch.long),
                        "attention_mask": torch.Tensor(tmp_attention_mask).to(torch.long),
                        "labels": torch.Tensor(tmp_labels).to(torch.long),
                    }
                )
                
            random.shuffle(data_iters)
            
        for d in self.data:
            for k in ["input_ids", "attention_mask", "labels"]:
                assert d[k].size(0) == max_length, f"{k}: {d[k].size(0)}"

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
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    training_args = transformers.HfArgumentParser(transformers.Seq2SeqTrainingArguments).parse_json_file(args.config_file)[0]
    training_args.deepspeed["fp16"]["enabled"] = training_args.fp16
    training_args.deepspeed["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
    training_args.deepspeed["train_micro_batch_size_per_gpu"] = training_args.per_device_train_batch_size
    if local_rank == 0:
        wandb.init(
            "self-instruct-gpt-j",
            config=dataclasses.asdict(training_args),
        )
    
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=False)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = bnb.optim.AdamW8bit(
        [p for n, p in model.named_parameters() if (n.find("bias") > 0)],
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        weight_decay=training_args.weight_decay,
    )
    # optimizer = None
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = SelfInstructDataset(args.data_file, tokenizer, model)
    model_engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=dataset,
        config=training_args.deepspeed,
    )

    pbar = tqdm(training_dataloader) if local_rank == 0 else training_dataloader
    for idx, batch in enumerate(pbar):
        output = model_engine(**{k: v.to(device) for k, v in batch.items()})
        model_engine.backward(output.loss)
        model_engine.step()
        step_idx = idx // training_args.gradient_accumulation_steps
        if idx % training_args.gradient_accumulation_steps == 0:
            if local_rank == 0:
                pbar.set_description(f"Loss: {output.loss}")
                wandb.log({"Train Loss": output.loss})
            if step_idx > 0 and step_idx % training_args.save_steps == 0 and :
                model_engine.save_checkpoint(training_args.output_dir)
                
    model_engine.save_checkpoint(training_args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
