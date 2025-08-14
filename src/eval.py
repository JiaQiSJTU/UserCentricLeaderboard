# encoding = "utf-8"

from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from tqdm import tqdm
import transformers
import random
from typing import List

from utils.seeding import set_global_seed
set_global_seed(42)
from utils.prompt import REWARD_MODEL_SYSTEM_PROMPT, PAIR_REWARD_MODEL_PROMPT
from utils import make_up_dialogue
from data import PreferenceDataset


class RMEvaluator:
    """Single-response reward model evaluator."""

    def __init__(self, model_name: str, batch_size: int = 1, use_criteria: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_criteria = use_criteria

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if "Tulu-3-8B-SFT-RM-RB2" in model_name or "Llama-3.1-8B-Base-RM-RB2" in model_name:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, revision="2", device_map="auto")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=1,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        self.model.eval()

    # ---------------------  collator  ---------------------
    def collator(self, batch):
        chosen_input_list, rejected_input_list = [], []

        for query, model_a, model_b, criteria, preference in zip(
            batch["query"], batch["model_a"], batch["model_b"], batch["criteria"], batch["preference"]
        ):
            chosen_model, rejected_model = (model_a, model_b) if preference == "A" else (model_b, model_a)

            if self.use_criteria:
                chosen = [
                    {
                        "role": "system",
                        "content": REWARD_MODEL_SYSTEM_PROMPT.format(
                            criteria="\n".join([f"* {c}" for c in criteria])
                        ),
                    }
                ]
                rejected = [
                    {
                        "role": "system",
                        "content": REWARD_MODEL_SYSTEM_PROMPT.format(
                            criteria="\n".join([f"* {c}" for c in criteria])
                        ),
                    }
                ]

                chosen += [{"role": item["role"], "content": item["content"]} for item in eval(chosen_model)]
                rejected += [{"role": item["role"], "content": item["content"]} for item in eval(rejected_model)]
            else:
                chosen = [{"role": item["role"], "content": item["content"]} for item in eval(chosen_model)]
                rejected = [{"role": item["role"], "content": item["content"]} for item in eval(rejected_model)]

            chosen_input_list.append(self.tokenizer.apply_chat_template(chosen, tokenize=False))
            rejected_input_list.append(self.tokenizer.apply_chat_template(rejected, tokenize=False))

        chosen_ids = self.tokenizer(chosen_input_list, return_tensors="pt", padding=True, truncation=False)
        rejected_ids = self.tokenizer(rejected_input_list, return_tensors="pt", padding=True, truncation=False)

        return {
            "chosen_input_ids": chosen_ids.input_ids,
            "chosen_attention_mask": chosen_ids.attention_mask,
            "rejected_input_ids": rejected_ids.input_ids,
            "rejected_attention_mask": rejected_ids.attention_mask,
        }

    # ---------------------  run  ---------------------
    def run(self, eval_data_file: str, output_file: str):
        dataset = PreferenceDataset(eval_data_file)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        start_idx, accuracy, pass_rate = 0, [], []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    sample = json.loads(line.strip())
                    start_idx += 1
                    accuracy.append(sample["accuracy"])
                    pass_rate.append(sample["pass_rate"])

        output_f = open(output_file, "a+", encoding="utf-8")

        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            if (i + 1) * self.batch_size <= start_idx:
                continue

            tokenized = self.collator(batch)
            tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}

            with torch.no_grad():
                rewards_chosen = self.model(
                    input_ids=tokenized["chosen_input_ids"],
                    attention_mask=tokenized["chosen_attention_mask"],
                    return_dict=True,
                )["logits"]

                rewards_rejected = self.model(
                    input_ids=tokenized["rejected_input_ids"],
                    attention_mask=tokenized["rejected_attention_mask"],
                    return_dict=True,
                )["logits"]

                logits = torch.stack((rewards_chosen, rewards_rejected)).mean(dim=2).softmax(dim=0).T

            for logit, idx in zip(logits, batch["idx"]):
                cur_acc = 1.0 if logit[0] > logit[1] else 0.0
                accuracy.append(cur_acc)
                pass_rate.append(1.0)

                output_f.write(
                    json.dumps({"id": idx, "pass_rate": 1.0, "accuracy": cur_acc, "logits": logit.tolist()})
                    + "\n"
                )

            torch.cuda.empty_cache()

        print(f"pass rate: {sum(pass_rate)/len(pass_rate)}")
        print(f"accuracy: {sum(accuracy)/len(accuracy)}")


class RMPairEvaluator:
    """Pairwise reward model evaluator."""

    def __init__(self, model_name: str, batch_size: int = 1, use_criteria: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_criteria = use_criteria

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def collator(self, batch):
        inputs: List[str] = []
        for query, model_a, model_b, criteria, _ in zip(
            batch["query"], batch["model_a"], batch["model_b"], batch["criteria"], batch["preference"]
        ):
            cri_list = eval(criteria) if self.use_criteria else ["General user preference"]
            prompt = PAIR_REWARD_MODEL_PROMPT.format(
                criteria="\n".join([f"* {c}" for c in cri_list]),
                user_query=json.dumps({"User": query}),
                model_a=make_up_dialogue(eval(model_a)[:]),
                model_b=make_up_dialogue(eval(model_b)[:]),
            )
            messages = [{"role": "user", "content": prompt}]
            inputs.append(self.tokenizer.apply_chat_template(messages, tokenize=False))

        tokenized = self.tokenizer(inputs, padding="longest", return_tensors="pt", add_special_tokens=False)
        return tokenized, inputs

    def run(self, eval_data_file: str, output_file: str):
        dataset = PreferenceDataset(eval_data_file)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        print("here")
        start_idx, accuracy, pass_rate = 0, [], []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    sample = json.loads(line.strip())
                    start_idx += 1
                    accuracy.append(sample["accuracy"])
                    pass_rate.append(sample["pass_rate"])

        output_f = open(output_file, "a+", encoding="utf-8")

        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            if (i + 1) * self.batch_size <= start_idx:
                continue

            tokenized, _ = self.collator(batch)
            tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}

            with torch.no_grad():
                logits = self.model(**tokenized).logits
                softmax_logits = torch.softmax(logits, dim=-1)
                predicted_indices = torch.argmax(softmax_logits, dim=-1)

            for pred_idx, ref, idx in zip(predicted_indices, batch["preference"], batch["idx"]):
                cur_acc = 1.0 if (pred_idx == 0 and ref == "A") or (pred_idx == 1 and ref == "B") else 0.0
                accuracy.append(cur_acc)
                pass_rate.append(1.0)
                output_f.write(
                    json.dumps(
                        {
                            "id": idx,
                            "pass_rate": 1.0,
                            "accuracy": cur_acc,
                            "predicted_index": pred_idx.item(),
                            "softmax_probabilities": softmax_logits.tolist(),
                            "raw_logits": logits.tolist(),
                        }
                    )
                    + "\n"
                )
            torch.cuda.empty_cache()

        print(f"pass rate: {sum(pass_rate)/len(pass_rate)}")
        print(f"accuracy: {sum(accuracy)/len(accuracy)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--eval_data_file", type=str, default="data/test_set/test_1_original.jsonl")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    # parser.add_argument("--enable_thinking", type=int, default=-1)
    parser.add_argument("--use_criteria", type=int, default=1)
    parser.add_argument("--eval_mode", type=str, default="rm", help="rm_pair, rm")
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name.split("/")[-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, args.eval_data_file.split("/")[-1].split(".")[0] + f"_use_criteria_{args.use_criteria}.jsonl")
     
    # if args.enable_thinking == -1:
    #     enable_thinking = None
    # elif args.enable_thinking == 0:
    #     enable_thinking = False
    # else:
    #     enable_thinking = True
    
    user_criteria = bool(args.use_criteria)

    if args.eval_mode == "rm":
        evaluator = RMEvaluator(args.model_name, batch_size=1, use_criteria=user_criteria)
        evaluator.run(args.eval_data_file, output_file)
    elif args.eval_mode == "rm_pair":
        print("here")
        evaluator = RMPairEvaluator(args.model_name, batch_size=4, use_criteria=user_criteria)
        print("ready for eval")
        evaluator.run(args.eval_data_file, output_file)