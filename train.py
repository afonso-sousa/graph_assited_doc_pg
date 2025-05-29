import argparse
import json
import logging
import math
import os

import datasets
import evaluate
import torch
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, SchedulerType, get_scheduler)

from custom_model import CustomLongT5ForConditionalGeneration

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a seq2seq model")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/long-t5-tglobal-base",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Use the custom LongT5 variant with graph support.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for input and target.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize the accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if args.with_graph:
        from transformers import LongT5Config
        config = LongT5Config.from_pretrained(args.model_name_or_path)
        model = CustomLongT5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    dataset = load_from_disk(args.dataset_name)

    column_names = dataset["train"].column_names

    # dataset["train"] = dataset["train"].select(range(100))
    # dataset["validation"] = dataset["validation"].select(range(100))

    def preprocess_function(examples):
        inputs = examples["source"]
        targets = examples["target"]

        # Tokenize inputs and targets
        tokenized_inputs = tokenizer(
            inputs,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
            # is_split_into_words=True,
        )

        tokenized_targets = tokenizer(
            targets,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        breakpoint()

        # Graph preprocessing
        if args.with_graph:
            graph_edges_batch = []

            for b, graph in enumerate(examples["semantic_graph"]):
                input_text = inputs[b]
                offset_mapping = tokenized_inputs["offset_mapping"][b].tolist()
                important_edges = []
                breakpoint()
                # === Step 1: Optional Node Filtering Heuristic ===
                # Keep only nodes with at least one edge and word length > 2
                node_indices = []
                for node in graph["nodes"]:
                    if len(node["index"]) > 0 and any(len(w) > 2 for w in node["word"]):
                        node_indices.append(node["index"])

                # === Step 2: Connect All Node Pairs as Edges ===
                for i, node_a in enumerate(node_indices):
                    for j in range(i + 1, len(node_indices)):
                        node_b = node_indices[j]
                        for idx_a in node_a:
                            for idx_b in node_b:
                                if idx_a != idx_b:
                                    important_edges.append((idx_a, idx_b))

                # === Step 3: Map Word-Level Indices to Token Indices ===
                char_to_token_map = {}
                for tok_id, (start, end) in enumerate(offset_mapping):
                    for i in range(start, end):
                        char_to_token_map[i] = tok_id

                # Convert word-based edges to subword token edges
                token_edges = set()
                words = input_text.split()

                for idx_a, idx_b in important_edges:
                    try:
                        word_a = words[idx_a]
                        word_b = words[idx_b]

                        # Find character spans for the words
                        start_a = input_text.find(word_a)
                        start_b = input_text.find(word_b)

                        # Find all token indices overlapping with these word spans
                        tokens_a = set()
                        tokens_b = set()
                        for i in range(len(input_text)):
                            if start_a <= i < start_a + len(word_a) and i in char_to_token_map:
                                tokens_a.add(char_to_token_map[i])
                            if start_b <= i < start_b + len(word_b) and i in char_to_token_map:
                                tokens_b.add(char_to_token_map[i])

                        # Create full cross-product edges
                        for ta in tokens_a:
                            for tb in tokens_b:
                                token_edges.add((ta, tb))
                                token_edges.add((tb, ta))  # bidirectional

                    except IndexError:
                        continue  # skip edge if word index is out of range

                graph_edges_batch.append(list(token_edges))

            model_inputs["graph_edges"] = graph_edges_batch

        return model_inputs

    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
    )
    eval_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with the accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                (step + 1) % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(args.checkpointing_steps, int):
                if completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        predictions = []
        references = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=args.max_seq_length,
                )
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    batch["labels"], skip_special_tokens=True
                )
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)

        # Compute sacrebleu score
        bleu_score = evaluate.load("sacrebleu").compute(
            predictions=predictions,
            references=[[ref] for ref in references],  # sacrebleu expects list of lists
        )
        accelerator.print(
            f"epoch {epoch}:",
            {
                "bleu": bleu_score["score"],
            },
        )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        with open(os.path.join(args.output_dir, "eval_scores.json"), "w") as f:
            json.dump(
                {
                    "bleu": bleu_score["score"],
                },
                f,
            )


if __name__ == "__main__":
    main()
