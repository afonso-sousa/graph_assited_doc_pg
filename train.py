import argparse
import json
import logging
import math
import os
from collections import defaultdict
from functools import partial

import datasets
import evaluate
import spacy
import torch
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BigBirdPegasusConfig,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)

from graph_bigbird import GraphBigBirdPegasusForConditionalGeneration
from graph_collator import GraphDataCollator

logger = get_logger(__name__)
nlp = spacy.load("en_core_web_lg")


def preprocess_function(examples, tokenizer, args):
    inputs = examples["source"]
    targets = examples["target"]

    # Tokenize source texts using SpaCy for proper matching with semantic graphs
    split_sources = [[token.text for token in nlp(text)] for text in inputs]

    # Tokenize inputs and targets
    tokenized_inputs = tokenizer(
        split_sources,
        max_length=args.max_seq_length,
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    tokenized_targets = tokenizer(
        targets,
        max_length=args.max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    model_inputs = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }

    # Graph preprocessing
    if args.with_graph:
        graph_edges_batch = []

        for b, graph in enumerate(examples["semantic_graph"]):
            word_ids = tokenized_inputs.word_ids(batch_index=b)

            # Map word indices to all subword token indices
            word_to_tokens = defaultdict(list)
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is not None:
                    word_to_tokens[word_idx].append(token_idx)

            token_edges = set()
            connected_word_indices = set()

            # Inter-word edges
            for edge in graph.get("edges", []):
                from_node_idx = edge["from"]
                to_node_idx = edge["to"]
                from_token_ids = word_to_tokens.get(from_node_idx, [])
                to_token_ids = word_to_tokens.get(to_node_idx, [])

                if from_token_ids and to_token_ids:
                    connected_word_indices.add(from_node_idx)
                    connected_word_indices.add(to_node_idx)
                    for tid1 in from_token_ids:
                        for tid2 in to_token_ids:
                            token_edges.add((tid1, tid2))
                            token_edges.add((tid2, tid1))  # bidirectional

            # Intra-word edges
            for word_idx in connected_word_indices:
                token_indices = word_to_tokens[word_idx]
                for i in range(len(token_indices)):
                    for j in range(i + 1, len(token_indices)):
                        tid1 = token_indices[i]
                        tid2 = token_indices[j]
                        token_edges.add((tid1, tid2))
                        token_edges.add((tid2, tid1))  # symmetric

            graph_edges_batch.append(list(token_edges))

        model_inputs["graph_edges"] = graph_edges_batch

    return model_inputs


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
        default="google/bigbird-pegasus-large-arxiv",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Use the custom BigBird variant with graph support.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=64,
        help="The block size for the BigBird attention mechanism.",
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

    config = BigBirdPegasusConfig.from_pretrained(args.model_name_or_path)
    config.block_size = args.block_size

    if args.with_graph:
        model = GraphBigBirdPegasusForConditionalGeneration.from_pretrained(
            args.model_name_or_path, config=config
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, config=config
        )

    dataset = load_from_disk(args.dataset_name)

    # import numpy as np
    # from datasets import DatasetDict

    # def check_dataset_nulls(dataset_dict: DatasetDict) -> dict:
    #     """
    #     Checks for null/empty/NaN values in all splits of a DatasetDict.
    #     Returns a summary dictionary with problematic columns and counts.
    #     """
    #     results = {}

    #     for split_name, dataset in dataset_dict.items():
    #         # Check for None, empty strings, or NaN
    #         def has_null(example):
    #             return {
    #                 col: (
    #                     example[col] is None
    #                     or (
    #                         isinstance(example[col], str) and example[col].strip() == ""
    #                     )
    #                     or (
    #                         isinstance(example[col], (int, float))
    #                         and np.isnan(example[col])
    #                     )
    #                 )
    #                 for col in example.keys()
    #             }

    #         # Apply null check
    #         null_results = dataset.map(
    #             has_null,
    #             batched=False,
    #             remove_columns=dataset.column_names,
    #         )

    #         # Summarize results
    #         split_stats = {
    #             col: sum(null_results[col]) for col in null_results.column_names
    #         }

    #         results[split_name] = split_stats

    #     return results

    # # Usage during breakpoint()
    # null_summary = check_dataset_nulls(dataset)
    # print("Null/Empty/NaN Summary:\n", null_summary)
    # breakpoint()

    column_names = dataset["train"].column_names

    # dataset["train"] = dataset["train"].select(range(100))
    # dataset["validation"] = dataset["validation"].select(range(100))

    preprocess_partial = partial(preprocess_function, tokenizer=tokenizer, args=args)

    train_dataset = dataset["train"].map(
        preprocess_partial,
        batched=True,
        remove_columns=column_names,
    )
    eval_dataset = dataset["validation"].map(
        preprocess_partial,
        batched=True,
        remove_columns=column_names,
    )

    if args.with_graph:

        def is_valid_graph_edge_entry(
            example, max_token_length: int, threshold: int = 0
        ):
            """
            Returns True if the example's 'graph_edges' are valid:
            - At least `threshold` edges
            - All token indices in-bounds
            """
            edges = example.get("graph_edges", [])
            if not isinstance(edges, list) or len(edges) <= threshold:
                return False

            for src, tgt in edges:
                if not (0 <= src < max_token_length) or not (
                    0 <= tgt < max_token_length
                ):
                    return False

            return True

        print(f"{len(train_dataset)} training examples before filtering")
        train_dataset = train_dataset.filter(
            lambda example: is_valid_graph_edge_entry(example, args.max_seq_length),
            desc="Filtering out invalid graph edge examples",
        )
        print(f"{len(train_dataset)} training examples after filtering")

        print(f"{len(eval_dataset)} evaluation examples before filtering")
        eval_dataset = eval_dataset.filter(
            lambda example: is_valid_graph_edge_entry(example, args.max_seq_length),
            desc="Filtering out invalid graph edge examples",
        )
        print(f"{len(eval_dataset)} evaluation examples after filtering")

    eval_dataset = eval_dataset.select(range(0, 100, 10))

    # Data collator
    if args.with_graph:
        data_collator = GraphDataCollator(tokenizer, model=model)
    else:
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
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
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
                if args.with_graph:
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        graph_edges=batch.get("graph_edges", None),
                        max_length=args.max_seq_length,
                    )
                else:
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
