import argparse
import logging
import os
import re
from functools import partial

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
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
)

from graph_bigbird import GraphBigBirdPegasusForConditionalGeneration
from graph_collator import GraphDataCollator
from train import preprocess_function

logger = get_logger(__name__)


# Parsing input arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a BigBird model for text generation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The dataset split to evaluate on (e.g., 'test').",
    )
    parser.add_argument(
        "--with_graph",
        action="store_true",
        help="Use the custom BigBird variant with graph support.",
    )
    parser.add_argument(
        "--predict_output_file",
        type=str,
        default=None,
        help="Where to store the final predictions.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The metric to use for evaluation.",
    )
    parser.add_argument(
        "--predict_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible evaluation.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

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

    # If passed along, set the evaluation seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.predict_output_file is not None:
            os.makedirs(os.path.dirname(args.predict_output_file), exist_ok=True)
    accelerator.wait_for_everyone()

    logger.info(f"Loading '{args.dataset_name}' dataset")
    dataset = load_from_disk(args.dataset_name)

    logger.info(f"Loading checkpoint '{args.model_name_or_path}'")
    path_components = args.model_name_or_path.split(os.sep)
    model_hub_path = os.path.join(*path_components[1:])
    # Matches:
    #   _with_graph_1, _with_graph_2, ..., _with_graph_10, _1, _2, _3, etc.
    clean_name = re.sub(r"(?:_with_graph)?_\d+$", "", model_hub_path)
    tokenizer = AutoTokenizer.from_pretrained(clean_name)

    config = BigBirdPegasusConfig.from_pretrained(args.model_name_or_path)
    if args.with_graph:
        model = GraphBigBirdPegasusForConditionalGeneration.from_pretrained(
            args.model_name_or_path, config=config
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path, config=config
        )

    # Resize embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    eval_dataset = dataset[args.split]
    eval_dataset = eval_dataset.select(range(0, 100, 10))
    preprocess_partial = partial(preprocess_function, tokenizer=tokenizer, args=args)

    with accelerator.main_process_first():
        eval_dataset = eval_dataset.map(
            preprocess_partial,
            batched=True,
            remove_columns=dataset[args.split].column_names,
            desc="Running tokenizer on dataset",
        )

    # eval_dataset = eval_dataset.select(range(10))

    # DataLoader creation
    label_pad_token_id = -100

    if args.with_graph:
        data_collator = GraphDataCollator(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
        )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.predict_batch_size,
    )

    # Prepare everything with `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    total_batch_size = args.predict_batch_size * accelerator.num_processes

    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.predict_batch_size}")
    logger.info(
        f"  Total eval batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )

    metric = evaluate.load(args.metric)

    model.eval()

    gen_kwargs = {
        "max_length": args.max_seq_length,
        "num_beams": 4,
    }

    sources = []
    predictions = []
    references = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        with torch.no_grad():
            if args.with_graph:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    graph_edges=batch.get("graph_edges", None),
                    **gen_kwargs,
                )
            else:
                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            inputs = accelerator.pad_across_processes(
                batch["input_ids"], dim=1, pad_index=tokenizer.pad_token_id
            )
            inputs = accelerator.gather(inputs).cpu().numpy()

            labels = accelerator.pad_across_processes(
                batch["labels"], dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.gather(labels).cpu().numpy()
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            sources.extend(decoded_inputs)
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)

            metric.add_batch(
                predictions=decoded_preds,
                references=decoded_labels,
                sources=decoded_inputs,
            )

    eval_results = metric.compute()
    logger.info(f"Evaluation results: {eval_results}")

    if args.predict_output_file is not None:
        result = pd.DataFrame(
            {
                "source": sources,
                "prediction": predictions,
                "reference": references,
            }
        )
        result.to_csv(args.predict_output_file, index=False, sep="\t")


if __name__ == "__main__":
    main()
