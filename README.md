# Graph-assisted Paraphrase Generation for Long Texts

TL;DR: We extend BigBird-Pegasus with a graph-aware attention mask. Instead of random block connections, we wire blocks using the connectivity of a pseudo-semantic graph extracted from the input, enabling long-context paraphrasing with structured inductive bias.

```
⚠️ Disclaimer
This project is ⚠️ Disclaimer
This project is experimental. In our initial trials, we did not achieve strong results compared to the baseline BigBird-Pegasus.
That said, the idea of replacing random attention with graph-assisted masks may still be useful, and this code could serve as a starting point for others exploring long-context paraphrasing with structural priors.. In our initial trials, we did not achieve strong results compared to the baseline BigBird-Pegasus.
That said, the idea of replacing random attention with graph-assisted masks may still be useful, and this code could serve as a starting point for others exploring long-context paraphrasing with structural priors.
```

Features:

✅ Plug-in graph mask replaces BigBird random sub-mask

✅ Works with pseudo-semantic graphs (cheap to compute, no gold AMR required)

✅ Scripts for data prep, training, testing, and inference

✅ Fair comparison toggles: with / without graph

## Installation

### 1. Create conda environment
To set up a fresh conda environment with all required dependencies, run:
```bash
conda env create -f environment.yml
```

### 2. Download pseudo-semantic graphs
We use pseudo-semantic graphs from [sem_para_gen](https://github.com/afonso-sousa/sem_para_gen.git).
1. Copy the /pseudo_semantic_graph folder to your project's root directory.
2. Note: You'll need to manually install `coreferee` for coreference resolution due to compatibility issues (see [this issue](https://github.com/richardpaulhudson/coreferee/issues/29)).


## Preprocess data

### Data Source
Original data collected from [Par3 repository](https://github.com/katherinethai/par3).

### Build Paraphrase Corpus
1. Place data in `\data` and navigate there:
```
cd data
```
2. Prepare dataset:
```
python prepare_paraphrase_dataset.py
```
3. Normalize the data:
```
python normalize_data.py
```

4. **(Optional)** Generate graph-enhanced dataset:
```
python create_data_with_graph.py
```

5. Split data:
```
sh ./scripts/split_dataset.sh
```
or
```
sh ./scripts/split_dataset_with_graph.sh
```
_Note: As it is, the former script randomly splits the data, while the latter uses the indices generated from the former scripts' execution to split the data for fair comparison._

## Train and test models
To train/test the model run the respective script at `./scripts`. For example:
```
sh ./scripts/train_with_graph.sh
```

```
sh ./scripts/test_with_graph.sh
```

### How the Graph Mask Works (1-minute read)

- We bucket tokens into blocks (BigBird style).

- For each example, we build a block-level adjacency list from the pseudo-semantic graph edges (token → token → block).

- During training, attention’s random block connections are replaced by these graph connections.

- During evaluation/inference, we return a zero mask (mirroring BigBird’s no-randomness at eval), ensuring deterministic behavior and comparable latency.