# SAPG: Semantically-Aware Paraphrase Generation with AMR Graphs


This repo contains the code for the paper SAPG: Semantically-Aware Paraphrase Generation with AMR Graphs, by Afonso Sousa & Henrique Lopes Cardoso (accepted at ICAART 2025).

Automatically generating paraphrases is crucial for various natural language processing tasks. Current approaches primarily try to control the surface form of generated paraphrases by resorting to syntactic graph structures. However, paraphrase generation is rooted in semantics, but there are almost no works trying to leverage semantic structures as inductive biases for the task of generating paraphrases. We propose SAPG, a semantically-aware paraphrase generation model, which encodes Abstract Meaning Representation (AMR) graphs into a pretrained language model using a graph neural network-based encoder. We demonstrate that SAPG enables the generation of more diverse paraphrases by transforming the input AMR graphs, allowing for control over the output generations' surface forms rooted in semantics. This approach ensures that the semantic meaning is preserved, offering flexibility in paraphrase generation without sacrificing fluency or coherence. Our extensive evaluation on two widely-used paraphrase generation datasets confirms the effectiveness of this method.

## Installation

### Create conda environment
To set up a fresh conda environment with all required dependencies, run:
```bash
conda env create -f environment.yml
```

### Download pseudo-semantic graphs
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
To train/test SAPG or any other model refered to in the paper you can run the corresponding script. For example:
```
sh ./scripts/train_graph_amr.sh
```

```
sh ./scripts/test_graph_amr.sh
```
