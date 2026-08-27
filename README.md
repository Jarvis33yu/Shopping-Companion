# Shopping Companion

Official implementation of **Shopping Companion: Benchmarking and Training LLM Agents for Long-Horizon Preference-Grounded E-Commerce Tasks**.

This release contains the benchmark construction and evaluation pipeline, agent implementations, tool servers, and the agentic reinforcement-learning components. The dataset is available on the [Hugging Face Hub](https://huggingface.co/datasets/yuzhan2205/Shopping-companion).

## Repository layout

```text
.
├── src/                 # data processing, agents, tools, servers, and evaluation
├── agentic_rl/          # multi-turn GRPO configuration, rewards, and RL utilities
├── resource/            # chat templates
├── run_*                # example preprocessing, serving, and evaluation scripts
├── install.sh
└── requirements.txt
```

The RL implementation builds on [verl](https://github.com/volcengine/verl). Its upstream source is intentionally not vendored here; install a compatible verl environment before running `agentic_rl/run_grpo.sh`.

## Installation

Python 3.11 is recommended.

```bash
bash install.sh
```

The code reads model-provider credentials from environment variables. For OpenAI-compatible endpoints, set:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

Do not commit credentials or private endpoint configuration to the repository.

## Data and indexes

The Shopping Companion dataset is hosted at:

> [yuzhan2205/Shopping-companion](https://huggingface.co/datasets/yuzhan2205/Shopping-companion)

Download the required files into `data/`, which is excluded from version control. Detailed preparation instructions and additional dataset artifacts will be documented in the dataset card as they are released.

To build the memory index after placing the conversation JSONL file at the path expected by the script:

```bash
bash run_mem_preprocess.sh
```

After downloading the product catalog as `data/products.jsonl`, build and start the local product index with:

```bash
bash run_product_preprocess.sh
```

If the catalog is stored elsewhere, set `PRODUCTS_FILE=/path/to/products.jsonl`.

## Running agents and evaluation

Start the memory and product services first, then adapt the model names and input paths in the example scripts:

```bash
bash run_agent.sh
```

The tool service endpoints default to localhost and can be overridden with the corresponding environment variables, including `MEM_BASE_URL`, `PRODUCT_BASE_URL`, and `REWARD_BASE_URL`.

## Agentic RL

Prepare the train and validation Parquet files with `agentic_rl/data_preprocess.py`, then run GRPO from the RL directory:

```bash
cd agentic_rl
MODEL_PATH=/path/to/model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/validation.parquet \
OUTPUT_DIR=/path/to/checkpoints \
bash run_grpo.sh
```

`NUM_GPUS` defaults to 8. Additional Hydra overrides may be appended to the command.

## Citation

Citation information will be added with the camera-ready publication metadata.

## License

License information will be added before the final public release.
