# Improving Long-Context Retrieval with Multi-Prefix Embedding

Project page: [https://farmerswrap.github.io/mpe/](https://farmerswrap.github.io/mpe/)

## Comparison with Perplexity Contextualized Embeddings

We compare MPE against [Perplexity Contextualized Embeddings](https://docs.perplexity.ai/docs/embeddings/contextualized-embeddings), a proprietary API that also produces context-aware chunk embeddings. We evaluate both the 0.6B and 4B variants on four LongEmbed datasets (nDCG@10):

| Method | QMSum | 2WikiMQA | SummScreen | NarrativeQA |
|---|---|---|---|---|
| Base no-chunk | 0.4504 | 0.8470 | 0.9662 | 0.5088 |
| Base chunk@64 (MV-Infer) | 0.5659 | 0.9021 | 0.9608 | 0.6090 |
| Single-vec fine-tuned | 0.5635 | 0.8648 | **0.9800** | 0.5435 |
| MV-Train+Infer | 0.5660 | 0.9427 | 0.8993 | 0.5553 |
| MPE Fixed-64 | 0.6521 | **0.9475** | 0.9278 | 0.5802 |
| Pplx Ctx 0.6b | 0.4452 | 0.8985 | 0.9363 | 0.5509 |
| Pplx Ctx 4b | **0.7116** | 0.9312 | 0.9734 | **0.6026** |

Perplexity Ctx 0.6b performs at base-model level; MPE Fixed-64 (0.6B) wins on 3 of 4 datasets. The larger Pplx Ctx 4b is strongest overall but uses a 7x larger model.

## Reproduction

All results in the paper can be reproduced end-to-end using the scripts in [`examples/mpe/`](https://github.com/texttron/tevatron/tree/main/examples/mpe). We also release pretrained LoRA adapters and precomputed embeddings on [HuggingFace](https://huggingface.co/FarmersWrap/mpe-repro) so you can skip training and/or encoding and jump straight to evaluation.

### Requirements

```bash
pip install transformers "datasets==2.21.0" peft faiss-cpu pyserini
git clone https://github.com/texttron/tevatron && cd tevatron && pip install -e .
```

Hardware: 8 GPUs recommended (tested on 8x RTX 5090). Training uses `torchrun`; corpus encoding is sharded across GPUs in parallel.

### Quick Start (Full Pipeline)

```bash
cd examples/mpe
python 00_prepare_data.py          # Download & prepare MLDR-en and BrowseComp-Plus data
bash 01_train.sh                   # Train 4 LoRA adapters on MLDR-en (1 epoch each)
bash 02_eval_mldr_en.sh            # Encode, search, evaluate on MLDR-en
bash 03_eval_browsecomp_plus.sh    # Encode, search, evaluate on BrowseComp-Plus
bash 04_eval_longembed.sh          # Encode, search, evaluate on 4 LongEmbed datasets
python 05_collect_results.py       # Aggregate all scores into a single table
```

### Pretrained Models

Four LoRA adapters (base model: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) are available at [`FarmersWrap/mpe-repro`](https://huggingface.co/FarmersWrap/mpe-repro):

| Model | Training Args | Used By |
|---|---|---|
| `models/nochunk-epoch1` | `--passage_chunk_size 0` | Single-vector, MaxP |
| `models/maxp-train-epoch1` | `--passage_chunk_size 64 --passage_chunk_independent` | MaxP-Train |
| `models/fixed-64-epoch1` | `--passage_chunk_size 64` | MPE Fixed-64 |
| `models/prand-32to1024-epoch1` | `--passage_chunk_size_range 32,1024` | MPE-Rand |

### Precomputed Embeddings

Precomputed query and corpus embeddings for all 5 configurations across all benchmarks are available under `encode/` in the same HuggingFace repo. To skip training and encoding and go directly to search + evaluation:

```bash
# Download embeddings
pip install huggingface_hub
huggingface-cli download FarmersWrap/mpe-repro --local-dir mpe-repro

# Run search + eval on MLDR-en (example for MPE-Rand)
python -m tevatron.retriever.driver.search \
    --query_reps mpe-repro/encode/mldr-en/mpe-rand-32to1024/queries.pkl \
    --passage_reps "mpe-repro/encode/mldr-en/mpe-rand-32to1024/corpus.*.pkl" \
    --depth 100 --batch_size 64 --save_text \
    --chunked --chunk_multiplier 10 \
    --save_ranking_to ranking.txt

python -m tevatron.utils.format.convert_result_to_trec \
    --input ranking.txt --output ranking.trec --remove_query

python -m pyserini.eval.trec_eval \
    -m ndcg_cut.10 -m recall.100 \
    data/qrels.tsv ranking.trec
```

### Available Embeddings

| Benchmark | Configs | Path |
|---|---|---|
| MLDR-en | all 5 | `encode/mldr-en/{config}/` |
| BrowseComp-Plus | all 5 | `encode/browsecomp-plus/{config}/` |
| NarrativeQA | all 5 | `encode/longembed/narrativeqa/{config}/` |
| 2WikiMQA | all 5 | `encode/longembed/2wikimqa/{config}/` |
| SummScreen | all 5 | `encode/longembed/summ_screen_fd/{config}/` |
| QMSum | all 5 | `encode/longembed/qmsum/{config}/` |

Configs: `single-vector`, `maxp`, `maxp-train`, `mpe-fixed64`, `mpe-rand-32to1024`. Each contains `queries.pkl` and `corpus.{0..7}.pkl`.

### Output Structure

```
{EXP_ROOT}/
├── models/                         # LoRA checkpoints
│   ├── nochunk-epoch1/
│   ├── maxp-train-epoch1/
│   ├── fixed-64-epoch1/
│   └── prand-32to1024-epoch1/
├── encode/                         # Pickled embeddings
│   ├── mldr-en/{config}/
│   ├── browsecomp-plus/{config}/
│   └── longembed/{dataset}/{config}/
├── results/                        # Rankings and TREC files
│   ├── mldr-en/{config}/
│   ├── browsecomp-plus/{config}/
│   └── longembed/{dataset}/{config}/
└── data/                           # Evaluation data
    ├── corpus.jsonl, queries.jsonl, qrels.tsv
    └── longembed/{dataset}/
```
