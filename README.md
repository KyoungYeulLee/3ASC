# 3ASC 3.0 - MIL-GFE

Reference implementation of the model described in "Integrating Large Language Models and Multiple Instance Learning for Comprehensive Prioritization of SNVs and Structural Variants in Rare Mendelian Diseases"

This repository is the reproduction package. It contains the data loading, the
architecture, the training loop and the benchmark evaluation of the published
model, and nothing else.

## The model

Every patient is a *bag*; every variant they carry - SNV, CNV, STR, BND or INS -
is an *instance*. Two towers see the same bag:

- **MIL tower.** Each variant's numeric feature vector (40 SNV / 8 CNV / 3 STR /
  18 BND / 16 INS columns, listed in `conf/config.yaml`) is embedded by a
  feature-attention projection: every scalar feature is expanded over learned
  radial basis functions and the resulting per-feature tokens are mixed by
  self-attention. A 16-block transformer then lets the variants of a bag attend
  to one another, with a CLS token summarising the bag.
- **GFE tower** (global feature encoder). The patient's symptom embedding
  cross-attends against the disease embedding of every variant, producing a
  per-variant phenotype-match feature.

The GFE feature is fused into the variant embeddings before the transformer
runs, and only for variants whose disease is known - the rest keep their
unfused embedding and are flagged in a mask. The model outputs a bag logit
(does this patient have a causal variant) and one logit per variant, which is
the ranking score.

Training minimises the sum of a bag-level BCE, a focal loss over the instances,
and a pointwise RankNet loss that pushes the causal variant above the rest of
its bag.

## Layout

```
conf/config.yaml     every hyperparameter, path and feature column
core/data_model.py   PatientDataSet: the pickled cohort format
core/datasets.py     cohort loading and the torch Dataset
core/losses.py       focal and pointwise RankNet losses
core/metric.py       top-k recall
core/trainer.py      one training / validation epoch
model/layers.py      transformer blocks, feature-attention projection
model/networks.py    the two towers and the model they compose into
train.py             training entry point
evaluate.py          benchmark entry point
make_example_data.py writes a template cohort in the on-disk input format
scripts/embedding/   GFE embedding model: LoRA training and extraction
```

## Reproducing the model

```bash
uv sync                      # or: pip install -e .
python train.py              # writes <data_root_path>/checkpoints/<experiment_name>/
python evaluate.py           # scores conf/config.yaml's evaluate.predictor_path
```

Both read `conf/config.yaml`; pass `--config` to point at another file. Training
writes `epoch_<n>_model.pth` for every epoch plus `best_model.pth` (lowest
validation loss), and stores `config.yaml` and the fitted scalers next to them.
Evaluation rebuilds the model from that saved config, so it can never disagree
with how the model was trained.

Training and evaluation each run on a single GPU, one patient per step. Set
`model.device` / `evaluate.device` to `cpu` to run without one.

Evaluation reports, for the whole benchmark and for the patients whose causal
variant is of each type: top-k recall over CPRAs up to k = 100, the pooled AUROC
and PR-AUC over every variant of every patient, and the mean per-patient AUROC
and PR-AUC.

### How top-k recall is counted

The ranking is over CPRAs, not over rows: a variant can be reported once per
associated disease, so the rows sharing a CPRA are first collapsed to their
best-scoring one. A patient's recall at k is then the share of its causal CPRAs
that land in the top k, and the reported number is the mean over patients.

A patient carrying more causal CPRAs than k cannot reach a recall of 1 there, so
its recall at that k is treated as undefined and it is **left out of the mean**
rather than counted as a partial hit. A patient with two causal CPRAs therefore
contributes to the mean from k = 2 upwards but not to k = 1, and the number of
patients behind the reported mean grows with k. This matters when comparing
these numbers against a definition that keeps every patient at every k.

## Data

Variant coordinates are GRCh38 throughout. The `cpra` of a variant is matched as
a string - against the keys of the disease map, and when collapsing rows to
CPRAs during evaluation - so a cohort called against another build cannot be
mixed in. `PatientData.genome_build` records the build a cohort was called
against, but the model itself never reads it.

A cohort is a pickled `PatientDataSet`, so it has to be created with these
dataclasses importable as `core.data_model` - assemble it with this repository
on the path, or the pickle cannot be reconstructed.

Unpickling executes arbitrary code, so load only the cohort and embedding files
you built yourself or received from a source you trust.

### Inputs

| config key | what it is |
|---|---|
| `model.train.train_data_path` | pickled `PatientDataSet` to train on |
| `model.train.exclude_from_train_data_path` | pickled `PatientDataSet` whose samples are held out of training |
| `model.disease_embeddings_path` | pickled `{disease_id: embedding}`, plus an `unknown_disease` entry - the fallback for every variant the disease map leaves unmapped, and for every CNV |
| `model.patient_symptom_embeddings_path` | pickled `{sample_id: embedding}`, plus an `unknown_sample` entry - the fallback for a patient with no symptom embedding |
| `model.disease_cpra_map_path` | JSON `{sample_id: {cpra: disease_id}}` |
| `evaluate.benchmark_data_path` | pickled `PatientDataSet` to score |

An embedding is a `np.ndarray`; its width has to match
`model.parameters.mil_gfe.patient_embedding_dim` and `variant_embedding_dim`,
which the published run set to 4096 for both.

The cohorts are clinical genomic data from patients referred for rare-disease
diagnosis and are not distributed with this code; the paths in
`conf/config.yaml` are placeholders named after what each input holds.
`core/data_model.py` documents the format precisely enough to assemble an
equivalent dataset from your own cases.

The instance labels of the published training run additionally mark externally
curated BND and INS true calls as causal on patients the source cohort had left
negative. Those labels are part of the dataset that is assembled - `VariantData.y`
carries them - and not a step this code performs.

### Example data

To see the format concretely, `make_example_data.py` writes every input of the
table above for a made-up one-patient cohort - a template for assembling your
own data, not something to train on:

```bash
python make_example_data.py    # writes example_data/
```

The patient carries one variant of every type, the SNV being the causal one,
at random (seeded) coordinates. Each variant's feature row holds the columns of
`conf/config.yaml` in order, with a few illustrative values filled in and `"."`
- the marker the loader reads as missing - everywhere else.

The GFE vectors are random by default. Pass `--gfe-model` to point at a merged
fine-tuned embedding model and the invented records are embedded through
`scripts/embedding/extract_embeddings.py` instead, so they carry the same
instruct prefix and layout as the real inputs.

## The GFE embedding model

`model.disease_embeddings_path` and `model.patient_symptom_embeddings_path` hold
precomputed vectors from a Qwen3-Embedding-8B model, LoRA-trained to place a
patient's symptom list next to the description of their disease. The GFE tower
consumes those vectors; it never runs the language model itself. The two training
stages live in `scripts/embedding/` and run in the same environment as the rest
of the repository.

### 1. Preparing the data

One JSONL row per (patient symptoms, disease) pair, with hard negatives that are immediately rejected as a response:

```json
{
  "query": "Instruct: Given a patient's symptoms, calculate the similarity to the symptoms of a disease\nQuery: Symptoms: Nystagmus, Night blindness, Reduced visual acuity",
  "response": "Disease Name: Night blindness, congenital stationary (complete), 1C, autosomal recessive\nGene Symbol: TRPM1\nSymptoms: Optic disc pallor, Nystagmus, Myopia",
  "rejected_responses": ["Disease Name: ...\nGene Symbol: ...\nSymptoms: ...", "..."]
}
```

- `query` - the patient's HPO term names behind the instruct prefix. The same
  prefix has to be used when the embeddings are extracted, or the vectors of a
  patient and a disease stop being comparable.
- `response` - the patient's own disease. Pre-training uses the free-text OMIM
  entry (`Disease Name` / `Text` / `Description`), fine-tuning the structured
  record the diagnostic pipeline holds (`Disease Name` / `Gene Symbol` /
  `Symptoms`).
- `rejected_responses` - other diseases, as hard negatives; the published run
  used five per row.

Write the two files to `$DATA_ROOT/embedding/pretraining_dataset.jsonl` and
`$DATA_ROOT/embedding/finetuning_dataset.jsonl`.

### 2. Training

Both stages are LoRA over an InfoNCE loss - one epoch of pre-training on the OMIM
text, three of fine-tuning on the in-house records. Each stage's adapter has to be
merged before the next step can load it:

```bash
DATA_ROOT=/path/to/3asc_data bash scripts/embedding/pretrain/train.sh
swift export --adapters ./output/embedding/pretrain/<run>/checkpoint-<n> --merge_lora true

PRETRAIN_MERGED=./output/embedding/pretrain/<run>/checkpoint-<n>-merged \
DATA_ROOT=/path/to/3asc_data bash scripts/embedding/finetune/train.sh
swift export --adapters ./output/embedding/finetune/<run>/checkpoint-<n> --merge_lora true
```

### 3. Extracting the embeddings

`extract_embeddings.py` runs the merged fine-tuned model over every disease and
every patient once and pickles the two maps the GFE tower reads, as
`disease_embeddings.pkl` and `patient_symptom_embeddings.pkl`. Point
`model.disease_embeddings_path` and `model.patient_symptom_embeddings_path` at
them.

`diseases.json` is keyed by disease ID, `patients.json` by sample ID with one
entry per HPO term of the patient:

```json
{
  "OMIM:610179": {
    "name": "Night blindness, congenital stationary (complete), 1C",
    "gene_symbol": "TRPM1",
    "symptoms": ["Optic disc pallor", "Nystagmus", "Myopia"]
  }
}
```

```json
{
  "SAMPLE-0001": [
    {"name": "Nystagmus", "onset_age": "Childhood"},
    {"name": "Night blindness", "onset_age": "Infancy"}
  ]
}
```

```bash
python scripts/embedding/extract_embeddings.py \
    --model ./output/embedding/finetune/<run>/checkpoint-<n>-merged \
    --diseases /path/to/diseases.json \
    --patients /path/to/patients.json \
    --output-dir /path/to/3asc_data/gfe
```
