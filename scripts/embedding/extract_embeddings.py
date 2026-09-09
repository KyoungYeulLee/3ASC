"""Extract the GFE embeddings that the 3ASC 3.0 model consumes.

    python scripts/embedding/extract_embeddings.py \
        --model ./output/embedding/finetune/<run>/checkpoint-<n>-merged \
        --diseases diseases.json \
        --patients patients.json \
        --output-dir /path/to/3asc_data/gfe

Writes `disease_embeddings.pkl` and `patient_symptom_embeddings.pkl`, the two
maps `model.disease_embeddings_path` and `model.patient_symptom_embeddings_path`
point the GFE tower at.

The two inputs are JSON maps of the records to embed:

    diseases.json  {"OMIM:610179": {"name": "Night blindness, ...",
                                    "gene_symbol": "TRPM1",
                                    "symptoms": ["Nystagmus", "Myopia"]}}

    patients.json  {"SAMPLE-0001": [{"name": "Nystagmus",
                                     "onset_age": "Childhood"}]}
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from vllm import LLM

# The prompt the embedding model was trained with. Reusing it verbatim is what
# keeps a patient vector and a disease vector comparable.
INSTRUCTION_TEMPLATE = (
    "Instruct: Given a patient's symptoms, calculate the similarity to the "
    "symptoms of a disease\nQuery: {query}"
)
# Fallbacks `VariantDataset` looks up for variants whose disease is not in the
# map, and for patients that have no embedding of their own.
UNKNOWN_DISEASE_TEXT = (
    "Disease Name: Unknown Disease\nGene Symbol: Unknown Gene\n"
    "Symptoms: Unknown Symptoms"
)
UNKNOWN_SAMPLE_TEXT = "Symptoms: Unknown Symptoms"
SEED = 2026


def build_disease_texts(diseases: dict[str, dict]) -> dict[str, str]:
    """Lay out every disease the way the fine-tuning responses were written.

    Args:
        diseases (dict[str, dict]): Disease records keyed by disease ID, each
            holding `name`, `gene_symbol` and `symptoms`.

    Returns:
        dict[str, str]: Text to embed per disease ID, plus `unknown_disease`.
    """
    texts = {
        disease_id: (
            f"Disease Name: {record['name']}\n"
            f"Gene Symbol: {record['gene_symbol']}\n"
            f"Symptoms: {', '.join(record['symptoms'])}"
        )
        for disease_id, record in diseases.items()
    }
    texts["unknown_disease"] = UNKNOWN_DISEASE_TEXT

    return texts


def build_patient_texts(patients: dict[str, list[dict]]) -> dict[str, str]:
    """Lay out every patient the way the training queries were written.

    Args:
        patients (dict[str, list[dict]]): Symptoms per sample ID, each symptom
            holding `name` and `onset_age`.

    Returns:
        dict[str, str]: Text to embed per sample ID, plus `unknown_sample`.
    """
    texts = {
        sample_id: "Symptoms: "
        + ", ".join(
            f"{symptom['name']}(Onset age: {symptom['onset_age']})"
            for symptom in symptoms
        )
        for sample_id, symptoms in patients.items()
        if symptoms
    }
    texts["unknown_sample"] = UNKNOWN_SAMPLE_TEXT

    return texts


def embed(model: LLM, texts: dict[str, str]) -> dict[str, np.ndarray]:
    """Embed every text behind the instruction prompt.

    Args:
        model (LLM): The merged fine-tuned embedding model.
        texts (dict[str, str]): Text to embed, keyed by the ID to store it under.

    Returns:
        dict[str, np.ndarray]: One embedding per key.
    """
    outputs = model.embed(
        [INSTRUCTION_TEMPLATE.format(query=text) for text in texts.values()]
    )

    return {
        key: np.asarray(output.outputs.embedding, dtype=np.float32)
        for key, output in zip(texts, outputs, strict=True)
    }


def main() -> None:
    """Parse the command line and write the two embedding maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="merged fine-tuned checkpoint"
    )
    parser.add_argument(
        "--diseases", type=Path, required=True, help="JSON of disease records"
    )
    parser.add_argument(
        "--patients", type=Path, required=True, help="JSON of patient symptoms"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="where to write the pickles"
    )
    args = parser.parse_args()

    with open(args.diseases) as fh:
        disease_texts = build_disease_texts(json.load(fh))
    with open(args.patients) as fh:
        patient_texts = build_patient_texts(json.load(fh))
    print(f"{len(disease_texts)} diseases and {len(patient_texts)} patients to embed")

    model = LLM(model=str(args.model), task="embed", seed=SEED)
    disease_embeddings = embed(model, disease_texts)
    patient_embeddings = embed(model, patient_texts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    disease_path = args.output_dir / "disease_embeddings.pkl"
    patient_path = args.output_dir / "patient_symptom_embeddings.pkl"
    with open(disease_path, "wb") as fh:
        pickle.dump(disease_embeddings, fh)
    with open(patient_path, "wb") as fh:
        pickle.dump(patient_embeddings, fh)
    print(f"wrote {disease_path} and {patient_path}")

    return


if __name__ == "__main__":
    main()
