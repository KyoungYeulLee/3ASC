"""Write a miniature example cohort in the exact on-disk format.

    python make_example_data.py [--config conf/config.yaml] [--out example_data]
                                [--gfe-model <merged checkpoint>]

The cohort is a template, not data to learn from: one patient carrying one
variant of every type, the SNV being the causal one. Every coordinate is drawn
at random (seeded, so the files come out the same every run) and every other
identifier is invented. Each variant's feature row holds the columns of
`conf/config.yaml` in order, with a few illustrative values filled in and "." -
the marker the loader maps to 0 - everywhere else. The GFE inputs (disease
embeddings, patient symptom embeddings and the CPRA-to-disease map) are written
next to it, so every path of the README's input table has a concrete
counterpart to inspect.

The GFE vectors are random by default. Pass `--gfe-model` to point at a merged
fine-tuned embedding model and the invented records go through
`scripts/embedding/extract_embeddings.py` instead, so the example carries the
same instruct prefix and text layout as the real inputs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from core.data_model import (
    BNDData,
    CNVData,
    INSData,
    PatientData,
    PatientDataSet,
    SNVData,
    STRData,
    Symptom,
    Variant,
    VariantData,
)

SAMPLE_ID = "SAMPLE-0001"

# Illustrative values per variant type; every configured feature that is not
# listed here is stored as ".", which the loader reads as missing.
EXAMPLE_VALUES = {
    "snv": {
        "ACMG_bayesian": 0.98,
        "symptom_similarity": 0.81,
        "vcf_info_QUAL": 812.7,
        "inhouse_freq": 0.00012,
        "vaf": 0.47,
        "is_incomplete_zygosity": 0,
        "gnomad_gene:pLI": 1.0,
        "gnomad_gene:loeuf": 0.28,
        "wes_AC": 3,
        "wgs_AC": 1,
        "clinvar_variant:scv:pathogenicity_n_p": 4,
        "clinvar_variant:scv:pathogenicity_n_b": 0,
        "PVS1": 1,
        "PM2": 1,
        "PP3": 1,
    },
    "cnv": {
        "ACMG_bayesian": 0.91,
        "symptom_similarity": 0.64,
        "num_genes": 12,
        "qual": 540.0,
        "caller_3bCNV": 1,
        "cnv_type_DEL": 1,
        "cnv_type_DUP": 0,
        "stix:tier": 1,
    },
    "str": {
        "repeat_num": 62,
        "inhouse:allele_count": 3,
        "classification": 2,
    },
    "bnd": {
        "symptom_similarity": 0.42,
        "quality": 310.0,
        "total_depth": 38,
        "vaf": 0.51,
        "genotype": 1,
        "inhouse:af": 0.0004,
        "sv_type": 1,
        "overlap_gene": 1,
        "mate_number": 1,
        "merged_acmg_class": 3,
        "merged_acmg_bayesian": 0.55,
        "cds_info_CDS": 1,
        "PM2": 1,
    },
    "ins": {
        "symptom_similarity": 0.38,
        "sv_type_INS": 1,
        "inhouse:af": 0.0009,
        "inhouse:an": 5614,
        "vaf": 0.33,
        "total_depth": 41,
        "closest_exon_dist": 128,
        "acmg_class": 3,
        "acmg_bayesian": 0.47,
        "PM2": 1,
    },
}

VARIANT_DATA_CLASSES = {
    "snv": SNVData,
    "cnv": CNVData,
    "str": STRData,
    "bnd": BNDData,
    "ins": INSData,
}

# Invented stand-ins for the records `extract_embeddings.py` embeds: the
# disease records keyed by disease ID, and the patient's HPO terms. `--gfe-model`
# lays these out and embeds them instead of drawing random vectors.
EXAMPLE_DISEASES = {
    "OMIM:000001": {
        "name": "Epileptic encephalopathy, early infantile, 1A",
        "gene_symbol": "GENE1",
        "symptoms": ["Seizure", "Global developmental delay", "Hypotonia"],
    },
    "OMIM:000002": {
        "name": "Spinocerebellar ataxia 1A",
        "gene_symbol": "GENE3",
        "symptoms": ["Gait ataxia", "Dysarthria", "Nystagmus"],
    },
    "OMIM:000003": {
        "name": "Developmental disorder with structural brain anomalies 1A",
        "gene_symbol": "GENE4",
        "symptoms": ["Global developmental delay", "Microcephaly"],
    },
    "OMIM:000004": {
        "name": "Retinal dystrophy 1A",
        "gene_symbol": "GENE5",
        "symptoms": ["Night blindness", "Reduced visual acuity"],
    },
}
EXAMPLE_PATIENT_SYMPTOMS = [{"name": "Seizure", "onset_age": "Infancy"}]


def random_cpra(rng: np.random.Generator, variant_type: str) -> str:
    """Draw a random identifier in the CPRA convention of one variant type.

    Args:
        rng (np.random.Generator): Source of randomness.
        variant_type (str): One of snv, cnv, str, bnd, ins.

    Returns:
        str: A random coordinate, e.g. "7-14352869-G-T" for an SNV.
    """
    chrom = str(rng.integers(1, 23))
    # Capped at 45 Mb, a position every autosome can hold.
    pos = int(rng.integers(1_000_000, 45_000_000))

    if variant_type == "snv":
        ref, alt = rng.choice(list("ACGT"), size=2, replace=False)
        return f"{chrom}-{pos}-{ref}-{alt}"
    if variant_type in ("cnv", "bnd"):
        end = pos + int(rng.integers(1_000, 500_000))
        return f"{chrom}:{pos}-{end}"
    if variant_type == "str":
        return f"{chrom}-{pos}-CAG"
    return f"{chrom}:{pos}"  # ins


def build_example_variants(rng: np.random.Generator) -> dict[str, Variant]:
    """Build one variant record per type, at random coordinates.

    The gene, disease and annotation values are invented: they only demonstrate
    the shape each field takes.

    Args:
        rng (np.random.Generator): Source of randomness for the coordinates.

    Returns:
        dict[str, Variant]: One record per variant type.
    """
    return {
        "snv": Variant(
            cpra=random_cpra(rng, "snv"),
            acmg_class="Pathogenic",
            acmg_rules=["PVS1", "PM2", "PP3"],
            gene_id="HGNC:00001",
            symbol="GENE1",
            disease_id="OMIM:000001",
            disease_symptom_similarity=0.81,
            inheritance="AD",
            onset_age="Infantile onset",
            sample_id=SAMPLE_ID,
            variant_type="SNV",
        ),
        "cnv": Variant(
            cpra=random_cpra(rng, "cnv"),
            acmg_class="VUS",
            gene_id=["HGNC:00002", "HGNC:00003"],
            symbol="GENE2",
            sample_id=SAMPLE_ID,
            variant_type="CNV",
        ),
        "str": Variant(
            cpra=random_cpra(rng, "str"),
            gene_id="HGNC:00004",
            symbol="GENE3",
            disease_id="OMIM:000002",
            disease_symptom_similarity=0.35,
            sample_id=SAMPLE_ID,
            variant_type="STR",
            repeat_seq="CAG",
            repeat_num=62,
        ),
        "bnd": Variant(
            cpra=random_cpra(rng, "bnd"),
            acmg_class="VUS",
            gene_id="HGNC:00005",
            symbol="GENE4",
            disease_id="OMIM:000003",
            disease_symptom_similarity=0.42,
            sample_id=SAMPLE_ID,
            variant_type="BND",
        ),
        "ins": Variant(
            cpra=random_cpra(rng, "ins"),
            acmg_class="VUS",
            gene_id="HGNC:00006",
            symbol="GENE5",
            disease_id="OMIM:000004",
            disease_symptom_similarity=0.38,
            sample_id=SAMPLE_ID,
            variant_type="INS",
        ),
    }


def build_variant_data(
    variant_type: str, features: list[str], variant: Variant
) -> VariantData:
    """Build the variant block of one type, holding a single variant.

    Args:
        variant_type (str): One of snv, cnv, str, bnd, ins.
        features (list[str]): Feature columns of the type, from the config.
        variant (Variant): The record the block holds.

    Returns:
        VariantData: One-variant block whose header is the configured feature
            list and whose label marks the SNV as the causal variant.
    """
    values = EXAMPLE_VALUES[variant_type]
    is_causal = variant_type == "snv"

    return VARIANT_DATA_CLASSES[variant_type](
        x=np.array([[values.get(feature, ".") for feature in features]], dtype=object),
        y=np.array([1.0 if is_causal else 0.0], dtype=np.float32),
        header=list(features),
        variants=[variant],
        causal_variant=[variant.cpra] if is_causal else [],
    )


def build_patient(cfg: DictConfig, variants: dict[str, Variant]) -> PatientData:
    """Build the example patient: the causal SNV plus one variant of each type.

    Args:
        cfg (DictConfig): Full configuration; `features` names the columns of
            every variant type.
        variants (dict[str, Variant]): One record per variant type.

    Returns:
        PatientData: A positive patient in the format the pickled cohorts use.
    """
    variant_blocks = {
        f"{variant_type}_data": build_variant_data(
            variant_type, list(features), variants[variant_type]
        )
        for variant_type, features in cfg.features.items()
    }

    return PatientData(
        sample_id=SAMPLE_ID,
        bag_label=True,
        **variant_blocks,
        symptoms=[Symptom(hpo="HP:0001250", title="Seizure", onsetAge="Infancy")],
        conclusion="positive",
        sequencing_type="wes",
        genome_build="GRCh38",
    )


def random_gfe_embeddings(
    cfg: DictConfig, disease_ids: list[str], rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Draw random GFE embeddings of the widths the config expects.

    Args:
        cfg (DictConfig): Full configuration; `model.parameters.mil_gfe` fixes
            the embedding widths.
        disease_ids (list[str]): Diseases to draw a vector for.
        rng (np.random.Generator): Source of the vectors.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: The disease and
            the patient embedding map, each with its fallback entry.
    """
    patient_dim = cfg.model.parameters.mil_gfe.patient_embedding_dim
    variant_dim = cfg.model.parameters.mil_gfe.variant_embedding_dim

    disease_embeddings = {
        disease_id: rng.standard_normal(variant_dim).astype(np.float32)
        for disease_id in disease_ids + ["unknown_disease"]
    }
    patient_embeddings = {
        sample_id: rng.standard_normal(patient_dim).astype(np.float32)
        for sample_id in (SAMPLE_ID, "unknown_sample")
    }

    return disease_embeddings, patient_embeddings


def model_gfe_embeddings(
    model_dir: Path, disease_ids: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Embed the invented example records with the fine-tuned GFE model.

    The text layout, the instruct prefix and the fallback records all come from
    `scripts/embedding/extract_embeddings.py`, so the example goes through the
    same conventions as the real inputs rather than a second implementation of
    them.

    Args:
        model_dir (Path): Merged fine-tuned embedding model directory, produced
            by `scripts/embedding/finetune/train.sh` and a LoRA merge.
        disease_ids (list[str]): Diseases whose invented record is embedded.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: The disease and
            the patient embedding map, each with its fallback entry.
    """
    from vllm import LLM

    from scripts.embedding.extract_embeddings import (
        SEED,
        build_disease_texts,
        build_patient_texts,
        embed,
    )

    disease_texts = build_disease_texts(
        {disease_id: EXAMPLE_DISEASES[disease_id] for disease_id in disease_ids}
    )
    patient_texts = build_patient_texts({SAMPLE_ID: EXAMPLE_PATIENT_SYMPTOMS})

    model = LLM(model=str(model_dir), task="embed", seed=SEED)

    return embed(model, disease_texts), embed(model, patient_texts)


def write_gfe_inputs(
    variants: dict[str, Variant],
    gfe_dir: Path,
    disease_embeddings: dict[str, np.ndarray],
    patient_embeddings: dict[str, np.ndarray],
) -> None:
    """Write the CPRA-to-disease map and the two embedding pickles.

    Args:
        variants (dict[str, Variant]): One record per variant type; their
            diseases populate the CPRA-to-disease map.
        gfe_dir (Path): Directory to write the three GFE files into.
        disease_embeddings (dict[str, np.ndarray]): Embedding per disease.
        patient_embeddings (dict[str, np.ndarray]): Embedding per patient.
    """
    # CNVs are never GFE-fused, so the map lists no disease for the CNV.
    cpra_to_disease = {
        SAMPLE_ID: {
            variant.cpra: variant.disease_id
            for variant_type, variant in variants.items()
            if variant_type != "cnv"
        }
    }

    gfe_dir.mkdir(parents=True, exist_ok=True)
    with open(gfe_dir / "disease_embeddings.pkl", "wb") as fh:
        pickle.dump(disease_embeddings, fh)
    with open(gfe_dir / "patient_symptom_embeddings.pkl", "wb") as fh:
        pickle.dump(patient_embeddings, fh)
    with open(gfe_dir / "cpra_to_disease.json", "w") as fh:
        json.dump(cpra_to_disease, fh, indent=4)


def main() -> None:
    """Parse the command line and write the example files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("conf/config.yaml"), help="config file"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("example_data"), help="output directory"
    )
    parser.add_argument(
        "--gfe-model",
        type=Path,
        default=None,
        help="merged fine-tuned GFE embedding model to embed the example "
        "records with; without it the vectors are random",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    rng = np.random.default_rng(0)
    variants = build_example_variants(rng)
    dataset = PatientDataSet([build_patient(cfg, variants)])

    disease_ids = [
        variant.disease_id
        for variant_type, variant in variants.items()
        if variant_type != "cnv"
    ]
    if args.gfe_model is None:
        embedding_maps = random_gfe_embeddings(cfg, disease_ids, rng)
    else:
        embedding_maps = model_gfe_embeddings(args.gfe_model, disease_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "example_cohort.pickle", "wb") as fh:
        pickle.dump(dataset, fh)
    write_gfe_inputs(variants, args.out / "gfe", *embedding_maps)

    source = "random vectors" if args.gfe_model is None else f"{args.gfe_model}"
    print(f"Wrote {args.out}/example_cohort.pickle: {dataset}")
    print(f"Wrote {args.out}/gfe/: embeddings ({source}) and the CPRA map")

    return


if __name__ == "__main__":
    main()
