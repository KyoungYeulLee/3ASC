"""Dataset loading and the torch Dataset feeding the 3ASC 3.0 (MIL-GFE) model."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from core.data_model import PatientData, PatientDataSet
from core.utils import load_pickle


def to_float_features(features: np.ndarray) -> np.ndarray:
    """Cast a raw feature block to float32, mapping missing values to zero.

    Both the scaler fit and the per-patient transform go through this function,
    so the two can never disagree on how a missing value is read.

    Args:
        features (np.ndarray): Raw feature block, as stored in `VariantData.x`.

    Returns:
        np.ndarray: The block as float32, with the "." the source matrix stores
            missing values as replaced by 0.0.
    """
    return np.where(features == ".", 0.0, features).astype(np.float32)


class VariantDataset(torch.utils.data.Dataset):
    """Serves one patient at a time: features, labels and GFE embeddings.

    Every variant is paired with the embedding of the disease the patient's
    disease map assigns to it. Variants without a mapped disease - and every
    CNV, which the GFE tower is not applied to - get the `unknown_disease`
    embedding and a False entry in the mask, so the encoder can tell them apart.

    Args:
        patient_dataset (PatientDataSet): Patients to serve.
        features (dict[str, list[str]]): Feature columns per variant type.
        scalers (dict[str, StandardScaler]): Fitted (or about to be fitted)
            scaler per variant type.
        variant_types (list[str]): Variant types to serve.
        disease_embeddings_path (str): Pickled {disease_id: embedding} map.
        patient_symptom_embeddings_path (str): Pickled {sample_id: embedding}
            map of patient symptom embeddings.
        disease_cpra_map_path (str): JSON {sample_id: {cpra: disease_id}} map.
    """

    def __init__(
        self,
        patient_dataset: PatientDataSet,
        features: dict[str, list[str]],
        scalers: dict[str, StandardScaler],
        variant_types: list[str],
        disease_embeddings_path: str,
        patient_symptom_embeddings_path: str,
        disease_cpra_map_path: str,
    ) -> None:
        self.patient_dataset = patient_dataset
        self.scalers = scalers
        self.variant_types = variant_types

        self.disease_embeddings = load_pickle(disease_embeddings_path)
        self.patient_symptom_embeddings = load_pickle(patient_symptom_embeddings_path)
        with open(disease_cpra_map_path, "r") as fh:
            self.disease_cpra_map = json.load(fh)

        # Feature columns are addressed by position in the stored matrix.
        self.feature_indices = {}
        for variant_type in self.variant_types:
            header = self._find_header(variant_type)
            self.feature_indices[variant_type] = np.array(
                [header.index(feature) for feature in features[variant_type]]
            )

    def __len__(self) -> int:
        return len(self.patient_dataset)

    def _find_header(self, variant_type: str) -> list[str]:
        """Return the feature column names of one variant type.

        Every patient of a cohort shares the same column layout, but a given
        patient can have no variant of a type at all - and then carries no
        header - so the first patient that has one is asked.

        Args:
            variant_type (str): Variant type to look the header up for.

        Returns:
            list[str]: Header of the first patient that has a variant of this
                type.

        Raises:
            ValueError: If no patient has a variant of this type, which leaves
                the position of its feature columns undefined.
        """
        for patient_data in self.patient_dataset:
            variant_data = getattr(patient_data, f"{variant_type}_data")
            if variant_data.x.size > 0:
                return variant_data.header

        raise ValueError(
            f"No patient of this dataset has a {variant_type} variant, so its "
            "feature columns cannot be located. Drop the type from "
            "model.variant_type, or pass a dataset that covers it."
        )

    def _make_variant_x(
        self, patient_data: PatientData, variant_type: str
    ) -> np.ndarray:
        """Select the configured feature columns and standardize them.

        Args:
            patient_data (PatientData): Patient to take the variants from.
            variant_type (str): Variant type to build the matrix for.

        Returns:
            np.ndarray: Array of shape (n_variants, n_features).
        """
        variant_x = getattr(patient_data, f"{variant_type}_data").x
        features = to_float_features(variant_x[:, self.feature_indices[variant_type]])

        return self.scalers[variant_type].transform(features)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Return one patient.

        Args:
            idx (int): Position of the patient in the dataset.

        Returns:
            tuple: (bag_label, patient_data_x, patient_data_y). `patient_data_x`
                holds one feature tensor per variant type the patient has
                variants of, plus the variant embeddings `v_embeddings`, their
                `gfe_known_mask` and the patient embedding `p_embeddings`.
                `patient_data_y` holds the matching instance labels.
        """
        patient_data: PatientData = self.patient_dataset[idx]

        bag_label = torch.tensor(patient_data.bag_label, dtype=torch.float32)
        patient_data_x = {"sample_id": patient_data.sample_id}
        patient_data_y = {}

        embeddings = []
        gfe_known_mask = []
        sample_cpra_map = self.disease_cpra_map.get(patient_data.sample_id, {})
        for variant_type in self.variant_types:
            variant_data = getattr(patient_data, f"{variant_type}_data")
            if len(variant_data.x) == 0:
                continue

            patient_data_y[variant_type] = torch.from_numpy(variant_data.y).to(
                torch.float32
            )
            patient_data_x[variant_type] = torch.from_numpy(
                self._make_variant_x(patient_data, variant_type)
            ).to(torch.float32)

            for variant in variant_data.variants:
                disease = sample_cpra_map.get(variant.cpra, "unknown_disease")
                variant_embeddings = self.disease_embeddings.get(disease)
                is_known = (
                    variant_type != "cnv"
                    and variant_embeddings is not None
                    and disease != "unknown_disease"
                )
                if not is_known:
                    variant_embeddings = self.disease_embeddings["unknown_disease"]
                embeddings.append(variant_embeddings)
                gfe_known_mask.append(is_known)

        if embeddings:
            # (n_variants, embedding_dim), cast because the model runs in
            # float32 while an embedding file may be stored as float64.
            patient_data_x["v_embeddings"] = torch.from_numpy(
                np.stack(embeddings, axis=0)
            ).to(torch.float32)
            patient_data_x["gfe_known_mask"] = torch.tensor(
                gfe_known_mask, dtype=torch.bool
            )
        patient_embedding = self.patient_symptom_embeddings.get(
            patient_data.sample_id,
            self.patient_symptom_embeddings["unknown_sample"],
        )
        patient_data_x["p_embeddings"] = torch.from_numpy(
            patient_embedding.reshape(1, -1)
        ).to(torch.float32)

        return bag_label, patient_data_x, patient_data_y


def load_training_dataset(
    train_data_path: str | Path,
    exclude_from_train_data_path: str | Path,
) -> PatientDataSet:
    """Load the training cohort and hold the benchmark samples out of it.

    Args:
        train_data_path (str | Path): Pickled `PatientDataSet` to train on.
        exclude_from_train_data_path (str | Path): Pickled `PatientDataSet`
            whose samples must not be trained on.

    Returns:
        PatientDataSet: The patients to train and validate on.
    """
    benchmark_dataset: PatientDataSet = load_pickle(exclude_from_train_data_path)
    train_dataset: PatientDataSet = load_pickle(train_data_path)

    return train_dataset.complement(benchmark_dataset.all_sample_ids)
