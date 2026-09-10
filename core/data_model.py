"""Data model of a patient cohort.

`PatientDataSet` is the object the released datasets are pickled as: a list of
`PatientData`, one per patient, each holding the feature matrix, the labels and
the variant records of every variant type.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np

VARIANT_TYPES = ("snv", "cnv", "str", "bnd", "ins")


@dataclass
class Variant:
    """A single variant and the gene/disease annotations attached to it.

    Attributes:
        cpra (str): Chromosome-position-reference-alternate identifier. It is
            the key the disease map is looked up with, and the level top-k
            recall is aggregated on.
        acmg_class (str): ACMG classification of the variant.
        acmg_rules (list[str]): ACMG rules assigned to the variant.
        gene_id (str | list[str]): Gene(s) the variant falls in.
        symbol (str): HGNC symbol of the gene.
        disease_id (str | list[str] | list[list[str]]): Associated disease(s).
        disease_symptom_similarity (float | list): Symptom similarity of the
            associated disease(s).
        inheritance (str | list): Inheritance mode(s) of the disease(s).
        onset_age (str | list): Onset age(s) of the disease(s).
        sample_id (str): Sample the variant was called in.
        score (float): 3ASC score, -1 when not scored yet.
        variant_type (str): One of SNV, CNV, STR, BND, INS.
        repeat_seq (str): Repeat unit, STR only.
        repeat_num (int): Repeat count, STR only.
    """

    cpra: str
    acmg_class: str = ""
    acmg_rules: list[str] = field(default_factory=list)
    gene_id: str | list[str] = ""
    symbol: str = ""
    disease_id: str | list[str] | list[list[str]] = ""
    disease_symptom_similarity: float | list[float] | list[list[float]] = -1.0
    inheritance: str | list[str] | list[list[str]] = ""
    onset_age: str | list[str] | list[list[str]] = ""

    sample_id: str = ""
    score: float = -1.0
    variant_type: str = ""

    repeat_seq: str = ""
    repeat_num: int = 0


@dataclass
class VariantData:
    """Variants of one type for one patient.

    Attributes:
        variant_type (str): The type of variant (SNV, CNV, STR, BND, INS).
        x (np.ndarray): Feature matrix of shape (n_variants, n_columns). The
            columns are named by `header`; the model consumes the subset listed
            in `conf/config.yaml`.
        y (np.ndarray): Binary causality label per variant.
        header (list[str]): Column names of `x`.
        variants (list[Variant]): Variant records, aligned with the rows of `x`.
        causal_variant (list): The causal variant(s) of the patient, if any.
        diseases (list[str] | None): Per-variant disease identifier.
    """

    variant_type: str = ""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    header: list[str] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    causal_variant: list = field(default_factory=list)
    diseases: list[str] | None = None

    @property
    def n_variants(self) -> int:
        """Number of variants held by this object."""
        return len(self.x)

    def __repr__(self) -> str:
        return (
            f"{self.variant_type}Data(n_variants={self.n_variants}, "
            + f"causal_variant={self.causal_variant})"
        )


@dataclass
class SNVData(VariantData):
    """Single nucleotide variants."""

    variant_type: str = field(default="SNV", init=False)


@dataclass
class CNVData(VariantData):
    """Copy number variants."""

    variant_type: str = field(default="CNV", init=False)


@dataclass
class STRData(VariantData):
    """Short tandem repeats."""

    variant_type: str = field(default="STR", init=False)


@dataclass
class BNDData(VariantData):
    """Breakends and inversions."""

    variant_type: str = field(default="BND", init=False)


@dataclass
class INSData(VariantData):
    """Mobile element insertions."""

    variant_type: str = field(default="INS", init=False)


@dataclass
class Symptom:
    """One HPO term of a patient, with the age its onset was observed at.

    The phenotype reaches the model through the patient symptom embeddings, not
    through this class - nothing here reads it. It is part of the format because
    the released cohorts carry the symptom list of every patient and unpickling
    them needs the class; `onsetAge` is the name the field has on disk, so it
    cannot be renamed for the same reason.
    """

    hpo: str
    title: str
    onsetAge: str


@dataclass
class PatientData:
    """One patient: the bag of the multiple-instance learning problem.

    Attributes:
        sample_id (str): Identifier of the sample.
        bag_label (bool): Whether the patient has a causal variant at all.
        snv_data (SNVData): SNV instances of the bag.
        cnv_data (CNVData): CNV instances of the bag.
        str_data (STRData): STR instances of the bag.
        bnd_data (BNDData): BND instances of the bag.
        ins_data (INSData): INS instances of the bag.
        symptoms (list[Symptom]): HPO terms of the patient. Carried by the
            released cohorts; the model sees the phenotype through the patient
            symptom embeddings instead.
        conclusion (str): Diagnostic conclusion of the case.
        sequencing_type (str): wes or wgs.
        genome_build (str): Reference build the variants were called against.
    """

    sample_id: str
    bag_label: bool
    snv_data: SNVData = field(default_factory=SNVData)
    cnv_data: CNVData = field(default_factory=CNVData)
    str_data: STRData = field(default_factory=STRData)
    bnd_data: BNDData = field(default_factory=BNDData)
    ins_data: INSData = field(default_factory=INSData)
    symptoms: list[Symptom] = field(default_factory=list)
    conclusion: Literal["positive", "inconclusive", "negative", ""] = ""
    sequencing_type: Literal["wes", "wgs"] = "wes"
    genome_build: str = ""

    def __repr__(self) -> str:
        return (
            f"PatientData(sample_id={self.sample_id}, bag_label={self.bag_label}, "
            + f"n_snv={self.snv_data.n_variants}, n_cnv={self.cnv_data.n_variants}, "
            + f"n_str={self.str_data.n_variants}, n_bnd={self.bnd_data.n_variants}, "
            + f"n_ins={self.ins_data.n_variants})"
        )


@dataclass
class MetaData:
    """Feature headers of a dataset, taken from its first non-empty patient."""

    snv_header: list[str] = field(default_factory=list)
    cnv_header: list[str] = field(default_factory=list)
    str_header: list[str] = field(default_factory=list)
    bnd_header: list[str] = field(default_factory=list)
    ins_header: list[str] = field(default_factory=list)

    n_snv_features: int = 0
    n_cnv_features: int = 0
    n_str_features: int = 0
    n_bnd_features: int = 0
    n_ins_features: int = 0


@dataclass
class PatientDataSet:
    """A cohort of patients, indexable by position or by sample ID.

    Example:
        >>> dataset = PatientDataSet([PatientData(...), PatientData(...)])
        >>> dataset[0]
        PatientData(sample_id=SAMPLE-0001, ...)
        >>> dataset["SAMPLE-0001"]
        PatientData(sample_id=SAMPLE-0001, ...)
        >>> dataset[[0, 1]]
        PatientDataSet(len=2)
    """

    _raw_data: list[PatientData] = field(default_factory=list)
    metadata: MetaData = field(default_factory=MetaData)

    _data: list[PatientData] = field(init=False, repr=False)

    def __post_init__(self):
        self._data = self._raw_data

        headers = {}
        for variant_type in VARIANT_TYPES:
            header = []
            for patient_data in self._raw_data:
                variant_data = getattr(patient_data, f"{variant_type}_data")
                if variant_data.x.size == 0:
                    continue
                header = variant_data.header
                break
            headers[f"{variant_type}_header"] = header
            headers[f"n_{variant_type}_features"] = len(header)

        self.metadata = MetaData(**headers)
        self.sample_id_to_idx = {
            p_data.sample_id: i for i, p_data in enumerate(self.data)
        }

    def __repr__(self) -> str:
        return f"PatientDataSet(len={self.__len__()})"

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx: str | int | Iterable[str] | Iterable[int]
    ) -> PatientData | PatientDataSet:
        # Rebuild sample_id_to_idx if missing or outdated (e.g., after pickle load)
        if not hasattr(self, "sample_id_to_idx") or len(self.sample_id_to_idx) != len(
            self.data
        ):
            self.sample_id_to_idx = {
                p_data.sample_id: i for i, p_data in enumerate(self.data)
            }

        if isinstance(idx, numbers.Integral):
            return self.data[idx]

        elif isinstance(idx, str):
            if idx not in self.sample_id_to_idx:
                raise IndexError(f"Passed sample_id({idx}) is not found.")
            return self.data[self.sample_id_to_idx[idx]]

        elif isinstance(idx, Iterable):
            if all([isinstance(x, str) for x in idx]):
                subset = []
                for sample_id in idx:
                    if sample_id not in self.sample_id_to_idx:
                        raise IndexError(f"Passed sample_id({sample_id}) not found.")
                    subset.append(self.data[self.sample_id_to_idx[sample_id]])
                return PatientDataSet(subset)

            elif all([isinstance(x, numbers.Integral) for x in idx]):
                return PatientDataSet([self.data[i] for i in idx])

        return PatientDataSet(self.data[idx])

    def complement(self, sample_ids: Iterable[str]) -> PatientDataSet:
        """Return the patients whose sample ID is not in `sample_ids`.

        Args:
            sample_ids (Iterable[str]): Sample IDs to drop.

        Returns:
            PatientDataSet: The remaining patients, in their original order.
        """
        sample_ids = set(sample_ids)
        return PatientDataSet(
            [p_data for p_data in self.data if p_data.sample_id not in sample_ids]
        )

    @property
    def data(self) -> list[PatientData]:
        """The patients of the dataset."""
        return self._data

    @data.setter
    def data(self, new_data: list[PatientData]):
        self._data = new_data
        self.sample_id_to_idx = {
            p_data.sample_id: i for i, p_data in enumerate(new_data)
        }

    @property
    def all_sample_ids(self) -> np.ndarray:
        """Sample IDs of every patient, in dataset order."""
        if not self.data:
            return np.zeros(shape=(0,))
        return np.array([data.sample_id for data in self.data])

    def feature_matrix(self, variant_type: str) -> np.ndarray:
        """Stack the feature matrices of one variant type over all patients.

        Args:
            variant_type (str): One of snv, cnv, str, bnd, ins.

        Returns:
            np.ndarray: Array of shape (total_n_variants, n_columns). Patients
                without a variant of this type contribute no row.
        """
        if not self.data:
            n_dim = getattr(self.metadata, f"n_{variant_type}_features")
            return np.zeros(shape=(0, n_dim))
        return np.concatenate(
            [
                getattr(patient_data, f"{variant_type}_data").x
                for patient_data in self.data
                if getattr(patient_data, f"{variant_type}_data").x.size > 0
            ]
        )
