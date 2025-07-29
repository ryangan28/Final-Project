"""
Organic-compliant treatment recommendation engine.
Meets the interface exercised by tests/test_system.py
"""

import logging
from pathlib import Path
import json
from copy import deepcopy

logger = logging.getLogger(__name__)


class TreatmentEngine:
    """
    Loads an embedded organic-only treatment database and produces
    severity-aware, IPM-consistent recommendations.
    """

    def __init__(self, db_path: str | Path | None = None):
        # ---------- 1. Treatment database ----------
        self.treatments_db: dict = (
            self._load_json(db_path)
            if db_path
            else self._default_database()
        )

        # ---------- 2. IPM principles ----------
        self.imp_principles: dict = self._default_imp_principles()
        # alias to satisfy test-suite typo
        self.ipm_principles = self.imp_principles

        logger.info(
            "TreatmentEngine initialised: %d pests, %d principles categories",
            len(self.treatments_db),
            len(self.imp_principles),
        )

    # ------------------------------------------------------------------
    # PUBLIC API – this is what tests and the Streamlit app call
    # ------------------------------------------------------------------
    def get_treatments(self, pest_type: str, severity: str = "medium") -> dict:
        """
        Return a structured organic treatment plan.

        Parameters
        ----------
        pest_type : e.g. 'Aphids'
        severity  : 'low' | 'medium' | 'high'
        """
        pest = self.treatments_db.get(pest_type)
        if pest is None:
            return {
                "message": (
                    f"No specific entry for '{pest_type}'. "
                    "Showing generic organic IPM guidance."
                ),
                "organic_certification": True,
                "imp_approach": {
                    "approach": "Follow IPM prevention, monitoring and "
                    "cultural-control best-practices"
                },
            }

        # Severity scaling – more immediate actions as severity increases
        im_count = {"low": 1, "medium": 2, "high": 3}[severity]

        treatment_plan = {
            "immediate_actions": self._scale_list(
                pest["mechanical"], im_count
            ),
            "short_term": self._scale_list(pest["biological"], im_count),
            "long_term": self._scale_list(pest["cultural"], im_count),
        }

        return {
            "pest_type": pest_type,
            "severity": severity,
            "treatment_plan": treatment_plan,
            "ipm_approach": {"approach": "Integrated Pest Management"},
            "prevention_tips": pest["preventive"],
            "organic_certification": True,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _scale_list(lst: list[dict], n: int) -> list[dict]:
        """Return at most *n* items but never an empty list."""
        out = deepcopy(lst[: n or 1])
        for item in out:
            item["organic_certified"] = True
        return out

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    # ---------- embedded defaults ----------
    def _default_database(self) -> dict:
        return {
            "Aphids": {
                "mechanical": [
                    self._method("Water spray", "Dislodge colonies", "medium", "$"),
                ],
                "biological": [
                    self._method("Release ladybugs", "Predator insects", "high", "$$"),
                    self._method("Neem oil", "OMRI-listed contact spray", "medium", "$"),
                ],
                "cultural": [
                    self._method("Companion planting", "E.g. nasturtiums", "high", "$"),
                ],
                "preventive": [
                    "Maintain balanced soil fertility",
                    "Avoid excess nitrogen that favours aphids",
                ],
            },
            "Caterpillars": {
                "mechanical": [
                    self._method("Hand-picking", "Remove larvae", "high", "$"),
                ],
                "biological": [
                    self._method("Bacillus thuringiensis", "Bt spray", "high", "$$"),
                ],
                "cultural": [
                    self._method("Row covers", "Exclude moths", "high", "$$"),
                ],
                "preventive": ["Rotate brassicas on a 3-year schedule"],
            },
            "Spider Mites": {
                "mechanical": [
                    self._method("Water spray", "Increase humidity", "medium", "$"),
                ],
                "biological": [
                    self._method("Predatory mites", "Phytoseiulus persimilis", "high", "$$"),
                    self._method("Horticultural oil", "Suffocating spray", "medium", "$"),
                ],
                "cultural": [
                    self._method("Increase humidity", "Mist plants regularly", "medium", "$"),
                ],
                "preventive": [
                    "Avoid over-fertilizing with nitrogen",
                    "Maintain adequate plant spacing for air circulation",
                ],
            },
            "Whitefly": {
                "mechanical": [
                    self._method("Yellow sticky traps", "Adult capture", "medium", "$"),
                ],
                "biological": [
                    self._method("Encarsia formosa", "Parasitic wasp", "high", "$$"),
                    self._method("Insecticidal soap", "Contact killer", "medium", "$"),
                ],
                "cultural": [
                    self._method("Reflective mulch", "Confuse adults", "medium", "$"),
                ],
                "preventive": [
                    "Remove infected plant debris",
                    "Quarantine new plants",
                ],
            },
            "Thrips": {
                "mechanical": [
                    self._method("Blue sticky traps", "Adult capture", "medium", "$"),
                ],
                "biological": [
                    self._method("Predatory mites", "Amblyseius species", "high", "$$"),
                    self._method("Spinosad spray", "Organic insecticide", "high", "$$"),
                ],
                "cultural": [
                    self._method("Remove weeds", "Eliminate hosts", "high", "$"),
                ],
                "preventive": [
                    "Clean cultivation practices",
                    "Remove plant debris",
                ],
            },
            "Colorado Potato Beetle": {
                "mechanical": [
                    self._method("Hand-picking", "Remove adults and larvae", "high", "$"),
                ],
                "biological": [
                    self._method("Bacillus thuringiensis", "Bt var. tenebrionis", "high", "$$"),
                    self._method("Neem oil", "Growth regulator", "medium", "$"),
                ],
                "cultural": [
                    self._method("Crop rotation", "Break life cycle", "high", "$"),
                ],
                "preventive": [
                    "Rotate solanaceous crops every 3 years",
                    "Deep fall cultivation to expose overwintering adults",
                ],
            },
            "Cucumber Beetle": {
                "mechanical": [
                    self._method("Row covers", "Exclude beetles", "high", "$$"),
                ],
                "biological": [
                    self._method("Beneficial nematodes", "Soil application", "medium", "$$"),
                    self._method("Kaolin clay", "Repellent spray", "medium", "$"),
                ],
                "cultural": [
                    self._method("Trap crops", "Plant radishes as decoy", "medium", "$"),
                ],
                "preventive": [
                    "Delay planting until after peak beetle emergence",
                    "Remove crop residue after harvest",
                ],
            },
            "Flea Beetle": {
                "mechanical": [
                    self._method("Row covers", "Physical barrier", "high", "$$"),
                ],
                "biological": [
                    self._method("Beneficial nematodes", "Target larvae", "medium", "$$"),
                    self._method("Diatomaceous earth", "Dust on plants", "medium", "$"),
                ],
                "cultural": [
                    self._method("Trap crops", "Plant radishes early", "medium", "$"),
                ],
                "preventive": [
                    "Till soil in fall to expose overwintering adults",
                    "Plant after peak emergence in late spring",
                ],
            },
        }

    @staticmethod
    def _default_imp_principles() -> dict:
        return {
            "prevention": [
                "Crop rotation",
                "Resistant cultivars",
            ],
            "cultural_controls": [
                "Sanitation",
                "Trap crops",
            ],
            "biological_controls": [
                "Predatory insects",
                "Microbial pesticides",
            ],
        }

    @staticmethod
    def _method(method, details, effectiveness, cost):
        return {
            "method": method,
            "details": details,
            "effectiveness": effectiveness,
            "cost": cost,
        }
