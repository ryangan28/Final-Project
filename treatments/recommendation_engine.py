"""
Organic-compliant treatment recommendation engine.
Meets the interface exercised by tests/test_system.py
"""

import logging
from pathlib import Path
import json
from copy import deepcopy
from typing import Union, Optional

logger = logging.getLogger(__name__)

# Treatment method constants
BACILLUS_THURINGIENSIS = "Bacillus thuringiensis"
ROW_COVERS = "Row covers"
CROP_ROTATION = "Crop rotation"
BENEFICIAL_NEMATODES = "Beneficial nematodes"
TRAP_CROPS = "Trap crops"
ENCOURAGE_PREDATORS = "Encourage predators"


class TreatmentEngine:
    """
    Loads an embedded organic-only treatment database and produces
    severity-aware, IPM-consistent recommendations.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
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
                    self._method(BACILLUS_THURINGIENSIS, "Bt spray", "high", "$$"),
                ],
                "cultural": [
                    self._method(ROW_COVERS, "Exclude moths", "high", "$$"),
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
                    self._method(BACILLUS_THURINGIENSIS, "Bt var. tenebrionis", "high", "$$"),
                    self._method("Neem oil", "Growth regulator", "medium", "$"),
                ],
                "cultural": [
                    self._method(CROP_ROTATION, "Break life cycle", "high", "$"),
                ],
                "preventive": [
                    "Rotate solanaceous crops every 3 years",
                    "Deep fall cultivation to expose overwintering adults",
                ],
            },
            "Cucumber Beetle": {
                "mechanical": [
                    self._method(ROW_COVERS, "Exclude beetles", "high", "$$"),
                ],
                "biological": [
                    self._method(BENEFICIAL_NEMATODES, "Soil application", "medium", "$$"),
                    self._method("Kaolin clay", "Repellent spray", "medium", "$"),
                ],
                "cultural": [
                    self._method(TRAP_CROPS, "Plant radishes as decoy", "medium", "$"),
                ],
                "preventive": [
                    "Delay planting until after peak beetle emergence",
                    "Remove crop residue after harvest",
                ],
            },
            "Flea Beetle": {
                "mechanical": [
                    self._method(ROW_COVERS, "Physical barrier", "high", "$$"),
                ],
                "biological": [
                    self._method(BENEFICIAL_NEMATODES, "Target larvae", "medium", "$$"),
                    self._method("Diatomaceous earth", "Dust on plants", "medium", "$"),
                ],
                "cultural": [
                    self._method(TRAP_CROPS, "Plant radishes early", "medium", "$"),
                ],
                "preventive": [
                    "Till soil in fall to expose overwintering adults",
                    "Plant after peak emergence in late spring",
                ],
            },
            "Ants": {
                "mechanical": [
                    self._method("Diatomaceous earth barriers", "Food-grade DE around plants", "medium", "$"),
                    self._method("Cinnamon powder", "Natural deterrent", "low", "$"),
                ],
                "biological": [
                    self._method(BENEFICIAL_NEMATODES, "Target ant larvae", "medium", "$$"),
                    self._method("Borax bait stations", "Organic-approved ant killer", "high", "$"),
                ],
                "cultural": [
                    self._method("Remove food sources", "Clean up fallen fruit", "high", "$"),
                    self._method("Eliminate moisture", "Fix irrigation leaks", "medium", "$"),
                ],
                "preventive": [
                    "Keep garden clean of debris and fallen fruit",
                    "Seal cracks in garden structures",
                ],
            },
            "Beneficial Insects": {
                "mechanical": [
                    self._method("Protection structures", "Provide nesting sites", "high", "$"),
                ],
                "biological": [
                    self._method("Encourage habitat", "Plant native flowers", "high", "$"),
                    self._method("Avoid disturbance", "Minimize soil disruption", "high", "$"),
                ],
                "cultural": [
                    self._method("Diverse plantings", "Year-round bloom succession", "high", "$"),
                    self._method("Reduce pesticide use", "Allow natural balance", "high", "$"),
                ],
                "preventive": [
                    "These are beneficial organisms - focus on protection and encouragement",
                    "Provide overwintering habitat and diverse food sources",
                ],
            },
            "Beetles": {
                "mechanical": [
                    self._method("Hand picking", "Remove adult beetles", "medium", "$"),
                    self._method(ROW_COVERS, "Physical exclusion", "high", "$$"),
                ],
                "biological": [
                    self._method(BENEFICIAL_NEMATODES, "Target soil-dwelling larvae", "medium", "$$"),
                    self._method("Spinosad spray", "Organic insecticide", "high", "$$"),
                ],
                "cultural": [
                    self._method(CROP_ROTATION, "Break pest cycle", "high", "$"),
                    self._method(TRAP_CROPS, "Concentrate beetles for removal", "medium", "$"),
                ],
                "preventive": [
                    "Clean up garden debris in fall",
                    "Use resistant plant varieties when available",
                ],
            },
            "Earwigs": {
                "mechanical": [
                    self._method("Newspaper traps", "Roll up newspaper, place in garden", "medium", "$"),
                    self._method("Oil pit traps", "Shallow containers with oil", "medium", "$"),
                ],
                "biological": [
                    self._method(ENCOURAGE_PREDATORS, "Birds and ground beetles", "medium", "$"),
                    self._method("Diatomaceous earth", "Apply around affected plants", "medium", "$"),
                ],
                "cultural": [
                    self._method("Remove hiding places", "Clear debris and mulch", "medium", "$"),
                    self._method("Improve drainage", "Reduce moist conditions", "medium", "$"),
                ],
                "preventive": [
                    "Keep garden areas dry and well-ventilated",
                    "Remove excessive organic matter near sensitive plants",
                ],
            },
            "Grasshoppers": {
                "mechanical": [
                    self._method(ROW_COVERS, "Physical barrier for young plants", "high", "$$"),
                    self._method("Hand collection", "Early morning when sluggish", "medium", "$"),
                ],
                "biological": [
                    self._method("Nosema locustae", "Biological control agent", "medium", "$$"),
                    self._method("Encourage birds", "Provide perches and habitat", "medium", "$"),
                ],
                "cultural": [
                    self._method("Habitat modification", "Remove weedy areas", "medium", "$"),
                    self._method("Timing of plantings", "Avoid vulnerable growth stages", "medium", "$"),
                ],
                "preventive": [
                    "Maintain border strips of grass away from crops",
                    "Encourage natural predators like birds and spiders",
                ],
            },
            "Moths": {
                "mechanical": [
                    self._method("Pheromone traps", "Monitor and trap adults", "medium", "$$"),
                    self._method("Light traps", "Attract and capture night-flying moths", "medium", "$$"),
                ],
                "biological": [
                    self._method(BACILLUS_THURINGIENSIS, "Bt spray for larvae", "high", "$$"),
                    self._method("Parasitic wasps", "Release beneficial insects", "high", "$$$"),
                ],
                "cultural": [
                    self._method(CROP_ROTATION, "Break reproduction cycle", "high", "$"),
                    self._method("Sanitation", "Remove crop residues", "medium", "$"),
                ],
                "preventive": [
                    "Time plantings to avoid peak moth activity",
                    "Use resistant varieties when available",
                ],
            },
            "Slugs and Snails": {
                "mechanical": [
                    self._method("Copper barriers", "Copper tape around plants", "high", "$$"),
                    self._method("Beer traps", "Shallow dishes with beer", "medium", "$"),
                    self._method("Hand picking", "Evening collection", "high", "$"),
                ],
                "biological": [
                    self._method("Iron phosphate bait", "OMRI-listed slug killer", "high", "$$"),
                    self._method(ENCOURAGE_PREDATORS, "Ground beetles, birds", "medium", "$"),
                ],
                "cultural": [
                    self._method("Reduce moisture", "Improve drainage and spacing", "medium", "$"),
                    self._method("Remove shelter", "Clear debris and hiding spots", "medium", "$"),
                ],
                "preventive": [
                    "Water in morning to reduce evening moisture",
                    "Maintain clean, well-drained garden beds",
                ],
            },
            "Wasps": {
                "mechanical": [
                    self._method("Physical removal", "Relocate nests if possible", "medium", "$"),
                    self._method("Protective barriers", "Fine mesh over vulnerable plants", "medium", "$$"),
                ],
                "biological": [
                    self._method("Encourage beneficial wasps", "Provide nectar sources", "high", "$"),
                    self._method("Natural deterrents", "Peppermint oil spray", "low", "$"),
                ],
                "cultural": [
                    self._method("Habitat management", "Balance beneficial vs. pest species", "medium", "$"),
                    self._method("Reduce attractants", "Cover sweet fruits", "medium", "$"),
                ],
                "preventive": [
                    "Many wasps are beneficial predators - identify species first",
                    "Provide alternative habitat away from sensitive areas",
                ],
            },
            "Weevils": {
                "mechanical": [
                    self._method("Trunk bands", "Sticky barriers on tree trunks", "medium", "$$"),
                    self._method("Shake and collect", "Early morning collection", "medium", "$"),
                ],
                "biological": [
                    self._method(BENEFICIAL_NEMATODES, "Target soil-dwelling larvae", "high", "$$"),
                    self._method(ENCOURAGE_PREDATORS, "Ground beetles and birds", "medium", "$"),
                ],
                "cultural": [
                    self._method("Sanitation", "Remove fallen nuts and fruits", "high", "$"),
                    self._method("Proper storage", "Sealed containers for grains", "high", "$"),
                ],
                "preventive": [
                    "Inspect stored products regularly",
                    "Maintain clean storage areas",
                ],
            },
        }

    @staticmethod
    def _default_imp_principles() -> dict:
        return {
            "prevention": [
                CROP_ROTATION,
                "Resistant cultivars",
            ],
            "cultural_controls": [
                "Sanitation",
                TRAP_CROPS,
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
