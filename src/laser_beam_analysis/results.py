"""Central container for the results of a full beam analysis.

Public API:
    BeamAnalysisResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BeamAnalysisResult:
    """Single container for all results of a beam analysis.

    Used as an intermediate API between the scientific core and any future
    GUI / reporting layer. All fields are optional so partial results can be
    constructed incrementally.
    """

    # Images
    raw_image: Any = None
    aligned_image: Any = None
    processed_image: Any = None
    basex_image: Any = None
    hansenlaw_image: Any = None

    # Metadata
    image_shape: Optional[tuple] = None

    # Pipeline stages
    alignment: dict = field(default_factory=dict)
    preprocessing: dict = field(default_factory=dict)
    center: dict = field(default_factory=dict)
    symmetry: dict = field(default_factory=dict)
    abel: dict = field(default_factory=dict)

    # Longitudinal profiles
    profiles_basex: list = field(default_factory=list)
    profiles_hansenlaw: list = field(default_factory=list)

    # Downstream analysis
    validation: dict = field(default_factory=dict)
    waist: dict = field(default_factory=dict)
    propagation: dict = field(default_factory=dict)

    # Configuration & metrology
    config: dict = field(default_factory=dict)
    psf_profiles_basex: list = field(default_factory=list)
    psf_profiles_hansenlaw: list = field(default_factory=list)
    metrology: dict = field(default_factory=dict)
    camera: dict = field(default_factory=dict)

    # Audit
    audit: dict = field(default_factory=dict)