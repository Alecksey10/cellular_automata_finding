"""Предметные модули: данные, инициализация, окрестности, правила, фильтры."""

from .data.discrete2d import DiscreteGrid2D
from .filters.band import ActivityFilter, FillBandFilter
from .filters.clouds import CloudFilter
from .filters.cycle import PeriodFilter
from .filters.density import DensityFilter
from .filters.drift import RotStructureFilter, ZeroShareFilter
from .filters.squares import MonochromeSquaresFilter
from .filters.stability import StabilityFilter
from .filters.structure import (
    ClusterFilter,
    EntropyFilter,
    PhaseBalanceFilter,
    SymmetryFilter,
)
from .neighborhoods.moore import Moore3Neighborhood, Moore5Neighborhood
from .neighborhoods.radius import RadiusNeighborhood
from .neighborhoods.ring import RingNeighborhood
from .neighborhoods.square import SquareNeighborhood
from .neighborhoods.two_rings import TwoRingNeighborhood
from .randomizers.simple import SparseRandomizer, UniformRandomizer
from .rules.aging import AgingFortressRule
from .rules.cyclic import CyclicRule
from .rules.life import LifeRule
from .rules.parity import ParityRule
from .rules.refractory import RefractoryRule
from .rules.rot import RotAgingRule
from .rules.thresholds import ThresholdRule
from .rules.totalistic import TotalisticRule
from .rules.vote import SymmetricVoteRule

__all__ = [
    "ActivityFilter",
    "AgingFortressRule",
    "ClusterFilter",
    "CloudFilter",
    "CyclicRule",
    "DensityFilter",
    "DiscreteGrid2D",
    "EntropyFilter",
    "FillBandFilter",
    "LifeRule",
    "MonochromeSquaresFilter",
    "Moore3Neighborhood",
    "Moore5Neighborhood",
    "ParityRule",
    "PeriodFilter",
    "PhaseBalanceFilter",
    "RadiusNeighborhood",
    "RefractoryRule",
    "RingNeighborhood",
    "RotAgingRule",
    "RotStructureFilter",
    "SparseRandomizer",
    "SquareNeighborhood",
    "StabilityFilter",
    "SymmetricVoteRule",
    "SymmetryFilter",
    "ThresholdRule",
    "TotalisticRule",
    "TwoRingNeighborhood",
    "UniformRandomizer",
    "ZeroShareFilter",
]
