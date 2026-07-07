"""
QT-2.23 — UI Pages Package

All 14 navigation pages. Each page is a CTkFrame subclass with
a refresh(experiment) method for live data updates.
"""

from ui.pages.mission_control import MissionControlPage
from ui.pages.radar import RadarPage
from ui.pages.thermal import ThermalPage
from ui.pages.acoustic import AcousticPage
from ui.pages.feature_space import FeatureSpacePage
from ui.pages.fusion import FusionPage
from ui.pages.baselines import BaselinesPage
from ui.pages.contribution import ContributionPage
from ui.pages.degradation import DegradationPage
from ui.pages.scaling import ScalingPage
from ui.pages.experiments import ExperimentsPage
from ui.pages.solver import SolverPage
from ui.pages.logs import LogsPage
from ui.pages.settings import SettingsPage

__all__ = [
    "MissionControlPage", "RadarPage", "ThermalPage", "AcousticPage",
    "FeatureSpacePage", "FusionPage", "BaselinesPage", "ContributionPage",
    "DegradationPage", "ScalingPage", "ExperimentsPage", "SolverPage",
    "LogsPage", "SettingsPage",
]
