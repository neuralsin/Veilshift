"""
QT-2.23 — Quantum Sensor Fusion & Stealth Detection Console
Design tokens extracted from Stitch MCP screens (Project 6046812853660105375).

This module is the single source of truth for all visual constants.
No other module should hardcode colors, font sizes, or spacing values.
"""


# ============================================================
# COLORS — extracted from Stitch Mission Control + all 5 screens
# ============================================================

class Colors:
    """Semantic color tokens. Every color in the UI traces here."""

    # Backgrounds / Surfaces (darkest → lightest)
    BG_DARKEST = "#0A0C0E"       # Application background, workspace
    BG_SURFACE = "#121416"       # Primary surfaces, topbar
    BG_CARD = "#15181C"          # Card backgrounds
    BG_ELEVATED = "#1C2025"      # Elevated surfaces, hover states, active nav

    # Borders
    BORDER_SUBTLE = "#2D333B"    # Card borders, dividers, separators
    BORDER_HOVER = "#3C494E"     # Hover borders, focus outlines

    # Text hierarchy
    TEXT_PRIMARY = "#E2E2E5"     # Primary text (soft off-white, never pure #FFF)
    TEXT_SECONDARY = "#BBC9CF"   # Secondary text, nav labels
    TEXT_MUTED = "#8B949E"       # Muted text, metadata, captions
    TEXT_DISABLED = "#5C6370"    # Disabled text

    # Sensor identity colors — MUST remain consistent across entire app
    RADAR = "#38BDF8"            # Cool technical cyan
    THERMAL = "#F59E0B"          # Warm amber
    ACOUSTIC = "#A78BFA"         # Distinct violet

    # Optimization / Result identity
    QUANTUM_GOLD = "#FACC15"     # Quantum-optimized / premium result accent
    CLASSICAL = "#8B949E"        # Classical baselines (desaturated neutral)

    # Semantic states
    SUCCESS = "#34C759"          # Validated, pass, online, completed
    WARNING = "#FFCC00"          # Warning, degraded sensor
    CRITICAL = "#FF3B30"         # Failed, error, critical

    # Chart-specific
    H0_COLOR = "#5C6370"         # H0/No-target (lower emphasis)
    H1_COLOR = "#E2E2E5"         # H1/Target (higher emphasis)
    STATIC_LINE = "#8B949E"      # Static fusion line in degradation
    ADAPTIVE_LINE = "#FACC15"    # Adaptive fusion line in degradation

    # Sensor colors as a list/dict for iteration
    SENSOR_COLORS = {
        "radar": "#38BDF8",
        "thermal": "#F59E0B",
        "acoustic": "#A78BFA",
    }

    SENSOR_COLORS_LIST = ["#38BDF8", "#F59E0B", "#A78BFA"]


# ============================================================
# TYPOGRAPHY — Inter (UI) + JetBrains Mono (monospace/scientific)
# ============================================================

class Typography:
    """Font families and type scale."""

    # Fonts
    UI_FONT = "Inter"
    MONO_FONT = "JetBrains Mono"

    # Fallbacks for systems without Inter/JetBrains Mono installed
    UI_FONT_FALLBACK = ("Inter", "Segoe UI", "Helvetica", "Arial", "sans-serif")
    MONO_FONT_FALLBACK = ("JetBrains Mono", "Consolas", "Courier New", "monospace")

    # Type scale (size, weight) — extracted from Stitch screens
    PAGE_TITLE = (22, "bold")          # Page title in topbar
    PAGE_SUBTITLE = (11, "normal")     # Subtitle below page title
    SECTION_HEADING = (14, "bold")     # Section/card headings
    CARD_HEADING = (14, "bold")        # Card title text
    PRIMARY_METRIC = (28, "bold")      # Large metric values (JetBrains Mono)
    SECONDARY_METRIC = (14, "normal")  # Smaller metric values (JetBrains Mono)
    LABEL = (11, "bold")              # Uppercase labels/metadata
    CAPTION = (11, "normal")           # Captions, muted text
    STATUS_TEXT = (11, "bold")         # Status pills
    BODY = (12, "normal")              # Body text, inference panels
    NAV_ITEM = (11, "bold")           # Navigation labels (uppercase)
    BUTTON_TEXT = (11, "bold")         # Button labels
    EQUATION = (13, "normal")          # Equation text (JetBrains Mono)
    TOOLTIP = (11, "normal")           # Tooltip text


# ============================================================
# SPACING — layout dimensions extracted from Stitch screens
# ============================================================

class Spacing:
    """Layout spacing constants."""

    # Shell dimensions
    SIDEBAR_WIDTH = 210
    TOPBAR_HEIGHT = 56
    STATUSBAR_HEIGHT = 32
    PAGE_PADDING = 20
    GRID_GAP = 12

    # Component dimensions
    CARD_PADDING = 16
    NAV_ITEM_HEIGHT = 36
    METRIC_CARD_HEIGHT = 100
    SENSOR_CARD_HEIGHT = 280
    BUTTON_HEIGHT = 36
    INPUT_HEIGHT = 32

    # Spacing scale
    XS = 4
    SM = 8
    MD = 12
    LG = 16
    XL = 20
    XXL = 24


# ============================================================
# CORNER RADII — restrained, desktop research software
# ============================================================

class Radii:
    """Corner radius constants. 6-10px standard, 12px max per Frontend manual."""

    SMALL = 4     # Buttons, inputs, status pills
    DEFAULT = 6   # Cards, panels
    MEDIUM = 8    # Larger containers
    LARGE = 10    # Major containers (rare)
    MAX = 12      # Maximum allowed


# ============================================================
# CHART STYLING — Matplotlib theme constants
# ============================================================

class ChartStyle:
    """Constants for Matplotlib chart rendering, consistent with app theme."""

    FIGURE_FACECOLOR = "#0A0C0E"
    AXES_FACECOLOR = "#15181C"
    AXES_EDGECOLOR = "#2D333B"
    GRID_COLOR = "#2D333B"
    GRID_ALPHA = 0.5
    TEXT_COLOR = "#E2E2E5"
    TICK_COLOR = "#8B949E"
    LABEL_COLOR = "#BBC9CF"
    TITLE_SIZE = 12
    LABEL_SIZE = 10
    TICK_SIZE = 9
    LEGEND_SIZE = 9
    DPI = 100
    DPI_EXPORT = 300

    # Line styles
    LINE_WIDTH = 1.5
    LINE_WIDTH_EMPHASIS = 2.0
    MARKER_SIZE = 4

    # H0/H1 styling
    H0_STYLE = {"color": "#5C6370", "linestyle": "--", "alpha": 0.7}
    H1_STYLE = {"color": "#E2E2E5", "linestyle": "-", "alpha": 0.9}


# ============================================================
# ICON MAPPING — Material Symbols Outlined equivalents for CTk
# Since CustomTkinter doesn't natively support Material icons,
# we map nav items to Unicode/text labels. CTk image-based icons
# can be loaded from PNG assets if available.
# ============================================================

NAV_ICONS = {
    "System Overview": "⊞",
    "Radar": "◎",
    "Thermal / IR": "◈",
    "Acoustic / Sonar": "≋",
    "Feature Space": "⊡",
    "Fusion Optimization": "⊕",
    "Baselines": "≡",
    "Contribution": "⊿",
    "Degradation Lab": "⊘",
    "Scaling": "⊶",
    "Experiments": "☰",
    "Solver": "⚙",
    "Logs": "≣",
    "Settings": "⚙",
}

# Navigation groups per Frontend manual Section 2
NAV_GROUPS = {
    "OVERVIEW": ["System Overview"],
    "SENSORS": ["Radar", "Thermal / IR", "Acoustic / Sonar"],
    "FUSION": ["Feature Space", "Fusion Optimization"],
    "EVALUATION": ["Baselines", "Contribution", "Degradation Lab", "Scaling"],
    "SYSTEM": ["Experiments", "Solver", "Logs", "Settings"],
}
