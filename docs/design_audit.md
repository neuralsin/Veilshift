# QT-2.23 Design Audit — Phase 1

## Stitch Screen Inventory (Project 6046812853660105375)

### Screen 1: System Overview - Quantum Sensor Fusion (Revised)
- **Purpose**: Landing page showing sensor evidence and fused detection outcome
- **Layout**: Sidebar (210px) | Topbar (56px) | Main workspace | Status strip (32px)
- **Panels**: 4 metric cards (Target Regime, Active Modalities, Fused AUC w/ CI, FAR), 3 sensor evidence cards (Radar/Thermal/Acoustic with mini-viz, score, weight, rank, SNR), central fusion node (SVG with dashed animated paths), trust distribution bar, system inference panel
- **Colors observed**: `#0A0C0E` (bg-dim), `#121416` (surface), `#15181C` (card), `#1C2025` (elevated), `#2D333B` (border), `#e2e2e5` (text-primary), `#bbc9cf` (text-variant), `#8B949E` (text-muted), `#38BDF8` (radar-cyan), `#F59E0B` (thermal-amber), `#A78BFA` (acoustic-violet), `#FACC15` (quantum-gold), `#34C759` (success), `#FFCC00` (warning), `#FF3B30` (critical)
- **Typography**: Inter (UI font, 400/500/600), JetBrains Mono (monospace), sizes: 28px (primary metric), 22px (page title), 14px (card heading/body), 11px (labels/metadata/captions, uppercase tracking-wide)
- **Spacing**: 20px page padding, 12px grid gap, 16px card internal padding
- **Radii**: ~4-6px for cards/buttons (rounded)
- **Nav items**: 36px height, 11px uppercase text, Material Symbols Outlined icons at 16px
- **Active nav**: right border 2px radar-cyan, bg-surface-elevated, text radar-cyan

### Screen 2: Radar Sensor - Physics Simulation & SNR Analysis
- **Purpose**: Radar parameter configuration and signal analysis workspace
- **Layout**: Sidebar | Topbar | Left parameter rail | Center chart workspace with tabs | Right equation/inspector panel
- **Tabs observed**: RANGE PROFILE, RANGE-DOPPLER, CFAR, DISTRIBUTION
- **Parameter groups**: Radar Hardware (Pt, G, freq, B, F, L, R), Target (type dropdown, RCS), Clutter (model, power, shape), CFAR (Pfa, ref cells, guard, MC trials)
- **Right panel**: Equation card with live substituted values, Physical Inference text, Validation panel with PASS/CHECK/FAILED status pills

### Screen 3: Feature Space - QUBO Selection (Revised)
- **Purpose**: Feature pool visualization, correlation matrix, QUBO construction and optimization
- **Layout**: Sidebar | Topbar | Summary cards at top | Main split (correlation heatmap left, relevance bars right) | Bottom QUBO workspace (Q matrix heatmap, controls, selected feature chips)
- **Feature chips**: Sensor-colored left border, with relevance/redundancy values, selected=high emphasis, rejected=low emphasis
- **Validation panel**: Brute force vs solver objective comparison with MATCH/INVESTIGATE status

### Screen 4: Baseline Comparison - Performance Metrics (Revised)
- **Purpose**: Side-by-side classical baseline comparison table, ROC curves, confusion matrix
- **Layout**: Sidebar | Topbar | Comparison table at top | ROC chart center-left | Confusion matrix center-right | Operating point chart bottom-left
- **Table columns**: Method, AUC, 95% CI, Detection Rate, FAR, Precision, Recall, F1, Objective, Solve Time
- **ROC chart style**: Dark background, sensor-colored lines, AUC in legend
- **Important note**: "QuantumNow and SLSQP optimize the same Rayleigh-LDA fusion objective" displayed as experiment note

### Screen 5: Degradation Lab - Adaptive Trust Migration (Revised)
- **Purpose**: Hero demonstration page — degradation sweep with synchronized charts
- **Layout**: Sidebar | Topbar | Scenario builder at top | Main split (Detection Retention chart left, Trust Migration chart right) | Side panel (before/after weights, trust migration %) | Bottom contribution analysis
- **Synchronized cursor**: Vertical cursor across both charts, updates weight bars and metrics
- **Severity slider**: 0-100%, shows physical parameter mapping
- **Two comparison lines**: Static Fusion (desaturated) vs Adaptive Optimized Fusion (quantum-gold accent)

---

## Consolidated Design Token Table

### Colors
| Token | Hex | Usage |
|-------|-----|-------|
| `bg_darkest` | `#0A0C0E` | Application background, workspace bg |
| `bg_surface` | `#121416` | Primary surfaces, topbar |
| `bg_card` | `#15181C` | Card backgrounds |
| `bg_elevated` | `#1C2025` | Elevated surfaces, hover states, active nav |
| `border_subtle` | `#2D333B` | Card borders, dividers |
| `border_hover` | `#3C494E` | Hover borders, outline variant |
| `text_primary` | `#E2E2E5` | Primary text (off-white) |
| `text_secondary` | `#BBC9CF` | Secondary text, nav labels |
| `text_muted` | `#8B949E` | Muted text, metadata, disabled |
| `radar` | `#38BDF8` | Radar identity, active accent |
| `thermal` | `#F59E0B` | Thermal/IR identity |
| `acoustic` | `#A78BFA` | Acoustic/Sonar identity |
| `quantum_gold` | `#FACC15` | Quantum-optimized / premium result |
| `classical` | `#8B949E` | Classical baseline (desaturated) |
| `success` | `#34C759` | Validated, pass, online |
| `warning` | `#FFCC00` | Warning, degraded |
| `critical` | `#FF3B30` | Failed, error, critical |

### Typography
| Level | Font | Size | Weight | Extras |
|-------|------|------|--------|--------|
| Page title | Inter | 22px | 600 | — |
| Page subtitle | Inter | 11px | 400 | text-secondary |
| Section heading | Inter | 14px | 600 | — |
| Card heading | Inter | 14px | 600 | — |
| Primary metric | JetBrains Mono | 28px | 600 | — |
| Secondary metric | JetBrains Mono | 14px | 400 | — |
| Label/metadata | Inter | 11px | 500 | UPPERCASE, tracking-wide |
| Caption | Inter | 11px | 400 | text-muted |
| Status text | Inter | 11px | 500 | Colored per state |
| Monospace values | JetBrains Mono | 14px | 400 | — |

### Spacing
| Token | Value |
|-------|-------|
| `sidebar_width` | 210px |
| `topbar_height` | 56px |
| `statusbar_height` | 32px |
| `page_padding` | 20px |
| `grid_gap` | 12px |
| `card_padding` | 16px |
| `nav_item_height` | 36px |
| `metric_card_height` | 100px |
| `sensor_card_height` | 280px |
| `button_height` | 36px |
| `input_height` | 32px |

### Corner Radii
| Element | Radius |
|---------|--------|
| Cards | 4-6px |
| Buttons | 4px |
| Status pills | 4px |
| Input fields | 4px |
| Major containers | 8px max |

### Component Styles
- **Card**: `bg_card` background, 1px `border_subtle` border, 4-6px radius
- **Active nav item**: Right border 2px `radar` color, `bg_elevated` background
- **Metric card**: Top accent border 2px for emphasis variant
- **Status pill**: `success`/`warning`/`critical` text on 10% opacity background
- **Button default**: `bg_elevated` background, `text_primary` text, 4px radius
- **Button hover**: Lighter bg, slight border change
- **Button running**: Shows elapsed time, subtle activity indicator

---

## Page Mapping — 14 Navigation Pages

### 5 Pages with Stitch Reference:
1. ✅ **System Overview** — Screen "System Overview - Quantum Sensor Fusion (Revised)"
2. ✅ **Radar** — Screen "Radar Sensor - Physics Simulation & SNR Analysis"
3. ✅ **Feature Space** — Screen "Feature Space - QUBO Selection (Revised)"
4. ✅ **Baselines** — Screen "Baseline Comparison - Performance Metrics (Revised)"
5. ✅ **Degradation Lab** — Screen "Degradation Lab - Adaptive Trust Migration (Revised)"

### 9 Pages Built from Manual + Extracted Token System:

6. **Thermal/IR** — Mirror the Radar page layout: left parameter rail (Background T/ε, Target T/ε, Detector NETD/noise type/alpha/frame size), center tabbed workspace (Thermal Frame, Contrast Map, Noise Spectrum, Features), right inspector (Stefan-Boltzmann equation, live values, inference). Same component library, same spacing.

7. **Acoustic/Sonar** — Mirror the Radar page layout: left parameter rail (Source Level, Environment range/sea state/NL/DI/DT/absorption, Signal duration/fs/tonals, STFT window/nperseg/overlap), center tabbed workspace (Time Series, LOFAR spectrogram, Tonal Analysis, Features), right inspector (Sonar equation, live values, inference).

8. **Fusion Optimization** — Top: 3 compact sensor score distribution density plots. Center-left: Fisher separation before/after visualization. Center-right: Trust distribution weight bars. Solver config section. Bottom: Optimizer comparison table (CRLB/SLSQP/DE/Backup). Right: Equation panel with collapsible derivation.

9. **Contribution** — Top: 3 sensor cards showing Weight/Ablation ΔAUC/Mean|SHAP|. Center: Normalized grouped bar chart (3 methods × 3 sensors) with Agreement Status indicator. Bottom: Feature-level SHAP chart with sensor filter. Right: Inference panel.

10. **Scaling** — Top: Configuration (problem sizes, solver selection). Main: Dual chart (Solve Time vs Problem Size log-scale + Normalized Objective Quality vs Problem Size). Bottom: Scaling results table. Inference panel.

11. **Experiments** — Scientific experiment registry table (ID, Name, Scenario, Target Type, Modalities, Solver, AUC w/ CI, Detection Rate, FAR, Created, Status). Search/filter/sort. Compare mode showing parameter diffs then metrics.

12. **Solver** — Top: Backend status (UNAVAILABLE/BACKUP ACTIVE with honest label). Solver modes list. Current job details (ID, experiment, problem type, variables, Q size, objective, elapsed, status). Solver history table. Raw metadata inspector.

13. **Logs** — Filter tabs (ALL/SIMULATION/FEATURES/OPTIMIZATION/EVALUATION/SOLVER/WARNING/ERROR). Log entries: timestamp, module, level, message, experiment ID. Monospace font. Copy/export.

14. **Settings** — Grouped settings: General (theme, UI scale, chart DPI, animation, export dir), Experiment Defaults (seed, sample count, folds, bootstrap, CI), Solver (default, fallback, timeout, auto-fallback), Visualization (scientific notation, annotation density, confidence bands, sensor color preview), Developer (debug mode, event logging, profiling).
