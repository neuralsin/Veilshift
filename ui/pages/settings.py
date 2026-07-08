"""QT-2.23 — Settings Page"""
from __future__ import annotations
import customtkinter as ctk
from ui.theme.tokens import Colors, Typography, Spacing, Radii
from ui.components import SectionFrame, ParameterField, ActionButton
from app.state import ExperimentState, TargetRegime


class SettingsPage(ctk.CTkScrollableFrame):
    def __init__(self, master, app_shell):
        super().__init__(master, fg_color=Colors.BG_DARKEST, scrollbar_button_color=Colors.BORDER_SUBTLE)
        self._app = app_shell
        self._build_ui()

    def _build_ui(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=Spacing.PAGE_PADDING, pady=Spacing.PAGE_PADDING)

        # Observability Level (Regime Selection)
        obs_section = SectionFrame(container, title="Observability Level")
        obs_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        ctk.CTkLabel(obs_section.content, text="TARGET SIGNATURE REGIME",
                     font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(anchor="w", padx=Spacing.XS, pady=(4, 2))

        self._regime_select = ctk.CTkOptionMenu(
            obs_section.content,
            values=[
                "CONVENTIONAL (100%)",
                "REDUCED SIGNATURE (75%)",
                "LOW OBSERVABILITY (50%)",
                "NEAR-BACKGROUND (25%)",
                "STEALTH (0%)",
            ],
            font=(Typography.UI_FONT, 11), fg_color=Colors.BG_DARKEST, button_color=Colors.BG_ELEVATED,
            dropdown_fg_color=Colors.BG_CARD, text_color=Colors.TEXT_PRIMARY, height=Spacing.INPUT_HEIGHT,
        )
        self._regime_select.set("STEALTH (0%)")
        self._regime_select.pack(fill="x", pady=2)

        # Global Settings
        global_section = SectionFrame(container, title="Global Settings")
        global_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._seed = ParameterField(global_section.content, label="Default Seed", default="42")
        self._seed.pack(fill="x", pady=2)

        self._samples = ParameterField(global_section.content, label="Sample Count", default="6000")
        self._samples.pack(fill="x", pady=2)

        self._bootstrap = ParameterField(global_section.content, label="Bootstrap N", default="1000")
        self._bootstrap.pack(fill="x", pady=2)

        self._ci_level = ParameterField(global_section.content, label="CI Level", default="0.95")
        self._ci_level.pack(fill="x", pady=2)

        self._folds = ParameterField(global_section.content, label="CV Folds", default="5")
        self._folds.pack(fill="x", pady=2)

        self._seeds = ParameterField(global_section.content, label="Multi-seed N", default="5")
        self._seeds.pack(fill="x", pady=2)

        # QUBO Settings
        qubo_section = SectionFrame(container, title="QUBO Configuration")
        qubo_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._alpha = ParameterField(qubo_section.content, label="α (relevance)", default="1.0")
        self._alpha.pack(fill="x", pady=2)

        self._beta = ParameterField(qubo_section.content, label="β (redundancy)", default="1.0")
        self._beta.pack(fill="x", pady=2)

        self._gamma = ParameterField(qubo_section.content, label="γ (cardinality)", default="2.0")
        self._gamma.pack(fill="x", pady=2)

        self._k_target = ParameterField(qubo_section.content, label="k (target features)", default="6")
        self._k_target.pack(fill="x", pady=2)

        ctk.CTkLabel(qubo_section.content, text="Selection Method", font=(Typography.UI_FONT, 11, "bold"),
                     text_color=Colors.TEXT_SECONDARY).pack(anchor="w", padx=Spacing.XS, pady=(4, 2))
        self._fs_method = ctk.CTkOptionMenu(
            qubo_section.content, values=["MID (Default)", "MIQ (Dinkelbach)"],
            font=(Typography.UI_FONT, 11), fg_color=Colors.BG_DARKEST, button_color=Colors.BG_ELEVATED,
            dropdown_fg_color=Colors.BG_CARD, text_color=Colors.TEXT_PRIMARY, height=Spacing.INPUT_HEIGHT,
        )
        self._fs_method.pack(fill="x", pady=2)

        self._db_damping = ParameterField(qubo_section.content, label="Dinkelbach Damping", default="0.5")
        self._db_damping.pack(fill="x", pady=2)

        self._db_max_iter = ParameterField(qubo_section.content, label="Dinkelbach Max Iterations", default="15")
        self._db_max_iter.pack(fill="x", pady=2)

        self._num_reads = ParameterField(qubo_section.content, label="SA reads", default="1000")
        self._num_reads.pack(fill="x", pady=2)

        # Fusion Settings
        fusion_section = SectionFrame(container, title="Fusion Configuration")
        fusion_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._lam = ParameterField(fusion_section.content, label="λ (Lagrangian)", default="0.5")
        self._lam.pack(fill="x", pady=2)

        self._restarts = ParameterField(fusion_section.content, label="SLSQP restarts", default="10")
        self._restarts.pack(fill="x", pady=2)

        self._bits = ParameterField(fusion_section.content, label="Bits/weight", default="4")
        self._bits.pack(fill="x", pady=2)

        self._penalty = ParameterField(fusion_section.content, label="Simplex penalty", default="5.0")
        self._penalty.pack(fill="x", pady=2)

        self._target_far = ParameterField(fusion_section.content, label="Target FAR", default="0.01")
        self._target_far.pack(fill="x", pady=2)

        # Display Settings
        display_section = SectionFrame(container, title="Display")
        display_section.pack(fill="x", pady=(0, Spacing.GRID_GAP))

        self._debug = ctk.CTkSwitch(
            display_section.content, text="Debug Mode",
            font=(Typography.UI_FONT, 11), text_color=Colors.TEXT_SECONDARY,
            fg_color=Colors.BG_DARKEST, progress_color=Colors.RADAR,
        )
        self._debug.pack(anchor="w", pady=2)

        self._animations = ctk.CTkSwitch(
            display_section.content, text="Animations Enabled",
            font=(Typography.UI_FONT, 11), text_color=Colors.TEXT_SECONDARY,
            fg_color=Colors.BG_DARKEST, progress_color=Colors.RADAR,
        )
        self._animations.pack(anchor="w", pady=2)
        self._animations.select()

        # Apply button
        ActionButton(container, text="APPLY SETTINGS", command=self._apply, primary=True).pack(
            anchor="w", pady=Spacing.LG)

    def refresh(self, exp: ExperimentState):
        from app.state import FeatureSelectionMethod

        # Sync regime selector
        regime_map = {
            TargetRegime.STEALTH: "STEALTH (0%)",
            TargetRegime.NEAR_BACKGROUND: "NEAR-BACKGROUND (25%)",
            TargetRegime.LOW_OBSERVABILITY: "LOW OBSERVABILITY (50%)",
            TargetRegime.REDUCED_SIGNATURE: "REDUCED SIGNATURE (75%)",
            TargetRegime.CONVENTIONAL: "CONVENTIONAL (100%)",
        }
        self._regime_select.set(regime_map.get(exp.target_regime, "STEALTH (0%)"))

        self._seed.set_value(str(exp.seed))
        self._samples.set_value(str(exp.radar_config.num_samples))
        self._alpha.set_value(str(exp.feature_config.alpha))
        self._beta.set_value(str(exp.feature_config.beta))
        self._gamma.set_value(str(exp.feature_config.gamma))
        self._k_target.set_value(str(exp.feature_config.k_target))
        self._fs_method.set("MIQ (Dinkelbach)" if exp.feature_config.method == FeatureSelectionMethod.MIQ_DINKELBACH else "MID (Default)")
        self._db_damping.set_value(str(exp.feature_config.dinkelbach_damping))
        self._db_max_iter.set_value(str(exp.feature_config.dinkelbach_max_iter))
        self._num_reads.set_value(str(exp.feature_config.num_reads))
        self._lam.set_value(str(exp.fusion_config.lam))
        self._restarts.set_value(str(exp.fusion_config.n_restarts))
        self._bits.set_value(str(exp.fusion_config.bits_per_weight))
        self._penalty.set_value(str(exp.fusion_config.simplex_penalty))
        self._target_far.set_value(str(exp.fusion_config.target_far))

        self._bootstrap.set_value(str(exp.evaluation_config.n_bootstrap))
        self._ci_level.set_value(str(exp.evaluation_config.ci_level))
        self._folds.set_value(str(exp.evaluation_config.n_folds))
        self._seeds.set_value(str(exp.evaluation_config.n_seeds))

    def _apply(self):
        from app.state import FeatureSelectionMethod
        exp = self._app.app_state.current_experiment
        try:
            # Apply regime preset
            regime_str = self._regime_select.get()
            regime_lookup = {
                "CONVENTIONAL (100%)": TargetRegime.CONVENTIONAL,
                "REDUCED SIGNATURE (75%)": TargetRegime.REDUCED_SIGNATURE,
                "LOW OBSERVABILITY (50%)": TargetRegime.LOW_OBSERVABILITY,
                "NEAR-BACKGROUND (25%)": TargetRegime.NEAR_BACKGROUND,
                "STEALTH (0%)": TargetRegime.STEALTH,
            }
            regime = regime_lookup.get(regime_str, TargetRegime.STEALTH)
            exp.apply_regime_preset(regime)

            exp.seed = int(self._seed.value)
            exp.radar_config.num_samples = int(self._samples.value)
            exp.thermal_config.num_samples = int(self._samples.value)
            exp.acoustic_config.num_samples = int(self._samples.value)
            exp.evaluation_config.n_bootstrap = int(self._bootstrap.value)
            exp.evaluation_config.ci_level = float(self._ci_level.value)
            exp.evaluation_config.n_folds = int(self._folds.value)
            exp.evaluation_config.n_seeds = int(self._seeds.value)
            exp.feature_config.alpha = float(self._alpha.value)
            exp.feature_config.beta = float(self._beta.value)
            exp.feature_config.gamma = float(self._gamma.value)
            exp.feature_config.k_target = int(self._k_target.value)
            
            method_str = self._fs_method.get()
            if "MIQ" in method_str:
                exp.feature_config.method = FeatureSelectionMethod.MIQ_DINKELBACH
            else:
                exp.feature_config.method = FeatureSelectionMethod.MID_DEFAULT
                
            exp.feature_config.dinkelbach_damping = float(self._db_damping.value)
            exp.feature_config.dinkelbach_max_iter = int(self._db_max_iter.value)
            exp.feature_config.num_reads = int(self._num_reads.value)
            exp.fusion_config.lam = float(self._lam.value)
            exp.fusion_config.n_restarts = int(self._restarts.value)
            exp.fusion_config.bits_per_weight = int(self._bits.value)
            exp.fusion_config.simplex_penalty = float(self._penalty.value)
            exp.fusion_config.target_far = float(self._target_far.value)
        except (ValueError, AttributeError):
            pass

        # Propagate changes to all pages
        self._app.propagate_experiment_state()
