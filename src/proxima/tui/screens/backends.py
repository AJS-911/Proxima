"""Backends screen for Proxima TUI.

Backend management and comparison.
Enhanced with LRET variants support as per TUI Integration Guide.
"""

from textual.containers import Horizontal, Vertical, Container, Grid, ScrollableContainer
from textual.widgets import Static, Button, DataTable, Checkbox, Label
from rich.text import Text
from pathlib import Path
import json

from .base import BaseScreen
from ..styles.theme import get_theme
from ..styles.icons import get_health_icon


try:
    from proxima.backends.registry import BackendRegistry
    from proxima.backends.base import BackendStatus
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

try:
    from ..dialogs.backends import BackendComparisonDialog, BackendMetricsDialog, BackendConfigDialog, AddCustomBackendDialog
    DIALOGS_AVAILABLE = True
    ADD_CUSTOM_AVAILABLE = True
except ImportError:
    DIALOGS_AVAILABLE = False
    ADD_CUSTOM_AVAILABLE = False

# LRET Variant Imports
try:
    from proxima.backends.lret.variant_registry import VariantRegistry
    from proxima.backends.lret.variant_analysis import VariantAnalyzer
    LRET_VARIANTS_AVAILABLE = True
except ImportError:
    LRET_VARIANTS_AVAILABLE = False

# LRET Dialogs
try:
    from ..dialogs.lret import VariantAnalysisDialog, Phase7Dialog
    LRET_DIALOGS_AVAILABLE = True
except ImportError:
    LRET_DIALOGS_AVAILABLE = False

# Wizards
try:
    from ..wizards import PennyLaneAlgorithmWizard
    WIZARDS_AVAILABLE = True
except ImportError:
    WIZARDS_AVAILABLE = False


# LRET Variant definitions for fallback display
LRET_VARIANTS_DATA = [
    {
        'id': 'lret_base',
        'name': 'LRET (Base)',
        'description': 'Core LRET quantum simulator',
        'type': 'Simulator',
        'status': 'healthy',
        'is_lret': True,
        'variant_type': 'base',
    },
    {
        'id': 'lret_cirq_scalability',
        'name': 'LRET Cirq Scalability',
        'description': 'Cirq FDM comparison & benchmarking',
        'type': 'LRET Variant',
        'status': 'healthy',
        'is_lret': True,
        'variant_type': 'cirq_scalability',
    },
    {
        'id': 'lret_pennylane_hybrid',
        'name': 'LRET PennyLane Hybrid',
        'description': 'VQE, QAOA, QNN algorithms',
        'type': 'LRET Variant',
        'status': 'healthy',
        'is_lret': True,
        'variant_type': 'pennylane_hybrid',
    },
    {
        'id': 'lret_phase7_unified',
        'name': 'LRET Phase 7 Unified',
        'description': 'Multi-framework with GPU support',
        'type': 'LRET Variant',
        'status': 'healthy',
        'is_lret': True,
        'variant_type': 'phase7_unified',
    },
]


class BackendsScreen(BaseScreen):
    """Backend management screen with LRET variants support.
    
    Shows:
    - List of backends with health status (including LRET variants)
    - Backend details and configuration
    - Performance comparison
    - Quick actions for LRET variants
    - Default/Fallback backend selection
    """
    
    SCREEN_NAME = "backends"
    SCREEN_TITLE = "📦 Backend Management"
    
    DEFAULT_CSS = """
    BackendsScreen .backends-list {
        height: 1fr;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
        margin-bottom: 1;
        overflow-y: auto;
        scrollbar-gutter: stable;
    }
    
    BackendsScreen .backend-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    BackendsScreen .backend-row:hover {
        background: $surface-lighten-1;
    }
    
    BackendsScreen .backend-row.-selected {
        background: $primary-darken-2;
    }
    
    BackendsScreen .backend-row.-lret {
        border-left: thick $success;
    }
    
    BackendsScreen .backend-checkbox {
        width: 5;
    }
    
    BackendsScreen .backend-name {
        width: 1fr;
        content-align: left middle;
    }
    
    BackendsScreen .backend-type {
        width: 15;
        content-align: center middle;
    }
    
    BackendsScreen .backend-status {
        width: 10;
        content-align: center middle;
    }
    
    BackendsScreen .backend-configure {
        width: 12;
    }
    
    BackendsScreen .quick-actions {
        height: 4;
        layout: horizontal;
        padding: 1;
        background: $surface-darken-1;
        margin-bottom: 1;
    }
    
    BackendsScreen .quick-actions-title {
        width: 15;
        content-align: left middle;
    }
    
    BackendsScreen .action-btn {
        margin-right: 1;
    }
    
    BackendsScreen .selection-section {
        height: 5;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
        margin-bottom: 1;
    }
    
    BackendsScreen .selection-row {
        height: 3;
        layout: horizontal;
    }
    
    BackendsScreen .selection-label {
        width: 18;
        content-align: left middle;
    }
    
    BackendsScreen .selection-value {
        width: 1fr;
        content-align: left middle;
        text-style: bold;
    }
    
    BackendsScreen .footer-actions {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }
    
    BackendsScreen .header-row {
        height: 3;
        layout: horizontal;
        padding: 0 1;
        background: $primary-darken-2;
        margin-bottom: 1;
    }
    
    BackendsScreen .header-label {
        content-align: center middle;
        text-style: bold;
    }
    
    BackendsScreen .backends-grid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1;
        padding: 1;
    }
    
    BackendsScreen .backend-card {
        height: 8;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    BackendsScreen .backend-card:hover {
        border: solid $primary;
    }
    
    BackendsScreen .backend-card.-selected {
        border: solid $primary;
        background: $surface-lighten-1;
    }
    
    BackendsScreen .backend-card:focus {
        border: double $primary;
    }
    
    BackendsScreen .backend-card.-lret-variant {
        border-left: thick $success;
    }
    
    BackendsScreen .actions-section {
        height: auto;
        layout: horizontal;
        padding: 1;
        border-top: solid $primary-darken-3;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the backends screen."""
        super().__init__(**kwargs)
        self._selected_backend = None
        self._enabled_backends = set()
        self._default_backend = "lret_phase7_unified"
        self._fallback_backend = "lret_cirq_scalability"
    
    def compose_main(self):
        """Compose the backends screen content with LRET variants."""
        with Vertical(classes="main-content"):
            # Title with refresh/help buttons
            with Horizontal(classes="header-row"):
                yield Static("📦 Available Backends", classes="header-label backend-name")
                yield Button("Help", id="btn-help", classes="action-btn")
                yield Button("Refresh", id="btn-refresh", classes="action-btn")
            
            # Backend list with checkboxes and configure buttons (scrollable)
            with ScrollableContainer(classes="backends-list"):
                # Get all backends including LRET variants
                backends_data = self._get_all_backends()
                
                for backend in backends_data:
                    row_classes = "backend-row"
                    if backend.get('is_lret'):
                        row_classes += " -lret"
                    
                    with Horizontal(classes=row_classes, id=f"row-{backend['id']}"):
                        yield Checkbox(
                            "",
                            value=True,
                            id=f"chk-{backend['id']}",
                            classes="backend-checkbox",
                        )
                        yield Static(backend['name'], classes="backend-name")
                        yield Static(backend.get('type', 'Simulator'), classes="backend-type")
                        yield Static(
                            self._get_status_indicator(backend.get('status', 'unknown')),
                            classes="backend-status",
                        )
                        yield Button(
                            "Configure",
                            id=f"cfg-{backend['id']}",
                            classes="backend-configure action-btn",
                        )
            
            # Quick Actions Section
            with Horizontal(classes="quick-actions"):
                yield Static("Quick Actions:", classes="quick-actions-title")
                yield Button("➕ Add Backend", id="btn-add-custom", classes="action-btn", variant="success")
                yield Button("🔄 Multi-Run", id="btn-multi-run", classes="action-btn", variant="primary")
                yield Button("Install Variant", id="btn-install-variant", classes="action-btn")
                yield Button("Run Benchmark", id="btn-run-benchmark", classes="action-btn", variant="primary")
                yield Button("View Comparison", id="btn-view-comparison", classes="action-btn")
                yield Button("Algorithm Wizard", id="btn-algorithm-wizard", classes="action-btn")
            
            # Current Selection Section
            with Container(classes="selection-section"):
                yield Static("Current Selection:", classes="section-title")
                with Horizontal(classes="selection-row"):
                    yield Static("Default Backend:", classes="selection-label")
                    yield Static(self._get_backend_display_name(self._default_backend), 
                                id="lbl-default-backend", classes="selection-value")
                with Horizontal(classes="selection-row"):
                    yield Static("Fallback:", classes="selection-label")
                    yield Static(self._get_backend_display_name(self._fallback_backend),
                                id="lbl-fallback-backend", classes="selection-value")
            
            # Footer Actions
            with Horizontal(classes="footer-actions"):
                yield Button("Apply Changes", id="btn-apply", variant="primary", classes="action-btn")
                yield Button("Restore Defaults", id="btn-restore", classes="action-btn")
                yield Button("Export Config", id="btn-export-config", classes="action-btn")
                yield Button("Run Health Check", id="btn-health", classes="action-btn")
    
    def _get_all_backends(self) -> list:
        """Get all available backends including LRET variants."""
        backends_data = []
        
        # Add LRET variants first
        backends_data.extend(LRET_VARIANTS_DATA)
        
        # Try to get real backends from registry
        if REGISTRY_AVAILABLE:
            try:
                registry = BackendRegistry()
                registry.discover()
                backend_names = registry.list_backends()
                
                for name in backend_names:
                    # Skip if already added as LRET variant
                    if any(b['id'] == name.lower().replace(' ', '_') for b in backends_data):
                        continue
                    
                    info = registry.get_backend_info(name)
                    health = registry.check_backend_health(name)
                    
                    backends_data.append({
                        'id': name.lower().replace(' ', '_'),
                        'name': name,
                        'type': info.get('type', 'Simulator') if info else 'Simulator',
                        'status': 'healthy' if health and health.get('status') == 'healthy' else 'unknown',
                        'description': info.get('description', f'{name} backend') if info else f'{name} backend',
                        'is_lret': False,
                    })
            except Exception:
                pass
        
        # Add standard backends if not from registry
        if len(backends_data) == len(LRET_VARIANTS_DATA):
            standard_backends = [
                {'id': 'cirq', 'name': 'Cirq', 'type': 'Simulator', 'status': 'healthy', 'description': 'Google Cirq Simulator', 'is_lret': False},
                {'id': 'qiskit_aer', 'name': 'Qiskit Aer', 'type': 'Simulator', 'status': 'healthy', 'description': 'IBM Qiskit Aer', 'is_lret': False},
                {'id': 'quest', 'name': 'QuEST', 'type': 'Simulator', 'status': 'unknown', 'description': 'QuEST High-Performance', 'is_lret': False},
                {'id': 'qsim', 'name': 'qsim', 'type': 'Simulator', 'status': 'unknown', 'description': 'Google qsim', 'is_lret': False},
                {'id': 'cuquantum', 'name': 'cuQuantum', 'type': 'GPU', 'status': 'unknown', 'description': 'NVIDIA cuQuantum', 'is_lret': False},
            ]
            backends_data.extend(standard_backends)
        
        # Add custom backends from persistent storage
        custom_backends = self._load_custom_backends()
        for backend_id, config in custom_backends.items():
            backends_data.append({
                'id': backend_id,
                'name': config.get('display_name', config.get('name', backend_id)),
                'type': config.get('type', 'Custom').replace('_', ' ').title(),
                'status': config.get('status', 'unknown'),
                'description': config.get('description', 'Custom backend'),
                'is_lret': False,
                'is_custom': True,
            })
        
        return backends_data
    
    def _get_status_indicator(self, status: str) -> str:
        """Get status indicator with icon."""
        if status == 'healthy':
            return "● Healthy"
        elif status == 'degraded':
            return "◐ Degraded"
        elif status == 'unavailable':
            return "○ Unavail"
        else:
            return "◌ Unknown"
    
    def _get_backend_display_name(self, backend_id: str) -> str:
        """Get display name for backend ID."""
        display_names = {
            'lret_base': 'LRET (Base)',
            'lret_cirq_scalability': 'LRET Cirq Scalability',
            'lret_pennylane_hybrid': 'LRET PennyLane Hybrid',
            'lret_phase7_unified': 'LRET Phase 7 Unified',
            'cirq': 'Cirq',
            'qiskit_aer': 'Qiskit Aer',
        }
        return display_names.get(backend_id, backend_id)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-health":
            self._run_health_check()
        elif button_id == "btn-compare" or button_id == "btn-view-comparison":
            self._compare_performance()
        elif button_id == "btn-metrics":
            self._view_metrics()
        elif button_id == "btn-configure":
            self._configure_backend()
        elif button_id == "btn-run-benchmark":
            self._open_benchmark_comparison()
        elif button_id == "btn-multi-run":
            self._open_multi_run()
        elif button_id == "btn-algorithm-wizard":
            self._open_algorithm_wizard()
        elif button_id == "btn-install-variant":
            self._install_variant()
        elif button_id == "btn-add-custom":
            self._open_add_custom_backend()
        elif button_id == "btn-apply":
            self._apply_changes()
        elif button_id == "btn-restore":
            self._restore_defaults()
        elif button_id == "btn-export-config":
            self._export_config()
        elif button_id == "btn-help":
            self._show_help()
        elif button_id == "btn-refresh":
            self._refresh_backends()
        elif button_id and button_id.startswith("cfg-"):
            # Configure specific backend
            backend_id = button_id[4:]  # Remove "cfg-" prefix
            self._configure_specific_backend(backend_id)
        elif button_id and button_id.startswith("edit-"):
            # Edit custom backend
            backend_id = button_id[5:]  # Remove "edit-" prefix
            self._edit_custom_backend(backend_id)
        elif button_id and button_id.startswith("del-"):
            # Delete custom backend
            backend_id = button_id[4:]  # Remove "del-" prefix
            self._delete_custom_backend(backend_id)
    
    def _open_multi_run(self) -> None:
        """Open multi-backend comparison run dialog."""
        try:
            from proxima.tui.dialogs.backends.multi_run import MultiBackendRunDialog
            
            def on_complete(results):
                if results:
                    self.notify(f"Comparison completed with {len(results)} backends", severity="success")
            
            self.app.push_screen(MultiBackendRunDialog(on_complete=on_complete))
        except ImportError as e:
            self.notify(f"Multi-run dialog not available: {e}", severity="warning")
        except Exception as e:
            self.notify(f"Error opening multi-run: {e}", severity="error")

    def _open_benchmark_comparison(self) -> None:
        """Open the benchmark comparison screen."""
        try:
            from .benchmark_comparison import BenchmarkComparisonScreen
            self.app.push_screen(BenchmarkComparisonScreen())
        except ImportError:
            self.notify("Benchmark comparison screen not available", severity="warning")
    
    def _open_algorithm_wizard(self) -> None:
        """Open the PennyLane Algorithm Wizard."""
        if WIZARDS_AVAILABLE:
            self.app.push_screen(PennyLaneAlgorithmWizard())
        else:
            self.notify("Algorithm wizard not available", severity="warning")
    
    def _install_variant(self) -> None:
        """Open variant installation dialog."""
        if LRET_DIALOGS_AVAILABLE:
            self.app.push_screen(VariantAnalysisDialog())
        else:
            self.notify("Variant installation dialog not available", severity="warning")
            self.notify("LRET variants are pre-installed", severity="information")
    
    def _open_add_custom_backend(self) -> None:
        """Open the Add Custom Backend dialog."""
        if ADD_CUSTOM_AVAILABLE:
            dialog = AddCustomBackendDialog(on_save=self._on_custom_backend_saved)
            self.app.push_screen(dialog)
        else:
            self.notify("Custom backend dialog not available", severity="warning")
    
    def _on_custom_backend_saved(self, config: dict) -> None:
        """Callback when a custom backend is saved."""
        if config:
            self.notify(f"Backend '{config.get('name', 'Unknown')}' added successfully!", severity="success")
            # Refresh the backend list
            self._refresh_backends()
    
    def _edit_custom_backend(self, backend_id: str) -> None:
        """Edit an existing custom backend."""
        # Load custom backends
        custom_backends = self._load_custom_backends()
        
        if backend_id in custom_backends:
            config = custom_backends[backend_id]
            # Open dialog with pre-filled config
            if ADD_CUSTOM_AVAILABLE:
                dialog = AddCustomBackendDialog(on_save=self._on_custom_backend_saved)
                self.app.push_screen(dialog)
                # TODO: Pre-fill dialog with existing config
            else:
                self.notify("Edit dialog not available", severity="warning")
        else:
            self.notify(f"Backend '{backend_id}' not found", severity="error")
    
    def _delete_custom_backend(self, backend_id: str) -> None:
        """Delete a custom backend."""
        try:
            backends_file = Path.home() / ".proxima" / "custom_backends.json"
            
            if backends_file.exists():
                with open(backends_file, 'r') as f:
                    custom_backends = json.load(f)
                
                if backend_id in custom_backends:
                    del custom_backends[backend_id]
                    
                    with open(backends_file, 'w') as f:
                        json.dump(custom_backends, f, indent=2)
                    
                    self.notify(f"Backend '{backend_id}' deleted", severity="success")
                    self._refresh_backends()
                else:
                    self.notify(f"Backend '{backend_id}' not found", severity="warning")
            else:
                self.notify("No custom backends found", severity="warning")
        except Exception as e:
            self.notify(f"Delete failed: {e}", severity="error")
    
    def _load_custom_backends(self) -> dict:
        """Load custom backends from storage."""
        try:
            backends_file = Path.home() / ".proxima" / "custom_backends.json"
            
            if backends_file.exists():
                with open(backends_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _apply_changes(self) -> None:
        """Apply backend configuration changes."""
        # Gather enabled backends
        enabled = []
        for backend in self._get_all_backends():
            try:
                checkbox = self.query_one(f"#chk-{backend['id']}", Checkbox)
                if checkbox.value:
                    enabled.append(backend['id'])
            except Exception:
                pass
        
        self._enabled_backends = set(enabled)
        self.notify(f"Configuration applied: {len(enabled)} backends enabled", severity="success")
    
    def _restore_defaults(self) -> None:
        """Restore default backend configuration."""
        self._default_backend = "lret_phase7_unified"
        self._fallback_backend = "lret_cirq_scalability"
        
        # Re-enable all backends
        for backend in self._get_all_backends():
            try:
                checkbox = self.query_one(f"#chk-{backend['id']}", Checkbox)
                checkbox.value = True
            except Exception:
                pass
        
        # Update display labels
        try:
            self.query_one("#lbl-default-backend", Static).update(
                self._get_backend_display_name(self._default_backend)
            )
            self.query_one("#lbl-fallback-backend", Static).update(
                self._get_backend_display_name(self._fallback_backend)
            )
        except Exception:
            pass
        
        self.notify("Defaults restored", severity="success")
    
    def _export_config(self) -> None:
        """Export backend configuration to file."""
        try:
            from pathlib import Path
            from datetime import datetime
            import json
            
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"backend_config_{timestamp}.json"
            
            # Gather configuration
            enabled = []
            for backend in self._get_all_backends():
                try:
                    checkbox = self.query_one(f"#chk-{backend['id']}", Checkbox)
                    if checkbox.value:
                        enabled.append(backend['id'])
                except Exception:
                    enabled.append(backend['id'])  # Default to enabled
            
            config = {
                'default_backend': self._default_backend,
                'fallback_backend': self._fallback_backend,
                'enabled_backends': enabled,
                'lret_variants': [b['id'] for b in LRET_VARIANTS_DATA],
            }
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.notify(f"Config exported to {filename}", severity="success")
        except Exception as e:
            self.notify(f"Export error: {e}", severity="error")
    
    def _show_help(self) -> None:
        """Show help for backend management."""
        from textual.screen import ModalScreen
        from textual.containers import Vertical, ScrollableContainer
        from textual.widgets import Static, Button, Markdown
        
        help_content = """# Backend Management Help

## Overview
This screen allows you to manage quantum computing backends for Proxima.

## Backend Types

### LRET Variants
- **LRET (Base)**: Core backend with standard features
- **Cirq Scalability**: Optimized for large circuits using Cirq
- **PennyLane Hybrid**: Supports VQE/QAOA/QNN algorithms
- **Phase 7 Unified**: Latest unified backend with all features

### Other Backends
- **Cirq**: Google's quantum computing framework
- **Qiskit Aer**: IBM's quantum simulator

## Quick Actions

| Button | Description |
|--------|-------------|
| **Configure** | Open settings for the selected backend |
| **Run Health Check** | Check if backends are responding |
| **Run Benchmark** | Compare LRET vs Cirq performance |
| **Algorithm Wizard** | Set up VQE, QAOA, or QNN algorithms |

## Tips
- Click on a backend row to select it
- Use checkboxes to enable/disable backends
- The default backend is used automatically
- Fallback is used if the default fails

## Keyboard Shortcuts
- **Escape**: Go back to previous screen
- **R**: Refresh backend list
- **H**: Show this help
"""
        
        class HelpDialog(ModalScreen):
            DEFAULT_CSS = """
            HelpDialog {
                align: center middle;
            }
            HelpDialog > .help-container {
                width: 80;
                height: 32;
                border: thick $accent;
                background: $surface;
                padding: 1 2;
            }
            HelpDialog .help-scroll {
                height: 1fr;
                scrollbar-gutter: stable;
                border: solid $primary-darken-2;
            }
            HelpDialog Markdown {
                margin: 1;
                height: auto;
            }
            HelpDialog .help-footer {
                height: 4;
                margin-top: 1;
                align: center middle;
            }
            HelpDialog .scroll-hint {
                color: $text-muted;
                text-align: center;
                height: 1;
            }
            """
            
            BINDINGS = [
                ("escape", "close", "Close"),
                ("down", "scroll_down", "Scroll Down"),
                ("up", "scroll_up", "Scroll Up"),
                ("pagedown", "page_down", "Page Down"),
                ("pageup", "page_up", "Page Up"),
            ]
            
            def compose(self):
                with Vertical(classes="help-container"):
                    with ScrollableContainer(classes="help-scroll", can_focus=True):
                        yield Markdown(help_content)
                    with Vertical(classes="help-footer"):
                        yield Static("↑/↓ or PageUp/PageDown to scroll", classes="scroll-hint")
                        yield Button("Close", variant="primary", id="btn-close-help")
            
            def on_mount(self):
                # Focus the scrollable container for keyboard navigation
                try:
                    scroll = self.query_one(".help-scroll", ScrollableContainer)
                    scroll.focus()
                except Exception:
                    pass
            
            def on_button_pressed(self, event):
                self.dismiss()
            
            def action_close(self):
                self.dismiss()
            
            def action_scroll_down(self):
                scroll = self.query_one(".help-scroll", ScrollableContainer)
                scroll.scroll_down(animate=False)
            
            def action_scroll_up(self):
                scroll = self.query_one(".help-scroll", ScrollableContainer)
                scroll.scroll_up(animate=False)
            
            def action_page_down(self):
                scroll = self.query_one(".help-scroll", ScrollableContainer)
                scroll.scroll_page_down(animate=False)
            
            def action_page_up(self):
                scroll = self.query_one(".help-scroll", ScrollableContainer)
                scroll.scroll_page_up(animate=False)
        
        self.app.push_screen(HelpDialog())
    
    def _refresh_backends(self) -> None:
        """Refresh backend list and status."""
        self.notify("Refreshing backends...", severity="information")
        # Trigger re-mount of the screen
        self.refresh()
        self.notify("Backend list refreshed", severity="success")
    
    def _configure_specific_backend(self, backend_id: str) -> None:
        """Configure a specific backend by ID."""
        # Check if it's an LRET variant
        variant_dialogs = {
            'lret_phase7_unified': 'Phase7Dialog',
            'lret_pennylane_hybrid': 'AlgorithmWizard',
            'lret_cirq_scalability': 'BenchmarkScreen',
        }
        
        if backend_id in variant_dialogs and LRET_DIALOGS_AVAILABLE:
            if backend_id == 'lret_phase7_unified':
                self.app.push_screen(Phase7Dialog())
            elif backend_id == 'lret_pennylane_hybrid':
                if WIZARDS_AVAILABLE:
                    self.app.push_screen(PennyLaneAlgorithmWizard())
                else:
                    self.notify("PennyLane wizard not available", severity="warning")
            elif backend_id == 'lret_cirq_scalability':
                self._open_benchmark_comparison()
            return
        
        # Regular backend configuration
        self._selected_backend = {'name': backend_id, 'type': 'Simulator'}
        self._configure_backend()


    def _run_health_check(self) -> None:
        """Run health check on selected backend or show selection dialog."""
        from textual.screen import ModalScreen
        from textual.containers import Vertical, Horizontal
        from textual.widgets import Static, Button, Select
        
        backend = self._get_selected_backend()
        if backend:
            # Run health check on selected backend
            self._execute_health_check(backend)
            return
        
        # Show backend selection dialog
        all_backends = self._get_all_backends()
        backend_options = [(b.get('name', b.get('id', 'Unknown')), b.get('id', b.get('name', ''))) for b in all_backends]
        
        class HealthCheckDialog(ModalScreen):
            DEFAULT_CSS = """
            HealthCheckDialog {
                align: center middle;
            }
            HealthCheckDialog > .dialog-container {
                width: 60;
                height: auto;
                border: thick $accent;
                background: $surface;
                padding: 1 2;
            }
            HealthCheckDialog .dialog-title {
                text-style: bold;
                color: $accent;
                text-align: center;
                margin-bottom: 1;
            }
            HealthCheckDialog Select {
                width: 100%;
                margin: 1 0;
            }
            HealthCheckDialog .dialog-footer {
                height: auto;
                margin-top: 1;
                layout: horizontal;
            }
            HealthCheckDialog Button {
                margin-right: 1;
            }
            """
            
            BINDINGS = [("escape", "close", "Close")]
            
            def __init__(self, options, **kwargs):
                super().__init__(**kwargs)
                self.options = options
            
            def compose(self):
                with Vertical(classes="dialog-container"):
                    yield Static("🏥 Run Health Check", classes="dialog-title")
                    yield Static("Select a backend to check, or check all:")
                    yield Select(
                        [("All Backends", "all")] + list(self.options),
                        value="all",
                        id="backend-select"
                    )
                    with Horizontal(classes="dialog-footer"):
                        yield Button("Run Check", variant="primary", id="btn-run")
                        yield Button("Cancel", variant="default", id="btn-cancel")
            
            def on_button_pressed(self, event):
                if event.button.id == "btn-run":
                    select = self.query_one("#backend-select", Select)
                    self.dismiss(select.value)
                else:
                    self.dismiss(None)
            
            def action_close(self):
                self.dismiss(None)
        
        def handle_selection(result):
            if result is None:
                return
            if result == "all":
                self._run_health_check_all()
            else:
                backend = {'name': result, 'id': result}
                self._execute_health_check(backend)
        
        self.app.push_screen(HealthCheckDialog(backend_options), handle_selection)
    
    def _execute_health_check(self, backend) -> None:
        """Execute health check on a specific backend."""
        backend_name = backend.get('name', backend.get('id', 'Unknown'))
        self.notify(f"Running health check on {backend_name}...")
        
        if REGISTRY_AVAILABLE:
            try:
                registry = BackendRegistry()
                health_result = registry.check_backend_health(backend_name)
                
                if health_result:
                    status = health_result.get('status', 'unknown')
                    response_time = health_result.get('response_time', 0)
                    
                    if status == 'healthy' or status == BackendStatus.HEALTHY:
                        self.notify(f"✓ {backend_name} is healthy ({response_time:.0f}ms)", severity="success")
                    elif status == 'degraded' or status == BackendStatus.DEGRADED:
                        self.notify(f"⚠ {backend_name} is degraded ({response_time:.0f}ms)", severity="warning")
                    else:
                        self.notify(f"✗ {backend_name} is unavailable", severity="error")
                else:
                    self.notify(f"✗ Health check failed for {backend_name}", severity="error")
            except Exception as e:
                self.notify(f"✗ Health check error: {e}", severity="error")
        else:
            # Simulated health check when core not available
            import random
            response_time = random.randint(50, 200)
            self.notify(f"✓ {backend_name} appears healthy ({response_time}ms)", severity="success")
            self.notify("Note: Using simulated check (core not available)", severity="information")

    def _run_health_check_all(self) -> None:
        """Run health check on all enabled backends."""
        backends = self._get_all_backends()
        healthy_count = 0
        total_count = len(backends)
        
        if REGISTRY_AVAILABLE:
            try:
                registry = BackendRegistry()
                for backend in backends:
                    backend_name = backend.get('name', backend.get('id', 'Unknown'))
                    try:
                        health_result = registry.check_backend_health(backend_name)
                        if health_result:
                            status = health_result.get('status', 'unknown')
                            if status == 'healthy' or status == BackendStatus.HEALTHY:
                                healthy_count += 1
                    except Exception:
                        pass
                self.notify(f"Health Check Complete: {healthy_count}/{total_count} backends healthy", 
                           severity="success" if healthy_count == total_count else "warning")
            except Exception as e:
                self.notify(f"Health check error: {e}", severity="error")
        else:
            # Simulated health check
            import random
            healthy_count = random.randint(total_count - 1, total_count)
            self.notify(f"Health Check Complete: {healthy_count}/{total_count} backends healthy", 
                       severity="success" if healthy_count == total_count else "warning")
            self.notify("Note: Using simulated checks (core not available)", severity="information")

    def _get_selected_backend(self):
        """Get the currently selected backend."""
        if hasattr(self, '_selected_backend') and self._selected_backend:
            return self._selected_backend
        
        # Try to get from UI selection
        try:
            table = self.query_one("#backends-table", DataTable)
            if table.cursor_row is not None:
                row = table.get_row_at(table.cursor_row)
                if row:
                    return {'name': str(row[0]), 'type': str(row[1]) if len(row) > 1 else 'Unknown'}
        except Exception:
            pass
        
        return None

    def _compare_performance(self) -> None:
        """Compare performance of backends using comparison dialog."""
        if DIALOGS_AVAILABLE:
            self.app.push_screen(BackendComparisonDialog())
        else:
            # Fallback to notifications
            self.notify("Performance Comparison", severity="information")
            
            if REGISTRY_AVAILABLE:
                try:
                    registry = BackendRegistry()
                    all_backends = registry.list_backends() if hasattr(registry, 'list_backends') else []
                    
                    comparison_data = []
                    for backend in all_backends[:5]:
                        health = registry.check_backend_health(backend)
                        if health:
                            comparison_data.append(f"{backend}: {health.get('response_time', 0):.0f}ms")
                    
                    if comparison_data:
                        for line in comparison_data:
                            self.notify(line)
                    else:
                        self.notify("No backends available for comparison", severity="warning")
                except Exception as e:
                    self.notify(f"Comparison failed: {e}", severity="error")
            else:
                self.notify("LRET: ~35ms | Cirq: ~55ms | Qiskit: ~70ms")
                self.notify("(Sample data - Install dialogs for full view)", severity="information")

    def _view_metrics(self) -> None:
        """View detailed metrics for selected backend using metrics dialog."""
        backend = self._get_selected_backend()
        backend_name = backend.get('name', None) if backend else None
        
        if DIALOGS_AVAILABLE:
            self.app.push_screen(BackendMetricsDialog(backend_name=backend_name))
        else:
            # Fallback to notifications
            if not backend:
                self.notify("Please select a backend first", severity="warning")
                return
            
            self.notify(f"Metrics for {backend_name}", severity="information")
            
            if REGISTRY_AVAILABLE:
                try:
                    registry = BackendRegistry()
                    metrics = registry.get_backend_metrics(backend_name) if hasattr(registry, 'get_backend_metrics') else None
                    
                    if metrics:
                        self.notify(f"Jobs Completed: {metrics.get('jobs_completed', 0)}")
                        self.notify(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
                        self.notify(f"Avg Queue Time: {metrics.get('avg_queue_time', 0):.1f}s")
                        self.notify(f"Uptime: {metrics.get('uptime', 0):.1%}")
                    else:
                        self.notify("No detailed metrics available", severity="warning")
                except Exception as e:
                    self.notify(f"Error fetching metrics: {e}", severity="error")
            else:
                self.notify("Jobs: 150 | Success: 94.2% | Queue: 2.3s")
                self.notify("(Sample data - Install dialogs for full view)", severity="information")

    def _configure_backend(self) -> None:
        """Configure the selected backend using configuration dialog."""
        backend = self._get_selected_backend()
        backend_name = backend.get('name', None) if backend else None
        
        if DIALOGS_AVAILABLE:
            self.app.push_screen(BackendConfigDialog(backend_name=backend_name))
        else:
            # Fallback to notifications
            if not backend:
                self.notify("Please select a backend first", severity="warning")
                return
            
            backend_type = backend.get('type', 'Unknown')
            self.notify(f"Configuration for {backend_name}", severity="information")
            self.notify(f"Type: {backend_type}")
            
            # Show configuration options based on backend type
            if 'simulator' in backend_name.lower() or 'sim' in backend_type.lower():
                self.notify("Options: shots, seed, noise_model, coupling_map")
                self.notify("Use Settings screen to modify simulator options")
            elif 'ibm' in backend_name.lower():
                self.notify("Options: hub, group, project, optimization_level")
                self.notify("Configure IBM credentials in Settings > LLM Settings")
            elif 'braket' in backend_name.lower() or 'amazon' in backend_name.lower():
                self.notify("Options: region, s3_bucket, device_arn")
                self.notify("Configure AWS credentials in Settings")
            else:
                self.notify("Options: endpoint, timeout, max_retries")
                self.notify("Use Settings screen for advanced configuration")



class BackendCard(Static):
    """A card displaying backend information."""
    
    can_focus = True  # Make card focusable for selection
    
    def __init__(
        self,
        backend_id: str,
        name: str,
        description: str,
        status: str,
        **kwargs,
    ):
        """Initialize the backend card."""
        super().__init__(**kwargs)
        self.backend_id = backend_id
        self.backend_name = name
        self.description = description
        self.status = status
        self.classes = "backend-card"
        self._selected = False
    
    def on_click(self) -> None:
        """Handle click to select this backend."""
        # Find parent screen and update selection
        screen = self.screen
        if hasattr(screen, '_selected_backend'):
            # Deselect previous
            if screen._selected_backend:
                try:
                    for card in screen.query(BackendCard):
                        card.remove_class("-selected")
                        card._selected = False
                except Exception:
                    pass
        
        # Select this card
        self._selected = True
        self.add_class("-selected")
        screen._selected_backend = {
            'name': self.backend_name,
            'type': 'Simulator',
            'status': self.status,
        }
        screen.notify(f"Selected: {self.backend_name}")
    
    def render(self) -> Text:
        """Render the backend card."""
        theme = get_theme()
        text = Text()
        
        # Status icon and name
        icon = get_health_icon(self.status)
        color = theme.get_health_color(self.status)
        
        text.append(icon, style=f"bold {color}")
        text.append(" ")
        text.append(self.backend_name, style=f"bold {theme.fg_base}")
        text.append("\n")
        
        # Description
        text.append(self.description, style=theme.fg_muted)
        text.append("\n\n")
        
        # Status text
        status_text = self.status.capitalize()
        if self.status == "healthy":
            text.append(f"● {status_text}", style=f"bold {color}")
        elif self.status == "unavailable":
            text.append(f"○ Not Available", style=color)
        else:
            text.append(f"○ {status_text}", style=color)
        
        return text
