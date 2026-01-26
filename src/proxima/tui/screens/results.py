"""Results screen for Proxima TUI.

Results browser with probability visualization.
"""

from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, DataTable, ListView, ListItem, Label
from rich.text import Text

from .base import BaseScreen
from ..styles.theme import get_theme
from ..styles.icons import PROGRESS_FILLED, PROGRESS_EMPTY
from pathlib import Path

try:
    from proxima.data.export import JSONExporter, HTMLExporter, export_to_json, export_to_html
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

try:
    from proxima.resources.session import SessionManager
    from proxima.data.results import ResultStore
    from pathlib import Path
    RESULTS_AVAILABLE = True
except ImportError:
    RESULTS_AVAILABLE = False

try:
    from ..dialogs.results import ResultStatsDialog, ResultCompareDialog
    RESULT_DIALOGS_AVAILABLE = True
except ImportError:
    RESULT_DIALOGS_AVAILABLE = False


class ResultsScreen(BaseScreen):
    """Results browser screen.
    
    Shows:
    - List of results
    - Result details with probability distribution
    - Export options
    """
    
    SCREEN_NAME = "results"
    SCREEN_TITLE = "Results Browser"
    
    DEFAULT_CSS = """
    ResultsScreen .results-list {
        width: 30;
        height: 100%;
        border-right: solid $primary-darken-2;
        background: $surface;
    }
    
    ResultsScreen .results-list-title {
        padding: 1;
        text-style: bold;
        border-bottom: solid $primary-darken-3;
    }
    
    ResultsScreen .result-detail {
        width: 1fr;
        height: 100%;
        padding: 1;
    }
    
    ResultsScreen .result-header {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    ResultsScreen .probability-section {
        height: 1fr;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface-darken-1;
        overflow-y: auto;
    }
    
    ResultsScreen .actions-section {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    
    ResultsScreen .action-btn {
        margin-right: 1;
    }
    """
    
    def compose_main(self):
        """Compose the results screen content."""
        with Horizontal(classes="main-content"):
            # Results list
            with Vertical(classes="results-list"):
                yield Static("Results", classes="results-list-title")
                yield ResultsListView()
            
            # Result detail
            with Vertical(classes="result-detail"):
                yield ResultHeaderPanel(classes="result-header")
                yield ProbabilityDistribution(classes="probability-section")
                
                with Horizontal(classes="actions-section"):
                    yield Button("View Full Stats", id="btn-stats", classes="action-btn")
                    yield Button("Export JSON", id="btn-json", classes="action-btn")
                    yield Button("Export HTML", id="btn-html", classes="action-btn")
                    yield Button("Compare", id="btn-compare", classes="action-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-stats":
            self._view_full_stats()
        elif button_id == "btn-json":
            self._export_json()
        elif button_id == "btn-html":
            self._export_html()
        elif button_id == "btn-compare":
            self._compare_results()



    def _view_full_stats(self) -> None:
        """View full statistics for selected result using stats dialog."""
        result = getattr(self, '_selected_result', None)
        
        if RESULT_DIALOGS_AVAILABLE:
            self.app.push_screen(ResultStatsDialog(result=result))
        else:
            # Fallback to notifications
            if not result:
                self.notify("Please select a result first", severity="warning")
                return
            
            stats_text = [
                f"=== Full Statistics for {result.get('name', 'Unknown')} ===",
                f"Execution ID: {result.get('id', 'N/A')}",
                f"Status: {result.get('status', 'Unknown')}",
                f"Backend: {result.get('backend', 'Unknown')}",
                f"Duration: {result.get('duration', 0):.2f}s",
                f"Total Shots: {result.get('total_shots', 0)}",
                f"Success Rate: {result.get('success_rate', 0):.1%}",
                f"Average Fidelity: {result.get('avg_fidelity', 0):.4f}",
            ]
            
            # Show detailed stats
            self.notify("
".join(stats_text))
            self.notify("Install dialogs for full statistics view", severity="information")

    def _export_json(self) -> None:
        """Export results to JSON file."""
        if not hasattr(self, '_selected_result') or not self._selected_result:
            self.notify("Please select a result first", severity="warning")
            return
        
        result = self._selected_result
        result_id = result.get('id', 'unknown')
        
        if EXPORT_AVAILABLE:
            try:
                export_path = Path.home() / f"proxima_result_{result_id}.json"
                export_to_json(result, export_path)
                self.notify(f"? Exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"? Export failed: {e}", severity="error")
        else:
            try:
                import json
                export_path = Path.home() / f"proxima_result_{result_id}.json"
                with open(export_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                self.notify(f"? Exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"? Export failed: {e}", severity="error")

    def _export_html(self) -> None:
        """Export results to HTML file."""
        if not hasattr(self, '_selected_result') or not self._selected_result:
            self.notify("Please select a result first", severity="warning")
            return
        
        result = self._selected_result
        result_id = result.get('id', 'unknown')
        
        if EXPORT_AVAILABLE:
            try:
                export_path = Path.home() / f"proxima_result_{result_id}.html"
                export_to_html(result, export_path)
                self.notify(f"? Exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"? Export failed: {e}", severity="error")
        else:
            try:
                export_path = Path.home() / f"proxima_result_{result_id}.html"
                html_content = f"""<!DOCTYPE html>
<html>
<head><title>Proxima Result {result_id}</title></head>
<body>
<h1>Execution Result: {result.get('name', 'Unknown')}</h1>
<h2>Summary</h2>
<ul>
<li>Status: {result.get('status', 'Unknown')}</li>
<li>Backend: {result.get('backend', 'Unknown')}</li>
<li>Duration: {result.get('duration', 0):.2f}s</li>
<li>Success Rate: {result.get('success_rate', 0):.1%}</li>
</ul>
</body>
</html>"""
                with open(export_path, 'w') as f:
                    f.write(html_content)
                self.notify(f"? Exported to {export_path}", severity="success")
            except Exception as e:
                self.notify(f"? Export failed: {e}", severity="error")

    def _compare_results(self) -> None:
        """Compare selected results using comparison dialog."""
        results = []
        
        # Try to get multiple results if available
        if hasattr(self, '_selected_results') and self._selected_results:
            results = self._selected_results
        elif hasattr(self, '_selected_result') and self._selected_result:
            results = [self._selected_result]
        
        if RESULT_DIALOGS_AVAILABLE:
            self.app.push_screen(ResultCompareDialog(results=results))
        else:
            # Fallback to notifications
            if not results:
                self.notify("No results available for comparison", severity="warning")
                return
            
            self.notify("Comparison Mode", severity="information")
            self.notify("Select multiple results using Shift+Click to compare")
            self.notify("Comparison metrics: execution time, success rate, fidelity")
            
            if results:
                for i, result in enumerate(results[:2]):
                    name = result.get('name', 'Unknown')
                    rate = result.get('success_rate', 0)
                    self.notify(f"Result {i+1}: {name} - {rate:.1%} success")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle result selection from the list view."""
        item = event.item
        
        # Check if this item has real result data attached
        if hasattr(item, '_result_data') and item._result_data:
            self._selected_result = item._result_data
            self.notify(f"Selected: {self._selected_result.get('name', 'Unknown')}")
        else:
            # Fallback for sample data - create a minimal result dict
            label = item.query_one(Label)
            result_name = str(label.renderable) if label else "Unknown"
            self._selected_result = {
                'id': result_name.replace('.json', ''),
                'name': result_name,
                'status': 'unknown',
                'backend': 'Unknown',
                'duration': 0,
                'success_rate': 0,
            }
            self.notify(f"Selected: {result_name} (sample data)")



class ResultsListView(ListView):
    """List of available results."""

    def on_mount(self) -> None:
        """Populate the results list."""
        # Try to load real results
        if RESULTS_AVAILABLE:
            try:
                storage_dir = Path.home() / ".proxima" / "results"
                self._load_real_results(storage_dir)
                return
            except Exception:
                pass
        
        # Fallback to sample data
        results = [
            "result_001.json",
            "result_002.json",
            "comparison_001.json",
            "bell_state_run.json",
            "ghz_4qubit.json",
        ]

        for result in results:
            self.append(ListItem(Label(result)))

    def _load_real_results(self, storage_dir: Path) -> None:
        """Load real results from storage."""
        if not storage_dir.exists():
            raise Exception("Storage directory does not exist")
        
        import json
        results_loaded = False
        
        for result_file in sorted(storage_dir.glob("*.json"), reverse=True)[:10]:
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                result_id = result_data.get('id', result_file.stem)[:8]
                name = result_data.get('name', 'Unnamed')
                backend = result_data.get('backend', 'Unknown')
                status = result_data.get('status', 'Unknown')
                
                # Format status with icons
                status_display = {
                    'completed': '? Done',
                    'success': '? Done',
                    'failed': '? Failed',
                    'error': '? Error',
                    'running': '? Running',
                }.get(status.lower() if isinstance(status, str) else 'unknown', status)
                
                # Format time
                import time
                created = result_data.get('created_at', result_data.get('timestamp', 0))
                if isinstance(created, str):
                    try:
                        from datetime import datetime
                        created = datetime.fromisoformat(created).timestamp()
                    except Exception:
                        created = time.time()
                
                elapsed = time.time() - created
                if elapsed < 60:
                    time_str = f"{int(elapsed)}s ago"
                elif elapsed < 3600:
                    time_str = f"{int(elapsed / 60)}m ago"
                else:
                    time_str = f"{int(elapsed / 3600)}h ago"
                
                # Store result data for selection and append to list
                display_text = f"{name} ({status_display})"
                item = ListItem(Label(display_text))
                item._result_data = result_data
                self.append(item)
                results_loaded = True
            except Exception:
                continue
        
        if not results_loaded:
            raise Exception("No results loaded")


class ResultHeaderPanel(Static):
    """Header panel showing result metadata."""
    
    def render(self) -> Text:
        """Render the result header."""
        theme = get_theme()
        text = Text()
        
        # Title line
        text.append("Simulation Results", style=f"bold {theme.primary}")
        text.append("\n\n")
        
        # Metadata
        text.append("Backend: ", style=theme.fg_muted)
        text.append("Cirq (StateVector)", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Qubits: ", style=theme.fg_muted)
        text.append("4", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Shots: ", style=theme.fg_muted)
        text.append("1024", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Time: ", style=theme.fg_muted)
        text.append("245ms", style=theme.fg_base)
        
        return text


class ProbabilityDistribution(Static):
    """Probability distribution visualization."""
    
    BAR_WIDTH = 40
    
    def render(self) -> Text:
        """Render the probability distribution."""
        theme = get_theme()
        text = Text()
        
        text.append("Probability Distribution:", style=f"bold {theme.fg_base}")
        text.append("\n\n")
        
        # Sample probability data
        probabilities = [
            ("|0000⟩", 48.2),
            ("|1111⟩", 47.1),
            ("|0011⟩", 2.4),
            ("|1100⟩", 1.8),
            ("|others⟩", 0.5),
        ]
        
        for state, prob in probabilities:
            # State label
            text.append(f"{state:<10}", style=f"bold {theme.accent}")
            
            # Bar
            filled = int(self.BAR_WIDTH * prob / 100)
            empty = self.BAR_WIDTH - filled
            
            text.append(PROGRESS_FILLED * filled, style=f"bold {theme.primary}")
            text.append(PROGRESS_EMPTY * empty, style=theme.fg_subtle)
            
            # Percentage
            text.append(f" {prob:>5.1f}%", style=theme.fg_muted)
            text.append("\n")
        
        # Pattern detection
        text.append("\n")
        text.append("Pattern: ", style=theme.fg_muted)
        text.append("GHZ State", style=f"bold {theme.success}")
        text.append(" (confidence: 96%)", style=theme.fg_muted)
        text.append("\n")
        
        # Statistics
        text.append("Entropy: ", style=theme.fg_muted)
        text.append("1.02", style=theme.fg_base)
        text.append("  │  ", style=theme.border)
        
        text.append("Fidelity: ", style=theme.fg_muted)
        text.append("99.2%", style=f"bold {theme.success}")
        text.append("  │  ", style=theme.border)
        
        text.append("Gini: ", style=theme.fg_muted)
        text.append("0.03", style=theme.fg_base)
        
        return text
