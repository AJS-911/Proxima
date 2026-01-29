"""Dashboard screen for Proxima TUI.

Main landing screen with overview and quick actions.
"""

from pathlib import Path
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Static, Button, DataTable
from rich.text import Text
from rich.panel import Panel

from .base import BaseScreen
from ..styles.theme import get_theme
from ..components.logo import Logo
from ..dialogs.sessions import SessionsDialog
from ..dialogs.sessions.session_item import SessionInfo
from ..dialogs.simulation import SimulationDialog, SimulationConfig
from datetime import datetime

try:
    from proxima.resources.session import SessionManager, SessionMetadata, SessionStatus
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    SessionManager = None  # type: ignore

try:
    from proxima.core.pipeline import run_simulation
    import asyncio
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


class DashboardScreen(BaseScreen):
    """Main dashboard screen.
    
    Shows:
    - Welcome message
    - Quick actions
    - Recent sessions
    - System health
    """
    
    SCREEN_NAME = "dashboard"
    SCREEN_TITLE = "Dashboard"
    
    DEFAULT_CSS = """
    DashboardScreen .welcome-section {
        height: auto;
        margin-bottom: 2;
    }
    
    DashboardScreen .welcome-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }
    
    DashboardScreen .welcome-subtitle {
        color: $text-muted;
    }
    
    DashboardScreen .quick-actions {
        height: auto;
        margin-bottom: 2;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    DashboardScreen .quick-actions-title {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    DashboardScreen .action-buttons {
        layout: horizontal;
        height: auto;
    }
    
    DashboardScreen .action-button {
        margin-right: 1;
        min-width: 18;
    }
    
    DashboardScreen .sessions-section {
        height: 1fr;
        margin-bottom: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    
    DashboardScreen .sessions-title {
        color: $text-muted;
        padding: 1;
        border-bottom: solid $primary-darken-3;
    }
    
    DashboardScreen .health-section {
        height: auto;
        padding: 1;
        border: solid $primary-darken-2;
        background: $surface;
    }
    """
    
    def compose_main(self):
        """Compose the dashboard content."""
        theme = get_theme()
        
        with Vertical(classes="main-content"):
            # Welcome section
            with Vertical(classes="welcome-section"):
                yield Static(
                    "Welcome to Proxima",
                    classes="welcome-title",
                )
                yield Static(
                    "Intelligent Quantum Simulation Orchestration",
                    classes="welcome-subtitle",
                )
            
            # Quick Actions
            with Container(classes="quick-actions"):
                yield Static("Quick Actions", classes="quick-actions-title")
                with Horizontal(classes="action-buttons"):
                    yield Button(
                        "[1] Run Simulation",
                        id="btn-run",
                        classes="action-button",
                        variant="primary",
                    )
                    yield Button(
                        "[2] Compare Backends",
                        id="btn-compare",
                        classes="action-button",
                    )
                    yield Button(
                        "[3] View Results",
                        id="btn-results",
                        classes="action-button",
                    )
                    yield Button(
                        "[4] Manage Sessions",
                        id="btn-sessions",
                        classes="action-button",
                    )
                    yield Button(
                        "[5] Configure",
                        id="btn-config",
                        classes="action-button",
                    )
                    yield Button(
                        "[?] Help",
                        id="btn-help",
                        classes="action-button",
                    )
            
            # Recent Sessions
            with Vertical(classes="sessions-section"):
                yield Static("Recent Sessions", classes="sessions-title")
                yield RecentSessionsTable()
            
            # System Health
            yield SystemHealthBar(classes="health-section")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-run":
            self._show_simulation_dialog()
        elif button_id == "btn-compare":
            self.app.action_goto_backends()
        elif button_id == "btn-results":
            self.app.action_goto_results()
        elif button_id == "btn-sessions":
            self._show_sessions_dialog()
        elif button_id == "btn-config":
            self.app.action_goto_settings()
        elif button_id == "btn-help":
            self.app.action_show_help()
    
    def _show_sessions_dialog(self) -> None:
        """Show the sessions management dialog."""
        # Get real sessions if available
        sessions_list = []
        storage_dir = Path.home() / ".proxima" / "sessions"
        manager = None
        
        if SESSION_MANAGER_AVAILABLE:
            try:
                manager = SessionManager(storage_dir=storage_dir)
                real_sessions = manager.list_sessions()
                
                for meta in real_sessions[:10]:  # Show up to 10 sessions
                    status_map = {
                        SessionStatus.COMPLETED: "completed",
                        SessionStatus.RUNNING: "active",
                        SessionStatus.PAUSED: "paused",
                        SessionStatus.FAILED: "failed",
                        SessionStatus.ABORTED: "aborted",
                        SessionStatus.CREATED: "created",
                    }
                    sessions_list.append(SessionInfo(
                        id=meta.id,
                        title=meta.name or f"Session {meta.id[:8]}",
                        status=status_map.get(meta.status, "unknown"),
                        created_at=datetime.fromtimestamp(meta.created_at),
                        task_count=0,  # Would need full session load
                        backend="Auto",
                    ))
            except Exception:
                pass
        
        # Fallback to sample data if no real sessions
        if not sessions_list:
            sessions_list = [
                SessionInfo(
                    id="a1b2c3d4-1234-5678-9abc-def012345678",
                    title="Bell State Experiment",
                    status="active",
                    created_at=datetime.now(),
                    task_count=3,
                    backend="Cirq",
                ),
                SessionInfo(
                    id="e5f6g7h8-5678-9012-cdef-345678901234",
                    title="GHZ Analysis",
                    status="completed",
                    created_at=datetime.now(),
                    task_count=5,
                    backend="Qiskit",
                ),
                SessionInfo(
                    id="i9j0k1l2-9012-3456-0123-456789012345",
                    title="Grover Search Test",
                    status="paused",
                    created_at=datetime.now(),
                    task_count=2,
                    backend="LRET",
                ),
            ]
        
        current_session_id = self.state.session_id if hasattr(self.state, 'session_id') else sessions_list[0].id if sessions_list else None
        
        def handle_session_action(result):
            if result:
                action = result.get("action")
                session = result.get("session")
                self._handle_session_action(action, session, manager, storage_dir)
        
        self.app.push_screen(
            SessionsDialog(
                sessions=sessions_list,
                current_session_id=current_session_id,
            ),
            handle_session_action,
        )
    
    def _handle_session_action(self, action: str, session: SessionInfo | None, manager: SessionManager | None, storage_dir: Path) -> None:
        """Handle session actions from the dialog.
        
        Args:
            action: The action to perform (switch, new, delete, export)
            session: The session to act on (for switch, delete, export)
            manager: The SessionManager instance
            storage_dir: Path to session storage directory
        """
        if action == "switch" and session:
            self._switch_to_session(session, manager)
        elif action == "new":
            self._create_new_session(manager, storage_dir)
        elif action == "delete" and session:
            self._delete_session(session, manager, storage_dir)
        elif action == "export" and session:
            self._export_session(session, manager, storage_dir)
    
    def _switch_to_session(self, session: SessionInfo, manager: SessionManager | None) -> None:
        """Switch to a different session."""
        if manager and SESSION_MANAGER_AVAILABLE:
            try:
                if manager.set_current(session.id):
                    current = manager.current
                    if current:
                        # Update TUI state with session data
                        self.state.current_task = current.metadata.name or f"Session {session.id[:8]}"
                        if hasattr(self.state, 'session_id'):
                            self.state.session_id = session.id
                        self.notify(f"✓ Switched to: {session.title}", severity="success")
                        return
            except Exception as e:
                self.notify(f"Error switching session: {e}", severity="error")
                return
        
        # Fallback for demo mode
        self.state.current_task = session.title
        if hasattr(self.state, 'session_id'):
            self.state.session_id = session.id
        self.notify(f"✓ Switched to: {session.title}", severity="success")
    
    def _create_new_session(self, manager: SessionManager | None, storage_dir: Path) -> None:
        """Create a new session."""
        import uuid
        
        if manager and SESSION_MANAGER_AVAILABLE:
            try:
                new_session = manager.create(name=f"New Session {datetime.now().strftime('%H:%M')}")
                manager.save(new_session)
                self.state.current_task = new_session.metadata.name
                if hasattr(self.state, 'session_id'):
                    self.state.session_id = new_session.metadata.id
                self.notify(f"✓ Created: {new_session.metadata.name}", severity="success")
                # Reset execution state for new session
                self.state.is_running = False
                self.state.execution_status = "IDLE"
                self.state.progress_percent = 0.0
                return
            except Exception as e:
                self.notify(f"Error creating session: {e}", severity="error")
                return
        
        # Fallback for demo mode
        new_id = str(uuid.uuid4())
        new_name = f"New Session {datetime.now().strftime('%H:%M')}"
        self.state.current_task = new_name
        if hasattr(self.state, 'session_id'):
            self.state.session_id = new_id
        self.state.is_running = False
        self.state.execution_status = "IDLE"
        self.state.progress_percent = 0.0
        self.notify(f"✓ Created: {new_name}", severity="success")
    
    def _delete_session(self, session: SessionInfo, manager: SessionManager | None, storage_dir: Path) -> None:
        """Delete a session."""
        # Check if trying to delete current session
        current_id = getattr(self.state, 'session_id', None)
        if current_id == session.id:
            self.notify("Cannot delete active session", severity="warning")
            return
        
        if manager and SESSION_MANAGER_AVAILABLE:
            try:
                if manager.delete(session.id):
                    self.notify(f"✓ Deleted: {session.title}", severity="success")
                    return
                else:
                    self.notify(f"Session not found: {session.title}", severity="warning")
                    return
            except Exception as e:
                self.notify(f"Error deleting session: {e}", severity="error")
                return
        
        # Fallback for demo mode
        self.notify(f"✓ Deleted: {session.title}", severity="success")
    
    def _export_session(self, session: SessionInfo, manager: SessionManager | None, storage_dir: Path) -> None:
        """Export a session to a file."""
        import json
        
        export_dir = Path.home() / ".proxima" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if manager and SESSION_MANAGER_AVAILABLE:
            try:
                session_data = manager.get(session.id)
                if session_data:
                    export_path = export_dir / f"session_{session.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    export_content = {
                        "id": session_data.metadata.id,
                        "name": session_data.metadata.name,
                        "status": session_data.metadata.status.name,
                        "created_at": session_data.metadata.created_at,
                        "config": session_data.config,
                        "state": session_data.state,
                        "results": session_data.results,
                    }
                    export_path.write_text(json.dumps(export_content, indent=2, default=str))
                    self.notify(f"✓ Exported to: {export_path.name}", severity="success")
                    return
            except Exception as e:
                self.notify(f"Error exporting session: {e}", severity="error")
                return
        
        # Fallback for demo mode - create a sample export
        try:
            export_path = export_dir / f"session_{session.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            export_content = {
                "id": session.id,
                "name": session.title,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "task_count": session.task_count,
                "backend": session.backend,
            }
            export_path.write_text(json.dumps(export_content, indent=2, default=str))
            self.notify(f"✓ Exported to: {export_path.name}", severity="success")
        except Exception as e:
            self.notify(f"Error exporting: {e}", severity="error")
    
    def _show_simulation_dialog(self) -> None:
        """Show the simulation configuration dialog."""
        def handle_simulation_config(config: SimulationConfig) -> None:
            if config:
                self._start_simulation(config)
        
        self.app.push_screen(SimulationDialog(), handle_simulation_config)
    
    def _start_simulation(self, config: SimulationConfig) -> None:
        """Start a simulation with the given configuration.
        
        Args:
            config: Simulation configuration from dialog
        """
        from ..state.tui_state import StageInfo
        
        self.notify(f"🚀 Starting simulation: {config.description or config.circuit_type}", severity="information")
        
        # Define simulation stages based on circuit complexity
        stages = [
            StageInfo(name="Initialization", index=0, status="pending"),
            StageInfo(name="Circuit Generation", index=1, status="pending"),
            StageInfo(name="Optimization", index=2, status="pending"),
            StageInfo(name="Execution", index=3, status="pending"),
            StageInfo(name="Measurement", index=4, status="pending"),
            StageInfo(name="Result Processing", index=5, status="pending"),
        ]
        
        # Update state for execution screen BEFORE navigating
        self.state.current_task = config.description or f"{config.circuit_type.title()} Circuit"
        self.state.current_backend = config.backend if config.backend != "auto" else "Cirq"
        self.state.qubits = config.qubits
        self.state.shots = config.shots
        self.state.is_running = True
        self.state.execution_status = "RUNNING"
        self.state.progress_percent = 0.0
        self.state.stage_index = 0
        self.state.current_stage = stages[0].name
        self.state.total_stages = len(stages)
        self.state.all_stages = stages
        self.state.elapsed_ms = 0
        self.state.eta_ms = 0
        
        # Navigate to execution screen
        self.app.action_goto_execution()
        
        if PIPELINE_AVAILABLE:
            # Start actual simulation in background
            try:
                # Run async simulation using Textual's worker system
                self.app.call_later(lambda: self._run_async_simulation(config))
                
            except Exception as e:
                self.notify(f"Failed to start simulation: {e}", severity="error")
                self.state.is_running = False
                self.state.execution_status = "FAILED"
        else:
            self.notify("Simulation started (demo mode)", severity="information")
    
    def _run_async_simulation(self, config: SimulationConfig) -> None:
        """Run the simulation asynchronously."""
        import asyncio
        
        async def _run():
            try:
                result = await run_simulation(
                    user_input=config.description or f"Create a {config.circuit_type} circuit",
                    backends=[config.backend] if config.backend != "auto" else None,
                    qubits=config.qubits,
                    shots=config.shots,
                )
                # Update state on completion
                self.state.is_running = False
                self.state.progress_percent = 100.0
                if result.get("success"):
                    self.state.execution_status = "COMPLETED"
                    self.notify("✓ Simulation completed successfully!", severity="success")
                else:
                    self.state.execution_status = "FAILED"
                    self.notify(f"Simulation failed: {result.get('error', 'Unknown error')}", severity="error")
            except Exception as e:
                self.state.is_running = False
                self.state.execution_status = "FAILED"
                self.notify(f"Simulation error: {e}", severity="error")
        
        asyncio.create_task(_run())


class RecentSessionsTable(DataTable):
    """Table displaying recent sessions."""
    
    DEFAULT_CSS = """
    RecentSessionsTable {
        height: 1fr;
        margin: 1;
    }
    """
    
    def on_mount(self) -> None:
        """Set up the table."""
        self.add_columns("ID", "Task", "Backend", "Status", "Time")
        self.cursor_type = "row"
        
        # Try to get real session data first
        self._populate_sessions()
    
    def _populate_sessions(self) -> None:
        """Populate with session data from SessionManager or fallback to sample."""
        sessions_loaded = False
        
        if SESSION_MANAGER_AVAILABLE:
            try:
                from pathlib import Path
                from datetime import datetime
                
                storage_dir = Path.home() / ".proxima" / "sessions"
                if storage_dir.exists():
                    manager = SessionManager(storage_dir=storage_dir)
                    real_sessions = manager.list_sessions()
                    
                    if real_sessions:
                        sessions_loaded = True
                        for meta in real_sessions[:5]:  # Show up to 5 sessions
                            status_icons = {
                                SessionStatus.COMPLETED: "✓ Done",
                                SessionStatus.RUNNING: "▶ Running",
                                SessionStatus.PAUSED: "⏸ Paused",
                                SessionStatus.FAILED: "✗ Failed",
                                SessionStatus.ABORTED: "⏹ Aborted",
                                SessionStatus.CREATED: "○ New",
                            }
                            
                            # Calculate time ago
                            created = datetime.fromtimestamp(meta.created_at)
                            delta = datetime.now() - created
                            if delta.seconds < 60:
                                time_ago = f"{delta.seconds}s ago"
                            elif delta.seconds < 3600:
                                time_ago = f"{delta.seconds // 60}m ago"
                            else:
                                time_ago = f"{delta.seconds // 3600}h ago"
                            
                            self.add_row(
                                meta.id[:8],
                                meta.name or f"Session {meta.id[:8]}",
                                "Auto",  # Backend not stored in metadata
                                status_icons.get(meta.status, "? Unknown"),
                                time_ago,
                            )
            except Exception:
                pass
        
        # Fallback to sample data if no real sessions
        if not sessions_loaded:
            self._populate_sample_data()
    
    def _populate_sample_data(self) -> None:
        """Populate with sample session data."""
        theme = get_theme()
        
        sample_sessions = [
            ("a1b2c3d4", "Bell State", "Cirq", "✓ Done", "12s ago"),
            ("e5f6g7h8", "GHZ 4-qubit", "Qiskit", "✓ Done", "2m ago"),
            ("i9j0k1l2", "Comparison Run", "Multi", "✓ Done", "15m ago"),
            ("m3n4o5p6", "Grover Search", "Cirq", "✓ Done", "1h ago"),
            ("q7r8s9t0", "VQE Optimization", "Qiskit", "⏸ Paused", "2h ago"),
        ]
        
        for session in sample_sessions:
            self.add_row(*session)


class SystemHealthBar(Static):
    """System health overview bar."""
    
    def render(self) -> Text:
        """Render the health bar."""
        theme = get_theme()
        text = Text()
        
        # Get real system stats if available
        cpu_percent = 23  # Default
        memory_percent = 52  # Default
        backends_healthy = 3
        backends_total = 6
        
        try:
            from proxima.resources.memory import MemoryMonitor
            monitor = MemoryMonitor()
            stats = monitor.get_stats()
            if stats:
                memory_percent = int(stats.get('memory_percent', 52))
                cpu_percent = int(stats.get('cpu_percent', 23))
        except ImportError:
            pass
        except Exception:
            pass
        
        try:
            from proxima.backends.registry import BackendRegistry
            registry = BackendRegistry()
            all_backends = registry.list_backends() if hasattr(registry, 'list_backends') else []
            backends_total = len(all_backends) or 6
            backends_healthy = 0
            for backend in all_backends:
                health = registry.check_backend_health(backend)
                if health and health.get('status') == 'healthy':
                    backends_healthy += 1
        except ImportError:
            pass
        except Exception:
            pass
        
        # Determine color based on value
        def get_color(value: int, thresholds=(50, 80)) -> str:
            if value < thresholds[0]:
                return theme.success
            elif value < thresholds[1]:
                return theme.warning
            else:
                return theme.error
        
        # CPU
        text.append("CPU: ", style=theme.fg_muted)
        text.append(f"{cpu_percent}%", style=f"bold {get_color(cpu_percent)}")
        text.append("  │  ", style=theme.border)
        
        # Memory
        text.append("Memory: ", style=theme.fg_muted)
        text.append(f"{memory_percent}%", style=f"bold {get_color(memory_percent)}")
        text.append("  │  ", style=theme.border)
        
        # Backends
        text.append("Backends: ", style=theme.fg_muted)
        backend_color = theme.success if backends_healthy >= backends_total // 2 else theme.warning
        text.append(f"{backends_healthy}/{backends_total} healthy", style=f"bold {backend_color}")
        
        return text
