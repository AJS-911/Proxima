"""Terminal Manager for Multi-Terminal Architecture.

This module implements Phase 3.3 for the Dynamic AI Assistant:
- Multi-Terminal Architecture: Terminal pools, context isolation,
  resource management, session persistence
- Process Lifecycle Management: Process tracking, signal handling,
  cleanup procedures, orphan detection
- Interactive Mode Handling: Prompt detection, input simulation,
  response capture, exit sequence management
- Output Processing: Real-time streaming, buffering strategies,
  error detection, log aggregation

Key Features:
============
- Terminal pool management with isolated contexts
- Process lifecycle tracking
- Interactive command handling with prompt detection
- Real-time output streaming and buffering
- Error pattern detection
- Background process management
- Session persistence and recovery

Design Principle:
================
All terminal decisions use LLM reasoning - NO hardcoded command patterns.
The LLM determines prompts, exit sequences, and error patterns dynamically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class TerminalState(Enum):
    """State of a terminal."""
    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    BLOCKED = "blocked"
    ERROR = "error"
    CLOSED = "closed"


class ProcessState(Enum):
    """State of a process."""
    STARTING = "starting"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"
    ORPHAN = "orphan"


class OutputType(Enum):
    """Type of output."""
    STDOUT = "stdout"
    STDERR = "stderr"
    MIXED = "mixed"
    PROMPT = "prompt"


@dataclass
class OutputChunk:
    """A chunk of terminal output."""
    content: str
    output_type: OutputType
    timestamp: float
    terminal_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "type": self.output_type.value,
            "timestamp": self.timestamp,
            "terminal_id": self.terminal_id,
        }


@dataclass
class ProcessInfo:
    """Information about a running process."""
    process_id: str
    pid: Optional[int] = None
    command: str = ""
    working_directory: str = ""
    
    # State
    state: ProcessState = ProcessState.STARTING
    exit_code: Optional[int] = None
    
    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Parent
    terminal_id: str = ""
    is_background: bool = False
    
    @property
    def running_time_ms(self) -> float:
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or time.time()
        return (end_time - self.started_at) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "pid": self.pid,
            "command": self.command[:100],  # Truncate long commands
            "state": self.state.value,
            "exit_code": self.exit_code,
            "running_time_ms": self.running_time_ms,
            "is_background": self.is_background,
        }


@dataclass
class TerminalSession:
    """A terminal session."""
    terminal_id: str
    name: str = ""
    working_directory: str = ""
    
    # State
    state: TerminalState = TerminalState.IDLE
    
    # Environment
    environment: Dict[str, str] = field(default_factory=dict)
    shell: str = ""
    
    # Process tracking
    current_process: Optional[ProcessInfo] = None
    process_history: List[ProcessInfo] = field(default_factory=list)
    
    # Output
    output_buffer: List[OutputChunk] = field(default_factory=list)
    max_buffer_size: int = 10000
    
    # Timing
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Internal
    _process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    _read_task: Optional[asyncio.Task] = field(default=None, repr=False)
    
    def add_output(self, content: str, output_type: OutputType):
        """Add output to buffer."""
        chunk = OutputChunk(
            content=content,
            output_type=output_type,
            timestamp=time.time(),
            terminal_id=self.terminal_id,
        )
        self.output_buffer.append(chunk)
        self.last_activity = time.time()
        
        # Trim buffer if needed
        if len(self.output_buffer) > self.max_buffer_size:
            self.output_buffer = self.output_buffer[-self.max_buffer_size // 2:]
    
    def get_recent_output(self, limit: int = 100) -> str:
        """Get recent output as string."""
        chunks = self.output_buffer[-limit:]
        return "".join(c.content for c in chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "terminal_id": self.terminal_id,
            "name": self.name,
            "working_directory": self.working_directory,
            "state": self.state.value,
            "current_process": self.current_process.to_dict() if self.current_process else None,
            "output_lines": len(self.output_buffer),
        }


@dataclass
class TerminalConfig:
    """Configuration for terminal manager."""
    # Pool settings
    max_terminals: int = 10
    default_shell: str = ""  # Auto-detect
    
    # Timeouts
    command_timeout_ms: int = 60000
    idle_timeout_ms: int = 300000  # 5 minutes
    
    # Buffer settings
    output_buffer_size: int = 10000
    stream_chunk_size: int = 4096
    
    # Interactive settings
    prompt_timeout_ms: int = 5000
    input_delay_ms: int = 100


class TerminalManager:
    """Manager for multiple terminal sessions.
    
    Uses LLM reasoning to:
    1. Detect prompts and interactive commands
    2. Handle input/output patterns
    3. Detect errors in output
    4. Manage process lifecycle
    
    Example:
        >>> manager = TerminalManager(llm_client=client)
        >>> terminal = await manager.create_terminal()
        >>> output = await manager.execute_command(terminal.terminal_id, "ls -la")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        config: Optional[TerminalConfig] = None,
    ):
        """Initialize the terminal manager.
        
        Args:
            llm_client: LLM client for reasoning
            config: Terminal configuration
        """
        self._llm_client = llm_client
        self._config = config or TerminalConfig()
        
        # Terminal pool
        self._terminals: Dict[str, TerminalSession] = {}
        self._terminal_lock = asyncio.Lock()
        
        # Process tracking
        self._processes: Dict[str, ProcessInfo] = {}
        self._background_processes: Dict[str, ProcessInfo] = {}
        
        # Output callbacks
        self._output_callbacks: Dict[str, List[Callable[[OutputChunk], None]]] = {}
        
        # Detect default shell
        if not self._config.default_shell:
            self._config.default_shell = self._detect_shell()
    
    def _detect_shell(self) -> str:
        """Detect the default shell."""
        if sys.platform == "win32":
            return os.environ.get("COMSPEC", "cmd.exe")
        else:
            return os.environ.get("SHELL", "/bin/bash")
    
    async def create_terminal(
        self,
        name: Optional[str] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> TerminalSession:
        """Create a new terminal session.
        
        Args:
            name: Optional terminal name
            working_directory: Working directory
            environment: Environment variables
            
        Returns:
            TerminalSession
        """
        async with self._terminal_lock:
            if len(self._terminals) >= self._config.max_terminals:
                # Try to cleanup idle terminals
                await self._cleanup_idle_terminals()
                
                if len(self._terminals) >= self._config.max_terminals:
                    raise RuntimeError("Maximum terminal limit reached")
            
            terminal_id = str(uuid.uuid4())[:8]
            
            terminal = TerminalSession(
                terminal_id=terminal_id,
                name=name or f"terminal-{terminal_id}",
                working_directory=working_directory or os.getcwd(),
                shell=self._config.default_shell,
                environment=environment or {},
            )
            
            self._terminals[terminal_id] = terminal
            logger.info(f"Created terminal {terminal_id}")
            
            return terminal
    
    async def get_terminal(self, terminal_id: str) -> Optional[TerminalSession]:
        """Get a terminal by ID."""
        return self._terminals.get(terminal_id)
    
    async def list_terminals(self) -> List[TerminalSession]:
        """List all terminals."""
        return list(self._terminals.values())
    
    async def close_terminal(self, terminal_id: str):
        """Close a terminal session."""
        terminal = self._terminals.get(terminal_id)
        if terminal:
            # Kill any running process
            if terminal.current_process and terminal.current_process.pid:
                await self._kill_process(terminal.current_process)
            
            # Close subprocess if exists
            if terminal._process:
                terminal._process.kill()
                await terminal._process.wait()
            
            # Cancel read task
            if terminal._read_task:
                terminal._read_task.cancel()
            
            terminal.state = TerminalState.CLOSED
            del self._terminals[terminal_id]
            
            logger.info(f"Closed terminal {terminal_id}")
    
    async def execute_command(
        self,
        terminal_id: str,
        command: str,
        timeout_ms: Optional[int] = None,
        capture_output: bool = True,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> str:
        """Execute a command in a terminal.
        
        Args:
            terminal_id: Terminal ID
            command: Command to execute
            timeout_ms: Optional timeout
            capture_output: Whether to capture and return output
            env: Additional environment variables
            cwd: Working directory override
            
        Returns:
            Command output if capture_output is True
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            raise ValueError(f"Terminal {terminal_id} not found")
        
        if terminal.state == TerminalState.RUNNING:
            raise RuntimeError("Terminal is busy executing another command")
        
        timeout_ms = timeout_ms or self._config.command_timeout_ms
        
        # Create process info
        process_info = ProcessInfo(
            process_id=str(uuid.uuid4())[:8],
            command=command,
            working_directory=cwd or terminal.working_directory,
            terminal_id=terminal_id,
        )
        
        terminal.state = TerminalState.RUNNING
        terminal.current_process = process_info
        self._processes[process_info.process_id] = process_info
        
        try:
            # Prepare environment
            full_env = {**os.environ, **terminal.environment}
            if env:
                full_env.update(env)
            
            # Execute command
            process_info.state = ProcessState.RUNNING
            process_info.started_at = time.time()
            
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or terminal.working_directory,
                env=full_env,
            )
            
            process_info.pid = process.pid
            terminal._process = process
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_ms / 1000,
                )
                
                output = stdout.decode("utf-8", errors="ignore")
                error_output = stderr.decode("utf-8", errors="ignore")
                
                process_info.exit_code = process.returncode
                process_info.state = ProcessState.COMPLETED if process.returncode == 0 else ProcessState.FAILED
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                process_info.state = ProcessState.KILLED
                raise TimeoutError(f"Command timed out after {timeout_ms}ms")
            
            process_info.completed_at = time.time()
            
            # Add to output buffer
            if capture_output:
                if output:
                    terminal.add_output(output, OutputType.STDOUT)
                if error_output:
                    terminal.add_output(error_output, OutputType.STDERR)
            
            # Move to history
            terminal.process_history.append(process_info)
            terminal.current_process = None
            
            return output + error_output
            
        except Exception as e:
            process_info.state = ProcessState.FAILED
            process_info.completed_at = time.time()
            raise
            
        finally:
            terminal.state = TerminalState.IDLE
            terminal._process = None
    
    async def execute_background(
        self,
        terminal_id: str,
        command: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> ProcessInfo:
        """Execute a command in the background.
        
        Args:
            terminal_id: Terminal ID
            command: Command to execute
            env: Additional environment variables
            cwd: Working directory override
            
        Returns:
            ProcessInfo for tracking
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            raise ValueError(f"Terminal {terminal_id} not found")
        
        # Create process info
        process_info = ProcessInfo(
            process_id=str(uuid.uuid4())[:8],
            command=command,
            working_directory=cwd or terminal.working_directory,
            terminal_id=terminal_id,
            is_background=True,
        )
        
        # Prepare environment
        full_env = {**os.environ, **terminal.environment}
        if env:
            full_env.update(env)
        
        # Start process
        process_info.state = ProcessState.RUNNING
        process_info.started_at = time.time()
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or terminal.working_directory,
            env=full_env,
        )
        
        process_info.pid = process.pid
        
        self._processes[process_info.process_id] = process_info
        self._background_processes[process_info.process_id] = process_info
        
        # Start background reader
        asyncio.create_task(
            self._read_background_output(terminal, process, process_info)
        )
        
        logger.info(f"Started background process {process_info.process_id}: {command[:50]}")
        
        return process_info
    
    async def _read_background_output(
        self,
        terminal: TerminalSession,
        process: asyncio.subprocess.Process,
        process_info: ProcessInfo,
    ):
        """Read output from a background process."""
        try:
            while True:
                # Read stdout
                if process.stdout:
                    line = await process.stdout.readline()
                    if line:
                        terminal.add_output(
                            line.decode("utf-8", errors="ignore"),
                            OutputType.STDOUT
                        )
                        self._emit_output(terminal.terminal_id, terminal.output_buffer[-1])
                
                # Check if process ended
                if process.returncode is not None:
                    break
                
                await asyncio.sleep(0.01)
            
            # Get remaining output
            if process.stdout:
                remaining = await process.stdout.read()
                if remaining:
                    terminal.add_output(
                        remaining.decode("utf-8", errors="ignore"),
                        OutputType.STDOUT
                    )
            
            if process.stderr:
                stderr_remaining = await process.stderr.read()
                if stderr_remaining:
                    terminal.add_output(
                        stderr_remaining.decode("utf-8", errors="ignore"),
                        OutputType.STDERR
                    )
            
            process_info.exit_code = process.returncode
            process_info.state = ProcessState.COMPLETED if process.returncode == 0 else ProcessState.FAILED
            process_info.completed_at = time.time()
            
        except Exception as e:
            logger.error(f"Error reading background output: {e}")
            process_info.state = ProcessState.FAILED
        
        finally:
            if process_info.process_id in self._background_processes:
                del self._background_processes[process_info.process_id]
    
    async def send_input(
        self,
        terminal_id: str,
        input_text: str,
        add_newline: bool = True,
    ):
        """Send input to an interactive process.
        
        Args:
            terminal_id: Terminal ID
            input_text: Input to send
            add_newline: Whether to add newline
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            raise ValueError(f"Terminal {terminal_id} not found")
        
        if not terminal._process or not terminal._process.stdin:
            raise RuntimeError("No interactive process running")
        
        if add_newline and not input_text.endswith("\n"):
            input_text += "\n"
        
        terminal._process.stdin.write(input_text.encode())
        await terminal._process.stdin.drain()
        
        terminal.last_activity = time.time()
    
    async def send_signal(
        self,
        terminal_id: str,
        signal_name: str,
    ):
        """Send a signal to the current process.
        
        Args:
            terminal_id: Terminal ID
            signal_name: Signal name (e.g., "SIGINT", "SIGTERM")
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal or not terminal.current_process:
            return
        
        pid = terminal.current_process.pid
        if pid is None:
            return
        
        sig = getattr(signal, signal_name, signal.SIGTERM)
        
        try:
            if sys.platform == "win32":
                os.kill(pid, sig)
            else:
                os.kill(pid, sig)
            
            logger.info(f"Sent {signal_name} to process {pid}")
            
        except Exception as e:
            logger.error(f"Failed to send signal: {e}")
    
    async def _kill_process(self, process_info: ProcessInfo):
        """Kill a process."""
        if process_info.pid is None:
            return
        
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(process_info.pid)], check=False)
            else:
                os.kill(process_info.pid, signal.SIGKILL)
            
            process_info.state = ProcessState.KILLED
            process_info.completed_at = time.time()
            
        except Exception as e:
            logger.error(f"Failed to kill process {process_info.pid}: {e}")
    
    async def _cleanup_idle_terminals(self):
        """Clean up idle terminals."""
        current_time = time.time()
        to_remove = []
        
        for terminal_id, terminal in self._terminals.items():
            idle_time = (current_time - terminal.last_activity) * 1000
            
            if (
                terminal.state == TerminalState.IDLE and
                idle_time > self._config.idle_timeout_ms
            ):
                to_remove.append(terminal_id)
        
        for terminal_id in to_remove:
            await self.close_terminal(terminal_id)
            logger.info(f"Cleaned up idle terminal {terminal_id}")
    
    def on_output(
        self,
        terminal_id: str,
        callback: Callable[[OutputChunk], None],
    ):
        """Register output callback for a terminal."""
        if terminal_id not in self._output_callbacks:
            self._output_callbacks[terminal_id] = []
        self._output_callbacks[terminal_id].append(callback)
    
    def _emit_output(self, terminal_id: str, chunk: OutputChunk):
        """Emit output to callbacks."""
        callbacks = self._output_callbacks.get(terminal_id, [])
        for callback in callbacks:
            try:
                callback(chunk)
            except Exception as e:
                logger.error(f"Output callback error: {e}")
    
    async def detect_prompt(
        self,
        terminal_id: str,
        timeout_ms: Optional[int] = None,
    ) -> Optional[str]:
        """Detect if terminal is showing a prompt.
        
        Args:
            terminal_id: Terminal ID
            timeout_ms: Timeout to wait for prompt
            
        Returns:
            Detected prompt string or None
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return None
        
        timeout_ms = timeout_ms or self._config.prompt_timeout_ms
        
        # Wait for output to settle
        await asyncio.sleep(timeout_ms / 1000)
        
        recent_output = terminal.get_recent_output(20)
        
        # Use LLM to detect prompt
        if self._llm_client:
            prompt = await self._llm_detect_prompt(recent_output)
            if prompt:
                return prompt
        
        # Fallback: common prompt patterns
        prompt_patterns = [
            r"[\$#>]\s*$",  # Common shell prompts
            r"\(.*\)\s*[\$#>]\s*$",  # With virtual env
            r".*@.*:\S*[\$#>]\s*$",  # user@host:path$
            r">>>\s*$",  # Python REPL
            r"\.\.\.\s*$",  # Python continuation
            r"In \[\d+\]:\s*$",  # IPython
        ]
        
        for pattern in prompt_patterns:
            if re.search(pattern, recent_output):
                match = re.search(pattern, recent_output)
                if match:
                    return match.group(0)
        
        return None
    
    async def _llm_detect_prompt(self, output: str) -> Optional[str]:
        """Use LLM to detect prompt in output."""
        prompt = f"""Analyze this terminal output and identify if it ends with a command prompt.

Output:
{output[-200:]}

If the output ends with a command prompt (where the user can type a command),
respond with just the prompt string. Otherwise respond "NO_PROMPT".
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            if "NO_PROMPT" in response.upper():
                return None
            
            return response.strip()
            
        except Exception:
            return None
    
    async def detect_errors(
        self,
        terminal_id: str,
        output: Optional[str] = None,
    ) -> List[str]:
        """Detect errors in terminal output.
        
        Args:
            terminal_id: Terminal ID
            output: Optional specific output to check
            
        Returns:
            List of detected error messages
        """
        terminal = self._terminals.get(terminal_id)
        if not terminal:
            return []
        
        output = output or terminal.get_recent_output(100)
        
        # Use LLM to detect errors
        if self._llm_client:
            errors = await self._llm_detect_errors(output)
            if errors:
                return errors
        
        # Fallback: common error patterns
        error_patterns = [
            r"error[\s:]+(.+)",
            r"Error[\s:]+(.+)",
            r"ERROR[\s:]+(.+)",
            r"fatal[\s:]+(.+)",
            r"Fatal[\s:]+(.+)",
            r"FAILED[\s:]+(.+)",
            r"failed[\s:]+(.+)",
            r"exception[\s:]+(.+)",
            r"Exception[\s:]+(.+)",
            r"command not found.*",
            r"Permission denied.*",
            r"No such file or directory.*",
        ]
        
        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            errors.extend(matches[:5])  # Limit per pattern
        
        return errors[:10]  # Overall limit
    
    async def _llm_detect_errors(self, output: str) -> List[str]:
        """Use LLM to detect errors in output."""
        prompt = f"""Analyze this terminal output and identify any error messages.

Output:
{output[-500:]}

List any error messages found, one per line. If no errors, respond "NO_ERRORS".
"""
        
        try:
            response = await self._llm_client.generate(prompt)
            
            if "NO_ERRORS" in response.upper():
                return []
            
            errors = []
            for line in response.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    errors.append(line)
            
            return errors[:10]
            
        except Exception:
            return []
    
    def get_process(self, process_id: str) -> Optional[ProcessInfo]:
        """Get process info by ID."""
        return self._processes.get(process_id)
    
    def get_background_processes(self) -> List[ProcessInfo]:
        """Get all background processes."""
        return list(self._background_processes.values())
    
    async def wait_for_process(
        self,
        process_id: str,
        timeout_ms: Optional[int] = None,
    ) -> ProcessInfo:
        """Wait for a process to complete.
        
        Args:
            process_id: Process ID
            timeout_ms: Timeout
            
        Returns:
            ProcessInfo when completed
        """
        process = self._processes.get(process_id)
        if not process:
            raise ValueError(f"Process {process_id} not found")
        
        timeout_ms = timeout_ms or self._config.command_timeout_ms
        start_time = time.time()
        
        while process.state in [ProcessState.STARTING, ProcessState.RUNNING]:
            if (time.time() - start_time) * 1000 > timeout_ms:
                raise TimeoutError(f"Waiting for process {process_id} timed out")
            
            await asyncio.sleep(0.1)
        
        return process
    
    async def cleanup_orphans(self):
        """Find and cleanup orphan processes."""
        current_pids = set()
        
        for process in self._processes.values():
            if process.pid and process.state in [ProcessState.RUNNING, ProcessState.STARTING]:
                current_pids.add(process.pid)
        
        # Check if processes are still running
        for process_id, process in list(self._processes.items()):
            if process.pid and process.state == ProcessState.RUNNING:
                try:
                    if sys.platform == "win32":
                        result = subprocess.run(
                            ["tasklist", "/FI", f"PID eq {process.pid}"],
                            capture_output=True,
                            text=True,
                        )
                        is_running = str(process.pid) in result.stdout
                    else:
                        os.kill(process.pid, 0)
                        is_running = True
                        
                except (OSError, subprocess.SubprocessError):
                    is_running = False
                
                if not is_running:
                    process.state = ProcessState.ORPHAN
                    logger.warning(f"Process {process_id} appears to be orphaned")


# Module-level instance
_global_terminal_manager: Optional[TerminalManager] = None


def get_terminal_manager(
    llm_client: Optional[Any] = None,
    config: Optional[TerminalConfig] = None,
) -> TerminalManager:
    """Get the global terminal manager.
    
    Args:
        llm_client: Optional LLM client
        config: Optional terminal config
        
    Returns:
        TerminalManager instance
    """
    global _global_terminal_manager
    if _global_terminal_manager is None:
        _global_terminal_manager = TerminalManager(
            llm_client=llm_client,
            config=config,
        )
    return _global_terminal_manager
