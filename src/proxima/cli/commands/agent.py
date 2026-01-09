"""
Agent CLI commands.

Execute proxima_agent.md files.
"""

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Execute proxima_agent.md files.")


@app.callback(invoke_without_command=True)
def agent_callback(ctx: typer.Context) -> None:
    """Show agent command help."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("run")
def run_agent(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without executing"),
    step: bool = typer.Option(False, "--step", help="Execute one task at a time"),
    resume_from: Optional[int] = typer.Option(None, "--resume", "-r", help="Resume from task N"),
) -> None:
    """Execute a proxima_agent.md file."""
    from proxima.core.agent_interpreter import AgentFileParser, AgentExecutor

    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)

    if not agent_file.suffix == ".md":
        typer.echo("Warning: Agent file should be a .md file", err=True)

    typer.echo(f"Loading agent file: {agent_file}")

    try:
        parser = AgentFileParser()
        agent_config = parser.parse(agent_file)
    except Exception as e:
        typer.echo(f"Failed to parse agent file: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Agent: {agent_config.metadata.name} v{agent_config.metadata.version}")
    typer.echo(f"Tasks: {len(agent_config.tasks)}")

    if dry_run:
        typer.echo("\n--- Execution Plan (Dry Run) ---")
        for i, task in enumerate(agent_config.tasks, 1):
            typer.echo(f"\n[Task {i}] {task.name}")
            typer.echo(f"  Type: {task.task_type.value}")
            typer.echo(f"  Description: {task.description[:80]}..." if len(task.description) > 80 else f"  Description: {task.description}")
        typer.echo("\n--- End of Plan ---")
        return

    # Execute the agent
    executor = AgentExecutor(agent_config)
    
    start_task = resume_from or 1
    if start_task > 1:
        typer.echo(f"Resuming from task {start_task}")

    try:
        for i, task in enumerate(agent_config.tasks, 1):
            if i < start_task:
                continue

            typer.echo(f"\n[Task {i}/{len(agent_config.tasks)}] {task.name}")
            
            if step:
                if not typer.confirm("Execute this task?"):
                    typer.echo("Skipped.")
                    continue

            result = executor.execute_task(task)
            
            if result.success:
                typer.echo(f"  ✓ Completed in {result.execution_time_ms:.1f}ms")
            else:
                typer.echo(f"  ✗ Failed: {result.error}")
                if not typer.confirm("Continue with next task?"):
                    raise typer.Exit(1)

        typer.echo("\n✓ Agent execution completed")

    except KeyboardInterrupt:
        typer.echo(f"\n⚠ Execution interrupted at task {i}")
        typer.echo(f"Resume with: proxima agent run {agent_file} --resume {i}")
        raise typer.Exit(130)


@app.command("validate")
def validate_agent(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
) -> None:
    """Validate a proxima_agent.md file without executing."""
    from proxima.core.agent_interpreter import AgentFileParser

    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Validating: {agent_file}")

    try:
        parser = AgentFileParser()
        agent_config = parser.parse(agent_file)
        
        errors = agent_config.validate()
        
        if errors:
            typer.echo("\n✗ Validation failed:")
            for error in errors:
                typer.echo(f"  - {error}")
            raise typer.Exit(1)
        
        typer.echo("\n✓ Agent file is valid")
        typer.echo(f"  Name: {agent_config.metadata.name}")
        typer.echo(f"  Version: {agent_config.metadata.version}")
        typer.echo(f"  Tasks: {len(agent_config.tasks)}")
        
        for i, task in enumerate(agent_config.tasks, 1):
            typer.echo(f"  [{i}] {task.name} ({task.task_type.value})")

    except Exception as e:
        typer.echo(f"\n✗ Parse error: {e}", err=True)
        raise typer.Exit(1)


@app.command("new")
def new_agent(
    output_file: Path = typer.Argument(
        Path("proxima_agent.md"),
        help="Output file path"
    ),
    name: str = typer.Option("My Agent", "--name", "-n", help="Agent name"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
) -> None:
    """Create a new proxima_agent.md template."""
    if output_file.exists() and not force:
        typer.echo(f"File already exists: {output_file}", err=True)
        typer.echo("Use --force to overwrite")
        raise typer.Exit(1)

    template = f'''# Proxima Agent Instructions

## Metadata
- name: {name}
- version: 1.0.0
- author: {typer.prompt("Author", default="Anonymous")}

## Configuration
- backend: auto
- shots: 1000
- output_format: xlsx

## Tasks

### Task 1: Bell State Preparation
Create and measure a Bell state circuit.

**Type:** simulation
**Backend:** auto

```quantum
H 0
CNOT 0 1
MEASURE ALL
```

### Task 2: Analyze Results
Generate insights from the simulation results.

**Type:** analysis
**Use LLM:** optional

Analyze the measurement distribution and provide insights.

## Output
- Format: XLSX
- Include: fidelity, execution time, insights
'''

    output_file.write_text(template)
    typer.echo(f"Created agent file: {output_file}")
    typer.echo(f"Edit the file and run with: proxima agent run {output_file}")


@app.command("list-tasks")
def list_tasks(
    agent_file: Path = typer.Argument(..., help="Path to proxima_agent.md file"),
) -> None:
    """List tasks in an agent file."""
    from proxima.core.agent_interpreter import AgentFileParser

    if not agent_file.exists():
        typer.echo(f"Agent file not found: {agent_file}", err=True)
        raise typer.Exit(1)

    try:
        parser = AgentFileParser()
        agent_config = parser.parse(agent_file)

        typer.echo(f"\nTasks in {agent_file}:\n")
        for i, task in enumerate(agent_config.tasks, 1):
            typer.echo(f"[{i}] {task.name}")
            typer.echo(f"    Type: {task.task_type.value}")
            if task.backend:
                typer.echo(f"    Backend: {task.backend}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Failed to parse agent file: {e}", err=True)
        raise typer.Exit(1)
