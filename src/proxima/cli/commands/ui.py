"""
TUI CLI commands.

Launch the Terminal User Interface.
"""

import typer

app = typer.Typer(help="Launch the Terminal User Interface.")


@app.callback(invoke_without_command=True)
def ui_callback(ctx: typer.Context) -> None:
    """Launch the TUI."""
    if ctx.invoked_subcommand is None:
        launch_tui()


@app.command("launch")
def launch(
    theme: str = typer.Option("dark", "--theme", "-t", help="UI theme (dark|light)"),
    screen: str = typer.Option("dashboard", "--screen", "-s", help="Initial screen"),
) -> None:
    """Launch the Proxima TUI."""
    launch_tui(theme, screen)


def launch_tui(theme: str = "dark", screen: str = "dashboard") -> None:
    """Launch the Proxima TUI with the specified theme and screen."""
    typer.echo("=" * 50)
    typer.echo("  Proxima TUI - Coming Soon!")
    typer.echo("=" * 50)
    typer.echo("")
    typer.echo("  The TUI is being rebuilt from scratch.")
    typer.echo("  Please check back soon for the new interface.")
    typer.echo("")
    typer.echo("  In the meantime, you can use the CLI commands:")
    typer.echo("    proxima run <circuit>")
    typer.echo("    proxima benchmark <suite>")
    typer.echo("    proxima config show")
    typer.echo("")
    raise typer.Exit(0)


@app.command("check")
def check_tui() -> None:
    """Check if TUI dependencies are available."""
    dependencies = {
        "textual": "TUI framework",
        "rich": "Rich text rendering",
    }

    all_ok = True
    for pkg, desc in dependencies.items():
        try:
            __import__(pkg)
            typer.echo(f"✓ {pkg}: {desc}")
        except ImportError:
            typer.echo(f"✗ {pkg}: {desc} (not installed)")
            all_ok = False

    if all_ok:
        typer.echo("\n✓ All TUI dependencies are available")
        typer.echo("Run 'proxima ui' to launch")
    else:
        typer.echo("\n✗ Some dependencies are missing")
        typer.echo("Install with: pip install proxima-agent[tui]")
        raise typer.Exit(1)
