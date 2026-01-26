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
    try:
        from proxima.tui.app import ProximaTUI
        typer.echo("Starting Proxima TUI...")
        app = ProximaTUI(theme=theme, initial_screen=screen)
        app.run()
    except ImportError as e:
        typer.echo("=" * 50)
        typer.echo("  TUI dependencies not installed!")
        typer.echo("=" * 50)
        typer.echo("")
        typer.echo(f"  Error: {e}")
        typer.echo("")
        typer.echo("  Install TUI dependencies with:")
        typer.echo("    pip install proxima-agent[ui]")
        typer.echo("")
        typer.echo("  Or install textual directly:")
        typer.echo("    pip install textual rich")
        typer.echo("")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error launching TUI: {e}")
        raise typer.Exit(1)


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
            typer.echo(f"? {pkg}: {desc}")
        except ImportError:
            typer.echo(f"? {pkg}: {desc} (not installed)")
            all_ok = False

    if all_ok:
        typer.echo("\n? All TUI dependencies are available")
        typer.echo("Run 'proxima ui' to launch")
    else:
        typer.echo("\n? Some dependencies are missing")
        typer.echo("Install with: pip install proxima-agent[ui]")
        raise typer.Exit(1)