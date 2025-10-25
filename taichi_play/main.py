"""Main CLI entry point for Taichi examples."""

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="taichi-play")
def cli():
    """Taichi playground with example demonstrations.

    Run various Taichi GPU programming examples and visualizations.
    """
    pass


@cli.command()
def julia():
    """Display an animated Julia Set fractal visualization."""
    from taichi_play.examples.julia_set import run

    run()


@cli.command()
def cloth():
    """Run a cloth and ball physics simulation."""
    from taichi_play.examples.cloth_simulation import run

    run()


@cli.command()
def widgets():
    """Display interactive GUI widgets demo."""
    from taichi_play.examples.gui_widgets import run

    run()


@cli.command()
def triangle():
    """Display a simple colored triangle using GGUI (OpenGL-style)."""
    from taichi_play.examples.triangle import run

    run()


@cli.command()
def list():
    """List all available examples."""
    examples = [
        ("julia", "Animated Julia Set fractal visualization"),
        ("cloth", "Cloth and ball physics simulation"),
        ("widgets", "Interactive GUI widgets demo"),
        ("triangle", "Simple colored triangle"),
    ]

    click.echo("\nAvailable Examples:")
    click.echo("=" * 50)
    for name, description in examples:
        click.echo(f"  {name:12} - {description}")
    click.echo("\nRun an example with: taichi-play <example-name>")
    click.echo()


if __name__ == "__main__":
    cli()
