import typer
from tools import logger

from cli import (
    evaluate,
    generate
)

app = typer.Typer()

app.command()(evaluate)
app.command()(generate)

if __name__ == '__main__':
    app()