from jet.commands import *
from jet.utils.cli import register_command  

import typer

app = typer.Typer()

register_command(app, download)
register_command(app, tokenize)
register_command(app, train)

if __name__ == "__main__":
    app()
