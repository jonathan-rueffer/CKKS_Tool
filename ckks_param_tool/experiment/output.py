from rich.console import Console
from rich.table import Table
from rich.panel import Panel

'''
This file contains the functions that print the tool results to the console using the rich library
'''

console = Console()

# helper function for bit conversion 
def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"

# function that formats the memory metric to show for each CKKS component
def memory_table(mem: dict[str, int]) -> Table:
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component")
    table.add_column("Size", justify="right")

    for k, v in mem.items():
        if k != "total":
            table.add_row(k.replace("_", " ").title(), format_bytes(v))

    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{format_bytes(mem['total'])}[/bold]"
    )

    return table

# function that prints the recommended params and their metrics
def print_experiment_summary(results):

    for sec, params, inference in results.recommended_params:

         # the parameters themselves
        table = Table(title=f"Recommended for >= {sec}-bit security", show_header=True, header_style="bold cyan")
        table.add_column("Parameter", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Polynomial degree", str(params.poly_degree))
        table.add_row("Scale", f"2^{int.bit_length(params.scale) - 1}")
        table.add_row("Modulus Chain", str(params.chain))
        table.add_row("Coefficient Modulus degree", str(sum(params.chain)))
        table.add_row("Security Level (bits)", str(params.security_score))

        # the associated metrics
        metrics = Table(show_header=False)
        metrics.add_column(style="dim")
        metrics.add_column(style="bold")

        metrics.add_row("Inference error", f"{inference.error:.4e}")
        metrics.add_row("Precision (bits)", f"{inference.precision_bits:.1f}")
        metrics.add_row("Median runtime (s)", f"{inference.runtime:.4f}")

        mem = params.memory

        console.print(Panel(table, title="Parameter Selection"))
        console.print(Panel(memory_table(mem), title="Estimated Memory Usage"))
        console.print(Panel(metrics, title="Encrypted Inference"))
        console.print("\n" + "-"*120 + "\n") 