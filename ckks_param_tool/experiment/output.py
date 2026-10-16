from rich.console import Console
from rich.table import Table
from rich.panel import Panel

'''
This file contains the functions that print the tool results to the console using the rich library
'''

console = Console()

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
        metrics.add_row("Max. relative error", f"{inference.max_relative_error:.2e}")
        if inference.prediction_agreement is not None:
            metrics.add_row("Prediction agreement", f"{inference.prediction_agreement:.1%}")

        console.print(Panel(table, title="Parameter Selection"))
        console.print(Panel(metrics, title="Encrypted Inference"))
        console.print("\n" + "-"*120 + "\n") 