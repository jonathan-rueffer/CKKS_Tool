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
        table = Table(title=f"Recommended (knee) for >= {sec}-bit security", show_header=True, header_style="bold cyan")
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

        metrics.add_row("Error (test vectors)", f"{inference.error:.4e}")
        metrics.add_row("Precision (bits)", f"{inference.precision_bits:.1f}")
        metrics.add_row("Median runtime (s)", f"{inference.runtime:.4f}")

        console.print(Panel(table, title="Parameter Selection"))
        console.print(Panel(metrics, title="Encrypted Inference"))
        console.print("\n" + "-"*120 + "\n") 

    # the two ends of each front, so the trade-off is visible at a glance
    if results.front_points:
        ends = Table(title="Front endpoints per security level", header_style="bold cyan")
        for col in ("Security", "Point", "N", "Chain", "Precision (bits)", "Median runtime (s)"):
            ends.add_column(col)
        for sec, label, params, inf in results.front_points:
            ends.add_row(f">= {sec}", label.replace("_", " "), str(params.poly_degree), str(params.chain),
                         f"{inf.precision_bits:.1f}", f"{inf.runtime:.4f}")
        console.print(ends)

    if results.estimated_required_bits is not None:
        console.print(f"Estimated precision this classifier needs: {results.estimated_required_bits} bits "
                      f"(from its decision margins; information only)")
