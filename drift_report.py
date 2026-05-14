"""
Generate visual drift reports for YOLOX Nano.
Produces both terminal output (via Rich) and matplotlib plots.
"""

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

from ..detectors.statistical_tests import DriftTestResult


STATUS_COLORS = {
    "HEALTHY": "green",
    "MINOR_DRIFT": "yellow",
    "WARNING": "dark_orange",
    "CRITICAL": "red",
}

STATUS_ICONS = {
    "HEALTHY": "[green]HEALTHY[/green]",
    "MINOR_DRIFT": "[yellow]MINOR DRIFT[/yellow]",
    "WARNING": "[dark_orange]WARNING[/dark_orange]",
    "CRITICAL": "[red]CRITICAL[/red]",
}


class DriftReportGenerator:
    """Generates comprehensive drift reports for YOLOX Nano."""

    def __init__(self, output_dir: str = "drift_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console() if RICH_AVAILABLE else None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate(
        self,
        image_results: Optional[dict] = None,
        prediction_results: Optional[dict] = None,
        ref_images: Optional[np.ndarray] = None,
        prod_images: Optional[np.ndarray] = None,
    ):
        """Generate full drift report."""
        self._print_header()

        if image_results:
            self._print_image_drift(image_results)
        if prediction_results:
            self._print_prediction_drift(prediction_results)

        if PLT_AVAILABLE:
            self._generate_plots(
                image_results, prediction_results, ref_images, prod_images
            )

        self._print_footer(image_results, prediction_results)

    def _print_header(self):
        if not self.console:
            print("=" * 70)
            print("  YOLOX Nano Data Drift Report")
            print(f"  Generated: {self.timestamp}")
            print("=" * 70)
            return

        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]YOLOX Nano — Data Drift Detection Report[/bold cyan]\n"
            f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="cyan",
        ))
        self.console.print()

    def _print_image_drift(self, results: dict):
        summary = results.get("summary", {})
        status = summary.get("overall_status", "UNKNOWN")

        if self.console:
            self.console.print(f"  [bold]Image-Level Drift:[/bold] {STATUS_ICONS.get(status, status)}")
            self.console.print(
                f"  Tests: {summary.get('drifted_tests', 0)}/{summary.get('total_tests', 0)} drifted  "
                f"| Critical: {summary.get('critical_count', 0)}  "
                f"| Warning: {summary.get('warning_count', 0)}"
            )
            self.console.print()

            self._print_test_table("Pixel-Level Tests", results.get("pixel_drift", {}))
            self._print_test_table("Embedding Tests", results.get("embedding_drift", {}))
            self._print_stats_table(results.get("statistics_drift", {}))
        else:
            print(f"\n  Image-Level Drift: {status}")
            for section_name, section in results.items():
                if isinstance(section, dict):
                    for k, v in section.items():
                        if isinstance(v, DriftTestResult):
                            print(f"    {v}")

    def _print_prediction_drift(self, results: dict):
        summary = results.get("summary", {})
        status = summary.get("overall_status", "UNKNOWN")

        if self.console:
            self.console.print(f"  [bold]Prediction-Level Drift:[/bold] {STATUS_ICONS.get(status, status)}")
            self.console.print(
                f"  Tests: {summary.get('drifted_tests', 0)}/{summary.get('total_tests', 0)} drifted  "
                f"| Critical: {summary.get('critical_count', 0)}"
            )
            self.console.print()

            self._print_test_table("Confidence Drift", results.get("confidence_drift", {}))
            self._print_test_table("BBox Drift", results.get("bbox_drift", {}))
            self._print_class_drift(results.get("class_distribution", {}))
            self._print_count_drift(results.get("detection_count_drift", {}))
        else:
            print(f"\n  Prediction-Level Drift: {status}")

    def _print_test_table(self, title: str, tests: dict):
        if not self.console or not tests:
            return

        table = Table(title=title, show_lines=True)
        table.add_column("Test", style="bold")
        table.add_column("Statistic", justify="right")
        table.add_column("P-Value", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Severity", justify="center")

        for key, result in tests.items():
            if not isinstance(result, DriftTestResult):
                continue
            status_str = "[red]DRIFT[/red]" if result.is_drift else "[green]OK[/green]"
            sev_color = {"critical": "red", "warning": "dark_orange", "none": "green"}.get(result.severity, "white")
            table.add_row(
                result.test_name,
                f"{result.statistic:.4f}",
                f"{result.p_value:.4f}",
                status_str,
                f"[{sev_color}]{result.severity.upper()}[/{sev_color}]",
            )

        self.console.print(table)
        self.console.print()

    def _print_stats_table(self, stats: dict):
        if not self.console or not stats:
            return

        table = Table(title="Image Statistics Comparison", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Reference", justify="right")
        table.add_column("Production", justify="right")
        table.add_column("Change %", justify="right")
        table.add_column("Alert", justify="center")

        for key, val in stats.items():
            if not isinstance(val, dict):
                continue
            alert_str = "[red]![/red]" if val.get("alert") else "[green]OK[/green]"
            change_color = "red" if val.get("pct_change", 0) > 10 else "green"
            table.add_row(
                key,
                f"{val['reference']:.4f}",
                f"{val['production']:.4f}",
                f"[{change_color}]{val['pct_change']:.1f}%[/{change_color}]",
                alert_str,
            )

        self.console.print(table)
        self.console.print()

    def _print_class_drift(self, class_data: dict):
        if not self.console or not class_data:
            return

        chi_result = class_data.get("chi_square")
        if isinstance(chi_result, DriftTestResult):
            status = "[red]DRIFT[/red]" if chi_result.is_drift else "[green]OK[/green]"
            self.console.print(
                f"  Class Distribution: {status} "
                f"(Chi²={chi_result.statistic:.2f}, p={chi_result.p_value:.4f})"
            )

        per_class = class_data.get("per_class", {})
        if per_class:
            table = Table(title="Top Class Changes", show_lines=True)
            table.add_column("Class", style="bold")
            table.add_column("Ref %", justify="right")
            table.add_column("Prod %", justify="right")
            table.add_column("Change", justify="right")

            sorted_classes = sorted(per_class.items(), key=lambda x: abs(x[1]["change"]), reverse=True)
            for name, vals in sorted_classes[:10]:
                change = vals["change"]
                color = "red" if abs(change) > 2 else "yellow" if abs(change) > 0.5 else "green"
                sign = "+" if change > 0 else ""
                table.add_row(
                    name,
                    f"{vals['ref_pct']:.1f}%",
                    f"{vals['prod_pct']:.1f}%",
                    f"[{color}]{sign}{change:.1f}%[/{color}]",
                )
            self.console.print(table)
            self.console.print()

    def _print_count_drift(self, count_data: dict):
        if not self.console or not count_data:
            return
        ref_mean = count_data.get("ref_mean_count", 0)
        prod_mean = count_data.get("prod_mean_count", 0)
        change_pct = count_data.get("count_change_pct", 0)
        color = "red" if abs(change_pct) > 20 else "yellow" if abs(change_pct) > 10 else "green"
        self.console.print(
            f"  Detections/Image: ref={ref_mean:.1f} → prod={prod_mean:.1f} "
            f"[{color}]({change_pct:+.1f}%)[/{color}]"
        )
        self.console.print()

    def _print_footer(self, image_results, prediction_results):
        statuses = []
        if image_results and "summary" in image_results:
            statuses.append(image_results["summary"].get("overall_status", "UNKNOWN"))
        if prediction_results and "summary" in prediction_results:
            statuses.append(prediction_results["summary"].get("overall_status", "UNKNOWN"))

        priority = ["CRITICAL", "WARNING", "MINOR_DRIFT", "HEALTHY"]
        overall = "HEALTHY"
        for p in priority:
            if p in statuses:
                overall = p
                break

        if self.console:
            self.console.print(Panel.fit(
                f"[bold]Overall Status: {STATUS_ICONS.get(overall, overall)}[/bold]\n"
                f"[dim]Report saved to: {self.output_dir}[/dim]",
                border_style=STATUS_COLORS.get(overall, "white"),
            ))
        else:
            print(f"\n  Overall Status: {overall}")
            print(f"  Report saved to: {self.output_dir}")

    def _generate_plots(self, image_results, prediction_results, ref_images, prod_images):
        if not PLT_AVAILABLE:
            return

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle("YOLOX Nano — Data Drift Report", fontsize=16, fontweight="bold")
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Plot 1: Channel histograms
        if ref_images is not None and prod_images is not None:
            for c, (name, color) in enumerate(
                [("Red", "red"), ("Green", "green"), ("Blue", "blue")]
            ):
                ax = fig.add_subplot(gs[0, c])
                ref_channel = ref_images[..., c].flatten()
                prod_channel = prod_images[..., c].flatten()
                ax.hist(ref_channel, bins=50, alpha=0.5, label="Reference", color=color, density=True)
                ax.hist(prod_channel, bins=50, alpha=0.5, label="Production", color="gray", density=True)
                ax.set_title(f"{name} Channel Distribution")
                ax.legend(fontsize=8)
                ax.set_xlabel("Pixel Value")

        # Plot 2: Drift test summary
        ax = fig.add_subplot(gs[1, 0])
        self._plot_drift_summary(ax, image_results, "Image Drift")

        ax = fig.add_subplot(gs[1, 1])
        self._plot_drift_summary(ax, prediction_results, "Prediction Drift")

        # Plot 3: Brightness comparison
        if ref_images is not None and prod_images is not None:
            ax = fig.add_subplot(gs[1, 2])
            ref_bright = np.mean(ref_images, axis=-1).mean(axis=(1, 2))
            prod_bright = np.mean(prod_images, axis=-1).mean(axis=(1, 2))
            ax.hist(ref_bright, bins=30, alpha=0.6, label="Reference", color="steelblue")
            ax.hist(prod_bright, bins=30, alpha=0.6, label="Production", color="coral")
            ax.set_title("Brightness Distribution")
            ax.legend()
            ax.set_xlabel("Mean Brightness")

        # Plot 4: Confidence drift
        if prediction_results and "confidence_drift" in prediction_results:
            ax = fig.add_subplot(gs[2, 0])
            conf = prediction_results["confidence_drift"]
            ref_mean = conf.get("ref_mean_confidence", 0)
            prod_mean = conf.get("prod_mean_confidence", 0)
            bars = ax.bar(["Reference", "Production"], [ref_mean, prod_mean],
                         color=["steelblue", "coral"])
            ax.set_title("Mean Confidence Score")
            ax.set_ylim(0, 1)
            for bar, val in zip(bars, [ref_mean, prod_mean]):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                       f"{val:.3f}", ha="center", fontweight="bold")

        # Plot 5: Detection count
        if prediction_results and "detection_count_drift" in prediction_results:
            ax = fig.add_subplot(gs[2, 1])
            count = prediction_results["detection_count_drift"]
            ref_mean = count.get("ref_mean_count", 0)
            prod_mean = count.get("prod_mean_count", 0)
            bars = ax.bar(["Reference", "Production"], [ref_mean, prod_mean],
                         color=["steelblue", "coral"])
            ax.set_title("Mean Detections per Image")
            for bar, val in zip(bars, [ref_mean, prod_mean]):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                       f"{val:.1f}", ha="center", fontweight="bold")

        # Plot 6: Overall status gauge
        ax = fig.add_subplot(gs[2, 2])
        self._plot_status_gauge(ax, image_results, prediction_results)

        save_path = self.output_dir / f"drift_report_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        if self.console:
            self.console.print(f"  [dim]Plot saved: {save_path}[/dim]")

    def _plot_drift_summary(self, ax, results, title):
        if not results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return

        summary = results.get("summary", {})
        total = summary.get("total_tests", 0)
        drifted = summary.get("drifted_tests", 0)
        healthy = total - drifted

        if total > 0:
            colors = ["#2ecc71", "#e74c3c"]
            ax.pie([healthy, drifted], labels=["OK", "Drift"],
                   colors=colors, autopct="%1.0f%%", startangle=90)
        ax.set_title(title)

    def _plot_status_gauge(self, ax, image_results, prediction_results):
        statuses = []
        if image_results:
            s = image_results.get("summary", {}).get("overall_status", "HEALTHY")
            statuses.append(s)
        if prediction_results:
            s = prediction_results.get("summary", {}).get("overall_status", "HEALTHY")
            statuses.append(s)

        priority = {"CRITICAL": 3, "WARNING": 2, "MINOR_DRIFT": 1, "HEALTHY": 0}
        score = max(priority.get(s, 0) for s in statuses) if statuses else 0
        color_map = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#e74c3c"}

        circle = plt.Circle((0.5, 0.5), 0.4, color=color_map[score], transform=ax.transAxes)
        ax.add_patch(circle)
        label_map = {0: "HEALTHY", 1: "MINOR\nDRIFT", 2: "WARNING", 3: "CRITICAL"}
        ax.text(0.5, 0.5, label_map[score], ha="center", va="center",
               fontsize=14, fontweight="bold", color="white", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Overall Status")
