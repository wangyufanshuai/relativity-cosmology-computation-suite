"""Entry point: run the full Mercury precession analysis."""

from mercury_precession.visualization import print_report, plot_precession_comparison
import matplotlib.pyplot as plt


def main():
    print_report()
    fig = plot_precession_comparison()
    plt.savefig("mercury_precession_analysis.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to mercury_precession_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
