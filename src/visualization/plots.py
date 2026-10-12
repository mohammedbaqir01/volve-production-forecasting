from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_forecast_plot(actual, predicted, output_path: str | Path, title: str) -> Path:
    """Save an actual-versus-forecast line plot and close its figure.

    Args:
        actual: Actual target series with a date-like index.
        predicted: Forecast values aligned to the same index.
        output_path: Destination image path. Parent folders are created as needed.
        title: Descriptive chart title.

    Returns:
        The destination path of the saved 300-dpi PNG or other Matplotlib-
        supported image format.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual.index, actual, label="Actual", color="#2878B5", linewidth=1.8)
    ax.plot(
        actual.index,
        predicted,
        label="Forecast",
        color="#E07A2D",
        linewidth=1.8,
        linestyle="--",
    )
    ax.set(title=title, xlabel="Date", ylabel="Bore oil volume")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
