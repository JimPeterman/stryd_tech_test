"""
Helper functions for assessing ML models performance.
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm


def calculate_r2_mae_rmse(actual: list, predicted: list) -> str:
    """
    Return the R2, MAE, and RMSE for a model.

    Args:
        actual (list): Actual target values.
        predicted (list): Predicted target values.
    """

    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)

    metrics = f"R2: {r2:.4f}\nMAE: {mae:,.0f}\nRMSE: {rmse:,.0f}"
    return metrics


def plot_residuals_qqplot_fit(actual: list, predicted: list):
    """
    Returns plots illustrating model fit.

    Args:
        actual (list): Actual target values.
        predicted (list): Predicted target values.
    """
    residuals = actual - predicted

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].scatter(x=predicted, y=residuals, alpha=0.2)
    axs[0].axhline(y=0.0, color="r")
    axs[0].set_title("Model Residuals", fontweight="bold", fontsize=16)
    axs[0].set_xlabel("Predicted Values", weight="bold")
    axs[0].set_ylabel("Residuals", weight="bold")

    sm.qqplot(residuals, line="q", ax=axs[1])
    axs[1].set_title("Q-Q Plot", fontweight="bold", fontsize=16)
    axs[1].set_xlabel("Theoretical Quantiles", weight="bold")
    axs[1].set_ylabel("Ordered Values", weight="bold")

    axs[2].scatter(x=actual, y=predicted, alpha=0.2, label=None)
    p1 = max(max(actual), max(predicted))
    p2 = min(min(actual), min(predicted))
    axs[2].plot([p1, p2], [p1, p2], "r--", linewidth=2.0, label="Line of Identity")
    axs[2].legend()
    axs[2].set_title("Model Fit", fontweight="bold", fontsize=16)
    axs[2].set_xlabel("Actual Values", weight="bold")
    axs[2].set_ylabel("Predicted Values", weight="bold")

    return fig
