"""Helper functions for evaluating predictive models"""

__version__ = "0.0.1"
__author__ = "Nathan Dixon"

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def classification_model_report(results_df, display_labels, threshold=0.50):
    """
    Generates common classification evaluation plots and statistics from a dataframe of actuals and predicted
    probabilities. Plots include ROC curve, calibration plots, and F1-curve.

    Args:
        df: dataframe with actuals and predicted probabilities; column expectations defined below
            actual: a column with the actual binary label (positive = 1, negative = 0)
            predicted_prob: a column with the predicted probability of the positive label from the model
        display_labels: class labels
        threshold: probability value for calculating precision and recall
    """

    report_df = results_df.copy()
    report_df["predicted"] = np.where(
        report_df.predicted_prob >= threshold, 1, 0
    )

    fpr, tpr, _ = roc_curve(report_df.actual, report_df.predicted_prob)
    roc_auc = roc_auc_score(report_df.actual, report_df.predicted_prob)
    plt.figure(figsize=(9, 6.5))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()
    print("AUC Score is: {:4f}".format(roc_auc))
    print("Classification Metrics Assuming 50% Predicted Prob Cutoff:")
    print(
        "Accuracy is: {:4f}".format(
            accuracy_score(report_df.actual, report_df.predicted)
        )
    )
    print(
        "Precision is: {:4f}".format(
            precision_score(report_df.actual, report_df.predicted)
        )
    )
    print(
        "Recall is: {:4f}".format(
            recall_score(report_df.actual, report_df.predicted)
        )
    )
    print(
        "F1-Score is: {:4f}".format(
            f1_score(report_df.actual, report_df.predicted)
        )
    )

    ConfusionMatrixDisplay(
        confusion_matrix(report_df.actual, report_df.predicted),
        display_labels=display_labels,
    ).plot(values_format="")
    plt.show()

    display(results_df.describe())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.kdeplot(
        data=report_df,
        x="predicted_prob",
        hue="actual",
        common_norm=False,
        ax=ax,
    ).set(title="Distribution of Predicted Probabilities by Actual Class")
    plt.show()
    prob_true, prob_pred = calibration_curve(
        report_df.actual, report_df.predicted_prob, n_bins=20
    )
    plt.figure(figsize=(11, 5.5))
    low = 2
    plt.plot(
        prob_pred, prob_true, color="blue", lw=lw, linestyle="-", marker="s"
    )
    plt.plot([0, 1], [0, 1], color="grey", lw=lw, linestyle="--")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean Predicted Probability")
    plt.title("Calibration Curve")
    plt.show()

    # make a list of values between 0 and 100, counting by 5s, for calculating confusion matrices
    cutoffs = [i / 100 for i in range(5, 90) if i % 2 == 0]

    # create empty data frame to append to
    class_metrics = list()

    # loop through cutoffs, grab metrics
    for c in cutoffs:
        # predict a class based on a cutoff
        y_pred = np.where(report_df.predicted_prob >= c, 1, 0)

        cutoff_dict = {
            "cutoff": c,
            "f1_score": f1_score(report_df.actual, y_pred),
        }

        class_metrics.append(cutoff_dict)

    class_metrics_df = pd.DataFrame(class_metrics)

    best_cutoff = class_metrics_df.sort_values(
        "f1_score", ascending=False
    ).cutoff.values[0]

    plt.figure(figsize=(11, 5.5))
    lw = 2
    plt.plot(
        class_metrics_df.cutoff,
        class_metrics_df.f1_score,
        color="red",
        lw=lw,
        linestyle="-",
        marker="s",
        label="F1-score is maximized at cutoff: %0.4f" % best_cutoff,
    )
    plt.ylabel("F1-Score")
    plt.xlabel("Probability Cutoff")
    plt.title("F1-Score Curve")
    plt.legend(loc="upper right")
    plt.show()
