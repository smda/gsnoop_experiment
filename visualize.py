#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from scipy.stats import wilcoxon
from cliffs_delta import cliffs_delta

st.set_page_config(layout="wide")


def create_diverging_colormap(hex1, hex2, midpoint="white", name="CustomDivergingMap"):
    """
    Create a custom diverging colormap using two specified colors and a midpoint color (white by default).

    Parameters:
    - hex1 (str): Hex code of the first color.
    - hex2 (str): Hex code of the second color.
    - midpoint (str): Hex code of the midpoint color, default is 'white'.
    - name (str): Name of the colormap.

    Returns:
    - matplotlib.colors.LinearSegmentedColormap: A custom diverging colormap.
    """
    # Create a colormap from the specified colors
    cmap = mcolors.LinearSegmentedColormap.from_list(name, [hex1, midpoint, hex2])
    return cmap


# Function to load data
options = [
    "features",
    "interaction_p",
    "rel_sample_size",
    # "relevant_options",
    "relevant_terms_level",
]
classification_metrix = [
    "lasso_precision",
    "lasso_recall",
    "lasso_f1",
    "group_precision",
    "group_recall",
    "group_f1",
    "baseline_precision",
    "baseline_recall",
    "baseline_f1",
    "jaccard",
]
prediction_metrix = [
    "lasso_mape-1",
    "lasso_mape-2",
    "lasso_mape-3",
    "lasso_expvar-1",
    "lasso_expvar-2",
    "lasso_expvar-3",
    "group_mape-1",
    "group_mape-2",
    "group_mape-3",
    "group_expvar-1",
    "group_expvar-2",
    "group_expvar-3",
]

metrix = prediction_metrix + classification_metrix

classification_labels = {"Precision": "precision", "Recall": "recall", "F1 Score": "f1"}

regression_labels = {
    "MAPE": "mape",
    "Explained Variance": "expvar",
}


def load_data():
    # This function will load data from a CSV file
    # Assuming the CSV file is in the same directory as the script
    df = pd.read_csv("results.csv")
    df_raw = df.copy()
    # mean over all repetitions
    collect = []
    for index, gdf in df.groupby(["system", "rel_sample_size"]):
        row = {}
        row["system"] = index[0]
        row["rel_sample_size"] = index[1]
        for option in options:
            row[option] = gdf[option].unique()[0]
        for metric in metrix:
            row[metric] = gdf[metric].mean()

        collect.append(row)
    collect = pd.DataFrame(collect)
    # st.write(collect)

    # mean over systems with similar number of features, interaction_p, relevant_terms_level
    data = []
    for index, gdf in collect.groupby(
        ["features", "interaction_p", "relevant_terms_level", "rel_sample_size"]
    ):
        row = {
            "features": index[0],
            "interaction_p": index[1],
            "relevant_terms_level": index[2],
            "rel_sample_size": index[3],
        }
        for metric in metrix:
            row[metric] = gdf[metric].mean()
        data.append(row)
    df = pd.DataFrame(data)
    return df, df_raw


# Main app function
def main():

    data, data_raw = load_data()
    # st.write(data)
    # Title of the dashboard
    # st.title("Group Screening vs Lasso Screening")

    with st.sidebar:
        # Show data frame on the page
        st.write("Select Axis Labels")
        x_axis = st.selectbox("x axis ", options=options)
        y_axis = st.selectbox("y axis ", options=options)

        st.divider()

        regression_metric = st.selectbox(
            "Regression Metric", options=regression_labels.keys()
        )

        st.divider()

        classification_metric = st.selectbox(
            "Classiciation Metric", options=classification_labels.keys()
        )

        recall_c = st.checkbox("Recall")
        precision_c = st.checkbox("Precision")
        f1_c = st.checkbox("F1 Score")

        st.divider()

        # interaction_t = st.slider("t", 1, 3, step=1, value=3)

    tab_screening, tab_prediction = st.tabs(["Solar Plots", "Statistics"])

    with tab_screening:
        col1, col2, col3 = st.columns(3)
        with col1:
            if recall_c:
                fig = plt.figure()
                plt.suptitle(f"Lasso Screening")
                metric = "lasso_recall"
                pivo = data.pivot_table(
                    index=x_axis, columns=y_axis, values=metric, aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="cividis_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
            if precision_c:
                fig = plt.figure()
                plt.suptitle(f"Lasso Screening")
                metric = "lasso_precision"
                pivo = data.pivot_table(
                    index=x_axis, columns=y_axis, values=metric, aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="cividis_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

        with col2:
            if recall_c:
                fig = plt.figure()
                plt.suptitle(f" Screening™")
                metric = "group_recall"
                pivo = data.pivot_table(
                    index=x_axis, columns=y_axis, values=metric, aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="cividis_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
            if precision_c:
                fig = plt.figure()
                plt.suptitle(f"Group Screening™")
                metric = "group_precision"
                pivo = data.pivot_table(
                    index=x_axis, columns=y_axis, values=metric, aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="cividis_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

        with col3:
            cmap = create_diverging_colormap("#d7191c", "#0c5c2d")
            if recall_c:
                # compute differences
                diff_col = "recall" + "_diff1"
                data[diff_col] = data["group_recall"] - data["lasso_recall"]

                fig = plt.figure()
                plt.suptitle(f"Group Screening™ vs Lasso Screening")
                metric = "group_" + classification_labels[classification_metric]
                pivo = data.pivot_table(
                    index=x_axis, columns=y_axis, values=diff_col, aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=cmap, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
            if precision_c:
                # compute differences
                diff_col = "precision" + "_diff1"
                data[diff_col] = data["group_precision"] - data["lasso_precision"]

                fig = plt.figure()
                plt.suptitle(f"Group Screening™ vs Lasso Screening")
                metric = "group_" + classification_labels[classification_metric]
                pivo = data.pivot_table(
                    index=x_axis, columns=y_axis, values=diff_col, aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=cmap, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)

        # compute p values per system
        systems = sorted(data_raw["system"].unique())
        for system in systems:
            st.write(system)

    with tab_prediction:
        col1, col2, col3 = st.columns(3)
        """with col1:
            systems = sorted(data_raw["system"].unique())
            
            for (system, rel_sample_size), g in data_raw.groupby(["system", "rel_sample_size"]):
                st.write(wilcoxon(g["lasso_recall"], g["baseline_recall"]), cliffs_delta(g["lasso_recall"], g["baseline_recall"]))

         """
        fig = plt.figure()
        sns.scatterplot(data=data_raw, x="lasso_precision", y="lasso_recall")
        st.pyplot(fig)
        with col2:
            pass


if __name__ == "__main__":
    main()
