#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
import os

import json

st.set_page_config(layout="wide")
plt.style.use("seaborn-v0_8-colorblind")


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


red_green = create_diverging_colormap("#05633e", "#990303")
gradual_cmap = create_diverging_colormap("#ffffff", "#043c61", midpoint="#79c1fc")


label_options = ["features", "rel_sample_size", "relevant_terms_level", "interaction_p"]
methods = {
    "Lasso with Hyperopt.": "lasso",
    "Group with Hyperopt.": "group",
    "Stepwise Group": "stepwise",
    "Minimal Hitting Set": "causal",
}


# Main app function
def main():
    df = pd.read_csv("results.csv")
    with st.sidebar:
        with st.expander("General", expanded=True):
            baseline = st.selectbox("Baseline", options=methods.keys())
            comparison = st.selectbox(
                "methods", options=sorted(list(set(methods.keys()) - set([baseline])))
            )

        st.divider()
        with st.expander("Fine-grained"):
            x_axis = st.selectbox("X axis", options=label_options)
            y_axis = st.selectbox(
                "Y axis", options=sorted(list(set(label_options) - set([x_axis])))
            )

    tab_general, tab_finegrained, tab_top = st.tabs(
        ["Overview", "Fine-grained", "Top Analysis"]
    )
    with tab_general:
        with st.expander("Precision, Recall, and F1 score"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision (sorted)")
                for k, v in methods.items():
                    plt.plot(sorted(df[f"precision_{v}"]), label=k)
                plt.legend()
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall (sorted)")
                for k, v in methods.items():
                    plt.plot(sorted(df[f"recall_{v}"]), label=k)
                plt.legend()
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score (sorted)")
                for k, v in methods.items():
                    plt.plot(sorted(df[f"f1_{v}"]), label=k)
                plt.legend()
                st.pyplot(fig)

        with st.expander("Precision, Recall, and F1 score DIFFERENCES"):
            cols = st.columns(3)

            with cols[0]:
                fig = plt.figure()
                plt.suptitle(
                    f"Precision Difference ({methods[baseline]} vs {methods[comparison]})"
                )
                plt.title(f"({methods[comparison]} better if < 0)")
                plt.plot(
                    sorted(
                        df[f"precision_{methods[baseline]}"]
                        - df[f"precision_{methods[comparison]}"]
                    ),
                    label=comparison,
                    color="indigo",
                )
                plt.axhline(0, linewidth=2, linestyle=":", color="black", alpha=0.5)
                plt.legend()
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle(
                    f"Recall Difference ({methods[baseline]} vs {methods[comparison]})"
                )
                plt.title(f"({methods[comparison]} better if < 0)")
                plt.plot(
                    sorted(
                        df[f"recall_{methods[baseline]}"]
                        - df[f"recall_{methods[comparison]}"]
                    ),
                    label=comparison,
                    color="indigo",
                )
                plt.axhline(0, linewidth=2, linestyle=":", color="black", alpha=0.5)
                plt.legend()
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle(
                    f"F1 Score Difference ({methods[baseline]} vs {methods[comparison]})"
                )
                plt.title(f"({methods[comparison]} better if < 0)")
                plt.plot(
                    sorted(
                        df[f"f1_{methods[baseline]}"] - df[f"f1_{methods[comparison]}"]
                    ),
                    label=comparison,
                    color="indigo",
                )
                plt.axhline(0, linewidth=2, linestyle=":", color="black", alpha=0.5)
                plt.legend()
                st.pyplot(fig)

    with tab_finegrained:

        with st.expander("Lasso Screening (Baseline)"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values="precision_lasso",
                    aggfunc="mean",
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="recall_lasso", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="f1_lasso", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

        with st.expander("Group Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values="precision_group",
                    aggfunc="mean",
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="recall_group", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="f1_group", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

        with st.expander("Stepwise Group Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values="precision_stepwise",
                    aggfunc="mean",
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values="recall_stepwise",
                    aggfunc="mean",
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="f1_stepwise", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

        with st.expander("Causal Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values="precision_causal",
                    aggfunc="mean",
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="recall_causal", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="f1_causal", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

        with st.expander("Comparison"):
            df["_precision"] = (
                df[f"precision_{methods[baseline]}"]
                - df[f"precision_{methods[comparison]}"]
            )
            df["_recall"] = (
                df[f"recall_{methods[baseline]}"] - df[f"recall_{methods[comparison]}"]
            )
            df["_f1"] = df[f"f1_{methods[baseline]}"] - df[f"f1_{methods[comparison]}"]

            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle(
                    f"Precision Difference ({methods[baseline]} vs {methods[comparison]})"
                )
                plt.title("(lower is better)")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="_precision", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle(
                    f"Recall Difference ({methods[baseline]} vs {methods[comparison]})"
                )
                plt.title("(lower is better)")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="_recall", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle(
                    f"F1  Difference ({methods[baseline]} vs {methods[comparison]})"
                )
                plt.title("(lower is better)")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="_f1", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)

        with tab_top:
            with st.expander("Precision"):
                x1 = df[
                    (df["precision_lasso"] > df["precision_group"])
                    & (df["precision_lasso"] > df["precision_causal"])
                    & (df["precision_lasso"] > df["precision_stepwise"])
                ]
                x2 = df[
                    (df["precision_group"] > df["precision_lasso"])
                    & (df["precision_group"] > df["precision_causal"])
                    & (df["precision_group"] > df["precision_stepwise"])
                ]
                x3 = df[
                    (df["precision_causal"] > df["precision_lasso"])
                    & (df["precision_causal"] > df["precision_group"])
                    & (df["precision_causal"] > df["precision_stepwise"])
                ]
                x4 = df[
                    (df["precision_stepwise"] > df["precision_lasso"])
                    & (df["precision_causal"] > df["precision_group"])
                    & (df["precision_stepwise"] > df["precision_causal"])
                ]

                # Add a 'source' column to each dataframe to label the data
                x1["Method"] = "Lasso Screening"
                x2["Method"] = "Group Screening"
                x3["Method"] = "Causal Screening"
                x4["Method"] = "Stepwise Screening"

                # Combine the dataframes
                combined_df = pd.concat([x1, x2, x3], ignore_index=True)
                # Setup the figure and axes for subplots
                fig, ax = plt.subplots(
                    1, len(label_options), figsize=(10 + len(label_options), 3)
                )  # Adjust width to accommodate legend

                # Iterate through each label and plot using the new style
                for i, column in enumerate(label_options):
                    sns.histplot(
                        data=combined_df,
                        x=column,
                        hue="Method",
                        multiple="stack",
                        legend=True,
                        shrink=0.8,
                        element="step",
                        stat="count",
                        common_norm=False,
                        ax=ax[i],
                        palette="pastel",
                    )
                st.pyplot(fig)

            with st.expander("Recall"):
                x1 = df[
                    (df["recall_lasso"] > df["recall_group"])
                    & (df["recall_lasso"] > df["recall_causal"])
                    & (df["recall_lasso"] > df["recall_stepwise"])
                ]
                x2 = df[
                    (df["recall_group"] > df["recall_lasso"])
                    & (df["recall_group"] > df["recall_causal"])
                    & (df["recall_group"] > df["recall_stepwise"])
                ]
                x3 = df[
                    (df["recall_causal"] > df["recall_lasso"])
                    & (df["recall_causal"] > df["recall_group"])
                    & (df["recall_causal"] > df["recall_stepwise"])
                ]
                x4 = df[
                    (df["recall_stepwise"] > df["recall_lasso"])
                    & (df["recall_causal"] > df["recall_group"])
                    & (df["recall_stepwise"] > df["recall_causal"])
                ]

                # Add a 'source' column to each dataframe to label the data
                x1["Method"] = "Lasso Screening"
                x2["Method"] = "Group Screening"
                x3["Method"] = "Causal Screening"
                x4["Method"] = "Stepwise Screening"

                # Combine the dataframes
                combined_df = pd.concat([x1, x2, x3], ignore_index=True)
                # Setup the figure and axes for subplots
                fig, ax = plt.subplots(
                    1, len(label_options), figsize=(10 + len(label_options), 3)
                )  # Adjust width to accommodate legend

                # Iterate through each label and plot using the new style
                for i, column in enumerate(label_options):
                    sns.histplot(
                        data=combined_df,
                        x=column,
                        hue="Method",
                        multiple="stack",
                        legend=True,
                        shrink=0.8,
                        element="step",
                        stat="count",
                        common_norm=False,
                        ax=ax[i],
                        palette="pastel",
                    )
                st.pyplot(fig)

            with st.expander("F1 Score"):
                x1 = df[
                    (df["f1_lasso"] > df["f1_group"])
                    & (df["f1_lasso"] > df["f1_causal"])
                    & (df["f1_lasso"] > df["f1_stepwise"])
                ]
                x2 = df[
                    (df["f1_group"] > df["f1_lasso"])
                    & (df["f1_group"] > df["f1_causal"])
                    & (df["f1_group"] > df["f1_stepwise"])
                ]
                x3 = df[
                    (df["f1_causal"] > df["f1_lasso"])
                    & (df["f1_causal"] > df["f1_group"])
                    & (df["f1_causal"] > df["f1_stepwise"])
                ]
                x4 = df[
                    (df["f1_stepwise"] > df["f1_lasso"])
                    & (df["f1_causal"] > df["f1_group"])
                    & (df["f1_stepwise"] > df["f1_causal"])
                ]

                # Add a 'source' column to each dataframe to label the data
                x1["Method"] = "Lasso Screening"
                x2["Method"] = "Group Screening"
                x3["Method"] = "Causal Screening"
                x4["Method"] = "Stepwise Screening"

                # Combine the dataframes
                combined_df = pd.concat([x1, x2, x3], ignore_index=True)
                # Setup the figure and axes for subplots
                fig, ax = plt.subplots(
                    1, len(label_options), figsize=(10 + len(label_options), 3)
                )  # Adjust width to accommodate legend

                # Iterate through each label and plot using the new style
                for i, column in enumerate(label_options):
                    sns.histplot(
                        data=combined_df,
                        x=column,
                        hue="Method",
                        multiple="stack",
                        legend=True,
                        shrink=0.8,
                        element="step",
                        stat="count",
                        common_norm=False,
                        ax=ax[i],
                        palette="pastel",
                    )
                st.pyplot(fig)


if __name__ == "__main__":
    main()
