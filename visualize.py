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

methods = [
    "baseline_normal",
    "baseline_group",
    "sizefit_normal",
    "sizefit_group",
    "causal_group",
    "stepsize-0.05_normal",
    "stepsize-0.05_group",
    "stepsize-0.025_normal",
    "stepsize-0.025_group",
    "stepsize-0.01_normal",
    "stepsize-0.01_group",
    "stepsize-0.005_normal",
    "stepsize-0.005_group",
]


# Main app function
def main():
    df = pd.read_csv("results.csv")
    with st.sidebar:
        with st.expander("General", expanded=True):
            baseline = st.selectbox("Baseline", options=methods)
            comparison = st.selectbox(
                "methods", options=sorted(list(set(methods) - set([baseline])))
            )

        st.divider()
        with st.expander("Fine-grained"):
            x_axis = st.selectbox("X axis", options=label_options)
            y_axis = st.selectbox(
                "Y axis", options=sorted(list(set(label_options) - set([x_axis])))
            )

    tab_general, tab_finegrained = st.tabs(
        ["Overview", "Fine-grained"]
    )
    with tab_general:
        with st.expander("Precision, Recall, and F1 score"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision (sorted)")
                for k in methods:
                    plt.plot(sorted(df[f"precision_{k}"]), label=k)
                plt.legend()
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall (sorted)")
                for k in methods:
                    plt.plot(sorted(df[f"recall_{k}"]), label=k)
                plt.legend()
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score (sorted)")
                for k in methods:
                    plt.plot(sorted(df[f"f1_{k}"]), label=k)
                plt.legend()
                st.pyplot(fig)

        
        with st.expander("Precision, Recall, and F1 score DIFFERENCES"):
            cols = st.columns(3)

            with cols[0]:
                fig = plt.figure()
                plt.suptitle(
                    f"Precision Difference ({baseline} vs {comparison})"
                )
                plt.title(f"({comparison} better if < 0)")
                plt.plot(
                    sorted(
                        df[f"precision_{baseline}"]
                        - df[f"precision_{comparison}"]
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
                    f"Recall Difference ({baseline} vs {comparison})"
                )
                plt.title(f"({comparison} better if < 0)")
                plt.plot(
                    sorted(
                        df[f"recall_{baseline}"]
                        - df[f"recall_{comparison}"]
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
                    f"F1 Score Difference ({baseline} vs {comparison})"
                )
                plt.title(f"({comparison} better if < 0)")
                plt.plot(
                    sorted(
                        df[f"f1_{baseline}"] - df[f"f1_{comparison}"]
                    ),
                    label=comparison,
                    color="indigo",
                )
                plt.axhline(0, linewidth=2, linestyle=":", color="black", alpha=0.5)
                plt.legend()
                st.pyplot(fig)
        
    with tab_finegrained:

        for mett in methods:
            with st.expander(mett):
                cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis,
                    columns=y_axis,
                    values=f"precision_{mett}",
                    aggfunc="mean",
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values=f"recall_{mett}", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)

            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values=f"f1_{mett}", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=gradual_cmap, vmin=0, vmax=1, annot=True)
                st.pyplot(fig)


        with st.expander("Comparison"):
            df["_precision"] = (
                df[f"precision_{baseline}"]
                - df[f"precision_{comparison}"]
            )
            df["_recall"] = (
                df[f"recall_{baseline}"] - df[f"recall_{comparison}"]
            )
            df["_f1"] = df[f"f1_{baseline}"] - df[f"f1_{comparison}"]

            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle(
                    f"Precision Difference ({baseline} vs {comparison})"
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
                    f"Recall Difference ({baseline} vs {comparison})"
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
                    f"F1  Difference ({baseline} vs {comparison})"
                )
                plt.title("(lower is better)")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values="_f1", aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)



if __name__ == "__main__":
    main()

