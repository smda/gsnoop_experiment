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
plt.style.use('seaborn-v0_8-colorblind')

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

red_green = create_diverging_colormap('#2f821e', '#ab2e0c')

def merge_data():
    files = ['results/' + d for d in os.listdir('results/')]

    df = []
    for file in files:
        with open(file, 'r') as f:
            row = json.load(f)
            
            # aggregate 
            metrics = {
                'precision_lasso': [],    
                'precision_group': [],  
                'precision_causal': [],  
                
                'recall_lasso': [],  
                'recall_group': [],  
                'recall_causal': [],  
                
                'f1_lasso': [],  
                'f1_group': [],  
                'f1_causal': [],  
            }
            for record in row['metrics']:
                metrics['precision_lasso'].append( record['precision']['lasso_screen'] )
                metrics['precision_group'].append( record['precision']['group_screen'] )
                metrics['precision_causal'].append( record['precision']['causal_screen'] )
                
                metrics['recall_lasso'].append( record['recall']['lasso_screen'] )
                metrics['recall_group'].append( record['recall']['group_screen'] )
                metrics['recall_causal'].append( record['recall']['causal_screen'] )
                
                metrics['f1_lasso'].append( record['f1_score']['lasso_screen'] )
                metrics['f1_group'].append( record['f1_score']['group_screen'] )
                metrics['f1_causal'].append( record['f1_score']['causal_screen'] )
            
            metrics = {k: np.mean(v) for k, v in metrics.items()}
            del row['metrics']
            del row['relevant_options']
            metrics.update(row)
            
            df.append(metrics)
        
    return pd.DataFrame(df)
    
label_options = ['features', 'rel_sample_size', 'relevant_terms_level', 'interaction_p']

# Main app function
def main():
    df = merge_data()
    with st.sidebar:
        st.header('Specific')
        x_axis = st.selectbox('X axis', options=label_options)
        y_axis = st.selectbox('Y axis', options=sorted(list(set(label_options) - set([x_axis]))))
        
        st.divider()

        
    tabs = st.tabs(['General', 'Fine-grained', 'Top Analysis'])
    with tabs[0]:
        with st.expander("Precision, Recall, and F1 score"):
            cols = st.columns(3)
            with cols[0]:
                
                
                fig = plt.figure()
                plt.suptitle('Precision (sorted)')
                plt.plot(sorted(df['precision_lasso']), label='Lasso Screening', color='black')
                plt.plot(sorted(df['precision_group']), label='Group Screening')
                plt.plot(sorted(df['precision_causal']), label='MHS Screening')
                plt.legend()
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle('Recall (sorted)')
                plt.plot(sorted(df['recall_lasso']), label='Lasso Screening', color='black')
                plt.plot(sorted(df['recall_group']), label='Group Screening')
                plt.plot(sorted(df['recall_causal']), label='MHS Screening')
                plt.legend()
                st.pyplot(fig)
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle('F1 Score (sorted)')
                plt.plot(sorted(df['f1_lasso']), label='Lasso Screening', color='black')
                plt.plot(sorted(df['f1_group']), label='Group Screening')
                plt.plot(sorted(df['f1_causal']), label='MHS Screening')
                plt.legend()
                st.pyplot(fig)
            
        with st.expander("Precision, Recall, and F1 score DIFFERENCES"):
            cols = st.columns(3)
            
            with cols[0]:
                fig = plt.figure()
                plt.suptitle('Precision Difference to Lasso (sorted)')
                plt.title('(lower is better)')
                plt.plot(sorted(df['precision_lasso'] - df['precision_group']), label='Group Screening')
                plt.plot(sorted(df['precision_lasso'] - df['precision_causal']), label='MHS Screening')
                plt.axhline(0, linewidth=1, linestyle=':', color='black', alpha=0.5)
                plt.legend()
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle('Recall Difference to Lasso (sorted)')
                plt.title('(lower is better)')
                plt.plot(sorted(df['recall_lasso'] - df['recall_group']), label='Group Screening')
                plt.plot(sorted(df['recall_lasso'] - df['recall_causal']), label='MHS Screening')
                plt.axhline(0, linewidth=1, linestyle=':', color='black', alpha=0.5)
                plt.legend()
                st.pyplot(fig)
            
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle('F1 Score Difference to Lasso (sorted)')
                plt.title('(lower is better)')
                plt.plot(sorted(df['f1_lasso'] - df['f1_group']), label='Group Screening')
                plt.plot(sorted(df['f1_lasso'] - df['f1_causal']), label='MHS Screening')
                plt.axhline(0, linewidth=1, linestyle=':', color='black', alpha=0.5)
                plt.legend()
                st.pyplot(fig)
                
    with tabs[1]:
        
        with st.expander("Lasso Screening (Baseline)"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='precision_lasso', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='recall_lasso', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='f1_lasso', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
                
        with st.expander("Group Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='precision_group', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='recall_group', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='f1_group', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
                
        with st.expander("Causal Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='precision_causal', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='recall_causal', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score")
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='f1_causal', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap="bone_r", vmin=0, vmax=1, annot=True)
                st.pyplot(fig)
                
                
        df['lasso_group_precision'] = df['precision_lasso'] - df['precision_group']
        df['lasso_group_recall'] = df['recall_lasso'] - df['recall_group']
        df['lasso_group_f1'] = df['f1_lasso'] - df['f1_group']
        
        df['lasso_causal_precision'] = df['precision_lasso'] - df['precision_causal']
        df['lasso_causal_recall'] = df['recall_lasso'] - df['recall_causal']
        df['lasso_causal_f1'] = df['f1_lasso'] - df['f1_causal']
        
        with st.expander("Lasso vs Group Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision Difference to Lasso")
                plt.title('(lower is better)')
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='lasso_group_precision', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall Difference to Lasso")
                plt.title('(lower is better)')
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='lasso_group_recall', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score Difference to Lasso")
                plt.title('(lower is better)')
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='lasso_group_f1', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
                
        with st.expander("Lasso vs Causal Screening"):
            cols = st.columns(3)
            with cols[0]:
                fig = plt.figure()
                plt.suptitle("Precision Difference to Lasso")
                plt.title('(lower is better)')
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='lasso_causal_precision', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
            
            with cols[1]:
                fig = plt.figure()
                plt.suptitle("Recall Difference to Lasso")
                plt.title('(lower is better)')
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='lasso_causal_recall', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
                
            with cols[2]:
                fig = plt.figure()
                plt.suptitle("F1 Score Difference to Lasso")
                plt.title('(lower is better)')
                pivo = df.pivot_table(
                    index=x_axis, columns=y_axis, values='lasso_causal_f1', aggfunc="mean"
                )
                sns.heatmap(pivo, cmap=red_green, vmin=-1, vmax=1, annot=True)
                st.pyplot(fig)
    
    with tabs[2]:
        with st.expander("Precision"):
            x1 = df[(df['precision_lasso'] > df['precision_group']) & (df['precision_lasso'] > df['precision_causal'])]
            x2 = df[(df['precision_group'] > df['precision_lasso']) & (df['precision_group'] > df['precision_causal'])]
            x3 = df[(df['precision_causal'] > df['precision_lasso']) & (df['precision_causal'] > df['precision_group'])]
            
            # Add a 'source' column to each dataframe to label the data
            x1['Method'] = 'Lasso Screening'
            x2['Method'] = 'Group Screening'
            x3['Method'] = 'Causal Screening'
            
            # Combine the dataframes
            combined_df = pd.concat([x1, x2, x3], ignore_index=True)
            # Setup the figure and axes for subplots
            fig, ax = plt.subplots(1, len(label_options), figsize=(10 + len(label_options), 3))  # Adjust width to accommodate legend
            
            # Iterate through each label and plot using the new style
            for i, column in enumerate(label_options):
                sns.histplot(data=combined_df, x=column, hue='Method', multiple="stack", legend=True, shrink=0.8, element='step', stat='count', common_norm=False, ax=ax[i], palette='pastel')
            
            st.pyplot(fig)
            
        with st.expander("Recall"):
            x1 = df[(df['recall_lasso'] > df['recall_group']) & (df['recall_lasso'] > df['recall_causal'])]
            x2 = df[(df['recall_group'] > df['recall_lasso']) & (df['recall_group'] > df['recall_causal'])]
            x3 = df[(df['recall_causal'] > df['recall_lasso']) & (df['recall_causal'] > df['recall_group'])]

            # Add a 'source' column to each dataframe to label the data
            x1['Method'] = 'Lasso Screening'
            x2['Method'] = 'Group Screening'
            x3['Method'] = 'Causal Screening'

            # Combine the dataframes
            combined_df = pd.concat([x1, x2, x3], ignore_index=True)
            fig, ax = plt.subplots(1, 4, figsize=(10,3))#plt.figure(figsize=(18, 6))

            # Create a list of the columns to plot
            columns_to_plot = label_options

            # Setup the figure and axes for subplots
            fig, ax = plt.subplots(1, len(label_options), figsize=(10 + len(label_options), 3))  # Adjust width to accommodate legend
            
            # Iterate through each label and plot using the new style
            for i, column in enumerate(label_options):
                sns.histplot(data=combined_df, x=column, hue='Method', multiple="stack", legend=True, shrink=0.8, element='step', stat='count', common_norm=False, ax=ax[i], palette='pastel')
            
            st.pyplot(fig)
        
        with st.expander("F1 Score"):
            
            x1 = df[(df['f1_lasso'] > df['f1_group']) & (df['f1_lasso'] > df['f1_causal'])]
            x2 = df[(df['f1_group'] > df['f1_lasso']) & (df['f1_group'] > df['f1_causal'])]
            x3 = df[(df['f1_causal'] > df['f1_lasso']) & (df['f1_causal'] > df['f1_group'])]
            
            # Add a 'Method' column to each dataframe to label the data
            x1['Method'] = 'Lasso Screening'
            x2['Method'] = 'Group Screening'
            x3['Method'] = 'Causal Screening'
            
            # Combine the dataframes
            combined_df = pd.concat([x1, x2, x3], ignore_index=True)
            
            # Setup the figure and axes for subplots
            fig, ax = plt.subplots(1, len(label_options), figsize=(10 + len(label_options), 3))  # Adjust width to accommodate legend
            
            # Iterate through each label and plot using the new style
            for i, column in enumerate(label_options):
                sns.histplot(data=combined_df, x=column, hue='Method', multiple="stack", legend=True, shrink=0.8, element='step', stat='count', common_norm=False, ax=ax[i], palette='pastel')
            
            st.pyplot(fig)
                    
            


if __name__ == "__main__":
    main()
