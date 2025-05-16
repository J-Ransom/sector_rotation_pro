import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

def run_pca_analysis(returns, n_components=2):
    """
    Run PCA analysis on returns.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
    n_components : int
        Number of principal components to extract
        
    Returns:
    -------
    tuple
        (transformed_data, explained_variance_ratio)
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns.T)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(scaled_data)
    
    return transformed_data, pca.explained_variance_ratio_

def run_spectral_clustering(returns, n_clusters=3):
    """
    Run spectral clustering on returns.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
    n_clusters : int
        Number of clusters to form
        
    Returns:
    -------
    numpy.ndarray
        Cluster labels for each ticker
    """
    # Create affinity matrix from correlation matrix
    corr_matrix = returns.corr().values
    affinity_matrix = (corr_matrix + 1) / 2  # Scale from [-1, 1] to [0, 1]
    
    # Run spectral clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=42
    )
    
    labels = clustering.fit_predict(affinity_matrix)
    
    return labels

def create_pca_figure(returns, title="PCA Analysis of Sectors"):
    """
    Create a PCA visualization.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        PCA visualization
    """
    # Run PCA
    transformed_data, explained_variance = run_pca_analysis(returns)
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(
        transformed_data,
        columns=['PC1', 'PC2'],
        index=returns.columns
    )
    
    # Add ticker as a column for easier plotting
    pca_df['Ticker'] = pca_df.index
    
    # Create the scatter plot
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        text='Ticker',
        title=f"{title}<br>Explained Variance: PC1 {explained_variance[0]:.2%}, PC2 {explained_variance[1]:.2%}"
    )
    
    # Update trace properties
    fig.update_traces(
        marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')),
        textposition='top center',
        textfont=dict(size=12)
    )
    
    # Run spectral clustering
    cluster_labels = run_spectral_clustering(returns)
    
    # Add cluster information
    pca_df['Cluster'] = cluster_labels
    
    # Create a new figure with clusters
    fig_clustered = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        text='Ticker',
        color='Cluster',
        title=f"{title} with Clustering<br>Explained Variance: PC1 {explained_variance[0]:.2%}, PC2 {explained_variance[1]:.2%}"
    )
    
    # Update trace properties
    fig_clustered.update_traces(
        marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')),
        textposition='top center',
        textfont=dict(size=12)
    )
    
    # Update layout for both figures
    for f in [fig, fig_clustered]:
        f.update_layout(
            template="plotly_white",
            xaxis_title=f"Principal Component 1 ({explained_variance[0]:.2%})",
            yaxis_title=f"Principal Component 2 ({explained_variance[1]:.2%})",
            height=700,
            width=800,
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        
        # Add a vertical and horizontal line at the origin
        f.add_shape(
            type="line", x0=0, y0=min(pca_df['PC2']), x1=0, y1=max(pca_df['PC2']),
            line=dict(color="black", width=1, dash="dash")
        )
        f.add_shape(
            type="line", x0=min(pca_df['PC1']), y0=0, x1=max(pca_df['PC1']), y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
    
    return fig, fig_clustered

def pca_cluster_fig(returns):
    """
    Create PCA and clustering visualizations.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
        
    Returns:
    -------
    tuple of plotly.graph_objects.Figure
        (pca_figure, pca_clustered_figure)
    """
    # Create the figures
    pca_fig, pca_clustered_fig = create_pca_figure(returns, title="PCA Analysis of Sectors")
    
    return pca_fig, pca_clustered_fig
