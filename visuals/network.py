import plotly.graph_objects as go
import networkx as nx
import sys
import os

# Add the parent directory to sys.path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_network_graph(corr_matrix, threshold=0.6, title="Correlation Network"):
    """
    Create a network graph from a correlation matrix.
    
    Parameters:
    ----------
    corr_matrix : pandas.DataFrame
        Correlation matrix
    threshold : float
        Correlation threshold for creating edges
    title : str
        Title for the figure
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Network graph
    """
    # Import directly from the analytics module
    from analytics.stats_tests import generate_network_data
    
    # Generate network data
    nodes, edges = generate_network_data(corr_matrix, threshold)
    
    if not nodes or not edges:
        # Return empty figure with message if no correlations above threshold
        fig = go.Figure()
        fig.add_annotation(
            text=f"No correlations above threshold ({threshold})",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'], color=edge['color'])
    
    # Use a layout algorithm to position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge traces
    edge_traces = []
    
    for edge in edges:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        weight = edge['weight']
        color = edge['color']
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=weight * 3, color=color),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in nodes:
        x, y = pos[node['id']]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node['name'])
        
        # Node size based on degree in graph
        neighbors = list(G.neighbors(node['id']))
        degree = len(neighbors)
        node_size.append(15 + degree * 5)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            size=node_size,
            color='blue',
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=node_text
    )
    
    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Add a legend for edge colors
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='green', width=4),
        name='Positive Correlation'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='red', width=4),
        name='Negative Correlation'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{title} (Threshold: {threshold})",
        template="plotly_white",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=700,
        width=800,
        margin=dict(b=20, l=5, r=5, t=40)
    )
    
    return fig

def network_fig(returns, threshold=0.6):
    """
    Create network graph from returns data.
    
    Parameters:
    ----------
    returns : pandas.DataFrame
        DataFrame with return data
    threshold : float
        Correlation threshold for creating edges
        
    Returns:
    -------
    plotly.graph_objects.Figure
        Network graph
    """
    # Import here to avoid circular imports
    from ..analytics.stats_tests import calculate_correlation_matrix
    
    # Calculate correlation matrix
    corr_matrix = calculate_correlation_matrix(returns)
    
    # Create the network graph
    fig = create_network_graph(corr_matrix, threshold, title="Sector Correlation Network")
    
    return fig
