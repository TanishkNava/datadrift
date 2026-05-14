import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def plot_drift_3d(original_emb, drift_emb, names=None):
    """
    Compare original vs drift embeddings in 3D
    """

    combined = np.vstack([original_emb, drift_emb])

    labels = (["Original"] * len(original_emb)) + (["Drift"] * len(drift_emb))

    # PCA to 3D
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(combined)

    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "z": reduced[:, 2],
        "type": labels
    })

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="type",
        title="Data Drift 3D Visualization (Original vs Drift)"
    )

    return fig