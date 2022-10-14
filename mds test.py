# Data manipulation
import pandas as pd # for data manipulation

# Visualization
import plotly.express as px # for data visualization

# Skleran
from sklearn.datasets import make_swiss_roll # for creating a swiss roll
from sklearn.manifold import MDS # for MDS dimensionality reduction

# Make a swiss roll
X, y = make_swiss_roll(n_samples=2000, noise=0.05)
# Make it thinner
X[:, 1] *= .5


# Create a 3D scatter plot
fig = px.scatter_3d(None, x=X[:,0], y=X[:,1], z=X[:,2], color=y,)

# Update chart looks
fig.update_layout(#title_text="Swiss Roll",
                  showlegend=False,
                  scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.25, y=1.5, z=1)),
                                        margin=dict(l=0, r=0, b=0, t=0),
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                         ),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                          ),
                               zaxis=dict(backgroundcolor='lightgrey',
                                          color='black', 
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                         )))

# Update marker size
fig.update_traces(marker=dict(size=3, 
                              line=dict(color='black', width=0.1)))

fig.update(layout_coloraxis_showscale=False)
fig.show()

### Step 1 - Configure MDS function, note we use default hyperparameter values for this example
model2d=MDS(n_components=2, 
          metric=True, 
          n_init=4, 
          max_iter=300, 
          verbose=0, 
          eps=0.001, 
          n_jobs=None, 
          random_state=42, 
          dissimilarity='euclidean')

### Step 2 - Fit the data and transform it, so we have 2 dimensions instead of 3
X_trans = model2d.fit_transform(X)
    
### Step 3 - Print a few stats
print('The new shape of X: ',X_trans.shape)
print('No. of Iterations: ', model2d.n_iter_)
print('Stress: ', model2d.stress_)

# Dissimilarity matrix contains distances between data points in the original high-dimensional space
#print('Dissimilarity Matrix: ', model2d.dissimilarity_matrix_)
# Embedding contains coordinates for data points in the new lower-dimensional space
#print('Embedding: ', model2d.embedding_)

# Create a scatter plot
fig = px.scatter(None, x=X_trans[:,0], y=X_trans[:,1], opacity=1, color=y)

# Change chart background color
fig.update_layout(dict(plot_bgcolor = 'white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', 
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey', 
                 showline=True, linewidth=1, linecolor='black')

# Set figure title
fig.update_layout(title_text="MDS Transformation")

# Update marker size
fig.update_traces(marker=dict(size=5,
                             line=dict(color='black', width=0.2)))

fig.show()