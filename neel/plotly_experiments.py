# %%

import neel
%load_ext autoreload
%autoreload 2
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go
# %%
import plotly.graph_objects as go
import numpy as np


# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    for c in range(10):
        fig.add_trace(
            go.Scattergl(
                visible=False,
                line=dict(width=6),
                name=f"𝜈 = {step:.2f}, const = {c/10:.2f}",
                x=np.arange(0, 10, 0.01),
                y=np.sin(step * np.arange(0, 10, 0.01)) + c/10))

# Make 10th trace visible
for c in range(10):
    fig.data[10*10+c].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)//10):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    for c in range(10):
        step["args"][0]["visible"][i*10+c] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)
fig.show()
print(fig)
# string = fig.to_html(plotly)
# fig.show()
# %%

tensor = torch.randn(50, 5, 100)

tensor = neel.to_numpy(tensor, flat=False)

# Create figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1000), y=np.arange(1000)))
# Add traces, one for each slider step
# for frame in 
# max_c = 10
# for step in range(11):
#     for c in range(max_c):
#         fig.add_trace(
#             go.Scattergl(
#                 visible=False,
#                 line=dict(width=6),
#                 name=f"𝜈 = {step:.2f}, const = {c/10:.2f}",
#                 x=np.arange(0, 10, 0.01),
#                 y=np.sin(step * np.arange(0, 10, 0.01)) + c/10))

# # Make 10th trace visible
# for c in range(max_c):
#     fig.data[10*max_c+c].visible = True

steps = []
for i in range(100):
    step = dict(
        method="update",
        args=[{"y": np.arange(1000)+i},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    # for c in range(max_c):
    #     step["args"][0]["visible"][i*10+c] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)
st = fig.to_html(include_plotlyjs='cdn')
print(len(st)/max_c)

# Create and add slider
steps = []
for i in range(len(fig.data)//max_c):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    for c in range(max_c):
        step["args"][0]["visible"][i*10+c] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)
st = fig.to_html(include_plotlyjs='cdn')
print(len(st)/max_c)
# fig.show()
# %%
import plotly.express as px

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", title="A Plotly Express Figure")

# If you print the figure, you'll see that it's just a regular figure with data and layout
# print(fig)

fig.show()
# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], marker_color='red'))

steps = []
for i in range(100):
    step = dict(
        # args=[{'data': [1, 2, 3], 'layout': {'title':'HELP ME!'}, 'traces': []}],
        method='restyle',
        args=['y', np.arange(1000)+i],
    )
    # for c in range(max_c):
    #     step["args"][0]["visible"][i*10+c] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)
st = fig.to_html(include_plotlyjs='cdn')
print(len(st))
fig.show()
# %%
import plotly.express as px

df = px.data.gapminder()

fig = px.bar(df, x="continent", y="pop", color="continent",
  animation_frame="year", animation_group="country", range_y=[0,4000000000])
fig.show()
print(fig)
# %%
# %%

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], marker_color='red'))
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.2,
            buttons=list([
                dict(label="None",
                     method="update",
                     args=[{"y": [4, 5, 7]}])]),
                
            ),]
        )
fig.show()
# %%
