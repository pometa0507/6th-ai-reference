import numpy as np
import plotly.graph_objects as go

def get_figure_data(pointcloud, color_idx_arr):
    
    data = [go.Scatter3d(
            x=pointcloud[:, 0],
            y=pointcloud[:, 1],
            z=pointcloud[:, 2],
            mode='markers',
            marker=dict(size=1, color=color_idx_arr),
            hoverinfo="skip")]
    return data

          
def view_pointcloud(pointcloud, color_idx_arr, mode="front"):

    assert mode in ["front", "top"]

    fig = go.Figure(data=get_figure_data(pointcloud, color_idx_arr))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-90, 90], visible=False),
            yaxis=dict(nticks=4, range=[-50, 50], visible=False),
            zaxis=dict(nticks=4, range=[-50, 50], visible=False),
        ),
        width=800,
        height=400,
        margin=dict(r=0, l=0, b=0, t=0),
        paper_bgcolor="rgb(50, 50, 50)"
    )

    if mode == "front":
        fig.update_layout(
            scene=dict(
                camera_center=dict(x=1),
                camera_eye=dict(x=-0.1, y=0, z=0)
            ))

    elif mode == "top":
        fig.update_layout(
            scene=dict(
                camera_center=dict(x=0, y=0, z=-1),
                camera_eye=dict(x=0.3, y=0, z=0.5),
                camera_up=dict(x=1, y=0, z=0),
            ))

    return fig


def view_pointcloud_3dbbox(pointcloud, color_idx_arr, corners, labels, mode="front"):

    assert mode in ["front", "top"]

    color_dict_label = {0: "orange", 1: "magenta", 2: "blue", 3: "yellow"}

    plot_data = []

    plot_data.append(
        go.Scatter3d(
            x=pointcloud[:, 0],
            y=pointcloud[:, 1],
            z=pointcloud[:, 2],
            mode='markers',
            marker=dict(size=1, color=color_idx_arr),
            hoverinfo="skip"))

    for bbox_corners, label in zip(corners, labels):
        frame = _bbox3d_frame(bbox_corners)
        plot_data.append(go.Scatter3d(x=frame[:, 0],
                                      y=frame[:, 1],
                                      z=frame[:, 2],
                                      mode='lines',
                                      line_color=color_dict_label[label],
                                      line_width=2,
                                      hoverinfo="skip"))
    fig = go.Figure(data=plot_data)

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-90, 90], visible=False),
            yaxis=dict(nticks=4, range=[-50, 50], visible=False),
            zaxis=dict(nticks=4, range=[-50, 50], visible=False),
        ),
        width=800,
        height=400,
        margin=dict(r=0, l=0, b=0, t=0),
        paper_bgcolor="rgb(50, 50, 50)",
        showlegend=False,
    )

    if mode == "front":
        fig.update_layout(
            scene=dict(
                camera_center=dict(x=1),
                camera_eye=dict(x=-0.1, y=0, z=0)
            ))

    elif mode == "top":
        fig.update_layout(
            scene=dict(
                camera_center=dict(x=0, y=0, z=-1),
                camera_eye=dict(x=0.3, y=0, z=0.5),
                camera_up=dict(x=1, y=0, z=0),
            ))

    return fig

def _bbox3d_frame(bbox_corners):
    frame = np.concatenate([
        bbox_corners[0:4], bbox_corners[0:1],
        bbox_corners[[0, 1, 2, 3, 0, 4, 5, 6, 7, 4]],
        np.array([[np.nan, np.nan, np.nan]]),
        bbox_corners[[1, 5]], 
        np.array([[np.nan, np.nan, np.nan]]),
        bbox_corners[[2, 6]], 
        np.array([[np.nan, np.nan, np.nan]]),
        bbox_corners[[3, 7]]
    ])
    return frame