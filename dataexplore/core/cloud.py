import pylab as plt
from math import sin, cos, pi
from random import random
import seaborn as sns
import numpy as np
from dataexplore.core.animation import LinearInterpolater, ThreeDKeyFrame
from matplotlib.animation import FuncAnimation
from pydantic import BaseModel, ConfigDict
from typing import List, Any

from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.axes3d import Axes3D


fig, ax = plt.subplots()
ax: Axes3D = plt.axes(projection="3d")


def two_d_set():
    return np.array([[random() for i in range(50)] for j in range(50)])


def one_d_set():
    return [random() for i in range(50)]


class Point(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pos: np.ndarray
    keyframe: ThreeDKeyFrame


cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)


kdeplot_kwargs = {"cmap": cmap, "fill": True, "multiple": "layer", "ax": ax}


class PointSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    points: List[Point]
    interp: LinearInterpolater = LinearInterpolater()

    def update_points(self, control_param: float) -> None:
        for point in self.points:
            point.pos = self.interp.calculate_value(point.keyframe, control_param)

    @property
    def x_data(self) -> List[float]:
        return [x.pos[0] for x in self.points]

    @property
    def y_data(self) -> List[float]:
        return [x.pos[1] for x in self.points]

    @property
    def z_data(self) -> List[float]:
        return [x.pos[2] for x in self.points]

    def plot_point_set(self):
        artist_out = []
        kdeplot_1 = sns.kdeplot(
            x=self.y_data,
            y=self.z_data,
            z=[1 for i in range(len(self.z_data))],
            zdir="x",
            offset=-0.2,
            **kdeplot_kwargs,
        ).collections
        artist_out.append(kdeplot_1)

        kdeplot_2 = sns.kdeplot(
            x=self.x_data,
            y=self.z_data,
            z=[1 for i in range(len(self.z_data))],
            zdir="y",
            offset=1.2,
            **kdeplot_kwargs,
        ).collections
        artist_out.append(kdeplot_2)

        kdeplot_3 = sns.kdeplot(
            x=self.x_data,
            y=self.y_data,
            z=[1 for i in range(len(self.x_data))],
            zdir="z",
            offset=-0.2,
            **kdeplot_kwargs,
        ).collections
        artist_out.append(kdeplot_3)
        out_ax = ax.scatter(
            self.x_data,
            self.y_data,
            self.z_data,
            color=(34.0 / 255, 139.0 / 255, 34.0 / 255, 0.6),
        )
        return [kdeplot_1, kdeplot_2, kdeplot_3], out_ax


data_set = [one_d_set() for i in range(3)]


n_points = 100
point_list = []
for i in range(n_points):
    start_pos = np.array([random(), random(), random()])
    end_pos = start_pos + np.array([random(), random(), random()]) * 0.5
    new_point = Point(
        pos=start_pos, keyframe=ThreeDKeyFrame(target_range=(start_pos, end_pos))
    )
    point_list.append(new_point)

point_set = PointSet(points=point_list)


""" ani = FuncAnimation(
    fig,
    update,
    frames=np.linspace(0.0, 1.0, 1),
    init_func=init,
    blit=False,
    interval=1000,
) """

media_folder = "/Users/johnwhitman/Projects/DataExplore/dataexplore/media/cloud_gif"

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
""" plt.axis("off")
plt.axis("image") """

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_zlim(-0.5, 1.5)


def format_frame(in_frame: int) -> str:
    frame_string = str(in_frame)
    while len(frame_string) < 4:
        frame_string = "0" + frame_string
    return frame_string


for frame_number, frame_value in enumerate(np.linspace(0, 1, 100)):
    if frame_number != 0:
        for collection in kdeplots:
            for artist in collection:
                artist.remove()
    point_set.update_points(frame_value)
    kdeplots, scatterplot = point_set.plot_point_set()

    string_frame_number = format_frame(frame_number)

    plt.savefig(f"{media_folder}/frame_{string_frame_number}.png", dpi=300)
