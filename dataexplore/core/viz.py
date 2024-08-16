import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class FigureProps:
    fig_size = (8, 8)
    dpi = 200
    title: str = "Default Title"


@dataclass
class AxisProps:
    label: str
    font_size: float = 14
    lim: List[float] = field(default_factory=lambda: [None, None])


@dataclass
class ScatterProps:
    x_axis: AxisProps
    y_axis: AxisProps
    point_size: int = None
    point_color: List[float] = field(default_factory=lambda: [0, 0, 0, 1])


@dataclass
class LineProps:
    x_axis: AxisProps
    y_axis: AxisProps
    line_width: float


class Visualizer:
    props: FigureProps = FigureProps()

    def __init__(self) -> None:
        fig, axs = plt.subplots()
        self.fig = fig
        fig.suptitle(self.props.title, fontweight="bold")
        self.axs: plt.Axes = axs

    def set_gen_props(self, scatter_props: ScatterProps) -> None:
        self.axs.set_xlabel(scatter_props.x_axis.label, fontweight="bold")
        self.axs.set_ylabel(scatter_props.y_axis.label, fontweight="bold")
        self.axs.set_xlim(*scatter_props.x_axis.lim)
        self.axs.set_ylim(*scatter_props.y_axis.lim)

    def create_scatter_plot(
        self,
        x_data: Iterable[float],
        y_data: Iterable[float],
        scatter_props: ScatterProps,
    ) -> plt.Axes:
        self.axs.scatter(
            x_data, y_data, s=scatter_props.point_size, color=scatter_props.point_color
        )
        self.set_gen_props(scatter_props=scatter_props)

        return self.axs


if __name__ == "__main__":
    import random

    base_viz = Visualizer()
    x_data = [random.random() for i in range(50)]
    y_data = [random.random() for i in range(50)]

    x_props = AxisProps(label="Fake X")
    y_props = AxisProps(label="Fake Y")
    scatter_props = ScatterProps(
        x_axis=x_props, y_axis=y_props, point_size=10, point_color=[0.75, 0.75, 0.75, 1]
    )

    base_viz.create_scatter_plot(
        x_data=x_data, y_data=y_data, scatter_props=scatter_props
    )
    plt.show()
