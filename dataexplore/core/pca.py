from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from sklearn.datasets import load_iris

import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.decomposition import PCA
from dataclasses import dataclass

import random


@dataclass
class PCADataPoint:
    row_id: int
    pca_indices: List[int]
    cluster_id: Optional[int] = None


class PCAExplorer:
    def __init__(self, full_data: np.ndarray, cluster_data: List[int] = list()) -> None:
        self.full_data = full_data
        self.full_pca = PCA(n_components=3).fit(full_data)
        self.full_pca_data = self.full_pca.fit_transform(self.full_data)
        self.sub_pcas: Dict[int, PCA] = {}
        self.data_points: Dict[int, PCADataPoint] = {
            i: PCADataPoint(
                row_id=i,
                pca_indices=[],
                cluster_id=cluster_data[i] if cluster_data is not None else None,
            )
            for i in range(len(self.full_data))
        }

    @property
    def main_basis(self) -> np.ndarray:
        return self.full_pca.components_

    def get_point_in_main_basis(self, index: int) -> np.ndarray:
        return self.full_data[index, :] @ self.main_basis.transpose()

    def generate_point_cloud(self, selected_point: int) -> List[np.ndarray]:
        basis_point = self.get_point_in_main_basis(selected_point)
        return [
            self.transform_to_pca_basis(basis_point, self.sub_pcas[pca_index])
            for pca_index in self.data_points[selected_point].pca_indices
        ]

    def generate_single_point_cross_validation(
        self, selected_point: int, folds: int = 5
    ) -> List[List[np.ndarray]]:
        random_partition = random.sample(
            [i for i in range(len(self.full_data))], k=len(self.full_data)
        )
        random_partition.pop(selected_point)

        partitions = [
            random_partition[i : i + int(len(random_partition) / folds)]
            for i in range(
                0, int(len(random_partition)), int(len(random_partition) / folds)
            )
        ]
        for partition in partitions:

            partition += [selected_point]

            indexer = np.array(
                [True if j in partition else False for j in range(len(self.full_data))],
                dtype=bool,
            )
            sub_sample = self.full_data[indexer, :]
            val_pca = PCA(n_components=3).fit(sub_sample)
            random_index = random.randint(10000, 20000)
            self.sub_pcas[random_index] = val_pca
            for value in partition:
                self.data_points[value].pca_indices.append(random_index)

    def transform_to_pca_basis(
        self, in_data: np.ndarray, relative_pca: PCA
    ) -> np.ndarray:
        return in_data @ (
            np.linalg.pinv(relative_pca.components_.transpose())
            @ self.main_basis.transpose()
        )

    def jackknife_single_pca(self, index: int) -> PCA:
        indexer = np.array(
            [True if j != index else False for j in range(len(self.full_data))],
            dtype=bool,
        )

        sub_sample = self.full_data[indexer, :]
        return PCA(n_components=3).fit(sub_sample)

    def jackknife_ensemble_pca(self) -> None:
        for i in range(len(self.full_data)):
            new_pca = self.jackknife_single_pca(i)
            self.sub_pcas[i] = new_pca
            for j in range(len(self.full_data)):
                if j != i:
                    self.data_points[j].pca_indices.append(i)


def viz_pca(in_data: np.ndarray):
    transform = PCA(n_components=3).fit(in_data)
    manual_mult = in_data @ np.array(transform.components_).transpose()

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    ax.scatter(
        manual_mult[:, 0],
        manual_mult[:, 1],
        manual_mult[:, 2],
        c=iris.target,
        s=40,
    )

    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])

    plt.show()


def get_target_index(
    xMin: float,
    xRange: float,
    xDiv: int,
    yMin: float,
    yRange: float,
    yDiv: int,
    pos: Tuple[float, float, float],
):
    xIndex = (pos[0] - xMin) // (xRange / xDiv)
    yIndex = (pos[1] - yMin) // (yRange / yDiv)
    return int(xIndex), int(yIndex)


if __name__ == "__main__":

    iris = load_iris()

    explorer = PCAExplorer(full_data=iris.data, cluster_data=iris.target)

    selected_point = 20

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    explorer.generate_single_point_cross_validation(selected_point, folds=5)
    points = explorer.generate_point_cloud(selected_point)
    basis_point = explorer.get_point_in_main_basis(selected_point)

    ax.scatter(
        [x[0] for x in points],
        [x[1] for x in points],
        [x[2] for x in points],
        c=(67 / 255.0, 75 / 255.0, 86 / 255.0, 1),
        s=150,
    )

    explorer.sub_pcas = {}
    explorer.data_points[selected_point].pca_indices = []

    explorer.generate_single_point_cross_validation(selected_point, folds=3)
    points = explorer.generate_point_cloud(selected_point)
    basis_point = explorer.get_point_in_main_basis(selected_point)

    ax.scatter(
        [x[0] for x in points],
        [x[1] for x in points],
        [x[2] for x in points],
        c=(73 / 255.0, 166 / 255.0, 166 / 255.0, 1),
        s=150,
    )

    explorer.sub_pcas = {}
    explorer.data_points[selected_point].pca_indices = []

    explorer.jackknife_ensemble_pca()
    points = explorer.generate_point_cloud(20)
    basis_point = explorer.get_point_in_main_basis(20)

    ax.scatter(
        basis_point[0], basis_point[1], basis_point[2], c="r", s=200, depthshade=False
    )
    ax.scatter(
        [x[0] for x in points],
        [x[1] for x in points],
        [x[2] for x in points],
        c=(0, 0, 1, 0.4),
    )
    divs = 50
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)

    for point in points:
        target_x, target_y = get_target_index(
            xMin=x[0],
            xRange=x[-1] - x[0],
            xDiv=50,
            yMin=y[0],
            yRange=y[-1] - y[0],
            yDiv=50,
            pos=point,
        )
        zz[target_y][target_x] += 1

    ax.contourf(xx, yy, zz, zdir="Z", offset=60)
    ax.set_title(
        "PCA Cloud Data Variability\nRed Dot is full data, Grey is 3-fold val\nTurquoise is 5-fold val, Blue is jackknife samp"
    )
    ax.set_xlabel("1st PCA")
    ax.set_ylabel("2nd PCA")
    ax.set_zlabel("3rd PCA")

    plt.show()
