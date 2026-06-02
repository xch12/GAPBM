#轨迹读取模块
import pandas as pd


def load_geolife_trajectory(
        file_path,
        max_points=1000
):
    """
    读取GeoLife轨迹文件

    Parameters
    ----------
    file_path : str
        .plt文件路径

    max_points : int
        最大读取轨迹点数

    Returns
    -------
    longitude : ndarray
    latitude : ndarray
    """

    df = pd.read_csv(
        file_path,
        skiprows=6,
        header=None
    )

    longitude = df[1].values[:max_points]
    latitude = df[0].values[:max_points]

    return longitude, latitude
