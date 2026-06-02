##轨迹生成
import pandas as pd

def load_geolife_trajectory(file_path):

    df = pd.read_csv(
        file_path,
        skiprows=6,
        header=None
    )

    df.columns = [
        "lat",
        "lon",
        "unused",
        "altitude",
        "date_days",
        "date",
        "time"
    ]

    x = df["lon"].values
    y = df["lat"].values

    return x, y