from __future__ import annotations

from pathlib import Path
import pandas as pd

HAPPY_COLS = [
    "id", "mag_r", "u_g", "g_r", "r_i", "i_z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5",
]

def load_happy_file(path: str | Path) -> pd.DataFrame:
    """Load the HAPPY catalog from a CSV file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"The file {path} does not exist.")
    
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=HAPPY_COLS,
        engine="python",
    )

    # Basic sanity check
    if df.shape[1] != len(HAPPY_COLS):
        raise ValueError(
            f"Unexpected number of columns in {path.name}: "
            f"got {df.shape[1]}, expected {len(HAPPY_COLS)}"
        )

    return df


def load_happy_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Happy A and Happy B from:
      data_dir/Happy/happy_A
      data_dir/Happy/happy_B
    """
    data_dir = Path(data_dir)
    a_path = data_dir / "Happy" / "happy_A"
    b_path = data_dir / "Happy" / "happy_B"

    dfA = load_happy_file(a_path)
    dfB = load_happy_file(b_path)

    return dfA, dfB


if __name__ == "__main__":
    # Project root 기준 실행을 가정: python src/load_happy.py
    # 데이터 위치: ./data/Happy/happy_A, ./data/Happy/happy_B
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    dfA, dfB = load_happy_dataset(data_dir)

    print("Happy A:", dfA.shape)
    print("Happy B:", dfB.shape)
    print(dfA.head())