import numpy as np


def get_train_test(df, fraction: float = 0.8):
    train_df = df.head(int(np.round((len(df) * fraction))))
    test_df = df.tail(int(len(df) * (1 - fraction)))
    return train_df, test_df


__all__ = ["get_train_test"]
