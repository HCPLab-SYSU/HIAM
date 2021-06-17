# part of this code are copied from DCRNN
import numpy as np

def masked_rmse_np(preds, labels):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels))


def masked_mse_np(preds, labels):
    rmse = np.square(np.subtract(preds, labels)).astype('float32')
    rmse = np.nan_to_num(rmse)
    return np.mean(rmse)


def masked_mae_np(preds, labels):
    mae = np.abs(np.subtract(preds, labels)).astype('float32')
    mae = np.nan_to_num(mae)
    return np.mean(mae)


def masked_mape_np(preds, labels):
    mape_net = np.divide(np.sum(np.abs(np.subtract(preds, labels)).astype('float32')), np.sum(labels))
    mape_net = np.nan_to_num(mape_net)
    return mape_net