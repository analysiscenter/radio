import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqn
from radio.models.utils import overlap_nodules

def binarization(arr, threshold=.5, upper=1, lower=0):
    return np.where(arr > threshold, upper, lower)

def calcualte_metrics(pipeline, **kwargs):
    true_stats = []
    pred_stats = []
    true_mask = []
    pred_mask = []
    for i, batch in tqn(enumerate(pipeline.gen_batch(batch_size=1))):
        nodules_true = pipeline.get_variable('nodules_true')
        nodules_pred = pipeline.get_variable('nodules_pred')
        pred_mask.append(pipeline.get_variable('predictions'))
        true_mask.append(pipeline.get_variable('targets'))
        result = overlap_nodules(batch, nodules_true, nodules_pred)
        true_stats.append(result['true_stats'])
        pred_stats.append(result['pred_stats'])

    true_mask_bin = binarization(np.array(true_mask))
    pred_mask_bin = binarization(np.array(pred_mask))

    true = pd.DataFrame(columns=['diam', 'locZ', 'locY', 'locX', 'overlap_index', 'source_id'])
    pred = pd.DataFrame(columns=['diam', 'locZ', 'locY', 'locX', 'overlap_index', 'source_id'])

    for i, tr in enumerate(true_stats):
        for j in tr:
            true = true.append(j)
        for k in pred_stats[i]:
            pred = pred.append(k)
    metrics = {}
    metrics['sensitivity'] = sensitivity(true, pred, **kwargs)
    metrics['specificity'] = specificity(true_mask_bin, pred_mask_bin, **kwargs)
    return metrics

def sensitivity(true_stats, pred_stats, threshold=.5, d_min=.5, **kwargs):
    predicted_nodules = 0
    for ix in true_stats.iterrows():
        true_diam, true_x, true_y, true_z  = ix[1][true_stats.columns[:4]]
        try:
            pred_diam, pred_x, pred_y, pred_z = pred_stats.loc[ix[1]['overlap_index']][:4]
        except TypeError:
            pass
        dist = (np.abs(pred_x - true_x)**2 + np.abs(pred_y - true_y) + np.abs(pred_z - true_z)**2)**.5
        r = pred_diam / 2
        R = true_diam / 2
        V = (np.pi * (R + r - dist)**2 * (dist**2 + 2*dist*r - 3*r*r + 2*dist*R + 6*r*R - 3*R*2)) / (12*dist)
        V_true = 4/3 * np.pi * R**3
        if V/V_true > d_min:
            predicted_nodules +=1
    return predicted_nodules/len(true_stats)

def specificity_2(true_mask, pred_mask, theshold=0.5, **kwargs):
    tnr = []
    for pred, target in zip(true_mask, pred_mask):
        tnr.append(np.sum((1 - target) * (1 - pred))/np.sum(1 - target))
    return np.mean(tnr)

def specificity(true_mask, pred_mask, **kwargs):
    return np.sum((1-true_mask) * (1 - pred_mask)) / np.sum(1 - true_mask)