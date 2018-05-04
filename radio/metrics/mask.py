""" Contains metrics """
import numpy as np

from .core import binarize


class MetricsByVolume:
    """ Contains evaluation metrics calculated by pixel volume involved """
    @staticmethod
    def sensitivity(target, prediction, threshold=.5, **kwargs):
        """ True positive rate

        Сalculates the percentage of correctly predicted masked pixels.

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold

        Returns
        -------
        float or None
            The percentage of correctly predicted values.
            None if there is nothing to predict (target contains zeros).
        """
        target, prediction = binarize([target, prediction], threshold)
        total_target = np.sum(target)
        if total_target > 0:
            total = np.sum(target * prediction) / total_target
        else:
            total = None
        return total

    @staticmethod
    def specificity(target, prediction, threshold=.5, **kwargs):
        """ True negative rate

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold

        Returns
        -------
        float or None
            True negative rate
            None if there is nothing to predict (target contains only ones).
        """
        target, prediction = binarize([target, prediction], threshold)
        total_target = np.sum(1 - target)
        if total_target > 0:
            total = np.sum((1 - target) * (1 - prediction)) / total_target
        else:
            total = None
        return total

    @staticmethod
    def false_discovery_rate(target, prediction, threshold=.5, **kwargs):
        """ False discovery rate

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold

        Returns
        -------
        float
            False discovery rate
        """
        target, prediction = binarize([target, prediction], threshold)
        total_prediction = np.sum(prediction)
        if total_prediction > 0:
            rate = np.sum((1 - target) * prediction) / total_prediction
        else:
            rate = 0.
        return rate

    @staticmethod
    def false_positive_rate(target, prediction, threshold=.5, **kwargs):
        """ False positive rate

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold

        Returns
        -------
        float
            False positive rate
        """
        target, prediction = binarize([target, prediction], threshold)
        total_prediction = np.sum(prediction)
        if total_prediction > 0:
            rate = np.sum((1 - target) * prediction) / total_prediction
        else:
            rate = 0.
        return rate


class MetricsByNodules:
    """ Contains evaluation metrics calculated by the nuber of nodules involved """
    @staticmethod
    def sensitivity(target, prediction, threshold=.5, iot=.5, **kwargs):
        """ True positive rate

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold
        iot : float
            The percentage of intersection between the predicted and the target nodules,
            at which the prediction is counted as correct

        Returns
        -------
        float or None
            The percentage of correctly predicted nodules
            None if there is nothing to predict (target contains zeros).
        """
        target = binarize(target, threshold)

        if np.sum(target) == 0:
            total = None
        else:
            target_nodules = get_nodules(target)
            intersection = prediction * target

            right = 0
            for coord in target_nodules:
                predicted_nodule = intersection[coord]
                if np.sum(predicted_nodule) / predicted_nodule.size >= iot:
                    right += 1
            total = right / len(target_nodules)

        return total

    @staticmethod
    def false_positive(target, prediction, threshold=.5, iot=.5, **kwargs):
        """ Calculate the number of falsely predicted nodules.

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold
        iot : float
            The percentage of intersection between the predicted and the target nodules,
            at which the prediction is counted as correct

        Returns
        -------
        int
            The number of falsely predicted nodules
        """
        prediction = binarize(prediction, threshold)

        if np.sum(prediction) == 0:
            total = 0
        else:
            predicted_nodules = get_nodules(prediction)
            target = binarize(target, threshold)

            total = 0
            for coord in predicted_nodules:
                nodule_true_mask = target[coord]
                if np.sum(nodule_true_mask) / nodule_true_mask.size < iot:
                    total += 1

        return total

    @staticmethod
    def false_positive_rate(target, prediction, threshold=.5, iot=.5, **kwargs):
        """ Calculate the ratio of falsely predicted nodules to all true nodules.

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold
        iot : float
            The percentage of intersection between the predicted and the target nodules,
            at which the prediction is counted as correct

        Returns
        -------
        float
            The share of falsely predicted nodules
        """
        false = false_positive_nodules(target, prediction, threshold=threshold, iot=iot)
        target_nodules = get_nodules(target)
        return false / len(target_nodules)


    @staticmethod
    def false_discovery_rate(target, prediction, threshold=.5, iot=.5, **kwargs):
        """ Calculate the ratio of falsely predicted nodules to all predicted nodules.

        Parameters
        ----------
        target : np.array
            Target mask
        prediction : np.array
            Predicted mask
        threshold : float
            Binarization threshold
        iot : float
            The percentage of intersection between the predicted and the target nodules,
            at which the prediction is counted as correct

        Returns
        -------
        float
            The share of falsely predicted nodules
        """
        false = false_positive_nodules(target, prediction, threshold=threshold, iot=iot)
        predicted_nodules = get_nodules(prediction)
        return false / len(predicted_nodules)
