""" contains class BatchCtMasked(BatchCt) for storing masked Ct-scans """
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import SimpleITK as sitk
from .ct_batch import CTImagesBatch
from .mask_ed2 import make_mask_patient
from .resize import resize_patient_numba
from dataset import action


class CTImagesBatchMasked(CTImagesBatch):
    """
    Class for storing masked batch of ct-scans

    in addition to batch itself, stores mask in
        self.mask (as BatchCt object)

    """

    def __init__(self, index):
        """
        initialization of BatchCtMasked
        """

        # initialize BatchCt itself
        super().__init__(index)

        # initialize mask to None
        self.mask = None

        # add origin and spacing attrs
        self.spacing = dict()
        self.origin = dict()

    # overload method _load_raw
    # add origin and spacing to self

    def _load_raw(self):
        """
        for sake of simplicity
            repeat part of the code from super-method
        read raw-images and save meta about pixel and spacing
            to self
        """
        raw_data = [sitk.ReadImage(self.index.get_fullpath(patient))
                    for patient in self.indices]

        list_of_arrs = [sitk.GetArrayFromImage(pat_data) for pat_data
                        in raw_data]

        # extract spacing and origin from raw_data
        # add them to attrs of self

        self.origin = {i: np.array(data.GetOrigin())
                       for i, data in zip(self.indices, raw_data)}

        self.spacing = {i: np.array(data.GetSpacing())
                        for i, data in zip(self.indices, raw_data)}

        return list_of_arrs

    @action
    def load_mask(self, nodules_df, num_threads=8):  # pylint: disable=too-many-locals
        """
        function for
            loading masks from dataframe with nodules

        nodules_df must contain columns
            seriesuid: index of patients
            coordX, coordY, coordZ: coords of nodule-center
            diameter_mm: diameter of nodule in mm (in 'world' units)
            *note that self._data is ordered as (z, y, x)
        """

        self.mask = np.zeros_like(self._data)

        # define list of args for multithreading
        args = []
        for index in self.indices:
            nods = nodules_df[nodules_df['seriesuid'] == index]
            nodules = nods[['coordX', 'coordY', 'coordZ', 'diameter_mm']].values
            ind_pos = self.index.get_pos(index)
            lower = self._lower_bounds[ind_pos]
            upper = self._upper_bounds[ind_pos]

            args_dict = {'pat_mask': self.mask[lower: upper, :, :],
                         'spacing': self.spacing[index],
                         'origin': self.origin[index],
                         'nodules': nodules}

            args.append(args_dict)

        # run threading procedure
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for arg in args:
                executor.submit(make_mask_patient, **arg)

        return self

    @action
    def resize(self, num_x_new=256, num_y_new=256,
               num_slices_new=128, order=3, num_threads=8):
        """
        resize masked batch
            -resize batch itself
            -resize mask (contained in self.mask)
            -recalculate spacing-dict
        """
        # recalculate new spacing

        new_spacing = dict()
        for index in self.spacing:
            # before-resize shapes
            # in x y z order
            old_shapes = np.asarray([self[index].shape[2],
                                     self[index].shape[1],
                                     self[index].shape[0]])

            # after resize shape, same order
            new_shapes = np.asarray([num_x_new, num_y_new, num_slices_new])

            # recalculate spacing
            new_spacing.update(
                {index: self.spacing[index] * old_shapes / new_shapes})

        self.spacing = new_spacing

        # resize batch
        args_res = dict(num_x_new=num_x_new, num_y_new=num_y_new, order=order,
                        num_slices_new=num_slices_new, num_threads=num_threads)

        super().resize(**args_res)

        # resize mask if loaded

        if self.mask is not None:
            result_mask = np.zeros((len(self) *
                                   num_slices_new, num_y_new, num_x_new))
            args = []
            for index in self.indices:
                ind_pos = self.index.get_pos(index)
                lower = self._lower_bounds[ind_pos]
                upper = self._upper_bounds[ind_pos]

                args_dict = {'chunk': self.mask,
                             'start_from': lower, 
                             'end_from': upper, 
                             'num_x_new': num_x_new,
                             'num_y_new': num_y_new,
                             'order': order,
                             'res': result_mask,
                             'start_to': ind_pos * num_slices_new}

                args.append(args_dict)

            # run multithreading
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for arg in args:
                executor.submit(resize_patient_numba, **arg)

            # change mask
            self.mask = result_mask

        return self
