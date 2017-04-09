""" contains class BatchCtMasked(BatchCt) for storing masked Ct-scans """

import numpy as np
import SimpleITK as sitk
from .ct_batch import CTImagesBatch
from .mask import make_mask
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

        # initialize BatchCt for mask
        self.mask = CTImagesBatch(index)

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
    def load_mask(self, nodules_df): # pylint: disable=too-many-locals
        """
        function for
            loading masks from dataframe with nodules

        nodules_df must contain columns
            seriesuid: index of patients
            coordX, coordY, coordZ: coords of nodule-center
            diameter_mm: diameter of nodule in mm (in 'world' units)
            *note that self._data is ordered as (z, y, x)
        """
        lst_masks = []

        # over patients in batch
        for index in self.indices:
            # get all nodules for the patient
            nodules = nodules_df[nodules_df['seriesuid'] == index]

            # view for patient data
            data = self[index]

            # future patient mask
            mask = np.zeros_like(data)

            if len(nodules) > 0:
                # data is ordered (z y x)
                num_slices, y_size, x_size = data.shape

                # fetch origin and spacing of patient
                # ordered x y z
                origin = self.origin[index]
                spacing = self.spacing[index]

                # over patient's nodules

                for _, row in nodules.iterrows():
                    # fetch nodule-params in world coords
                    node_x, node_y, node_z = row.coordX, row.coordY, row.coordZ
                    diam = row.diameter_mm

                    # center of nodule in world and pix coords
                    world_center = np.array([node_x, node_y, node_z])

                    # the order of axes in spacing/origin is
                    # x, y, z
                    pix_center = np.rint((world_center - origin) / spacing)

                    # outline range of slices to add mask on
                    # range for iter

                    slices = (np.arange(int(pix_center[2] - diam / 2 / spacing[2] - 2),
                                        int(pix_center[2] + diam / 2 / spacing[2] + 2)).
                              clip(0, num_slices - 1))

                    for i_z in slices:
                        # create mask
                        mask_2d = make_mask(world_center, diam, x_size, y_size,
                                            i_z * spacing[2] + origin[2],
                                            spacing, origin)
                        # add it to patient-mask
                        mask[i_z, :, :] = mask_2d

            lst_masks.append(mask)

        # set params for load batch with masks from array
        bounds_masks = np.cumsum([mask.shape[0] for mask in lst_masks])
        src_masks = np.concatenate(lst_masks)

        # fill self.mask with data
        self.mask.load(fmt='ndarray', src=src_masks,
                       upper_bounds=bounds_masks)

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

        # resize batch and mask
        args_res = dict(num_x_new=num_x_new, num_y_new=num_y_new, order=order,
                        num_slices_new=num_slices_new, num_threads=num_threads)

        super().resize(**args_res)
        self.mask = self.mask.resize(**args_res)

        return self
