""" contains class CTImagesMaskedBatch(CTImagesBatch) for storing masked Ct-scans """
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import SimpleITK as sitk
from .ct_batch import CTImagesBatch
from .mask import make_mask_patient
from .resize import resize_patient_numba
from dataset import action


class CTImagesMaskedBatch(CTImagesBatch):
    """
    Class for storing masked batch of ct-scans

    in addition to batch itself, stores mask in
        self.mask as ndarray type

    new attrs:
        1. mask: ndarray of masks
        2. spacing: dict with keys = self.indices
            stores distances between pixels in mm for patients
            order is x, y, z
        3. origin: dict with keys = self.indices
            stores world coords of [0, 0, 0]-pixel of data for
            all patients

    Important methods:
        1. load_mask(self, nodules_df, num_threads=8)
            function for
            loading masks from dataframe with nodules
            multithreading is supported
        2. resize(self, num_x_new=256, num_y_new=256,
                  num_slices_new=128, order=3, num_threads=8)
            transform shape of all patients to
            (num_slices_new, num_y_new, num_x_new)
            if masks are loaded, they are are also resized

        *Note: spacing, origin are recalculated when resize is executed
            As a result, load_mask can be also executed after resize
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
    # load origin and spacing from .raw files

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
        load masks from dataframe with nodules

        args:
            nodules_df: dataframe that contain columns
                seriesuid: index of patients
                coordX, coordY, coordZ: coords of nodule-center
                diameter_mm: diameter of nodule in mm (in 'world' units)
                *note that self._data is ordered as (z, y, x)
            num_threds: number of threads for parallelism
        """

        self.mask = np.zeros_like(self._data)

        # define list of args for multithreading
        args = []
        for index in self.indices:
            nods = nodules_df[nodules_df['seriesuid'] == index]
            nodules = nods[['coordX', 'coordY',
                            'coordZ', 'diameter_mm']].values
            ind_pos = self.index.get_pos(index)
            lower = self._lower_bounds[ind_pos]
            upper = self._upper_bounds[ind_pos]

            args_dict = {'pat_mask': self.mask[lower: upper, :, :],
                         'spacing': np.asarray(self.spacing[index]),
                         'origin': np.asarray(self.origin[index]),
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
            -resize mask if loaded
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
                             'num_slices_new': num_slices_new,
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

        # resize batch
        args_res = dict(num_x_new=num_x_new, num_y_new=num_y_new, order=order,
                        num_slices_new=num_slices_new, num_threads=num_threads)

        super().resize(**args_res)

        return self

    def dump_mask(self, dst, fmt='blosc'):
        """Dump mask on hard drive.

        dump mask on specified path and format
            create folder corresponding to each patient

        example:
            # initialize batch and load data
            ind = ['1ae34g90', '3hf82s76', '2ds38d04']
            batch = CTImagesMaskedBatch(ind)

            batch.load_mask(...)

            batch.dump_mask('./data/blosc_mask_preprocessed')
        """
        if fmt != 'blosc':
            raise NotImplementedError(
                'Dump to {} not implemented yet'.format(fmt))

        for patient_id in self.indices:
            # view on patient data
            patient_num = self.index.get_pos(patient_id)
            patient_mask = self.mask[self._lower_bounds[patient_num]:
                                     self._upper_bounds[patient_num], :, :]
            # pack the data
            packed = blosc.pack_array(patient_mask, cname='zstd', clevel=1)

            # remove directory if exists
            if os.path.exists(os.path.join(dst, patient_id)):
                shutil.rmtree(os.path.join(dst, patient_id))

            # put blosc on disk
            os.makedirs(os.path.join(dst, patient_id))

            with open(os.path.join(dst, patient_id, 'data.blk'), mode='wb') as file:
                file.write(packed)

        # add info in self.history
        info = {}
        info['method'] = 'dump'
        info['params'] = {'path': dst}
        self.history.append(info)

        return self

    def get_axial_slice(self, person_number, slice_height):
        """
        get tuple of slices (data slice, mask slice)

        args:
            person_number: person position in the batch
            slice_height: height, take slices with number
                int(0.7 * number of slices for person) from
                patient's scan and mask
        """
        margin = int(slice_height * self[person_number].shape[0])

        patch = (self[person_number][margin, :, :],
                 self.mask[self._lower_bounds[person_number] + margin, :, :])
        return patch
