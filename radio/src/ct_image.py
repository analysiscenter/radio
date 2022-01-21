import os
import pydicom
import numpy as np
import glob
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy

class CTFile:
    DICOM_ALIASES = ['dcm', 'dicom']
    MHD_ALIASES = ['mhd', 'raw']

    def __new__(cls, path, *args, **kwargs):
        """ Select the type of geometry based on file extension.
        Breaks the autoreload magic (but only for this class).
        """
        #pylint: disable=import-outside-toplevel
        _ = args, kwargs
        fmt = None
        if os.path.isdir(path):
            for ext in [*cls.DICOM_ALIASES]:
                if len(glob.glob(os.path.join(path, f'*.{ext}'))) > 0:
                    fmt = ext
        else:
            fmt = os.path.splitext(path)[1][1:]

        if fmt in cls.DICOM_ALIASES:
            new_cls = DicomFile
        elif fmt in cls.MHD_ALIASES:
            new_cls = MetaImageFile
        else:
            raise TypeError(f'Unknown format of the image: {fmt}')

        instance = super().__new__(new_cls)
        return instance        

    def __init__(self, path):
        self.path = path
        self.data = None
        self.spacing = None
        self.origin = None
    
    def load(self):
        pass

    @property
    def shape(self):
        return self.data.shape

    def unify_spacing(self, order=3):
        if self.spacing[0] == self.spacing[1] == self.spacing[2]:
            return
        factor = np.array([self.spacing[0] / self.spacing[2], self.spacing[1] / self.spacing[2], 1])
        output_shape = self.shape * factor
        
        self.data = scipy.ndimage.interpolation.zoom(self.data, factor, order=order)
        self.spacing = [self.spacing[2]] * 3
    
    def show_slide(self, loc=0, axis=0):
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = loc
        plt.imshow(self.data[tuple(slices)], cmap='gray')
        plt.show()
    
class DicomFile(CTFile):
    def get_slide_files(self, slides, image_type):
        if image_type is not None:
            slides = filter(lambda item: set(image_type) <= set(item.ImageType), slides)
        return list(slides)

    def load(self, image_type=['PRIMARY', 'AXIAL']):
        results = []
        list_of_dicoms = [pydicom.read_file(s) for s in glob.glob(os.path.join(self.path, '*'))]
        slides = self.get_slide_files(list_of_dicoms, image_type)
        slides.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)
        
        dicom_slide = slides[0]
        intercept_pat = dicom_slide.RescaleIntercept
        slope_pat = dicom_slide.RescaleSlope

        patient_data = np.stack([s.pixel_array for s in slides]).astype(np.int16)

        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)
        
        self.data = patient_data
        self.spacing = np.asarray([float(dicom_slide.SliceThickness),
                                  float(dicom_slide.PixelSpacing[0]),
                                  float(dicom_slide.PixelSpacing[1])], dtype=np.float)
        self.origin = np.asarray([float(dicom_slide.ImagePositionPatient[2]),
                                  float(dicom_slide.ImagePositionPatient[0]),
                                  float(dicom_slide.ImagePositionPatient[1])], dtype=np.float)

class MetaImageFile(CTFile):        
    def load(self):
        raw_data = sitk.ReadImage(self.path)
        self.data = sitk.GetArrayFromImage(raw_data)
        self.spacing = np.array(raw_data.GetSpacing())[::-1]
        self.origin = np.array(raw_data.GetOrigin())[::-1]
