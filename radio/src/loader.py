import os
import pydicom
import numpy as np
import glob

class Loader:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.spacing = None
        self.origin = None

        self.shape = None

    def make_array(self):
        pass

class DicomLoader(Loader):
    def get_slide_files(self, slides, image_type):
        if image_type is not None:
            slides = filter(lambda item: set(image_type) <= set(item.ImageType), slides)
        return list(slides)

    def load(self, image_type=['PRIMARY', 'AXIAL']):
        results = []
        list_of_dicoms = [pydicom.read_file(s) for s in glob.glob(os.path.join(self.path, '*'))]
        slides = self.get_slide_files(list_of_dicoms, image_type)
        slides.sort(key=lambda x: int(x.ImagePositionPatient[2]), reverse=True)
        
        self._slide = slides[0]
        intercept_pat = self._slide.RescaleIntercept
        slope_pat = self._slide.RescaleSlope

        patient_data = np.stack([s.pixel_array for s in slides]).astype(np.int16)

        if slope_pat != 1:
            patient_data = slope_pat * patient_data.astype(np.float64)
            patient_data = patient_data.astype(np.int16)

        patient_data += np.int16(intercept_pat)
        
        self.data = patient_data
        self.spacig = np.asarray([float(dicom_slide.SliceThickness),
                                  float(dicom_slide.PixelSpacing[0]),
                                  float(dicom_slide.PixelSpacing[1])], dtype=np.float)
        self.origin = np.asarray([float(dicom_slide.ImagePositionPatient[2]),
                                  float(dicom_slidee.ImagePositionPatient[0]),
                                  float(dicom_slide.ImagePositionPatient[1])], dtype=np.float)
        self.shape = self.data.shape
