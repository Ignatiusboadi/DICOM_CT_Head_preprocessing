from pydicom import dcmread

import ants
import cv2
import json
import nibabel
import numpy as np
import os
import pandas as pd
import scipy


def read_dicom_files(folder_path):
    """
    Reads and processes DICOM files from a specified folder.

    The function performs the following steps:
    - Loads all DICOM files from the folder.
    - Sorts them based on their position along the z-axis.
    - Computes a uniform slice thickness from the first two slices.
    - Assigns the computed slice thickness to all slices.

    Parameters:
        folder_path (str): Path to the folder containing DICOM files.

    Returns:
        list: A sorted list of pydicom Dataset objects with consistent SliceThickness.
    """
    dicom_files = [dcmread(os.path.join(folder_path, file)) for file in os.listdir(folder_path)]
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(dicom_files[0].ImagePositionPatient[2] - dicom_files[1].ImagePositionPatient[2])
    except AttributeError:
        slice_thickness = np.abs(dicom_files[0].SliceLocation - dicom_files[1].SliceLocation)
    for dicom_file in dicom_files:
        dicom_file.SliceThickness = slice_thickness
    return dicom_files

