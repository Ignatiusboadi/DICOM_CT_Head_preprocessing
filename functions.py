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


def compute_hu(dicom_files):
    """
    Compute Hounsfield Units (HU) from a list of DICOM files.

    Parameters
    ----------
    dicom_files : list
        List of pydicom Dataset objects representing DICOM slices.

    Returns
    -------
    numpy.ndarray
        3D numpy array of the same shape as the stacked pixel arrays,
        containing the pixel values converted to Hounsfield Units.
    """
    pixels_stack = np.stack([metadata.pixel_array for metadata in dicom_files])
    pixels_stack = pixels_stack.astype(np.int16)
    pixels_stack[pixels_stack == -2000] = 0

    hu_stack = np.zeros(pixels_stack.shape, dtype=np.float32)

    for index, metadata in enumerate(dicom_files):
        intercept = getattr(metadata, 'RescaleIntercept', 0)
        slope = getattr(metadata, 'RescaleSlope', 1)
        hu_stack[index] = pixels_stack[index].astype(np.float32) * slope + intercept

    return hu_stack


def resize_hu_array(hu_array, new_size):
    """
    Resize a 3D numpy array of HU values slice-by-slice in XY dimensions.

    Args:
        hu_array (np.ndarray): 3D array with shape (Z, H, W).
        new_size (tuple): Desired output size for each slice (width, height).

    Returns:
        np.ndarray: Resized 3D array with shape (Z, new_height, new_width).
    """
    resized_slices = []
    for i in range(hu_array.shape[0]):
        slice_ = hu_array[i]
        resized_slice = cv2.resize(slice_, new_size, interpolation=cv2.INTER_LINEAR)
        resized_slices.append(resized_slice)
    resized_array = np.stack(resized_slices, axis=0)
    return resized_array


def reorient_image(np_image, spacing, output_prefix):
    """
    Convert a numpy image array to an ANTs image, save it,
    reorient to RAS orientation, save the reoriented image,
    and return the reoriented image as a numpy array.

    Parameters
    ----------
    np_image : numpy.ndarray
        The input image as a NumPy array.
    spacing : tuple or list of float
        The physical spacing of the image voxels in each dimension (z, y, x).
    output_prefix : str
        Prefix for saving the output NIfTI files.

    Returns
    -------
    numpy.ndarray
        The reoriented image as a NumPy array.
    """
    ants_img = ants.from_numpy(np_image, spacing=spacing)
    ants.image_write(ants_img, f"{output_prefix}_input.nii.gz")
    reoriented_img = ants.reorient_image2(ants_img, orientation='RAS')
    ants.image_write(reoriented_img, f"{output_prefix}_reoriented.nii.gz")
    return reoriented_img.numpy()
