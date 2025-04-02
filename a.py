import tensorflow as tf
import torch
import cv2
import numpy as np
import scipy
import sklearn
import time

def check_tensorflow_gpu():
    print("Checking TensorFlow GPU...")
    try:
        # Check if TensorFlow can access the GPU
        if tf.config.experimental.list_physical_devices('GPU'):
            print("TensorFlow is using GPU!")
        else:
            print("TensorFlow is NOT using GPU.")
    except Exception as e:
        print(f"Error checking TensorFlow GPU: {e}")

def check_pytorch_gpu():
    print("\nChecking PyTorch GPU...")
    try:
        # Check if PyTorch can access the GPU
        if torch.cuda.is_available():
            print("PyTorch is using GPU!")
            print(f"CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("PyTorch is NOT using GPU.")
    except Exception as e:
        print(f"Error checking PyTorch GPU: {e}")

def check_opencv_gpu():
    print("\nChecking OpenCV GPU...")
    try:
        # Check if OpenCV has CUDA support
        if cv2.ocl.haveOpenCL():
            print("OpenCV is using GPU!")
        else:
            print("OpenCV is NOT using GPU.")
    except Exception as e:
        print(f"Error checking OpenCV GPU: {e}")

def check_numpy_scipy_gpu():
    print("\nChecking NumPy/Scipy GPU...")
    try:
        # NumPy and Scipy do not directly use GPU, so we will test if we are running on a system with GPU
        if np.cuda.is_available():  # This is more relevant for libraries like CuPy, which use GPU
            print("NumPy/Scipy can use GPU (via CuPy)!")
        else:
            print("NumPy/Scipy is NOT using GPU.")
    except Exception as e:
        print(f"Error checking NumPy/Scipy GPU: {e}")

def check_sklearn_gpu():
    print("\nChecking scikit-learn GPU...")
    try:
        # scikit-learn does not directly use GPU but some estimators can use it via libraries like cuML
        print("scikit-learn does not directly use GPU, but can be used with libraries like cuML.")
    except Exception as e:
        print(f"Error checking scikit-learn GPU: {e}")

def check_all_libraries():
    print("Library GPU Check Report:")
    check_tensorflow_gpu()
    check_pytorch_gpu()
    check_opencv_gpu()
    check_numpy_scipy_gpu()
    check_sklearn_gpu()

if __name__ == "__main__":
    start_time = time.time()
    check_all_libraries()
    end_time = time.time()
    print(f"\nTotal check time: {end_time - start_time:.2f} seconds")
