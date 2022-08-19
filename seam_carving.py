"""
Runing this file will produce all the required output in the order as listed
in Midterm Project README 4.III
"""


import numpy as np
import scipy as sp
import cv2
import scipy.ndimage as nd
import copy


def getBackwardEnergyMap(image):
    """
    Generate energy map by simple energy function.

    Parameters
    ----------
    image: A grayscale image.

    Returns
    -------
    energy: An array of the same size as the input image. The values represent
    the energy of that pixel in the input image.
    """
    img = copy.deepcopy(image)
    Ix = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType = cv2.BORDER_CONSTANT))
    Iy = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType = cv2.BORDER_CONSTANT))
    energy = Ix + Iy
    return energy


def getBackwardPathMap(energy):
    """
    Generate cumulative BACKWARD path map using dynamic programming.

    Parameters
    ----------
    energy: A backward energy map from getBackwardEnergyMap

    Returns
    -------
    path_map: An array of the same size as the energy map. The values represent
    the cumulative minimum energy for all possible connected seams of that pixel.
    """
    shape = energy.shape
    height, width = shape[0], shape[1]
    path_map = np.zeros(shape, dtype = np.float64)
    path_map[0, :] = energy[0, :]
    for i in range(1, height):
        for j in range(0, width):
            middle = path_map[i-1, j]
            if j == 0:
                left = middle + 1
                right = path_map[i-1, j+1]
            elif j == width - 1:
                left = path_map[i-1, j-1]
                right = middle + 1
            else:
                left = path_map[i-1, j-1]
                right = path_map[i-1, j+1]
            path_map[i, j] = energy[i, j] + min(left, middle, right)
    return path_map


def getBackwardSeam(path_map):
    """
    Find the least energy seam by backtracking.

    Parameters
    ----------
    path_map: A cumulative backward path energy map from getBackwardPathMap.

    Returns
    -------
    seam: A list of column indexs representing a vertical seam with the lowest
    cumulative backward path energy.
    """
    shape = path_map.shape
    height, width = shape[0], shape[1]
    index = np.argmin(path_map[-1,:])
    seam = [index]
    j = index
    for i in range(height-1, 0, -1):
        middle = path_map[i-1, j]
        if j == 0:
            left = middle + 1
            right = path_map[i-1, j+1]
        elif j == width - 1:
            left = path_map[i-1, j-1]
            right = middle + 1
        else:
            left = path_map[i-1, j-1]
            right = path_map[i-1, j+1]
        if left <= min(middle, right):
            j = j - 1
        elif right <= middle:
            j = j + 1
        seam.append(j)
    seam.reverse()
    return seam


def getForwardPathMap(image):
    """
    Generate cumulative FORWARD path energy map from cost of each pixel.

    Parameters
    ----------
    image: A color image padded with 1 pixel border.

    Returns
    -------
    path_map: An array of the same size as the input image excluding the border.
    The values represent the cumulative minimum energy for all possible
    connected seams of that pixel.
    """
    img = copy.deepcopy(image)
    shape = img.shape
    height, width = shape[0], shape[1]
    path_map = np.zeros((height, width), dtype = np.float64)
    for j in range(1, width-1):
        path_map[1,j] = sum(np.absolute(img[1,j+1,:] - img[1,j-1,:]))
    for i in range(2, height-1):
        for j in range(1, width-1):
            cu = sum(np.absolute(img[i,j+1,:] - img[i,j-1,:]))
            cl = cu + sum(np.absolute(img[i-1,j,:] - img[i,j-1,:]))
            cr = cu + sum(np.absolute(img[i-1,j,:] - img[i,j+1,:]))
            if j == 1:
                path_map[i,j] = min((path_map[i-1,j] + cu), (path_map[i-1,j+1] + cr))
            elif j == width-2:
                path_map[i,j] = min((path_map[i-1,j] + cu), (path_map[i-1,j-1] + cl))
            else:
                path_map[i,j] = min((path_map[i-1,j+1] + cr), (path_map[i-1,j] + cu), (path_map[i-1,j-1] + cl))
    path_map = path_map[1:height-1, 1:width-1]
    return path_map


def getForwardSeam(path_map, image):
    """
    Find the least energy seam by backtracking.

    Parameters
    ----------
    path_map: A cumulative forward path energy map from getForwardPathMap.
    image: A color image padded with 1 pixel border.

    Returns
    -------
    seam: A list of column index representing a vertical seam with the lowest
    cumulative forward path energy.
    """
    img = copy.deepcopy(image)
    shape = path_map.shape
    height, width = shape[0], shape[1]
    index = np.argmin(path_map[-1,:])
    seam = [index]
    j = index
    for i in range(height-1, 0, -1):
        cu = sum(np.absolute(img[i+1,j+2,:] - img[i+1,j,:]))
        if j == 0:
            cr = cu + sum(np.absolute(img[i,j+1,:] - img[i+1,j+2,:]))
            if path_map[i,j] - cr == path_map[i-1,j+1]:
                j=j+1
        elif j == width-1:
            cl = cu + sum(np.absolute(img[i,j+1,:] - img[i+1,j,:]))
            if path_map[i,j] - cl == path_map[i-1,j-1]:
                j=j-1
        else:
            cr = cu + sum(np.absolute(img[i,j+1,:] - img[i+1,j+2,:]))
            cl = cu + sum(np.absolute(img[i,j+1,:] - img[i+1,j,:]))
            if path_map[i,j] - cl == path_map[i-1,j-1]:
                j=j-1
            elif path_map[i,j] - cr == path_map[i-1,j+1]:
                j=j+1
        seam.append(j)
    seam.reverse()
    return seam


def getOriginalSeams(seams):
    """
    Find the corresponding pixel of each seam element in the ORIGINAL image.

    Parameters
    ----------
    seams: A list of seams in the order of removal, each element in a seam
    represents its column index in the image before the seam is removed.

    Returns
    -------
    seam_map: A list of seams in the same order of the input seams, each
    element in a seam represents its column index in the original image.
    """
    seam_map = copy.deepcopy(seams)
    n = len(seams)
    height = len(seams[0])
    for i in range(n-1, 0, -1):
        for j in range(i-1, -1, -1):
            for k in range(0, height):
                if seam_map[i][k] >= seam_map[j][k]:
                    seam_map[i][k] += 1
    return seam_map


def backwardReduceWidth(image, ratio):
    """
    Reduce width of the input image using BACKWARD energy method.

    Parameters
    ----------
    image: A color image.
    ratio: A decimal number, represents the ratio to be reduced.

    Returns
    -------
    new_img: A new image reduced in width.
    seams: A list of seams in the order of removal.
    """
    img = copy.deepcopy(image)
    column_number = int(ratio*img.shape[1])
    seams = []
    for i in range(0, column_number):
        height, width = img.shape[0], img.shape[1]
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        energy = getBackwardEnergyMap(gray_img)
        path_map = getBackwardPathMap(energy)
        seam = getBackwardSeam(path_map)
        seams.append(seam)
        new_img = img[np.arange(width) != np.array(seam)[:,None]].reshape(height,-1, 3)
        img = new_img
    return new_img, seams


def forwardReduceWidth(image, ratio):
    """
    Reduce width of the input image using FORWARD energy method.

    Parameters
    ----------
    image: A color image.
    ratio: A decimal number, represents the ratio to be reduced.

    Returns
    -------
    new_img: A new image reduced in width.
    seams: A list of seams in the order of removal.
    """
    img = copy.deepcopy(image)
    column_number = int(ratio*img.shape[1])
    seams = []
    for i in range(0, column_number):
        height, width = img.shape[0], img.shape[1]
        img = img.astype(np.float64)
        patch_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        path_map = getForwardPathMap(patch_img)
        seam = getForwardSeam(path_map, patch_img)
        seams.append(seam)
        new_img = img[np.arange(width) != np.array(seam)[:,None]].reshape(height,-1, 3)
        img = new_img
    return new_img, seams


def enlargeWidth(image, ratio, seams):
    """
    Enlarge width of the input image. Works for BOTH BACKWARD and FORWARD energy method.

    Parameters
    ----------
    image: A color image.
    ratio: A decimal number, represents the ratio to be enlarged.
    seams: A list of seams if the image is to be reduced by ratio.

    Returns
    -------
    new_img: A new image enlarged in width.
    """
    img = copy.deepcopy(image)
    column_number = int(ratio*img.shape[1])
    height, width = img.shape[0], img.shape[1]
    img = img.astype(np.float64)
    original_seams = np.array(getOriginalSeams(seams))
    sorted_index = np.argsort(original_seams, 0)
    sorted_seams = np.take_along_axis(original_seams, sorted_index, 0)
    new_img = np.zeros((height, width + column_number, 3), dtype=np.float64)
    for i in range(0, height):
        j = 0
        k = 0
        while j < width + column_number:
            if k not in sorted_seams[:,i]:
                new_img[i,j,:] = img[i,k,:]
                j += 1
            else:
                new_img[i,j,:] = img[i,k,:]
                if k != width - 1:
                    new_img[i,j+1,:] = 0.5*(img[i,k,:] + img[i,k+1,:])
                else:
                    new_img[i,j+1,:] = img[i,k,:]
                j += 2
            k += 1
    return new_img


def drawSeams(image, ratio, seams):
    """
    Draw red seams on the ENLARGED image. Works for BOTH BACKWARD and FORWARD energy method

    Parameters
    ----------
    image: A color image.
    ratio: A decimal number, represents the ratio to be enlarged.
    seams: A list of seams if the image is to be reduced by ratio.

    Returns
    -------
    new_img: A new image enlarged in width with red seams.
    """
    img = copy.deepcopy(image)
    column_number = int(ratio*img.shape[1])
    height, width = img.shape[0], img.shape[1]
    img = img.astype(np.float64)
    original_seams = np.array(getOriginalSeams(seams))
    sorted_index = np.argsort(original_seams, 0)
    sorted_seams = np.take_along_axis(original_seams, sorted_index, 0)
    new_img = np.zeros((height, width + column_number, 3), dtype=np.float64)
    for i in range(0, height):
        j = 0
        k = 0
        while j < width + column_number:
            if k not in sorted_seams[:,i]:
                new_img[i,j,:] = img[i,k,:]
                j += 1
            else:
                new_img[i,j,:] = (0,0,255)
                if k != width - 1:
                    new_img[i,j+1,:] = 0.5*(img[i,k,:] + img[i,k+1,:])
                else:
                    new_img[i,j+1,:] = img[i,k,:]
                j += 2
            k += 1
    return new_img

def drawSeamsOriginal(image, seams):
    """
    Draw red seams on the original image. Works for BOTH BACKWARD and FORWARD energy method

    Parameters
    ----------
    image: A color image.
    seams: A list of seams if the image is to be reduced.
    color: Red intensity that is used for the seam.

    Returns
    -------
    img: Original image with red seams.
    """
    img = copy.deepcopy(image)
    height, width = img.shape[0], img.shape[1]
    original_seams = np.array(getOriginalSeams(seams))
    sorted_index = np.argsort(original_seams, 0)
    sorted_seams = np.take_along_axis(original_seams, sorted_index, 0)
    for i in range(0, height):
        j = 0
        while j < width:
            if j in sorted_seams[:,i]:
                img[i,j,:] = (0,0,200)
            j += 1
    return img


#waterfall
print("Processing waterfall image…………")
waterfall = cv2.imread("fig5_07_base.png", cv2.IMREAD_COLOR)
backward_reduce_waterfall, seams = backwardReduceWidth(waterfall, 0.5)
cv2.imwrite("fig5_07_result.png", backward_reduce_waterfall)
print("fig5_07_result.png completed\n")


#dolphin
print("Processing dolphin image…………")
dolphin = cv2.imread("fig8_07_base.png", cv2.IMREAD_COLOR)
backward_reduce_dolphin, seams = backwardReduceWidth(dolphin, 0.5)
seam_dolphin = drawSeams(dolphin, 0.5, seams)
cv2.imwrite("fig8c_07_result.png", seam_dolphin)
print("fig8c_07_result.png completed")

enlarge_dolphin = enlargeWidth(dolphin, 0.5, seams)
cv2.imwrite("fig8d_07_result.png", enlarge_dolphin)
print("fig8d_07_result.png completed")

double_reduce_dolphin, double_seams = backwardReduceWidth(enlarge_dolphin.astype(np.uint8), 0.34)
double_dolphin = enlargeWidth(enlarge_dolphin, 0.34, double_seams)
cv2.imwrite("fig8f_07_result.png", double_dolphin)
print("fig8f_07_result.png completed\n")


#bench
print("Processing bench image…………")
bench = cv2.imread("fig8_08_base.png", cv2.IMREAD_COLOR)
backward_reduce_bench, seams = backwardReduceWidth(bench, 0.5)
backward_seam = drawSeamsOriginal(bench, seams)
cv2.imwrite("fig8_08_back_seam_result.png", backward_seam)
print("fig8_08_back_seam_result.png completed")
cv2.imwrite("fig8_08_backward_result.png", backward_reduce_bench)
print("fig8_08_backward_result.png completed")

forward_reduce_bench, seams = forwardReduceWidth(bench, 0.5)
forward_seam = drawSeamsOriginal(bench, seams)
cv2.imwrite("fig8_08_forward_seam_result.png", forward_seam)
print("fig8_08_forward_seam_result.png completed")
cv2.imwrite("fig8_08_forward_result.png", forward_reduce_bench)
print("fig8_08_forward_result.png completed\n")


#car
print("Processing car image…………")
car = cv2.imread("fig9_08_base.png", cv2.IMREAD_COLOR)
backward_reduce_car, seams = backwardReduceWidth(car, 0.5)
backward_enlarge_car = enlargeWidth(car, 0.5, seams)
cv2.imwrite("fig9_08_backward_result.png", backward_enlarge_car)
print("fig9_08_backward_result.png completed")

forward_reduce_car, seams = forwardReduceWidth(car, 0.5)
forward_enlarge_car = enlargeWidth(car, 0.5, seams)
cv2.imwrite("fig9_08_forward_result.png", forward_enlarge_car)
print("fig9_08_forward_result.png completed\n")


print("All images completed")
