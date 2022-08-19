import numpy as np
import scipy as sp
import cv2
import scipy.ndimage as nd
import copy





def getAverageEnergy(image):
    """
    Generate energy map by simple energy function.

    Parameters
    ----------
    image: A color image.

    Returns
    -------
    average_energy: The average pixel energy of the input image.
    """
    img = copy.deepcopy(image)
    height, width = img.shape[0], img.shape[1]
    pixels = height*width
    Ix = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, borderType = cv2.BORDER_CONSTANT))
    Iy = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, borderType = cv2.BORDER_CONSTANT))
    energy = Ix + Iy
    average_energy = np.sum(energy)/pixels
    return average_energy


def compareEnergy(image1, image2):
    """
    Compare the average energy difference of two images.

    Parameters
    ----------
    image1: A color image.
    image2: A color image.

    Returns
    -------
    ratio: The ratio of the average pixel energy of image1 and image2,
    calculated by dividing the larger value with the smaller value. ratio is in the range of 0-1,
    while 0 represents the largest differences and 1 represents the smallest differences.
    """
    average_energy1 = getAverageEnergy(image1)
    average_energy2 = getAverageEnergy(image2)
    ratio = min(average_energy1, average_energy2)/max(average_energy1, average_energy2)
    return ratio


def compareHis(image1, image2):
    """
    Compare histograms of two images, using correlation method.

    Parameters
    ----------
    image1: A color image.
    image2: A color image.

    Returns
    -------
    correlation: The correlation of histograms of image1 and image2, in the range
    of 0-1, while 0 represents the lowest similarity and 1 represents the highest similarity
    """
    gray_img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    his1 = cv2.calcHist([gray_img1], [0], None, [256], [0, 255])
    his2 = cv2.calcHist([gray_img2], [0], None, [256], [0, 255])
    correlation = cv2.compareHist(his1, his2, method=cv2.HISTCMP_CORREL)
    return correlation

#fig5_07
print("Comparing fig5_07…………")
fig5_07_1 = cv2.imread("fig5_07_result.png", cv2.IMREAD_COLOR)
fig5_07_2 = cv2.imread("fig5_07_seam_removal.png", cv2.IMREAD_COLOR)
fig5_07_ratio = compareEnergy(fig5_07_1, fig5_07_2)
fig5_07_corr = compareHis(fig5_07_1, fig5_07_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig5_07_ratio, fig5_07_corr))

#fig8c_07
print("Comparing fig8c_07…………")
fig8c_07_1 = cv2.imread("fig8c_07_result.png", cv2.IMREAD_COLOR)
fig8c_07_2 = cv2.imread("fig8c_07_seams.png", cv2.IMREAD_COLOR)
fig8c_07_ratio = compareEnergy(fig8c_07_1, fig8c_07_2)
fig8c_07_corr = compareHis(fig8c_07_1, fig8c_07_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8c_07_ratio, fig8c_07_corr))

#fig8d_07
print("Comparing fig8d_07…………")
fig8d_07_1 = cv2.imread("fig8d_07_result.png", cv2.IMREAD_COLOR)
fig8d_07_2 = cv2.imread("fig8d_07_insert50.png", cv2.IMREAD_COLOR)
fig8d_07_ratio = compareEnergy(fig8d_07_1, fig8d_07_2)
fig8d_07_corr = compareHis(fig8d_07_1, fig8d_07_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8d_07_ratio, fig8d_07_corr))

#fig8f_07
print("Comparing fig8f_07…………")
fig8f_07_1 = cv2.imread("fig8f_07_result.png", cv2.IMREAD_COLOR)
fig8f_07_2 = cv2.imread("fig8f_insert50-50.png", cv2.IMREAD_COLOR)
fig8f_07_ratio = compareEnergy(fig8f_07_1, fig8f_07_2)
fig8f_07_corr = compareHis(fig8f_07_1, fig8f_07_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8f_07_ratio, fig8f_07_corr))

#fig8_08_back_seam
print("Comparing fig8_08_back_seam…………")
fig8_08_back_seam_1 = cv2.imread("fig8_08_back_seam_result.png", cv2.IMREAD_COLOR)
fig8_08_back_seam_2 = cv2.imread("fig8_08_backward_seams.png", cv2.IMREAD_COLOR)
fig8_08_back_seam_ratio = compareEnergy(fig8_08_back_seam_1, fig8_08_back_seam_2)
fig8_08_back_seam_corr = compareHis(fig8_08_back_seam_1, fig8_08_back_seam_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8_08_back_seam_ratio, fig8_08_back_seam_corr))

#fig8_08_backward
print("Comparing fig8_08_backward…………")
fig8_08_backward_1 = cv2.imread("fig8_08_backward_result.png", cv2.IMREAD_COLOR)
fig8_08_backward_2 = cv2.imread("fig8_08_backward_energy.png", cv2.IMREAD_COLOR)
fig8_08_backward_ratio = compareEnergy(fig8_08_backward_1, fig8_08_backward_2)
fig8_08_backward_corr = compareHis(fig8_08_backward_1, fig8_08_backward_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8_08_backward_ratio, fig8_08_backward_corr))

#fig8_08_forward_seam
print("Comparing fig8_08_forward_seam…………")
fig8_08_forward_seam_1 = cv2.imread("fig8_08_forward_seam_result.png", cv2.IMREAD_COLOR)
fig8_08_forward_seam_2 = cv2.imread("fig8_08_forward_seams.png", cv2.IMREAD_COLOR)
fig8_08_forward_seam_ratio = compareEnergy(fig8_08_forward_seam_1, fig8_08_forward_seam_2)
fig8_08_forward_seam_corr = compareHis(fig8_08_forward_seam_1, fig8_08_forward_seam_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8_08_forward_seam_ratio, fig8_08_forward_seam_corr))

#fig8_08_forward
print("Comparing fig8_08_forward…………")
fig8_08_forward_1 = cv2.imread("fig8_08_forward_result.png", cv2.IMREAD_COLOR)
fig8_08_forward_2 = cv2.imread("fig8_08_ forward_energy.png", cv2.IMREAD_COLOR)
fig8_08_forward_ratio = compareEnergy(fig8_08_forward_1, fig8_08_forward_2)
fig8_08_forward_corr = compareHis(fig8_08_forward_1, fig8_08_forward_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig8_08_forward_ratio, fig8_08_forward_corr))

#fig9_08_backward
print("Comparing fig9_08_backward…………")
fig9_08_backward_1 = cv2.imread("fig9_08_backward_result.png", cv2.IMREAD_COLOR)
fig9_08_backward_2 = cv2.imread("fig9_08_backward_energy.png", cv2.IMREAD_COLOR)
fig9_08_backward_ratio = compareEnergy(fig9_08_backward_1, fig9_08_backward_2)
fig9_08_backward_corr = compareHis(fig9_08_backward_1, fig9_08_backward_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig9_08_backward_ratio, fig9_08_backward_corr))

#fig9_08_forward
print("Comparing fig9_08_forward…………")
fig9_08_forward_1 = cv2.imread("fig9_08_forward_result.png", cv2.IMREAD_COLOR)
fig9_08_forward_2 = cv2.imread("fig9_08_forward_energy.png", cv2.IMREAD_COLOR)
fig9_08_forward_ratio = compareEnergy(fig9_08_forward_1, fig9_08_forward_2)
fig9_08_forward_corr = compareHis(fig9_08_forward_1, fig9_08_forward_2)
print("The average energy ratio is {}.\nThe histogram correlation is {}.\n".format(fig9_08_forward_ratio, fig9_08_forward_corr))
