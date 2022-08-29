from depth_normals.NormalizationMethods import NORMALIZE_IMAGE
from numpy import zeros, uint8, asarray
from os import listdir, mkdir
from os.path import abspath, dirname, exists
from cv2 import imread
import pytest

PATH = dirname(abspath(__file__))
FILE = r"./depth_images/"
def directories(path = FILE):

    dirs = listdir(path)
    images = []
    if not (exists(PATH + "/normalized_images/")):
        mkdir(PATH + "/normalized_images/")

    for i in range(len(dirs)):
        img = imread(path + dirs[i], 0)
        images.append(img)
    return asarray(images, dtype=uint8)

@pytest.mark.parametrize("mode, path", [
    (b"conv", FILE), (b"cross", FILE),
    (b"crossfast", FILE), (b"jetmap", FILE)
])
def test_depth_normals(mode, path):

    arrays = directories(path)
    normal = NORMALIZE_IMAGE()
    normals = normal.loadNumPyArray(mode, b"./normalized_images/", arrays)
    print("Normals ->", normals.shape)
    #pyNormals = normal.from_array("conv", array)

if (__name__ == "__main__"):
    arrays = directories()
    normal = NORMALIZE_IMAGE()
    normals = normal.loadNumPyArray(b"crossfast",b"./normalized_images/", arrays)
    print("Normals ->", normals.shape)
