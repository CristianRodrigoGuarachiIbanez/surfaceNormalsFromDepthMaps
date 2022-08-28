from depth_normals.normals import NORMALIZE_IMAGE
from numpy import zeros, uint8, asarray
from os import listdir
from cv2 import imread
import pytest

def directories(path = r"../../depth_images/"):

    dirs = listdir(path)
    images = []
    for i in range(len(dirs)):
        img = imread(path + dirs[i], 0)
        images.append(img)
    return asarray(images, dtype=uint8)
@pytest.mark.parametrize("mode, path", [
    (b"conv", r"../../depth_images/"), (b"cross", r"../../depth_images/"),
    (b"crossfast", r"../../depth_images/"), (b"jetmap", r"../../depth_images/")
])
def test_depth_normals(mode, path):

    arrays = directories()
    normal = NORMALIZE_IMAGE()
    normals = normal.loadNumPyArray(mode, arrays)
    print("Normals ->", normals.shape)
    #pyNormals = normal.from_array("conv", array)

if (__name__ == "__main__"):
    arrays = directories()
    normal = NORMALIZE_IMAGE()
    normals = normal.loadNumPyArray(b"crossfast", arrays)
    print("Normals ->", normals.shape)