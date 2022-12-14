#distutils: language = c++
from libcpp.vector cimport vector
from libcpp.string cimport string



# For cv::Mat usage
cdef extern from "opencv2/core/core.hpp":
  cdef int  CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_8U
  cdef int CV_32F

cdef extern from "opencv2/core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t len, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO


cdef extern from "../mk_normals.h" namespace "NORM":
    cdef cppclass NormalizeImage:
        NormalizeImage(string filepath, string MODE) except +
        NormalizeImage(string filepath) except +
        NormalizeImage() except +
        void loadImage(string filepath);
        void selectMethod(string MODE);
        void setImage(Mat image);
        Mat getNormalizedImage();
        Mat getEmbeddedImage();
        void pprintImage(string PATH, string MODE)

        Mat depth;
        void convMethod();
        void crossMethod();
        void crossfastMethod();
        void jetmap();