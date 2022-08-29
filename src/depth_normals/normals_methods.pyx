from lib.normals_method cimport *
from cython.operator cimport dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdio cimport FILE, fopen, feof, fgetc, getline,fclose
from cython cimport boundscheck, wraparound, cdivision
from libc.string cimport memcpy
from numpy import asarray, dstack, uint8, float32, zeros, ascontiguousarray, any
from numpy cimport ndarray, uint8_t


cdef class NORMALIZE_IMAGE:
    cdef:
         NormalizeImage * thisptr
         Mat normalizedImage
         vector[Mat] normalizedImages
    def __cinit__(self, string MODE="conv",  string FILE="", ndarray imageArrays=None):
        if(FILE !=""):
            self.thisptr = new NormalizeImage(FILE, MODE)
        else:
             self.thisptr = new NormalizeImage()
             if (imageArrays is not None): #(imageArrays.size!=0):
                self.loadImageArray(imageArrays, MODE, "./normalized_images/")

    def __dealloc__(self):
        del self.thisptr

    @boundscheck(False)
    @wraparound(False)
    @staticmethod
    cdef NORMALIZE_IMAGE from_array(string MODE, ndarray imageArrays, string output="./normalized_images/"):
        cdef NORMALIZE_IMAGE wrapper = NORMALIZE_IMAGE.__new__(NORMALIZE_IMAGE) # , MODE, b"", imageArrays)
        wrapper.loadImageArray(imageArrays, MODE, output)
        return wrapper

    @boundscheck(False)
    @wraparound(False)
    cdef inline Mat getNormalizedImage(self):
        return self.thisptr.getNormalizedImage()

    @boundscheck(False)
    @wraparound(False)
    cdef inline Mat getEmbeddedImage(self):
        return self.thisptr.getEmbeddedImage()

    @boundscheck(False)
    @wraparound(False)
    cdef vector[Mat] getImages(self):
        return self.normalizedImages

    @boundscheck(False)
    @wraparound(False)
    cdef inline void setImage(self, ndarray image):
        cdef:
            Mat img
        if(image.ndim==2):
            img = self.np2Mat2D(image)
        elif(image.ndim==3 and image.shape[2]==3):
            img = self.np2Mat3D(image)
        self.thisptr.setImage(img)

    @boundscheck(False)
    @wraparound(False)
    cdef inline void crossfastMethod(self):
        self.thisptr.crossfastMethod()

    @boundscheck(False)
    @wraparound(False)
    cdef inline void crossMethod(self):
        self.thisptr.crossMethod()

    @boundscheck(False)
    @wraparound(False)
    cdef inline void convMethod(self):
        self.thisptr.convMethod()

    @boundscheck(False)
    @wraparound(False)
    cdef inline void jetmapMethod(self):
        self.thisptr.jetmap()

    @boundscheck(False)
    @wraparound(False)
    cdef inline void writer(self, string PATH, string MODE):
        self.thisptr.pprintImage(PATH, MODE)

    @boundscheck(False)
    @wraparound(False)
    cdef vector[Mat] loadNumpyArray(self, string MODE="conv", string output="./normalized_images/", ndarray imageArrays=None):

        cdef:
            vector[Mat] images

        cdef NORMALIZE_IMAGE wrapper = NORMALIZE_IMAGE.from_array(MODE, imageArrays, output)
        images = wrapper.getImages()
        return images

    @boundscheck(False)
    @wraparound(False)
    cdef void loadImageArray(self, ndarray imageArrays, string MODE, string output="./normalized_images/"):

        cdef:
            int i, length
            vector[Mat] iamges
        length = imageArrays.shape[0]
        for i in range(length):
            self.setImage(imageArrays[i])
            if (MODE == "jetmap"):
                self.jetmapMethod()
                self.normalizedImages.push_back(self.getEmbeddedImage())
            else:

                if(MODE=="conv"):
                    self.convMethod()
                elif(MODE=="cross"):
                    self.crossMethod()
                elif(MODE=="crossfast"):
                    self.crossfastMethod()
                self.normalizedImages.push_back(self.getNormalizedImage())
                if(output!=""):
                    self.writer(output + b"_" + str(i).encode("utf-8"), MODE)

    @boundscheck(False)
    @wraparound(False)
    cdef ndarray pyNormals(self, vector[Mat] images):
        cdef:
            int i, length
            Mat temp
        output = []
        length = images.size()
        for i in range(length):
            temp = images[i]
            pyImage = self.Mat2np(temp)
            output.append(pyImage)
        return asarray(output, dtype=uint8)

    @boundscheck(False)
    @wraparound(False)
    cdef inline Mat np2Mat2D(self, ndarray image ):
        assert (image.ndim==2) , "ASSERT::1 channel grayscale only!!"
        cdef ndarray[uint8_t, ndim=2, mode ='c'] np_buff = ascontiguousarray(image, dtype=uint8)

        cdef unsigned int* im_buff = <unsigned int*> np_buff.data
        cdef int r = image.shape[0]
        cdef int c = image.shape[1]
        cdef Mat m
        m.create(r, c, CV_8UC1)
        memcpy(m.data, im_buff, r*c)
        return m

    @boundscheck(False)
    @wraparound(False)
    cdef inline Mat np2Mat3D(self, ndarray ary):
        assert (ary.ndim==3 and ary.shape[2]==3), "ASSERT::3channel RGB only!!"
        ary = dstack((ary[...,2], ary[...,1], ary[...,0])) #RGB -> BGR

        cdef ndarray[uint8_t, ndim=3, mode ='c'] np_buff = ascontiguousarray(ary, dtype=uint8)
        cdef unsigned int* im_buff = <unsigned int*> np_buff.data
        cdef int r = ary.shape[0]
        cdef int c = ary.shape[1]
        cdef Mat m
        m.create(r, c, CV_8UC3)
        memcpy(m.data, im_buff, r*c*3)
        return m

    @boundscheck(False)
    @wraparound(False)
    cdef inline object Mat2np(self, Mat&m):
        # Create buffer to transfer data from m.data
        cdef Py_buffer buf_info

        # Define the size / len of data
        cdef size_t len = m.rows*m.cols*m.elemSize()  #m.channels()*sizeof(CV_8UC3)

        # Fill buffer
        PyBuffer_FillInfo(&buf_info, NULL, m.data, len, 1, PyBUF_FULL_RO)

        # Get Pyobject from buffer data
        Pydata  = PyMemoryView_FromBuffer(&buf_info)

        # Create ndarray with data
        # the dimension of the output array is 2 if the image is grayscale
        if (m.channels()==2 ):
            shape_array = (m.rows, m.cols, m.channels())
        elif(m.channels()==3):
            shape_array = (m.rows, m.cols, m.channels())
        else:
            shape_array = (m.rows, m.cols)

        if m.depth() == CV_32F :
            array = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=float32)
        else :
            #8-bit image
            array = ndarray(shape=shape_array, buffer=Pydata, order='c', dtype=uint8)

        if m.channels() == 3:
            # BGR -> RGB
            array = dstack((array[...,2], array[...,1], array[...,0]))

        return asarray(array, dtype=uint8)

    def pyNormalizedImage(self):
        return self.pyNormals(self.getImages())

    def loadNumPyArray(self, MODE, output, array):
        if(output is None):
            return self.pyNormals(self.loadNumpyArray(MODE, "", array))
        else:
            return self.pyNormals(self.loadNumpyArray(MODE, output, array))


