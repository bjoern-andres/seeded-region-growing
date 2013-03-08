# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "helper.hxx" namespace "andres":
    cdef enum CoordinateOrder:
        FirstMajorOrder
        LastMajorOrder
    cdef cppclass npView[T]:
        npView(np.intp_t *dimensions_start,
               np.intp_t *dimensions_end,
               np.intp_t *strides_start,
               T *data,
               CoordinateOrder order)
    cdef cppclass SeededRegionGrower[T]:
         void grow_regions(npView[np.uint8_t],
                           npView[T])

ctypedef fused seedtype:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.int8_t
    np.int16_t
    np.int32_t


cpdef _inplace_region_growing(np.ndarray elevation,
                              np.ndarray seeds,
                              seedtype ignore):
    cdef npView[np.uint8_t] *elevation_view
    cdef npView[seedtype] *seed_view
    cdef SeededRegionGrower[seedtype] grower
    cdef np.ndarray[np.intp_t, ndim=1] seed_strides

    elevation_view = new npView[np.uint8_t](elevation.shape,
                                            elevation.shape + <int> elevation.ndim,
                                            elevation.strides,
                                            <np.uint8_t *> elevation.data,
                                            LastMajorOrder if elevation.flags['F_CONTIGUOUS'] else FirstMajorOrder)

    # strides in andres::npView are based on elements size, while numpy strides
    # are bytes, so we adjust by itemsize
    seed_strides = np.array([seeds.strides[j] for j in range(seeds.ndim)]).astype(np.intp) / seeds.itemsize
    seed_view = new npView[seedtype](seeds.shape,
                                     seeds.shape + <int> seeds.ndim,
                                     <np.intp_t *> seed_strides.data,
                                     <seedtype *> seeds.data,
                                     LastMajorOrder if seeds.flags['F_CONTIGUOUS'] else FirstMajorOrder)
    grower.grow_regions(cython.operator.dereference(elevation_view),
                        cython.operator.dereference(seed_view))
    del elevation_view
    del seed_view

def inplace_region_growing(elevation,
                           seeds):
    assert elevation.shape == seeds.shape
    # Unfortunately, Cython doesn't support dispatch on typed arrays of
    # arbitrary dimension, so we fetch the function directly
    try:
        _inplace_region_growing.__signatures__[str(seeds.dtype) + '_t'](elevation, seeds, 0)
    except KeyError:
        print "region growing only implemented for the following types:"
        for k in _inplace_region_growing.__signatures__.keys():
            print "    ", k
        raise
