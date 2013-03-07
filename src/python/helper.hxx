// helper.cpp - wrap templated functions in classes for cython to call

#include "andres/vision/seeded-region-growing.hxx"
#include <numpy/arrayobject.h>

namespace andres {
    template <class T>
    class SeededRegionGrower {
    public:
        void grow_regions(andres::View<unsigned char> &elevationView,
                          andres::View<T> &seeds) {
            andres::vision::seededRegionGrowing(elevationView, seeds);
        }
    };

    template <class T>
    class npView : public View<T> {
    public:
      npView(npy_intp *shape_start,
                 npy_intp *shape_end,     
                 npy_intp *strides_start,
                 T *data,
                 CoordinateOrder order) : 
          View<T>(shape_start, shape_end, strides_start, data, order)
      { }
    };
}
