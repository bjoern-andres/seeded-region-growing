// Seeded region growing in n-dimensional grid graphs, in linear time.
//
// Copyright (c) 2013 by Bjoern Andres.
// 
// This software was developed by Bjoern Andres.
// Enquiries shall be directed to bjoern@andres.sc.
//
// All advertising materials mentioning features or use of this software must
// display the following acknowledgement: ``This product includes andres::vision 
// developed by Bjoern Andres. Please direct enquiries concerning andres::vision 
// to bjoern@andres.sc''.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice, 
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - All advertising materials mentioning features or use of this software must 
//   display the following acknowledgement: ``This product includes 
//   andres::vision developed by Bjoern Andres. Please direct enquiries 
//   concerning andres::vision to bjoern@andres.sc''.
// - The name of the author must not be used to endorse or promote products 
//   derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
#include "mex.h"

#include "andres/vision/seeded-region-growing.hxx"

// for editing in-place without risking conflicts with copy-on-write
extern "C" bool mxUnshareArray(mxArray*, bool);

// this function operates in-place on its parameter "seeds"
template<class T>
inline void 
helper(
    const mxArray* elevation,
    mxArray* seeds
) {
    andres::View<unsigned char> elevationView(
        mxGetDimensions(elevation), 
        mxGetDimensions(elevation) + mxGetNumberOfDimensions(elevation), 
        static_cast<unsigned char*>(mxGetData(elevation))
    );
    andres::View<T> seedsView(
        mxGetDimensions(seeds), 
        mxGetDimensions(seeds) + mxGetNumberOfDimensions(seeds), 
        static_cast<T*>(mxGetData(seeds))
    );
    andres::vision::seededRegionGrowing(elevationView, seedsView);
}

// Seeded region growing in linear time, in n dimensions.
//
// This function operates in-place on the second input parameter using safe 
// copy-on-write.
//
// prhs[0]: elevation. numeric array (uint8).
// prhs[1]: seeds. numeric array having the same shape as prhs[0] (an integer type).
//
void mexFunction(
    int nlhs, 
    mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[]
) {
    if(nrhs != 2) {
        mexErrMsgTxt("incorrect number of input parameters. expecting two:\n\
                     1. elevation. array (uint8).\n\
                     2. seeds. array (an integer type).\n\
                     this function operates in-place on the second\
                     input parameter using safe copy-on-write.\n");
    }
    if(nlhs != 0) {
        mexErrMsgTxt("incorrect number of return parameters. expecting none.\n");
    }
    if(mxGetNumberOfDimensions(prhs[0]) != mxGetNumberOfDimensions(prhs[1])) { 
        mexErrMsgTxt("dimension mismatch between elevation (parameter 1) and seeds (parameter 2).\n");
    }   
    for(size_t j = 0; j < mxGetNumberOfDimensions(prhs[0]); ++j) {
        if(mxGetDimensions(prhs[0])[j] != mxGetDimensions(prhs[1])[j]) {
            mexErrMsgTxt("shape mismatch between elevation (parameter 1) and seeds (parameter 2).\n");
        }
    }
    if(mxGetClassID(prhs[0]) != mxUINT8_CLASS) {
        mexErrMsgTxt("data type not supported for elevation (parameter 1). expecting uint8.\n");
    }

    // edit in-place without risking conflicts with copy-on-write
    mxUnshareArray(const_cast<mxArray*>(prhs[1]), true);

    switch(mxGetClassID(prhs[1])) {
    case mxUINT8_CLASS:
        helper<unsigned char>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxINT8_CLASS:
        helper<signed char>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxUINT16_CLASS:
        helper<unsigned short>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxINT16_CLASS:
        helper<short>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxUINT32_CLASS:
        helper<unsigned int>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxINT32_CLASS:
        helper<int>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxUINT64_CLASS:
        helper<unsigned long>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    case mxINT64_CLASS:
        helper<long>(prhs[0], const_cast<mxArray*>(prhs[1]));
        break;
    default:
        mexErrMsgTxt("seed map (parameter 2) data type not supported. expecting an integer type.\n");
        break;
    }
}
