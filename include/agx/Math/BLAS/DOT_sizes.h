/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGX_MAX_BLAS_DOT_SIZES_H
#define AGX_MAX_BLAS_DOT_SIZES_H

//Used by the OpenCL version of the dot-product. The number has been chosen for NVIDIA G80. We need a way to parametrize this on a per device basis.
#define SDOT_GROUP_SIZE 128
#define SDOT_NUM_GROUPS 80

#endif // AGX_MAX_BLAS_DOT_SIZES_H