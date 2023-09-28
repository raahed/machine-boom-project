/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXDATA_INTEGERTEMPLATES_H
#define AGXDATA_INTEGERTEMPLATES_H

#include <agx/Integer.h>

AGX_TYPE_BINDING(agx::Int8 , "Int")
AGX_TYPE_BINDING(agx::Int16, "Int")
AGX_TYPE_BINDING(agx::Int32, "Int")
AGX_TYPE_BINDING(agx::Int64, "Int")

AGX_TYPE_BINDING(agx::UInt8 , "UInt")
AGX_TYPE_BINDING(agx::UInt16, "UInt")
AGX_TYPE_BINDING(agx::UInt32, "UInt")
AGX_TYPE_BINDING(agx::UInt64, "UInt")

#ifdef __APPLE__
AGX_TYPE_BINDING(size_t, "UInt")
#endif

AGX_TYPE_BINDING(agx::Bool, "Bool")


#endif /* _AGXDATA_INTEGERTEMPLATES_H_ */
