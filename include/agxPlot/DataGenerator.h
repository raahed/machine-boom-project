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

#ifndef AGXPLOT_DATAGENERATOR_H
#define AGXPLOT_DATAGENERATOR_H

#include <agx/Real.h>
#include <agx/Referenced.h>

namespace agxPlot
{
  class AGXPHYSICS_EXPORT DataGenerator : public agx::Referenced
  {
    public:
      struct Result
      {
        bool hasValue;
        agx::Real value;
      };

    public:
      virtual Result getValue() = 0;

    protected:
      virtual ~DataGenerator() {};
  };
  AGX_DECLARE_POINTER_TYPES(DataGenerator);

}




#endif
