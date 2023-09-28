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

#ifndef AGXIO_HDF5EXPORTER_H
#define AGXIO_HDF5EXPORTER_H

#include <agx/config/AGX_USE_HDF5.h>
#include <agx/config.h>
#if AGX_USE_HDF5()

#include <agx/AgxH5.h>
#include <agxSDK/StepEventListener.h>

namespace agxIO
{

  /**
  The Hdf5Exported allows for export of body, contacts and constraint data.
  */
  class AGXPHYSICS_EXPORT Hdf5Exporter : public agxSDK::StepEventListener
  {
    public:

      /**
      Constructor
      */
      Hdf5Exporter( const agx::String& filename = "data.h5" );

      /// Inherited from agxSDK::StepEventListener.
      virtual void pre(  const agx::TimeStamp& /* time */ ) override;

      /// Inherited from agxSDK::StepEventListener.
      virtual void post( const agx::TimeStamp& /* time */ ) override;

    protected:
      /**
      Destructor
      */
      virtual ~Hdf5Exporter();

    private:
      agx::String m_filename;
      H5::H5File *m_f;
      H5::Group  m_g;
  };

}

#endif

#endif

