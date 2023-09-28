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

#ifndef AGXOSG_SIMPLEDEPTHBUFFERLIDAR_H
#define AGXOSG_SIMPLEDEPTHBUFFERLIDAR_H

#include <agxOSG/export.h>
#include <agx/Vec3.h>
#include <agxOSG/RenderToImage.h>
#include <agxOSG/RenderTarget.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(SimpleDepthBufferLidar);

  class AGXOSG_EXPORT SimpleDepthBufferLidar : public agxOSG::RenderToImage
  {
    public:
      /**
      Construct 2D lidar
      \param width - resolution width for the camera
      \param height - resolution height for the camera
      \param eyeCoordinates - boolean for the scan points to be in eye coordinates. If false the points will be in world coordinates
      */
      SimpleDepthBufferLidar(agx::UInt width, agx::UInt height, bool eyeCoordinates = true);

      /**
      Get the latest scan
      \return - A vector of all the scan points
      */
      agx::Vec3fVector& getScan(void);

    protected:
      ~SimpleDepthBufferLidar();

    private:
      bool m_eyeCoordinates;
      agx::Vec3fVector m_scan;
  };
}

#endif /* AGXOSG_SIMPLEDEPTHBUFFERLIDAR_H */
