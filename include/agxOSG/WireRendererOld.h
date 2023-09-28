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

#ifndef AGXOSG_WIRERENDERER_OLD_H
#define AGXOSG_WIRERENDERER_OLD_H

#include <agxOSG/SplineRendererOld.h>
#include <agxWire/Wire.h>

namespace agxOSG
{
  class AGXOSG_EXPORT WireRendererOld : public SplineRendererOld
  {
    public:
      /**
      Construct wire renderer given a wire and root (parent) node for this renderer.
      \param wire - wire to render
      \param root - root/parent node
      */
      WireRendererOld( agxWire::Wire* wire, osg::Group* root );

    protected:
      virtual ~WireRendererOld() {}

      virtual void update() override;
      virtual void update( const agx::RealVector& radiiVector ) override { SplineRendererOld::update( radiiVector ); }

    protected:
      agx::observer_ptr< agxWire::Wire >  m_wire;
  };

  typedef agx::ref_ptr< WireRendererOld > WireRendererOldRef;
}

#endif
