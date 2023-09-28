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

#ifndef AGXOSG_COMPOSITERENDERER_H
#define AGXOSG_COMPOSITERENDERER_H

#include <agx/config/AGX_USE_COMPOSITE.h>
#include <agx/config.h>

#if AGX_USE_COMPOSITE()

#include <agxOSG/SplineRendererOld.h>
#include <agxWire/Composite/MultiWire.h>

namespace agxOSG
{
  class AGXOSG_EXPORT CompositeRenderer : public agxSDK::StepEventListener
  {
    public:
      /**
      Construct Composite renderer given a Composite and root (parent) node for this renderer.
      \param wire - wire to render
      \param root - root/parent node
      */
      CompositeRenderer( agxWire::Composite::MultiWire* wire, osg::Group* root );

      void setNumSegmentsPerUnitLength( agx::Real numSegmentsPerUnitLength );
      void setColor( const agx::Vec4f& color );

    protected:
      virtual ~CompositeRenderer() {}

      virtual void post(const agx::TimeStamp&);
      virtual void removeNotification();

    protected:
      void update();

      agx::Vector<agxOSG::SplineRendererOldRef> m_splineRenderers;
      agxWire::Composite::MultiWireRef m_wire;
      osg::ref_ptr< osg::Group > m_root;

      agx::Real m_segmentsPerUnitLength;
      agx::Vec4f m_color;
  };

  typedef agx::ref_ptr< CompositeRenderer > CompositeRendererRef;
}

#endif
#endif // AGX_USE_COMPOSITE
