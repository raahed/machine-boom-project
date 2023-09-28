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

#ifndef AGXOSG_WIRERENDERER_H
#define AGXOSG_WIRERENDERER_H

#include <agxOSG/SplineRenderer.h>
#include <agxWire/Wire.h>
#include <agxUtil/Spline.h>

namespace agxOSG
{
  class AGXOSG_EXPORT WireRenderer : public SplineRenderer
  {
  public:
    /**
    Construct wire renderer given a wire and root (parent) node for this renderer.
    \param wire - wire to render
    \param root - root/parent node
    \param spline - spline model to be used. Default is agxUtil::ParameterizedCatmullRomSpline(0.5)
    */
    WireRenderer( agxWire::Wire* wire, osg::Group* root, agxUtil::Spline* spline = nullptr );

    void setReferenceMaxTension( agx::Real maxTension );
    agx::Real getReferenceMaxTension() const;

  protected:
    virtual ~WireRenderer() {}

    virtual void addNotification() override;

    virtual void update() override;
    virtual void update( const agx::RealVector& radiiVector ) override { SplineRenderer::update( radiiVector ); }

  protected:
    agx::observer_ptr< agxWire::Wire >  m_wire;
    agx::Real m_refMaxTension;
  };

  typedef agx::ref_ptr< WireRenderer > WireRendererRef;
}

#endif

