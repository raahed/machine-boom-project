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

#ifndef AGXOSG_POINTSPRITEDRAWABLE_H
#define AGXOSG_POINTSPRITEDRAWABLE_H

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Task.h>
#include <agxData/Buffer.h>
#include <agxData/Value.h>
#include <agxData/Array.h>
// #include <agx/Bound.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>

namespace agxOSG
{
  class AGXOSG_EXPORT PointSpriteDrawable : public osg::Drawable
  {
  public:
    PointSpriteDrawable(agxData::Buffer *positions, agxData::Buffer *rotations, agxData::Buffer *radii,
                        agxData::Buffer *colors,    agxData::Buffer *enableRendering,
                        agxData::Value  *bound,     agx::Component  *context);

    // void set( agxData::Array<agx::Vec3> positions, agxData::Array<agx::Real> radii, agx::Bound3 bound );

    void setColor(const agx::Vec4& color);
    void setEnable(bool flag);


    /** Compute the bounding box around Drawables's geometry.*/
# if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
    virtual osg::BoundingSphere computeBound() const;
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    /// The actual draw method for the particle system
    virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

  public:
    virtual osg::Object* cloneType() const { return new PointSpriteDrawable (m_positions, m_rotations, m_radii, m_colors, m_enableRendering, m_bound, m_renderTask->getContext()->as<agx::Component>()); }
    virtual osg::Object* clone(const osg::CopyOp&) const { return new PointSpriteDrawable (*this); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const PointSpriteDrawable *>(obj)!=nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "PointSpriteDrawable"; }

  protected:
    virtual ~PointSpriteDrawable();

  private:
    agx::TaskRef m_renderTask;
    agxData::BufferRef m_positions;
    agxData::BufferRef m_rotations;
    agxData::BufferRef m_radii;
    agxData::BufferRef m_colors;
    agxData::BufferRef m_enableRendering;
    agxData::ValueRef m_bound;
    bool m_enabled;

    // agxData::Array< agx::Vec3 > m_positions;
    // agxData::Array< agx::Real > m_radii;
    // agx::Bound3 m_bound;
  };

  typedef osg::ref_ptr<PointSpriteDrawable> PointSpriteDrawableRef;
}


#endif /* AGXOSG_POINTSPRITEDRAWABLE_H */
