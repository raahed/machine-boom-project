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
#ifndef AGXOSG_RENDERTASKDRAWABLE
#define AGXOSG_RENDERTASKDRAWABLE

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.


#include <agxOSG/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>

namespace agxOSG {

class ParticleDrawableCallback;

  class AGXOSG_EXPORT RenderTaskDrawable : public osg::Drawable
  {
  public:

    RenderTaskDrawable( agx::Task *renderTask );
    RenderTaskDrawable( const RenderTaskDrawable&c, const osg::CopyOp& copyOp );

    /// Set the drawable that will be drawn for each particle
    void setDrawable(osg::Drawable *drawable ) { m_drawable = drawable; }

    /// Return the drawable that will be drawn for each particle
    osg::Drawable *getDrawable( ) { return m_drawable.get(); }

    virtual osg::Object* cloneType() const { return new RenderTaskDrawable ( m_renderTask ); }
    virtual osg::Object* clone(const osg::CopyOp& copyop) const { return new RenderTaskDrawable (*this,copyop); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const RenderTaskDrawable *>(obj)!=nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "RenderTaskDrawable"; }


    /** Compute the bounding box around Drawables's geometry.*/
# if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
    virtual osg::BoundingSphere computeBound() const;
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    /// The actual draw method for the particle system
    virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

  protected:
    friend class ParticleDrawableCallback;
    virtual ~RenderTaskDrawable() {}


    osg::ref_ptr<osg::Drawable> m_drawable;
    mutable agx::observer_ptr<agx::Task> m_renderTask;
    mutable osg::BoundingBox m_bbox;
  };
}
#endif
