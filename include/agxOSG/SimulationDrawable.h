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

#ifndef AGXOSG_SIMULATIONDRAWABLE_H
#define AGXOSG_SIMULATIONDRAWABLE_H

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/export.h>
#include <agx/Task.h>
#include <agx/observer_ptr.h>
#include <agxSDK/Simulation.h>

namespace agxOSG
{
  class AGXOSG_EXPORT SimulationDrawable : public osg::Drawable
  {
  public:

    SimulationDrawable( agxSDK::Simulation *simulation);
    SimulationDrawable( const SimulationDrawable&c, const osg::CopyOp& copyOp );

    virtual osg::Object* cloneType() const { return new SimulationDrawable ( m_simulation.get() ); }
    virtual osg::Object* clone(const osg::CopyOp& copyop) const { return new SimulationDrawable (*this,copyop); }
    virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const SimulationDrawable *>(obj)!=nullptr; }
    virtual const char* libraryName() const { return "agxOSG"; }
    virtual const char* className() const { return "SimulationDrawable"; }


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
    virtual ~SimulationDrawable() {}


    mutable agxSDK::SimulationObserver m_simulation;
    // mutable osg::BoundingBox m_bbox;
  };

}


#endif /* AGXOSG_SIMULATIONDRAWABLE_H */
