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

#pragma once


#include <agxOSG/Node.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osgViewer/Viewer>
#include <osgSim/ColorRange>
#include <osgSim/ScalarsToColors>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.
#include <agxCollide/HeightField.h>

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(HeightField);

  /**
  Rendering class for the agxCollide::HeightField
  The visual representation will be updated if the HeightField is modified
  */
  class AGXOSG_EXPORT HeightField : public osg::Geode
  {

    public:
      /**
      Construct a renderable height field from the specified collision HeightField

      \param heightField - The collision height field that we want to render
      */
      HeightField( const agxCollide::HeightField* heightField);

      /**
      \return the collision HeightField
      */
      const agxCollide::HeightField* getHeightField() const;


      osg::Geometry *getGeometry();
      const osg::Geometry *getGeometry() const ;


    protected:
      bool createVisual(const agxCollide::HeightField *heightField);
      virtual void traverse(osg::NodeVisitor& nv);

      virtual ~HeightField();

    protected:
      agxCollide::HeightFieldConstObserver        m_heightField;
      osg::ref_ptr<osg::Geometry>                 m_geometry;
      osg::StateSet*                              m_stateSet;
      agx::UInt32                                 m_modifiedCount;
  };
}

