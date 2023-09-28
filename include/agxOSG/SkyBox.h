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

#ifndef AGXOSG_SKYBOX_H
#define AGXOSG_SKYBOX_H

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <agx/Vector.h>
#include <osg/TextureCubeMap>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{

  AGXOSG_EXPORT osg::Geode*
    createSkyBox( agx::Vector<osg::ref_ptr<osg::Image> >& images, float size );

  AGXOSG_EXPORT osg::Geode*
    createSkySphere( const std::string& name,  agx::Vector<osg::ref_ptr<osg::Image> >& images, float radius );

  AGXOSG_EXPORT osg::TextureCubeMap *createTextureCubeMap( agx::Vector<osg::ref_ptr<osg::Image> >& images);

  AGXOSG_EXPORT osg::TextureCubeMap *createTextureCubeMap( const std::string& imagename, const std::string& filetype );

  /// Sky box
  class AGXOSG_EXPORT SkyBox : public osg::Group
  {
    public:
      /// Constructor
      SkyBox(  const std::string& name,
        const std::string& imagename,
        const std::string& filetype,
        double size = 1000,
        bool use_glslang = false );

    protected:
      double m_size;
  };

}
#endif
