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

#ifndef AGXOSG_NODE_H
#define AGXOSG_NODE_H

#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <agxOSG/SceneDecorator.h>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.
#include <agx/Vec3.h>
#include <agx/AffineMatrix4x4.h>



namespace agxOSG
{

  class AGXOSG_EXPORT Texture2D : public agx::Referenced {
  public:
    Texture2D(osg::Texture2D *texture) : m_texture(texture) {}

    operator osg::Texture2D* () { return m_texture.get(); }
    operator const osg::Texture2D* () const { return m_texture.get(); }

    osg::Texture2D *getTexture() { return m_texture.get(); }

  protected:
    virtual ~Texture2D() {}

    osg::ref_ptr<osg::Texture2D> m_texture;
  };


  class GeometryNode;




  class AGXOSG_EXPORT Transform : public osg::MatrixTransform
  {
  public:
    Transform();
    void setName( const std::string& name );
    bool addChild( osg::Node *child );
    bool addChild( agxOSG::GeometryNode *child );
    void setMatrix( const agx::AffineMatrix4x4& m);
    void setScale( const agx::Vec3& scale );
  protected:
    virtual ~Transform();

  };
}
#endif
