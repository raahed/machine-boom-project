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

#ifndef AGXOSG_PRESSUREFIELDRENDERER_H
#define AGXOSG_PRESSUREFIELDRENDERER_H

#include <agxOSG/export.h>

#include <agxModel/PressureFieldRenderer.h>

#include <agx/Referenced.h>
#include <agx/Vec4.h>

#include <agx/PushDisableWarnings.h>
#include <osg/Array>
#include <osg/PrimitiveSet>
#include <osg/Geometry>
#include <agx/PopDisableWarnings.h>

#include <mutex>

namespace osg
{
  class Group;
}

namespace agxCollide
{
  class Mesh;
}

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(PressureFieldRenderer);

  class AGXOSG_EXPORT PressureFieldRenderer : public agxModel::PressureFieldRenderer
  {
    public:
      /**
      Construct given root node, base color and a scale of the mesh (to have it a bit larger/smaller than the visual mesh).
      */
      PressureFieldRenderer( osg::Group* root, agx::Real scale = agx::Real( 1 ) );

      virtual agx::UInt addTriangle( const agxModel::TriangleData& tData, const agx::AffineMatrix4x4& worldToShape ) override;
      virtual agx::UInt addPressure( agx::UInt index, agx::Real pressure ) override;
      virtual void clear() override;
      virtual void done() override;

    protected:
      PressureFieldRenderer();
      ~PressureFieldRenderer();

    private:
      osg::ref_ptr< osg::FloatArray > m_heightsArray;
      osg::ref_ptr< osg::FloatArray > m_pressureArray;
      osg::ref_ptr< osg::Vec3Array >  m_vertices;
      osg::ref_ptr< osg::Vec3Array >  m_normals;
      osg::ref_ptr< osg::DrawArrays > m_primitiveSet;
      osg::ref_ptr< osg::Geometry >   m_geometry;
      agx::UInt m_activeVertices;
      float m_maxPressure;
      agx::Real m_scale;
      std::mutex m_mutex;
  };
}

#endif

