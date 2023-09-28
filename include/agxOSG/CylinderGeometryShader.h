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

#ifndef AGXOSG_CYLINDER_GEOMETRY_SHADER_H
#define AGXOSG_CYLINDER_GEOMETRY_SHADER_H


#include <agxOSG/export.h>

#include <agx/agx_vector_types.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Geometry>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxCollide
{
  class Geometry;
}

namespace agxOSG
{
  class AGXOSG_EXPORT CylinderGeometryShader : public osg::Geometry
  {
  public:
    CylinderGeometryShader(size_t initialBufferSize = 256);

    /**
    Add cylinder given world start- and end point, and radius.
    */
    void add(const agx::Vec3& startPoint, const agx::Vec3& endPoint, agx::Real radius, agx::Vec2 redShift = agx::Vec2(0, 0));

    /**
    Continuous mesh, for splines. Add point and direction towards the next.
    */
    void add(const agx::Vec3& point, const agx::Quat& zToDir, agx::Real radius, agx::Real redShift = agx::Real(0));

    /**
    All compatible shapes in \p geometry will be rendered.
    \param geometry - geometry with one or several compatible shapes
    */
    void add(const agxCollide::Geometry* geometry);

    /**
    Push the data to the graphics card.
    */
    void update();

    /**
    Resets all internal counters. I.e., all old data will be forgotten.
    */
    void resetCounters();

    /**
    \return the program
    */
    osg::Program* getProgram() { return m_program; }
    const osg::Program* getProgram() const { return m_program; }

    /**
    \return the uniform holding the number of cylinder segments in the shader
    */
    osg::Uniform* getNumCylinderSegmentsUniform() { return m_numCylinderSegments; }
    const osg::Uniform* getNumCylinderSegmentsUniform() const { return m_numCylinderSegments; }

    /**
    \return the uniform holding the color of the cylinders
    */
    osg::Uniform* getCylinderColorUniform() { return m_cylinderColor; }
    const osg::Uniform* getCylinderColorUniform() const { return m_cylinderColor; }

    osg::Vec3Array* getVertexArray() { return m_vertexArray; }
    osg::Vec4Array* getQuatsInColorArray() { return m_quatsInColorArray; }
    osg::FloatArray* getRadiusArray() { return m_radiusArray; }
    void setNumActiveElements(size_t numActiveElements) { m_numActiveElements = numActiveElements; }

  protected:
    virtual ~CylinderGeometryShader();

  protected:
    size_t              m_numActiveElements;

    osg::DrawArrays*    m_primitiveSet;
    osg::Vec3Array*     m_vertexArray;
    osg::Vec4Array*     m_quatsInColorArray;
    osg::FloatArray*    m_radiusArray;
    osg::FloatArray*    m_redShiftArray;
    osg::Program*       m_program;
    osg::Uniform*       m_numCylinderSegments;
    osg::Uniform*       m_cylinderColor;
  };

  typedef osg::ref_ptr< CylinderGeometryShader > CylinderGeometryShaderRef;

}

#endif
