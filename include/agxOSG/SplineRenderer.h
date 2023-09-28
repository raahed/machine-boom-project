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


#include <agxOSG/export.h>
#include <agxOSG/Node.h>

#include <agxUtil/Spline.h>

#include <agxSDK/StepEventListener.h>

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
  /**
  Class to encapsulate reading shader files and setting up rendering of a Spline
  */
  class AGXOSG_EXPORT SplineShader : public osg::Geometry
  {
  public:
    SplineShader(size_t initialBufferSize = 256);

    /**
    Add cylinder given world start- and end point, and radius.
    */
    void add(const agx::Vec3& startPoint, const agx::Vec3& endPoint, agx::Real radius, agx::Vec2 redShift = agx::Vec2(0, 0));

    /**
    Continuous mesh, for splines. Add point and direction towards the next.
    */
    void add(const agx::Vec3& point, const agx::Quat& zToDir, agx::Real radius, agx::Real redShift = agx::Real(0));

    void add2(const agx::Vec3& point, const agx::Quat& quat, agx::Real radius, agx::Real xTexCoord, agx::Real redShift);

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
    osg::Uniform* getCircleResolutionUniform() { return m_circleResolutionUniform; }
    const osg::Uniform* getCircleResolutionUniform() const { return m_circleResolutionUniform; }

    /**
    \return the uniform holding the color of the cylinders
    */
    osg::Uniform* getCylinderColorUniform() { return m_cylinderColorUniform; }
    const osg::Uniform* getCylinderColorUniform() const { return m_cylinderColorUniform; }

    osg::Uniform* getCircleNormalUniform() { return m_circleNormalUniform; }
    const osg::Uniform* getCircleNormalUniform() const { return m_circleNormalUniform; }

    osg::Vec3Array* getVertexArray() { return m_vertexArray; }
    osg::Vec4Array* getQuatsInColorArray() { return m_quatsInColorArray; }
    osg::FloatArray* getRadiusArray() { return m_radiusArray; }
    void setNumActiveElements(size_t numActiveElements) { m_numActiveElements = numActiveElements; }

    void setUseRedShiftScale(bool);

  protected:
    virtual ~SplineShader();

  protected:
    size_t              m_numActiveElements;

    osg::DrawArrays*    m_primitiveSet;
    osg::Vec3Array*     m_vertexArray;
    osg::FloatArray*    m_xTexCoordArray;
    osg::Vec4Array*     m_quatsInColorArray;
    osg::FloatArray*    m_radiusArray;
    osg::FloatArray*    m_redShiftArray;
    osg::Program*       m_program;
    osg::ref_ptr<osg::Uniform>       m_circleNormalUniform;
    osg::ref_ptr<osg::Uniform>       m_circleResolutionUniform;
    osg::ref_ptr<osg::Uniform>       m_cylinderColorUniform;
    osg::ref_ptr<osg::Uniform>       m_useRedShiftUniform;
  };

  typedef osg::ref_ptr< SplineShader > SplineShaderRef;

  /**
  Class for rendering a spline model
  */
  class AGXOSG_EXPORT SplineRenderer : public agxSDK::StepEventListener
  {
    public:
      /**
      Construct spline (geometry shader) renderer. Update is by default POST_STEP,
      but can be set to STEP_NONE with explicit calls to the update method when
      the control points have been changed.
      \param spline - spline to be used
      \param root - root/parent node
      */
      SplineRenderer( agxUtil::Spline* spline, osg::Group* root );

      /**
      Given the current spline control points, data will be pushed for new update.
      */
      virtual void update();

      /**
      Use this update if this spline has variable radii. Number of values in \p radiiVector must be identical
      to the number of control points in the spline.
      \param radiiVector - radius of each control point (radiiVector.size == spline->getNumPoints())
      */
      virtual void update( const agx::RealVector& radiiVector );

      /**
      Assign spline model.
      \param spline - spline
      */
      void setSplineModel( agxUtil::Spline* spline ) { m_spline = spline; }

      /**
      \return the currently used spline model
      */
      agxUtil::Spline* getSplineModel() { return m_spline; }
      const agxUtil::Spline* getSplineModel() const { return m_spline; }

      /**
      Assign new radius of the cylinder segments.
      \param radius - new radius
      */
      void setRadius( agx::Real radius ) { m_radius = radius; }

      /**
      \return the current radius used for the cylinder segments
      */
      agx::Real getRadius() const { return m_radius; }

      /**
      Assign how many cylinder segments that should be used per unit length.
      \param segmentsPerUnitLength - number of cylinder segments per unit length
      */
      void setNumSegmentsPerUnitLength( agx::Real segmentsPerUnitLength ) { m_numSegmentsPerUnitLength = segmentsPerUnitLength; }

      /**
      \return number of segments per unit length
      */
      agx::Real getNumSegmentsPerUnitLength() const { return m_numSegmentsPerUnitLength; }

      /**
      Assign color to this spline. Default color is white.
      \param color - new color
      */
      void setColor( const agx::Vec4f& color );

      /**
      \return color of the spline
      */
      agx::Vec4f getColor() const;

      virtual void removeNotification();

  protected:
    virtual ~SplineRenderer() {}

    virtual void addNotification();

    void initialize(osg::Group* root);
    virtual void last(const agx::TimeStamp&) { update(); }

    agx::RealVector& getRedShiftScales() { return m_redShiftScales; }
    const agx::RealVector& getRedShiftScales() const { return m_redShiftScales; }

  protected:
    osg::ref_ptr< osg::Group > m_node;
    agxUtil::SplineRef         m_spline;
    SplineShaderRef            m_splineGeometry;

    agx::RealVector            m_redShiftScales; /**< Values from 0 to 1 how red the color of the node is (0 default color, 1 red color, redShiftScale.size == spline->getNumPoints()) */

    agx::Real                  m_radius;
    agx::Real                  m_numSegmentsPerUnitLength;
  };

  typedef agx::ref_ptr< SplineRenderer > SplineRendererRef;
}

