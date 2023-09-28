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

#ifndef AGXOSG_SIMULATIONOBJECT_H
#define AGXOSG_SIMULATIONOBJECT_H

#include <agxOSG/export.h>

#include <agx/observer_ptr.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Node>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxSDK/SimulationObject.h>
#include <agx/Bound.h>
#include <agx/Material.h>

#define DEFAULT_DETAIL_RATIO 1.0f

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxCollide {
  class Geometry;
  class Trimesh;
}

namespace agx {
  class RigidBody;
}

namespace osg {
  class Geode;
}


namespace agxOSG
{




  /**
  Utility class for storing a agx::RigidBody, agxCollide::Geometry and a osg::Node in the same class.

  *Deprecated, used only for internal examples*

  Makes it easier to create simple functions that create all of the above and return them.
  Also casting operators are overloaded so that this object can be interpretated as a Geometry *, RigidBody *
  depending on component.
  */
  class AGXOSG_EXPORT SimulationObject : public agxSDK::SimulationObject
  {
  public:
    /// Constructor
    explicit SimulationObject( agxCollide::Geometry *geom, agx::RigidBody *body, osg::Node *node=nullptr ) : agxSDK::SimulationObject( geom, body), m_node( node )
    {
      //      LOGGER_WARNING() << "SimulationObject( agxCollide::" << std::endl << LOGGER_END();
    }
    explicit SimulationObject(  )  {
      //      LOGGER_WARNING() << "SimulationObject()" << std::endl << LOGGER_END();
    }

    /// Destructor

    ~SimulationObject() {
      //      LOGGER_WARNING() << "~SimulationObject()" << std::endl << LOGGER_END();
    }

    /// Copy constructor
    SimulationObject( const SimulationObject& copy) : agxSDK::SimulationObject( copy ) {
      m_node = copy.m_node;
      //      LOGGER_WARNING() << "SimulationObject( const SimulationObject& copy) " << std::endl << LOGGER_END();
    }

    SimulationObject& operator= (const SimulationObject& copy )
    {
      if (this == &copy)
        return *this;


      this->m_geometry = copy.m_geometry;
      this->m_body = copy.m_body;
      m_node = copy.m_node;

      return *this;
    }


    osg::Node *getNode() { return m_node.get(); }

    /// Cast operator. \return The contained Node
    operator const osg::Node *() const { return m_node.get(); }
    /// Cast operator. \return The contained Node
    operator osg::Node *() { return m_node.get(); }

    //     virtual bool operator ==(const SimulationObject& obj)
    //     {
    //       return ( m_geometry.get() == obj.m_geometry.get() && m_body.get() == obj.m_body.get() && m_node.get() == obj.m_node.get());
    //     }

    /**
    Utility function for creating a box including:

    \deprecated used only for internal examples*

    - a agxCollide::Geometry including a agxCollide::Box agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param halfExtent - Half size of the box
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a box shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \addAxes - If true, a coordinate axes geometry will be added
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createBox( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::Vec3& halfExtent,
      osg::Group *root,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true,
      bool addAxes=false
      );


    /**

    \deprecated used only for internal examples*

    Utility function for creating a plane including:

    - a agxCollide::Geometry including a agxCollide::Plane agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The plane will point upwards in z direction in its local coordinate system.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createPlane( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::STATIC,
      bool createGeometry=true,
      bool createBody=true
      );

    /**

    \deprecated used only for internal examples*

    Utility function for creating a HeightField including:

    - a agxCollide::Geometry including a agxCollide::HeightField agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The HeightField will point upwards in z direction in its local coordinate system.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param filename  - The path to an image file (PNG format).
    \param sizeX, sizeY - The size in world coordinates that the image file/heightfield will be mapped to.
    \param low, high - The range of heights which the data in the image file will be mapped to.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createHeightField( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::String& filename, const agx::Real& sizeX, const agx::Real& sizeY,
      const agx::Real& low, const agx::Real& high,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::STATIC,
      bool createGeometry=true,
      bool createBody=true
      );

    /**

    \deprecated used only for internal examples*

    Utility function for creating a HeightField including:

    - a agxCollide::Geometry including a agxCollide::HeightField agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The HeightField will point upwards in z direction in its local coordinate system.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the Geometry
    \param heights - The heights values in
    \param resolutionX  The number of sample points in dimension x. resolutionX * resolutionY should be heights.size()
    \param resolutionY  The number of sample points in dimension y. resolutionX * resolutionY should be heights.size()
    \param sizeX The physical size in dimension x.
    \param sizeY The physical size in dimension y.
    \param bottomMargin How deep is the HeightField under its lowest point?
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createHeightField( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::RealVector& heights,
      size_t resolutionX,
      size_t resolutionY,
      agx::Real sizeX,
      agx::Real sizeY,
      agx::Real bottomMargin,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::STATIC,
      bool createGeometry=true,
      bool createBody=true
      );



    /**

    \deprecated used only for internal examples*

    Utility function for creating a cylinder including:

    - a agxCollide::Geometry including a agxCollide::Cylinder agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param radius - Radius of the cylinder
    \param radius - Height of the cylinder
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a cylinder shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \addAxes - If true, a coordinate axes geometry will be added
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createCylinder(  const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agx::Real radius, agx::Real height,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true,
      bool addAxes=false
      );

    /**

    \deprecated used only for internal examples*

    Utility function for creating a capsule including:

    - a agxCollide::Geometry including a agxCollide::Capsule agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param radius - Radius of the capsule
    \param height - height of the capsule
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createCapsule( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agx::Real radius, agx::Real height,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );


    /**

    \deprecated used only for internal examples*

    Utility function for creating a sphere including:

    - a agxCollide::Geometry including a agxCollide::Sphere agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param radius - Radius of the sphere
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createSphere(  const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agx::Real radius,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );

    /**

    *Deprecated, used only for internal examples*
    \deprecated used only for internal examples*

    Utility function for creating a trimesh including:

    - a agxCollide::Geometry including a agxCollide::Trimesh agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param filename - The name of the file containing the mesh data. Currently, .obj files are supported.
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createTrimesh( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::String& filename,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );

    /**

    \deprecated used only for internal examples*

    Utility function for creating a trimesh including:

    - a agxCollide::Geometry including a agxCollide::Trimesh agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param cloneTrimesh - A pointer to a agxCollide::Trimesh that the Geometry data should be reused from.
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createTrimesh( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agxCollide::Trimesh* cloneTrimesh,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );


    /**

    \deprecated used only for internal examples*

    Utility function for creating a trimesh including:

    - a agxCollide::Geometry including a agxCollide::Trimesh agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl
    - A agxOSG::GeometryNode including visual geometry. The GeometryNode will be updated with the transformation
    of the created geometry.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame()::setLocalTranslate()

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param filename - The name of the file containing the mesh data. Currently, .obj files are supported.
    \param bottomMargin A safety threshold for catching collisions below the terrain surface.
    \param root - a pointer to a osg::Group where the visual geometry will be added (if root != nullptr)
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICSor KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createTrimesh( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::String& filename,
      agx::Real bottomMargin,
      osg::Group *root,
      agxSDK::Simulation* simulation=nullptr,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );


    // static agx::ParticleSystem *createParticleSystem(const agx::String& name, osg::Group *root, agxSDK::Simulation* simulation=nullptr);

  protected:
    osg::ref_ptr<osg::Node> m_node;
  };


  AGXOSG_EXPORT agx::RigidBody *buildContainer(const agx::Bound3& bound, agx::Real thickness, agxSDK::Simulation* simulation, osg::Group *root, bool addCeiling = false, agx::Material *material = agx::Material::getDefaultMaterial(), osg::Vec4 color = osg::Vec4(0, 1, 0, 0.1f));

  // Create a osg Geode from a specified shape
  AGXOSG_EXPORT osg::Geode* createGeode( osg::Shape *shape, const agx::String& name, float detailRatio = DEFAULT_DETAIL_RATIO );

}

DOXYGEN_END_INTERNAL_BLOCK()

#endif /* _AGXOSG_SIMULATIONOBJECT_H_ */
