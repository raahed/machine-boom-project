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

#ifndef AGXSDK_SIMULATIONOBJECT_H
#define AGXSDK_SIMULATIONOBJECT_H

#include <agx/agxPhysics_export.h>

#include <agxCollide/Geometry.h>
#include <agx/RigidBody.h>

namespace agxSDK {
  class Simulation;
}


namespace agxSDK
{


  /**
  *Deprecated, used only for internal examples*

  Utility class for storing a body and a geometry
  */
  class AGXPHYSICS_EXPORT SimulationObject
  {
  public:

    /// Constructor for a SimulationObject that will store a reference to a Geometry and a RigidBody.
    SimulationObject( agxCollide::Geometry *geom, agx::RigidBody *body ) : m_geometry( geom ), m_body( body ) {}

    /// Constructor for a SimulationObject that will store a reference to a Geometry and a RigidBody.
    //explicit SimulationObject( agxCollide::GeometryRef geom, agx::RigidBodyRef body ) : m_geometry( geom ), m_body( body ) {}

    explicit SimulationObject(  )  {}

    /// Copy constructor
    SimulationObject( const SimulationObject& copy )  {
      m_geometry = copy.m_geometry;
      m_body = copy.m_body;
    }


    SimulationObject& operator= (const SimulationObject& copy )
    {
      if (this == &copy)
        return *this;

      m_geometry = copy.m_geometry;
      m_body = copy.m_body;
      return *this;
    }

    ~SimulationObject( ) {}

    /// \return the geometry
    agxCollide::Geometry *getGeometry() { return m_geometry.get(); }
    void setGeometry( agxCollide::Geometry *geom ) { m_geometry = geom; }

    /// \return the rigidbody
    agx::RigidBody *getRigidBody() { return m_body.get(); }
    void setRigidBody( agx::RigidBody * body ) { m_body = body; }

    /// Cast operator. \return The contained Rigidbody
    operator agx::RigidBody *() { return m_body.get(); }
    /// Cast operator. \return The contained Geometry
    operator agxCollide::Geometry *() { return m_geometry.get(); }
    /// Cast operator. \return The contained Rigidbody
    operator const agx::RigidBody *() const { return m_body.get(); }
    /// Cast operator. \return The contained Geometry
    operator const agxCollide::Geometry *() const { return m_geometry.get(); }


    // / \return true if obj contains the same geometry and body as this object.
    //     virtual bool operator ==(const SimulationObject& obj)
    //     {
    //       return ( m_geometry.get() == obj.m_geometry.get() && m_body.get() == obj.m_body.get());
    //     }


    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a box including:

    - a agxCollide::Geometry including a agxCollide::Cylinder agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param halfExtent - Half extents of the Box.
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a cylinder shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createBox(  const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::Vec3& halfExtent,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody = true
      );


    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a cylinder including:

    - a agxCollide::Geometry including a agxCollide::Cylinder agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param radius - Radius of the cylinder
    \param height - Height of the cylinder
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a cylinder shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createCylinder(  const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agx::Real radius, agx::Real height,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );

    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a sphere including:

    - a agxCollide::Geometry including a agxCollide::Sphere agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param radius - Radius of the sphere
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createSphere(  const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agx::Real radius,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );


    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a plane including:

    - a agxCollide::Geometry including a agxCollide::Plane agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The plane will point upwards in z direction in its local coordinate system.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createPlane( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::STATIC,
      bool createGeometry=true,
      bool createBody=false
      );

    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a HeightField including:

    - a agxCollide::Geometry including a agxCollide::HeightField agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The HeightField will point upwards in z direction in its local coordinate system.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param filename  - The path to an image file (PNG format).
    \param sizeX, sizeY - The size in world coordinates that the image file/heightfield will be mapped to.
    \param low, high - The range of heights which the data in the image file will be mapped to.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a sphere shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createHeightField( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::String& filename, const agx::Real& sizeX, const agx::Real& sizeY,
      const agx::Real& low, const agx::Real& high,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::STATIC,
      bool createGeometry=true,
      bool createBody=false
      );

    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a HeightField including:

    - a agxCollide::Geometry including a agxCollide::HeightField agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The HeightField will point upwards in z direction in its local coordinate system.

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the Geometry
    \param heights - The heights values in
    \param resolutionX  The number of sample points in dimension x. resolutionX * resolutionY should be heights.size()
    \param resolutionY  The number of sample points in dimension y. resolutionX * resolutionY should be heights.size()
    \param sizeX The physical size in dimension x.
    \param sizeY The physical size in dimension y.
    \param bottomMargin How deep is the HeightField under its lowest point?
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
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
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::STATIC,
      bool createGeometry=true,
      bool createBody=false
      );

    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a capsule including:

    - a agxCollide::Geometry including a agxCollide::Capsule agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param radius - Radius of the capsule
    \param height - height of the capsule
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createCapsule(  const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agx::Real radius, agx::Real height,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );

    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a trimesh including:

    - a agxCollide::Geometry including a agxCollide::Trimesh agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param filename - The name of the file containing the mesh data. Currently, .obj files are supported.
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createTrimesh( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::String& filename,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );


    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a trimesh including:

    - a agxCollide::Geometry including a agxCollide::Trimesh agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param cloneTrimesh - A pointer to a agxCollide::Trimesh that the Geometry data should be reused from. The cloneTrimesh has to be finalized, if not =nullptr.
    \param simulation - A pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createTrimesh( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      agxCollide::Shape* cloneTrimesh,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );

    /**
    *Deprecated, used only for internal examples*

    Utility function for creating a trimesh including:

    - a agxCollide::Geometry including a agxCollide::Trimesh agxCollide::Shape
    - a agx::RigidBody with the specified agx::RigidBody::MotionControl

    The Geometry will be placed at the center of mass of the RigidBody.
    If you want to move the Geometry, relative to the RigidBody COM, just use the Geometry::getFrame::setLocalTranslate

    \param name - The name that the RigidBody and the Geometry will get.
    \param transform - The transformation that will be applied to the RigidBody and the Geometry
    \param filename - The name of the file containing the mesh data. Currently, .obj files are supported.
    \param bottomMargin A safety threshold for catching collisions below the terrain surface.
    \param simulation - a pointer to a agxSDK::Simulation, where the RigidBody and the Geometry will be added, if simulation != nullptr
    \param motionControl - Determines whether the RigidBody should be STATIC, DYNAMICS or KINEMATICS
    \param createGeometry - If true a Geometry including a capsule shape will be created.
    \param createBody - If true, a RigidBody will be created.
    \return a \p SimulationObject with the created Geometry and RigidBody.
    */
    static SimulationObject createTrimesh( const agx::String& name,
      const agx::AffineMatrix4x4& transform,
      const agx::String& filename,
      agx::Real bottomMargin,
      agxSDK::Simulation* simulation,
      agx::RigidBody::MotionControl motionControl=agx::RigidBody::DYNAMICS,
      bool createGeometry=true,
      bool createBody=true
      );


  protected:
    agxCollide::GeometryRef m_geometry;
    agx::RigidBodyRef m_body;
  };

}



#endif

