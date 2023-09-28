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


#include <agx/agx.h>
#include <agx/agxPhysics_export.h>
#include <agx/Singleton.h>
#include <agxCollide/agxcollide_vector_types.h>
#include <agx/agx_vector_types.h>
#include <agx/AffineMatrix4x4.h>
#include <agx/Constraint.h>

#include <agxCollide/Convex.h>

#include <agxSDK/StepEventListener.h>
#include <agxSDK/ContactEventListener.h>
#include <agxSDK/Simulation.h>


namespace agx
{
  typedef std::pair< Real, Real > RealPair;
  class RigidBody;
  class Material;
  class ConstraintImplementation;

}

namespace agxSDK {
  class Assembly;
}

namespace agxCollide {
  class Mesh;
}

namespace agxControl
{
  class EventSensor;
}


/**
\namespace agxUtil
\brief The agxUtil namespace contain classes and methods for utility functionality
*/
namespace agxUtil {

  /**
  Enable or disable the collision between all geometries in RigidBody rb1 and RigidBody rb2.
  \param rb1 - First rigid body
  \param rb2 - Second rigid body
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if both bodies have geometries, false if one or both of the bodies does not have any geometries.
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agx::RigidBody *rb1, agx::RigidBody *rb2, bool enable );

  /**
  Enable or disable the collision between all geometries in RigidBody \p body the geometry \p geom.
  \param body - rigid body
  \param geom - Geometry
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if \p body has geometries that was disabled against \p geom
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agx::RigidBody *body, const agxCollide::Geometry *geom, bool enable );

  /**
  Enable or disable the collision between all geometries in RigidBody \p body the geometry \p geom.
  \param geom - Geometry
  \param body - rigid body
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if \p body has geometries that where disabled against \p geom
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( const agxCollide::Geometry *geom, agx::RigidBody *body, bool enable );

  /**
  Enable or disable the collision between all geometries found in Assembly \p g1 and in Assembly \p g2.

  \param a1 - First Assembly containing bodies and geometries
  \param a2 - Second Assembly containing bodies and geometries
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if both assemblies have geometries which could be disabled/enabled for collisions
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agxSDK::Assembly *a1, agxSDK::Assembly *a2, bool enable );

  /**
  Enable or disable the collision between all geometries found in Assembly \p assembly and in RigidBody \p body.

  \param assembly - Assembly containing bodies and geometries
  \param body - Body containing geometries
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if both assembly and body has geometries which could be disabled/enabled for collisions
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agxSDK::Assembly *assembly, agx::RigidBody *body, bool enable );

  /**
  Enable or disable the collision between all geometries found in Assembly \p assembly and in RigidBody \p body.

  \param body - Body containing geometries
  \param assembly - Assembly containing bodies and geometries
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if both assembly and body has geometries which could be disabled/enabled for collisions
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agx::RigidBody *body, agxSDK::Assembly *assembly, bool enable );

  /**
  Enable or disable the collision between all geometries found in Assembly \p assembly and a Geometry \p geo.

  \param assembly - Assembly containing bodies and geometries
  \param geo - Geometry
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if both assembly and body has geometries which could be disabled/enabled for collisions
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agxSDK::Assembly *assembly, agxCollide::Geometry *geo, bool enable );

  /**
  Enable or disable the collision between all geometries found in Assembly \p assembly and a Geometry \p geo.

  \param geo - Geometry
  \param assembly - Assembly containing bodies and geometries
  \param enable - If true, collisions will be enabled, if false, collisions will be disabled.
  \return true if both assembly and body has geometries which could be disabled/enabled for collisions
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions( agxCollide::Geometry *geo, agxSDK::Assembly *assembly, bool enable );

  /**
  Enable or disable all geometries in a RigidBody
  \param body - rigid body body with geometries
  \param enable - If true, geometries will be enabled, if false, collisions will be disabled.
  \return true if body has any geometries that where disabled
  */
  AGXPHYSICS_EXPORT bool setEnableGeometries( agx::RigidBody *body, bool enable );

  /**
  Enable or disable collision for all geometries in a RigidBody
  \param body - rigid body body with geometries
  \param enable - If true, geometries will be enabled, if false, collisions will be disabled.
  \return true if body has any geometries for which the collision where disabled
  */
  AGXPHYSICS_EXPORT bool setEnableCollisions(agx::RigidBody* body, bool enable);


  /**
  Utility function to extract all geometries from an assembly (and its sub-assemblies), including geometries
  found in RigidBodies which are a part of the assembly (and sub-assemblies).
  Geometries of wires (that are added to the assembly) will also be extracted.

  \param geometries - The vector where all the geometries found in the assembly will be placed
  \param assembly - The assembly where to find all geometries.
  \return true if geometries are added to the vector, otherwise false.
  */
  AGXPHYSICS_EXPORT bool extractGeometries( agxCollide::GeometryPtrVector& geometries, agxSDK::Assembly *assembly );

  /**
  Utility function to extract all geometries from a RigidBody

  \param geometries - The vector where all the geometries found in the assembly will be placed
  \param body - The RigidBody where to find all geometries.
  \return true if geometries are added to the vector, otherwise false.
  */
  AGXPHYSICS_EXPORT bool extractGeometries( agxCollide::GeometryPtrVector& geometries, agx::RigidBody *body );


  /**
  Utility function to extract all Rigid Bodies from an assembly (and its sub-assemblies).
  It will not extract bodies that has isPowerlineBody()==true.
  Bodies from wires will also be extracted
  \param bodies - The vector where all the bodies found in the assembly will be placed
  \param assembly - The assembly where to find all bodies.
  \return true if bodies are added to the vector, otherwise false.
  */
  AGXPHYSICS_EXPORT bool extractRigidBodies( agx::RigidBodyPtrVector& bodies, agxSDK::Assembly *assembly );

  /**
  Utility function to extract all constraints from the system
  \param vec - The vector where all the constraints in the system will be added.
  \param system - The system which contains the constraints.
  \return true if constraints were added to the vector, otherwise false.
  */
  AGXPHYSICS_EXPORT bool extractConstraints( agx::ConstraintPtrVector& vec, agx::DynamicsSystem *system );

  /**
  Utility function to extract all geometries from the space into a vector
  \param vec - The vector where all the geometries in the system will be added.
  \param space - The space which contains the geometries.
  \return true if geometries were added to the vector, otherwise false.
  */
  AGXPHYSICS_EXPORT bool extractGeometries( agxCollide::GeometryPtrVector& vec, agxCollide::Space *space );

  /**
  Utility function to  loop through a body and set the material for associated geometry.
  \param body - The body for which all geometries will get a new material
  \param material - The material that will be assigned to all geometries
  \return true if the body had any geometries that could get a material assigned.
  */
  AGXPHYSICS_EXPORT bool setBodyMaterial( agx::RigidBody *body, agx::Material *material );

  /**
  Utility function to set motion control for a vector of rigid bodies
  \param bodies - Vector of rigid bodies that will be modified.
  \param motionControl - The desired motion control that all the \p bodies will get.
  */
  AGXPHYSICS_EXPORT void setMotionControl( agx::RigidBodyPtrVector& bodies, agx::RigidBody::MotionControl motionControl );

  /**
  \p body will get the parents velocity added to its velocity as if it was rigidly attached.

  \param parentBody - Body for which all of the other bodies velocities will be matched.
  \param body - this body will get a new velocity that equals to its current velocity+parent velocity
  */
  AGXPHYSICS_EXPORT void addParentVelocity( const agx::RigidBody *parentBody, agx::RigidBody *body );

  /**
  For the specified \p body, add the collision group ID to its geometries.
  \param body - The body for which its geometries will have the \p id added.
  \param id - The group id to be added
  */
  AGXPHYSICS_EXPORT void addGroup( const agx::RigidBody *body, unsigned id );

  /**
  For the specified \p body, remove the collision group ID to its geometries.
  \param body - The body for which its geometries will have the \p id removed.
  \param id - The group id to be removed
  */
  AGXPHYSICS_EXPORT void removeGroup( const agx::RigidBody *body, unsigned id );

  /**
  Calculates transform that can be used for cylinders/capsules like geometries. E.g.;
  aCylinder = createCylinder( length, radius )
  aCylinder->setTransform( calculateCylinderTransform( startPoint, endPoint ) )
  where 'length' has to be length( startPoint - endPoint ).
  */
  AGXPHYSICS_EXPORT agx::AffineMatrix4x4 calculateCylinderTransform( const agx::Vec3& startPoint, const agx::Vec3& endPoint );


  /**
  Utillity function for creating a "sink" on a geometry that will remove rigid bodies
  and particles that come into contact with it. The sink is created by placing an
  agxControl::EventSensor on the geometry with operations that remove particles and bodies.
  \param geometry - the specified geometry to place the sink on.
  \param onlyIncludeEmittedBodies - true if only rigid bodies that have been emitted should
                                    be removed, false otherwise. Note that particles will
                                    always be removed.
  \note "emitted" means that the agx::RigidBodyEmitter::rigidBodyIsEmitted(body) returns true.
  \return the EventSensor object that contains the operations for removing bodies and particles.
  */
  AGXPHYSICS_EXPORT agxControl::EventSensor* createSinkOnGeometry(agxCollide::Geometry* geometry,
                                                                  bool onlyIncludeEmittedBodies);

  /**
  Extract vertices and indices from \p mesh.
  \param mesh - mesh with data to extract
  \param[out] vertices - vertices
  \param[out] indices - indices
  \return number of triangles added
  */
  AGXPHYSICS_EXPORT size_t getMeshData( const agxCollide::Mesh* mesh, agx::Vec3Vector& vertices, agx::UInt32Vector& indices );

  /**
  Algorithm for determining if a mesh is convex.

  For each triangle face, project all vertices on the triangle plane. If any vertex
  is above the plane with respect to some threshold value, we have detected a concavity and
  the mesh is considered non-convex.

  \param mesh - the mesh to test.
  \param threshold - the minimum distance that a vertex can be above any triangle plane. ( Default: 1E-5 )
  \return true if the mesh is convex within the specified threshold, false otherwise.
  */
  AGXPHYSICS_EXPORT bool isConvexMesh( const agxCollide::Mesh* mesh, agx::Real threshold=1E-5 );

  /**
  Get all contact materials from the material manager and add them to the supplied vector
  \return number of extracted contact materials
  */
  AGXPHYSICS_EXPORT size_t getContactMaterialVector(agxSDK::MaterialManager* mgr, agx::ContactMaterialPtrVector& contactMaterials);

  /**
  Get all  materials from the material manager and add them to the supplied vector
  \return number of extracted  materials
  */
  AGXPHYSICS_EXPORT size_t getMaterialVector(agxSDK::MaterialManager* mgr, agx::MaterialPtrVector& materials);

  /**
  Vector transform from rigid body \p fromBody to rigid body \p toBody. If a rigid body is null
  that body is assumed to be world (e.g, transformVectorFromTo( null, null, vec ) == vec).
  \param vec - vector in \p fromBody frame
  \param fromBody - rigid body to transform from
  \param toBody - rigid body to transform to
  */
  AGXPHYSICS_EXPORT agx::Vec3 transformVectorFromTo( const agx::Vec3 vec, const agx::RigidBody* fromBody, const agx::RigidBody* toBody );

  /**
  Point transform from rigid body \p fromBody to rigid body \p toBody. If a rigid body is null
  that body is assumed to be world (e.g, transformVectorFromTo( null, null, point ) == point).
  \param point - point in \p fromBody frame
  \param fromBody - rigid body to transform from
  \param toBody - rigid body to transform to
  */
  AGXPHYSICS_EXPORT agx::Vec3 transformPointFromTo( const agx::Vec3 point, const agx::RigidBody* fromBody, const agx::RigidBody* toBody );

  /**
  Jump request options for wire nodes which parent isn't the given assembly/collection.
  */
  struct JumpRequestWireOption
  {
    enum Option : agx::UInt32
    {
      DETACH_CONTACT_NODES = 1 << 0, /**< Contact nodes which parent isn't included in the assembly will become lumped nodes instead. */
      DETACH_EYE_NODES     = 1 << 1  /**< Eye nodes which parent isn't included in the assembly will become lumped nodes instead. */
    };
  };

  /**
  This method will perform a "jump request" will all bodies in the assembly, including bodies which are part of wires.
  \p parentBody is the main body which will work as a parent, all other bodies will move relative to this rigid body.
  \p the transform parentBodyWorldTransform defines the new transform for the parent body, the target transform for \p parentBody.
  Also, make sure \p parentBody is part of the assembly \p collection
  \param collection - An Assembly which contain all bodies (including parentBody) and wires.
  \param parentBody - The main rigid body, all other bodies will move relative to this body
  \param parentBodyWorldTransform - The new requested transformation for the parentBody.
  \param wireOptions - options how to handle wire nodes
  \return the number of bodies that where moved, including parentBody
  */
  AGXPHYSICS_EXPORT size_t jumpRequest( agxSDK::Assembly* collection,
                                        agx::RigidBody* parentBody,
                                        const agx::AffineMatrix4x4& parentBodyWorldTransform,
                                        agx::UInt32 wireOptions = agx::UInt32( 0 ) );


  /**
  Less than operator to e.g., sort functions.
  */
  template< typename T >
  struct LessThanPred
  {
    agx::Bool operator() ( const T& o1, const T& o2 ) { return o1 < o2; }
  };

  /**
  Larger than operator to e.g., sort functions.
  */
  template< typename T >
  struct LargerThanPred
  {
    agx::Bool operator() ( const T& o1, const T& o2 ) { return o1 > o2; }
  };

  /**
  Insertion sort, supports std::valarray.
  \param container - container to sort
  \param pred - Predicator used for comparison
  */
  template< typename ContainerT, typename Pred >
  void insertionSort( ContainerT& container, Pred pred )
  {
    int size = (int)container.size();
    if ( size < 2 )
      return;

    for ( int i = 1; i < size; ++i ) {
      typename ContainerT::value_type tmp = container[ i ];
      int j = 0;
      for ( j = i; j > 0; --j )
        if ( pred( tmp, container[ j - 1 ] ) )
          container[ j ] = container[ j - 1 ];
        else break;
      container[ j ] = tmp;
    }
  }

  /**
  Insertion sort, supports std::valarray.
  \param container - container to sort
  */
  template< typename ContainerT >
  void insertionSort( ContainerT& container )
  {
    insertionSort( container, LessThanPred< typename ContainerT::value_type >() );
  }

  /**
  std::valarray friendly is sorted function.
  \return true if container is sorted
  */
  template< typename ContainerT >
  agx::Bool isSorted( const ContainerT& container )
  {
    if ( container.size() < 2 )
      return true;

    for ( size_t i = 0; i < container.size() - 1; ++i )
      if ( container[ i + 1 ] < container[ i ] )
        return false;
    return true;
  }


  /**
  Free the memory held by the given container by swapping it with a
  default-constructed instance.

  This is required when an application is using a different system allocator
  than the AGX Dynamics shared library and a container is passed by-value
  from AGX Dynamics to the application. When the container goes out of scope
  the application's system allocator will try to deallocate the container's
  buffer, a buffer that was allocated with AGX Dynamics' system allocator.
  This will either cause memory errors or a crash. By first calling
  freeContainerMemory the buffer will be freed inside AGX Dynamics, with
  AGX Dynamics' system allocator, and the application's system allocator will
  not need to do anything.
  */
  template<typename ContainerT>
  void freeContainerMemory(ContainerT& container);

  /**
  Returns a copy of the passed container.
  This is useful in situations where an application that is using a different system
  allocator than the AGX Dynamics shared library needs to obtain a copy of a container
  that was allocated by AGX Dynamics. One example being where a container is created
  in the application and passed-by-value to some AGX Dynamics function. In this case
  the container will be allocated using the application system allocator, and destroyed
  by AGX Dynamics, which may cause runtime crashes. Obtaining a copy of the container
  through this function before passing it to AGX Dynamics will fix that issue.


  ** Example usage in application with different system allocator than AGX Dynamics **

  const agx::String path = "path/to/a/service/license/file/agx.lfx";

  // The below function call may produce a runtime crash since the copy passed to
  // loadLicenseFile() (passed by value) is allocated using the application system allocator,
  // and is destroyed inside AGX Dynamics.
  // agx::Runtime::instance()->loadLicenseFile(path); // <- may produce runtime crash.

  // Instead, we can pass a temporary agx::String object allocated by AGX Dynamics that will
  // be moved from, thus the string is both allocated and deallocated by AGX Dynamics, avoiding
  // the runtime crash.
  agx::Runtime::instance()->loadLicenseFile(agxUtil::copyContainerMemory(path));
  */
  template<typename ContainerT>
  ContainerT copyContainerMemory(const ContainerT& container);

  /**
  Utility class to hold custom implementations of constraints. Add this constraint to
  a simulation then by adding implementation to this holder the implementations
  makes it to the solver.
  */
  class AGXPHYSICS_EXPORT ConstraintHolder : public agx::Constraint
  {
    public:
      ConstraintHolder();

      /**
      Add 'interface less' constraint implementation. This constraint will
      be added to the solver.
      \param constraint - constraint to add
      \return true if add were successful - otherwise false
      */
      bool add( agx::ConstraintImplementation* constraint );

      /**
      Remove constraint implementation.
      \param constraint - constraint to remove
      \return true if the constraint was actually removed - otherwise false
      */
      bool remove( agx::ConstraintImplementation* constraint );

      /**
      Remove all constraints
      */
      void clear();

    protected:
      virtual ~ConstraintHolder();

      virtual int getNumDOF() const override
      {
        return 0;
      }
      virtual void render( class agxRender::RenderManager* , float /*scale = 1.0f */ ) const override
      {}

      AGXSTREAM_DECLARE_SERIALIZABLE( agxUtil::ConstraintHolder );

    private:
      class ConstraintHolderImplementation* m_implementation;
  };

  typedef agx::ref_ptr< ConstraintHolder > ConstraintHolderRef;

  /**
  Utility class to get step event callbacks to any class having the methods implemented.
  Example:
    class Foo
    {
      public:
        Foo();

        void preCollide( const agx::TimeStamp& );
        void pre( const agx::TimeStamp& );
        void post( const agx::TimeStamp& );

      private:
        agx::ref_ptr< agxUtil::GeneralStepListener< Foo > > m_stepCallback;
    };
  m_stepCallback has to be added to the simulation in some way.
  */
  template< typename T >
  class GeneralStepListener : public agxSDK::StepEventListener
  {
    public:
      GeneralStepListener( T* obj, int mask = agxSDK::StepEventListener::DEFAULT )
        : agxSDK::StepEventListener( mask ), m_obj( obj ) {}

    protected:
      virtual ~GeneralStepListener() {}

      DOXYGEN_START_INTERNAL_BLOCK()
#define callback_implementation( callback_name )            \
      virtual void callback_name( const agx::TimeStamp& t ) \
      {                                                     \
        if ( !m_obj.isValid() ) {                           \
          getSimulation()->remove( this );                  \
          return;                                           \
        }                                                   \
        m_obj->callback_name( t );                          \
      }

      callback_implementation( preCollide )
      callback_implementation( pre )
      callback_implementation( post )
#undef callback_implementation
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      agx::observer_ptr< T > m_obj;
  };

  /**
  Utility class to get contact event callbacks to any class having the methods implemented.
  Example:
    class Foo
    {
      public:
        Foo();

        agxSDK::ContactEventListener::KeepContactPolicy impact( const agx::TimeStamp&, agxCollide:GeometryContact* );
        agxSDK::ContactEventListener::KeepContactPolicy contact( const agx::TimeStamp&, agxCollide:GeometryContact* );
        void                                            separation( const agx::TimeStamp&, agxCollide:GeometryPair& );

      private:
        agx::ref_ptr< agxUtil::GeneralContactEventListener< Foo > > m_contactCallback;
    };
  m_contactCallback has to be added to the simulation in some way.
  */
  template< typename T >
  class GeneralContactEventListener : public agxSDK::ContactEventListener
  {
    public:
      GeneralContactEventListener( T* obj, int mask = agxSDK::ContactEventListener::DEFAULT )
        : agxSDK::ContactEventListener( mask ), m_obj( obj ) {}

    protected:
      virtual ~GeneralContactEventListener() {}

#define callback_implementation( callback_name, ArgT )                                                           \
      virtual agxSDK::ContactEventListener::KeepContactPolicy callback_name( const agx::TimeStamp& t, ArgT obj ) \
      {                                                                                                          \
        if ( !m_obj.isValid() ) {                                                                                \
          getSimulation()->remove( this );                                                                       \
          return agxSDK::ContactEventListener::KEEP_CONTACT;                                                     \
        }                                                                                                        \
        return m_obj->callback_name( t, obj );                                                                   \
      }

      callback_implementation( impact,  agxCollide::GeometryContact* )
      callback_implementation( contact, agxCollide::GeometryContact* )
#undef callback_implementation

      virtual void separation( const agx::TimeStamp& t, agxCollide::GeometryPair& gp )
      {
        if ( !m_obj.isValid() ) {
          getSimulation()->remove( this );
          return;
        }

        m_obj->separation( t, gp );
      }

    protected:
      agx::observer_ptr< T > m_obj;
  };



  class AGXPHYSICS_EXPORT SceneRoot : public agx::Singleton
  {
    public:
      static SceneRoot* object();

      virtual bool createVisual( agx::RigidBody* /*rb*/, float /*detailRation*/ = 0.30f ) { return false; };
      virtual bool createVisual( agxCollide::Geometry* /*geometry*/, float /*detailRation*/ = 0.30f ) { return false; }

      static void setInstance( SceneRoot* baseInstance );

      SINGLETON_CLASSNAME_METHOD();

    protected:
      virtual ~SceneRoot() {}

    protected:
      static SceneRoot* s_baseInstance;
  };



}

