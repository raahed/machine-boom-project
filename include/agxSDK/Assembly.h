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
#include <agx/RigidBody.h>
#include <agx/MergedBody.h>
#include <agxCollide/agxCollide.h>
#include <agxSDK/agxSDK.h>

#include <agx/Frame.h>
#include <agxSDK/EventListener.h>
#include <agx/Constraint.h>
#include <agxCollide/Geometry.h>
#include <agx/ObserverFrame.h>
#include <agx/ParticleSystem.h>
#include <agxStream/Serializable.h>
#include <agx/SetVector.h>
#include <agx/DynamicsSystem.h>

namespace agxSDK
{
  typedef agx::SetVector< agx::ref_ptr<agxSDK::Assembly> > AssemblyRefSetVector;

  class AssemblyVisitor;

  AGX_DECLARE_POINTER_TYPES(Assembly);
  /**
  An assembly is a collection of basic simulation objects, such as rigid bodies,
  constraints, geometries. The assembly may also contain other assemblies,
  which enables a hierarchical structuring of the simulation, allowing
  manipulation at different conceptual levels.

  The assembly has a local reference frame which is parent to the transformations
  of all contained bodies and sub-assemblies. When an object is added to an
  assembly, it is disconnected from any previous parent frames and is added as
  a child of the assembly.

  The assembly tree can be traversed using an AssemblyVisitor.

  Note that even though wires (agxWire::Wire) can be added to the assembly, it will
  not follow transformations or velocity operations such as:

  assembly->add(wire);
  assembly->setPosition(pos); // Wire will not be affected
  assembly->getRigidBodies(); // Bodies in wires are not included
  assembly->getGeometries(); // Geometries in wires are not included.

  To include wires you need to write an AssemblyVisitor that include the wires.
  One such example is the class agxUtil::CollectBodiesAndWiresVisitor
  */
  class AGXPHYSICS_EXPORT Assembly : public virtual agx::Referenced, public virtual agxStream::Serializable
  {
  public:
    /// Default constructor
    Assembly();

    /**
    Add a constraint to the assembly.
    \return true when the constraint has been successfully added - otherwise false (constraint
            is nullptr or already present in this assembly)
    */
    bool add(agx::Constraint* constraint);

    /**
    Remove a constraint from this assembly (does not recurse down in tree).
    \return true when the constraint has been successfully removed - otherwise
            false (constraint is nullptr or not in this assembly)
    */
    bool remove(agx::Constraint* constraint);

    /**
    Add an interaction to this assembly.
    \return true when the interaction has been successfully added - otherwise false (interaction
            is nullptr or already present in this assembly)
    */
    bool add(agx::Interaction* interaction);

    /**
    Remove an interaction from this assembly.
    \return true when the interaction has been successfully removed - otherwise false (interaction
            is nullptr or not in this assembly)
    */
    bool remove(agx::Interaction* interaction);

    /**
    Add a listener to the assembly.
    \return true when the listener has been successfully added - otherwise false (listener
            is nullptr or already present in this assembly)
    */
    bool add(agxSDK::EventListener* listener);

    /**
    Remove an EventListener from the assembly.
    \return true when the listener has been successfully removed - otherwise false (listener
            is nullptr or not in this assembly)
    */
    bool remove(EventListener* listener);

    /**
    Add a rigid body to the assembly.
    \return true when the rigid body has been successfully added - otherwise false (the rigid body
            is nullptr or already present in this assembly)
    */
    bool add(agx::RigidBody* body);

    /**
    Remove a rigid body from this assembly (does not recurse down in tree).
    \param body - The body to be removed
    \param removeAssociatedGeometries - If true, the geometries associated to this body will also be removed
    \return true if the body been successfully removed
    */
    bool remove(agx::RigidBody* body, bool removeAssociatedGeometries = false);

    /**
    Add a geometry to the assembly.
    \return true when the geometry has been successfully added - otherwise false (the geometry
            is nullptr or already present in this assembly)
    */
    bool add(agxCollide::Geometry* geometry);

    /**
    Remove a geometry from this assembly (does not recurse down in tree).
    \return true when the geometry has been successfully removed - otherwise false (geometry
            is nullptr or not in this assembly)
    */
    bool remove(agxCollide::Geometry* geometry);

    /**
    Add another assembly as child to this assembly.
    \return true when the assembly has been successfully added - otherwise false (the assembly
            is nullptr or already present in this assembly)
    */
    bool add(Assembly* assembly);

    /**
    Remove a child assembly from a parent.
    \return true when the assembly has been successfully removed - otherwise false (assembly
            is nullptr or not in this assembly)
    */
    bool remove(Assembly* assembly);

    /**
    Add a contact material to this assembly
    \return true when the contact material has been successfully added - otherwise false (the
            contact material is nullptr or already present in this assembly)
    */
    bool add(agx::ContactMaterial* contactMaterial);

    /**
    Remove a contact material from this assembly
    \return true when the contact material has been successfully removed - otherwise false (contact
            material is nullptr or not in this assembly)
    */
    bool remove(agx::ContactMaterial* contactMaterial);

    /**
    Add an emitter to this assembly
    \return true when the emitter has been successfully added - otherwise false (the
            emitter is nullptr or already present in this assembly)
    */
    bool add(agx::Emitter* emitter);

    /**
    Remove an emitter from this assembly
    \return true when the emitter has been successfully removed - otherwise false (emitter
            is nullptr or not in this assembly)
    */
    bool remove(agx::Emitter* emitter);

    /**
    Add an ObserverFrame to this assembly
    \return true when the observer frame has been successfully added - otherwise false (the
            observer frame is nullptr or already present in this assembly)
    */
    bool add(agx::ObserverFrame* observerFrame);

    /**
    Remove an ObserverFrame from this assembly
    \return true when the observer frame has been successfully removed - otherwise false (observer frame
            is nullptr or not in this assembly)
    */
    bool remove(agx::ObserverFrame* observerFrame);


    /**
    Add a MergedBody to this assembly
    \return true when the MergedBody has been successfully added - otherwise false (the
            MergedBody is nullptr or already present in this assembly)
    */
    bool add(agx::MergedBody* mergedBody);

    /**
    Remove a MergedBody from this assembly
    \return true when the MergedBody has been successfully removed - otherwise false (MergedBody
            is nullptr or not in this assembly)
    */
    bool remove(agx::MergedBody *mergedBody);

    /**
    Add an object of type serialized. The object will be dynamically cast to the types supported by add() methods an
    appropriately added to this Assembly.
    \return true when the object has been successfully added - otherwise false (the
    object is nullptr or already present in this assembly)
    */
    bool add(agxStream::Serializable* object);


    /////////////////////////////////////////////////

    /**
    Find (linear search) and return named RigidBody
    \param name - name of the body to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found RigidBody body, null if not found
    */
    agx::RigidBody* getRigidBody(const agx::Name& name, bool recursive = true);

    /**
    Find (linear search) and return named RigidBody
    \param name - name of the body to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found RigidBody body, null if not found
    */
    const agx::RigidBody* getRigidBody(const agx::Name& name, bool recursive = true) const;

    /**
    Find (linear search) and return a pointer to a RigidBody with the given uuid
    \param uuid - uuid of the requested rigidbody
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found RigidBody, null if not found
    */
    agx::RigidBody* getRigidBody(const agx::Uuid& uuid, bool recursive = true);

    /**
    Find (linear search) and return a pointer to a RigidBody with the given uuid
    \param uuid - uuid of the requested rigidbody
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found RigidBody, null if not found
    */
    const agx::RigidBody* getRigidBody(const agx::Uuid& uuid, bool recursive = true) const;

    /**
    Find (linear search) and return a pointer to a Constraint with the given uuid
    \param uuid - uuid of the requested constraint
    \param recursive - if true (default) the search will include sub-assemblies
    \return const pointer to the found constraint, null if not found
    */
    const agx::Constraint* getConstraint(const agx::Uuid& uuid, bool recursive = true) const;

    /**
    Find and return a pointer to a Constraint with the given uuid
    \param uuid - uuid of the requested constraint
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Constraint, null if not found
    */
    agx::Constraint* getConstraint(const agx::Uuid& uuid, bool recursive = true);

    /**
    Find (linear search) and return named Constraint
    \param name - name of the Constraint to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Constraint, null if not found
    */
    const agx::Constraint* getConstraint(const agx::Name& name, bool recursive = true) const;

    /**
    Find (linear search) and return named Constraint
    \param name - name of the Constraint to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Constraint, null if not found
    */
    agx::Constraint* getConstraint(const agx::Name& name, bool recursive = true);

    /**
    Find (linear search) and return named Assembly
    \param name - name of the Assembly to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Assembly, null if not found
    */
    const agxSDK::Assembly* getAssembly(const agx::Name& name, bool recursive = true) const;

    /**
    Find (linear search) and return named Assembly
    \param name - name of the Assembly to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Assembly, null if not found
    */
    agxSDK::Assembly* getAssembly(const agx::Name& name, bool recursive = true);

    /**
    Find (linear search) and return an Assembly matching the given uuid
    \param uuid - uuid of the Assembly to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Assembly, null if not found
    */
    const agxSDK::Assembly* getAssembly(const agx::Uuid& uuid, bool recursive = true) const;

    /**
    Find (linear search) and return an Assembly matching the given uuid
    \param uuid - uuid of the Assembly to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found Assembly, null if not found
    */
    agxSDK::Assembly* getAssembly(const agx::Uuid& uuid, bool recursive = true);

    /**
    Find and return named Constraint of a template type, for example Hinge:

    agx::Hinge *hinge = simulation->getConstraint<agx::Hinge>("FrontWheelHinge");

    \param name - name of the Constraint to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return typed pointer to the found Constraint, null if not found
    */
    template < typename T >
    T* getConstraint(const agx::Name& name, bool recursive = true);

    /**
    Find and return named Constraint of a template type, for example Hinge:

    const agx::Hinge* hinge = simulation->getConstraint<agx::Hinge>("FrontWheelHinge");

    \param name - name of the Constraint to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return typed pointer to the found Constraint, null if not found
    */
    template < typename T >
    const T* getConstraint(const agx::Name& name, bool recursive = true) const;

    /**
    Find (linear search) and return named collision Geometry
    \param name - name of the collision Geometry to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found collision Geometry, null if not found
    */
    const agxCollide::Geometry* getGeometry(const agx::Name& name, bool recursive = true) const;

    /**
    Find (linear search) and return named collision Geometry
    \param name - name of the collision Geometry to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found collision Geometry, null if not found
    */
    agxCollide::Geometry* getGeometry(const agx::Name& name, bool recursive = true);

    /**
    Find (linear search) and return a Geometry matching the given uuid
    \param uuid - uuid of the collision Geometry to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found collision Geometry, null if not found
    */
    const agxCollide::Geometry* getGeometry(const agx::Uuid& uuid, bool recursive = true) const;

    /**
    Find (linear search) and return a Geometry matching the given uuid
    \param uuid - uuid of the collision Geometry to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found collision Geometry, null if not found
    */
    agxCollide::Geometry* getGeometry(const agx::Uuid& uuid, bool recursive = true);


    /**
    Find (linear search) and return an ObserverFrame matching the given uuid
    \param name - name of the ObserverFrame to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found ObserverFrame, null if not found
    */
    const agx::ObserverFrame* getObserverFrame(const agx::Name& name, bool recursive = true) const;

    /**
    Find (linear search) and return an ObserverFrame matching the given uuid
    \param name - name of the ObserverFrame to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found ObserverFrame, null if not found
    */
    agx::ObserverFrame* getObserverFrame(const agx::Name& name, bool recursive = true);

    /**
    Find (linear search) and return an ObserverFrame matching the given uuid
    \param uuid - uuid of the ObserverFrame to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found ObserverFrame, null if not found
    */
    const agx::ObserverFrame* getObserverFrame(const agx::Uuid& uuid, bool recursive = true) const;

    /**
    Find (linear search) and return an ObserverFrame matching the given uuid
    \param uuid - uuid of the ObserverFrame to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found ObserverFrame, null if not found
    */
    agx::ObserverFrame* getObserverFrame(const agx::Uuid& uuid, bool recursive = true);

    /**
    Find (linear search) and return an MergedBody matching the given uuid
    \param uuid - uuid of the MergedBody to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found MergedBody, null if not found
    */
    const agx::MergedBody* getMergedBody (const agx::Uuid& uuid, bool recursive = true) const;

    /**
    Find (linear search) and return an MergedBody matching the given uuid
    \param uuid - uuid of the MergedBody to find
    \param recursive - if true (default) the search will include sub-assemblies
    \return pointer to the found MergedBody, null if not found
    */
    agx::MergedBody* getMergedBody(const agx::Uuid& uuid, bool recursive = true);

    /////////////////////////////////////////////////

    /**
    Traverse the assembly tree recursively using a visitor.
    */
    virtual void traverse(AssemblyVisitor* visitor);
    virtual void traverse(const AssemblyVisitor* visitor) const;

    /**
    \return a const reference to the RigidBodies in this Assembly (not tracing recursively into sub-assemblies)
    */
    const agx::RigidBodyRefSetVector& getRigidBodies() const;
    /**
    \return a const reference to the Geometries in this Assembly (not tracing recursively into sub-assemblies)
    */
    const agxCollide::GeometryRefSetVector& getGeometries() const;

    /**
    \return a const reference to the EventListeners in this Assembly (not tracing recursively into sub-assemblies)
    */
    const EventListenerRefSetVector& getEventListeners() const;

    /**
    \return a const reference to the Assemblies in this Assembly (not tracing recursively into sub-assemblies)
    */
    const AssemblyRefSetVector& getAssemblies() const;

    /**
    \return a const reference to the ParticleSystems in this Assembly (not tracing recursively into sub-assemblies)
    */
    const agx::ParticleSystemRefSetVector& getParticleSystems() const;

    /**
    \return a const reference to the Emitters in this Assembly (not tracing recursively into sub-assemblies)
    */
    const agx::EmitterRefSetVector& getEmitters() const;

    /**
    \return a const reference to the Constraints in this Assembly (not tracing recursively into sub-assemblies)
    */
    const agx::ConstraintRefSetVector& getConstraints() const;

    /**
    \return interactions in this assembly (not tracing recursively into sub-assemblies)
    */
    const agx::InteractionRefSetVector& getInteractions() const;

    /**
    \return observers in this assembly (not tracing recursively into sub-assemblies)
    */
    const agx::ObserverFrameRefSetVector& getObserverFrames() const;

    /**
    \return MergedBodies in this assembly (not tracing recursively into sub-assemblies)
    */
    const agx::MergedBodyRefSetVector& getMergedBodies() const;

    /**
    Return a reference to the frame containing transformation and velocity information
    for this assembly.
    */
    virtual agx::Frame* getFrame();
    virtual const agx::Frame* getFrame() const;

    /**
    \return the parent assembly if this assembly is part of another assembly.
    */
    Assembly* getParent();

    /**
    \return the parent assembly if this assembly is part of another assembly.
    */
    const Assembly* getParent() const;

    /**
    Set the transform of the assembly. Its frame will move to the specified transform,
    which is given in world coordinates.
    \param matrix - desired transform for the frame in world coordinates.
    */
    void setTransform( const agx::AffineMatrix4x4& matrix );

    /**
    \return - the local transformation matrix of the assembly's frame, relative to the parent's frame.
    */
    const agx::AffineMatrix4x4& getLocalTransform() const;

    /**
    Assign the local transformation matrix for this assembly, ignoring any eventual parent transformation.
    \param matrix - transformation matrix relative to parent transform for the assembly's frame
    */
    void setLocalTransform( const agx::AffineMatrix4x4& matrix );

    /**
    \return - the relative translate to the parent frame of the assembly's frame.
    */
    agx::Vec3 getLocalPosition() const;

    /**
    Set the position of the assembly relative to its frame's parent frame.
    \param p - local translate of assembly as a 3D vector
    */
    void setLocalPosition( const agx::Vec3& p );

    /**
    Set the position of the assembly relative to its frame's parent frame.
    \param x - local x translate
    \param y - local y translate
    \param z - local z translate
    */
    void setLocalPosition( agx::Real x, agx::Real y, agx::Real z );

    /**
    Set the rotation of the assembly relative to world frame.
    \param q - rotation given as a quaternion
    */
    void setRotation( const agx::Quat& q );

    /**
    Set the rotation of the assembly relative to world frame.
    \param e - rotation given as Euler angles
    */
    void setRotation( const agx::EulerAngles& e);

    /**
    Set the rotation of the assembly relative to world frame.
    \param m - rotation given as an orthogonal transformation matrix
    */
    void setRotation( const agx::OrthoMatrix3x3& m );

    /**
    \return - the assembly's rotation relative to its frame's parent frame.
    */
    agx::Quat getLocalRotation() const;

    /**
    Set the rotation of the assembly relative to its frame's parent frame.
    \param q - rotation given as a quaternion
    */
    void setLocalRotation( const agx::Quat& q );

    /**
    Set the rotation of the assembly relative to its frame's parent frame.
    \param e - rotation given as Euler angles
    */
    void setLocalRotation( const agx::EulerAngles& e );

    /**
    \return - the parent frame of the assembly's frame, or 0 if this frame has no parent
    */
    agx::Frame* getParentFrame();

    /**
    \return - the parent frame of the assembly's frame, or 0 if this frame has no parent
    */
    const agx::Frame* getParentFrame() const;

    /**
    Set the parent frame of this assembly's frame.

    This means that getLocalTranslate, getLocalRotate, getLocalTransform will be given
    in the parents coordinate frame. I.e., this frame's transformation
    will be concatenated with the parents.
    \param frame - new parent frame, 0 to remove parent
    \return true if parent is changed (not the same as before) - otherwise false
    */
    bool setParentFrame( agx::Frame* frame );

    /**
    Set the position of the frame in world coordinates.
    \param p - desired position in world coordinates.
    */
    void setPosition( const agx::Vec3& p );

    /**
    Set the position of the frame in world coordinates.
    \param x - desired x-coordinate in world frame
    \param y - desired y-coordinate in world frame
    \param z - desired z-coordinate in world frame
    */
    void setPosition( agx::Real x, agx::Real y, agx::Real z );

    AGX_ADD_FRAME_TRANSFORM_INTERFACE()

    /**
    Called when this assembly is added to a simulation
    (given that it not already is in the simulation).
    */
    virtual void addNotification(agxSDK::Simulation*) {}

    /**
    Called when this assembly is removed from a simulation
    */
    virtual void removeNotification(agxSDK::Simulation*) {}

    /**
    Called when this assembly is added to another assembly
    */
    virtual void addNotification(Assembly*) {}

    /**
    Called when this assembly is removed from another assembly
    */
    virtual void removeNotification(Assembly*) { }

    /// \return the simulation this assembly is added to. nullptr if none.
    agxSDK::Simulation* getSimulation();

    /// \return the simulation this assembly is added to. nullptr if none.
    const agxSDK::Simulation* getSimulation() const;

    /**
    Set the linear velocity (in world frame) recursively for all rigid bodies stored in assembly.
    \param velocity - The linear velocity in world frame that will be set for all rigid bodies.
    */
    virtual void setVelocity(const agx::Vec3& velocity);

    /**
    Set the angular velocity (in world frame) recursively for all rigid bodies stored in assembly.
    \param velocity - The angular velocity in world frame that will be set for all rigid bodies.
    */
    virtual void setAngularVelocity(const agx::Vec3& velocity);

    /// Specifies the type an assembly is
    enum Type {
      ASSEMBLY,  /**< This is a full fledged Assembly which uses Frame for the hierarchy */
      COLLECTION /**< This is only a collection of objects NOT using Frame for the hierarchy */
    };

    /// \return the type of this Assembly/Collection
    Type getType() const;

    /**
    Set the name of this Assembly.
    \param name - new name of this Assembly.
    */
    void setName(const agx::Name& name);

    /// \return the name of this Assembly
    agx::Name getName() const;

    agx::Constraint1DOF* getConstraint1DOF(const agx::Name& name, bool recursive=true) { return this->getConstraint<agx::Constraint1DOF>(name, recursive); }
    agx::Constraint2DOF* getConstraint2DOF(const agx::Name& name, bool recursive=true) { return this->getConstraint<agx::Constraint2DOF>(name, recursive); }

    /// \return a pointer to an Assembly if obj is infact an Assembly
    static Assembly* asAssembly(agxStream::Serializable *obj) { return dynamic_cast<Assembly *>(obj); }

  public:
    AGXSTREAM_DECLARE_SERIALIZABLE(agxSDK::Assembly);

  protected:


    /**
    Add a particle system to this assembly.
    ParticleSystems should only be added to Collections.
    */
    bool addParticleSystem(agx::ParticleSystem* particleSystem);

    /**
    Remove a particle system from this assembly
    */
    bool removeParticleSystem(agx::ParticleSystem* particleSystem);


    Assembly(Type type);

    /// Destructor
    virtual ~Assembly();
    friend class Simulation;

    void addToSimulation(Simulation* simulation);
    void removeFromSimulation(Simulation* simulation);

  private:

    agx::FrameRef m_frame;

    agxCollide::GeometryRefSetVector m_geometries;
    agx::RigidBodyRefSetVector m_bodies;
    agx::ConstraintRefSetVector m_constraints;
    agx::EmitterRefSetVector m_emitters;
    EventListenerRefSetVector m_listeners;
    agxSDK::Simulation* m_simulation;
    agx::InteractionRefSetVector m_interactions;

    agx::ContactMaterialRefSetVector m_contactMaterials;
    agx::ParticleSystemRefSetVector m_particleSystems;

    AssemblyRefSetVector m_assemblies;
    AssemblyObserver m_parent;

    agx::ObserverFrameRefSetVector m_observers;
    agx::MergedBodyRefSetVector m_mergedBodies;

    Type m_type;
    agx::Name m_name;
  };


  inline agx::Frame* Assembly::getFrame()
  {
    return m_frame.get();
  }

  inline const agx::Frame* Assembly::getFrame() const
  {
    return m_frame.get();
  }

  /**
  Class for visiting all elements in a tree of Assemblies.
  For each Body/constraint/geometry/assembly/EventListener a virtual call to visit
  (with the specified type) is performed.

  This class can be used to enable all bodies in a structure/disable collisions between all geometries etc..
  Just derive from this class and implement the visit method that you are interested in using.
  */
  class AGXPHYSICS_EXPORT AssemblyVisitor
  {
  public:

    /// \return true if the visitor succeeded.
    bool succeed() const;

    // Constructor
    AssemblyVisitor();
    virtual ~AssemblyVisitor();

  protected:

    friend class Assembly;
    virtual void visit(agx::Constraint*) {}
    virtual void visit(agx::Interaction*) {}
    virtual void visit(agx::RigidBody*) {}
    virtual void visit(agxCollide::Geometry*) {}
    virtual void visit(Assembly*);
    virtual void visit(EventListener*) {}
    virtual void visit(agx::ContactMaterial*) {}
    virtual void visit(agx::ParticleSystem*) {}
    virtual void visit(agx::Emitter*) {}
    virtual void visit(agx::ObserverFrame*) {}
    virtual void visit(agx::MergedBody*) {}

    virtual void visit(const agx::Constraint*) const {}
    virtual void visit(const agx::Interaction*) const {}
    virtual void visit(const agx::RigidBody*) const {}
    virtual void visit(const agxCollide::Geometry*) const {}
    virtual void visit(const Assembly*)const;
    virtual void visit(const EventListener*) const {}
    virtual void visit(const agx::ContactMaterial*) const {}
    virtual void visit(const agx::ParticleSystem*) const {}
    virtual void visit(const agx::Emitter*) const {}
    virtual void visit(const agx::ObserverFrame*) const {}
    virtual void visit(const agx::MergedBody*) const {}

    void setSucceed(bool flag) {
      m_succeed = flag;
    }

    bool m_succeed;
  };



  /* Implementation */


  template< typename T >
  T* Assembly::getConstraint(const agx::Name& name, bool recursive)
  {
    return dynamic_cast< T* >(this->getConstraint(name, recursive));
  }

  template< typename T >
  const T* Assembly::getConstraint(const agx::Name& name, bool recursive) const
  {
    return dynamic_cast< const T* >(this->getConstraint(name, recursive));
  }


}
