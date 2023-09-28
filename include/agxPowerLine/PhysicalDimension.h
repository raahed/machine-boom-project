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

#ifndef AGXPOWERLINE_PHYSICAL_DIMENSION_H
#define AGXPOWERLINE_PHYSICAL_DIMENSION_H

#include <agxModel/export.h>
#include <agxPowerLine/Sides.h>
#include <agxPowerLine/detail/DimensionState.h>

#include <agxSDK/StepEventListener.h>
#include <agxStream/Serializable.h>

#include <agxUtil/agxUtil.h>

#include <agx/RigidBody.h>
#include <agx/agx_hash_types.h>
#include <agx/ElementaryConstraint.h>
#include <agx/RegularizationParameters.h>

#include <string>



#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4589 ) // Disable: 4589: Constructor of abstract class 'agx::FrictionModel' ignores initializer for virtual base class 'agxStream::Serializable'
#endif


namespace agxPowerLine
{
  namespace detail
  {
    class ConnectionOperations;
    class AbstractDimensionState1Dof;
    class AbstractDimensionState3Dof;
    using PhysicalDimensionStateRef = agx::ref_ptr<DimensionState>;
  }
}

namespace agxPowerLine
{
  class Connection;
  class Connector;
  class Unit;
  class PowerLine;
  class PowerGenerator;
  class LookupTable;
  class PhysicalDimensionMultiBodyConstraintImplementation;

  typedef agx::ref_ptr<PowerGenerator> PowerGeneratorRef;
  typedef agx::ref_ptr< Connection > ConnectionRef;
  typedef agx::ref_ptr< Connector > ConnectorRef;
  typedef agx::observer_ptr<PowerGenerator> PowerGeneratorObserver;
  typedef agx::observer_ptr< Connector > ConnectorObserver;
  typedef agx::ref_ptr< LookupTable > LookupTableRef;
  typedef agx::Vector<ConnectionRef> ConnectionRefVector;
  typedef Connection* ConnectionPtr;
  typedef agx::Vector<ConnectionPtr> ConnectionPtrVector;
  typedef agx::Vector<Unit*> UnitPtrVector;
  typedef agx::HashVector< agx::RigidBody*, int > RigidBodyPtrIntHashVector;


  class PhysicalDimension;
  typedef std::pair<agxPowerLine::PhysicalDimension*, agxPowerLine::Side> DimensionAndSide;

  /**
  Pure virtual class.
  Base class for representation of one physical dimension.
  It contains one 1D body(right now it is a agx::RigidBody that is potentially
  shared with other PhysicalDimensions), that is used in the dynamics solve.

  The dimension could be for example a 1D position or a 1D rotation.
  The dimension has a:
  1. value (position for translational position)
  2. gradient (speed for translational position)
  3. second gradient (acceleration for translational position)
  */
  class AGXMODEL_EXPORT PhysicalDimension : public virtual agx::Referenced, public virtual agxStream::Serializable
  {
    public:
      typedef int Type;
      typedef agx::Vector<Type> TypeVector;

      /**
      Calculates jacobian for a physical dimension.
      \param G - Array of Jacobian elements to be filled in.
      \param ratio - Scaling to apply in order to convert the dimension's unit direction to a constraint Jacobian.
      \param index - Index in G to write to.
      */
      virtual void calculateJacobian(
          agx::Jacobian6DOFElement* G,
          const agx::Real ratio,
          const size_t index) const;

      /**
      Connect to another dimension for possible power transfer.
      \param otherDimension - a physical dimension with a body defining another dof.
      \param connector - defines the constraint between the dimensions
      */
      bool connect(agxPowerLine::PhysicalDimension* otherDimension, agxPowerLine::Connector* connector);



      /**
      Physical properties of the dimension.
      Implement these in child class. (value is for example angle or position. gradient is velocity ...)
      */
      virtual agx::Real getValue() const ;
      virtual void setValue(agx::Real value);
      virtual agx::Real getGradient() const;
      virtual void setGradient( agx::Real value);
      virtual agx::Real getSecondGradient() const;

      /**
      Returns the name of this PhysicalDimension instance.

      There are two types of names for a PhysicalDimension: the instance name
      and the type name. The instance name can be set by the user and the type
      name is an indentifier for a type of PhysicalDimension. For example,
      all RotationalDimensions have the same type name but may have different
      instance names.

      @see getTypeName
      @see getActualTypeName
      @return This PhysicalDimension's name.
      */
      const agx::Name& getName() const;
      void setName(const agx::Name& name);


      /**
      Ensure that this PhysicalDimension does not share it's body with any other
      PhysicalDimension. A new body may be created for this PhysicalDimension in
      order to make that guarantee. Body sharing becomes disabled for this
      PhysicalDimension.
      */
      void reserveBody();

      /**
      Return a body that is guaranteed to not be shared with any other
      PhysicalDimension. Will return nullptr if no such guarantee can be made.

      \see reserveBody
      */
      agx::RigidBody* getReservedBody();
      const agx::RigidBody* getReservedBody() const;

      /**
      Return a body that is guaranteed to not be shared with any other
      PhysicalDimension. May create a new body for this PhysicalDimension.

      \see reserveBody
      \see getReservedBody
      */
      agx::RigidBody* getOrReserveBody();


      /**
      \return True if this PhysicalDimension has a reserved body, false otherwise.
      */
      bool hasReservedBody() const;

      /**
      Returns pointer to the Unit that contains this dimension.
      */
      const agxPowerLine::Unit* getUnit() const;
      agxPowerLine::Unit* getUnit();

      /**
      Return true if the body that defines the dimension was given to the Power-
      Line, false if the body was created by the PowerLine.
      */
      bool isExternalBody() const;

      /**
      returns the mass property of the dimension body.( Inertia for the rotational dimension, mass for translational dimension, ... )
      */
      virtual agx::Real getMassProperty() const;

      /**
      Set the mass property of the dimension.
      */
      virtual void setMassProperty(agx::Real massProperty);


      /**
      return the load in/out
      */
      virtual agx::Real getInputLoad() const;//Torque(rotational dimension), Force(translational dimension)...
      virtual agx::Real getOutputLoad() const;//Torque(rotational dimension), Force(translational dimension)...


      /**
      Add a load to the dimension. The effect depends on the type of the dimension.
      */
      virtual void addLoad(agx::Real load);


      /**
      return the power in/out
      */
      virtual agx::Real getPowerIn() const;
      virtual agx::Real getPowerOut() const;

      /**
      \returns pointer to a constraint going out from this dimension (one constraint for each output connection)
      */
      agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* getConstraint(size_t outputConnection = 0) const;

      /**
      \returns a constraint going into this dimension. It's properties are defined by the parent dimension.
      */
      agxPowerLine::PhysicalDimensionMultiBodyConstraintImplementation* getInputConstraint(size_t inputConnection = 0) const;

      agx::Vec3 getWorldDirection() const;

      void setVelocityDamping(float damping);
      float getVelocityDamping() const;

      /**
      Implement this to render Your physical dimension.
      */
      virtual void renderDebug();

      /**
      Clear all reference pointers
      */
      virtual void clear();

      size_t getNumInputConnections() const;
      size_t getNumOutputConnections() const;
      size_t getNumConnections(agxPowerLine::Side side) const;
      size_t getNumConnections() const;

      /**
      Traversing helper functions. Has to be public. Do NOT use them.
      */
      const agxPowerLine::Connection* getInputConnection(size_t inputIndex = 0) const;
      const agxPowerLine::Connection* getOutputConnection(size_t outputIndex = 0) const;

      Connection* getOutputConnection(const Connector* connector );
      Connection* getInputConnection(const Connector* connector );
      Connection* getOutputConnection(const Unit* unit);
      Connection* getInputConnection(const Unit* unit );

      const Connection* getOutputConnection(const Connector* connector ) const;
      const Connection* getInputConnection(const Connector* connector ) const;
      const Connection* getOutputConnection(const Unit* unit) const;
      const Connection* getInputConnection(const Unit* unit ) const;

      const agxPowerLine::ConnectionRefVector& getOutputConnections() const;
      const agxPowerLine::ConnectionRefVector& getInputConnections() const;
      agxPowerLine::Connection* getInputConnection(size_t inputIndex = 0);
      agxPowerLine::Connection* getOutputConnection(size_t outputIndex = 0);

      const agxPowerLine::ConnectionRefVector& getConnections(agxPowerLine::Side side) const;


      size_t getBodyIndexIn(const RigidBodyPtrIntHashVector& bodyToIndexTable);

      /**
      Provides direct access to the underlying DimensionState.
      This should be considered an implementation detail and not called in user
      code.
      */
      agxPowerLine::detail::DimensionState* getState();
      const agxPowerLine::detail::DimensionState* getState() const;

      /**
      Access the DimensionState as a 1-dimensional state. Will return nullptr
      if the PhysicalDimension has a 3-dimensional DimensionState.

      \see get3DofState
      */
      agxPowerLine::detail::AbstractDimensionState1Dof* get1DofState();
      const agxPowerLine::detail::AbstractDimensionState1Dof* get1DofState() const;


      /**
      Access the DimensionState as a 3-dimensional state. Will return nullptr
      if the PhysicalDimension has a 1-dimensional DimensionState.

      \see get1DofState
      */
      agxPowerLine::detail::AbstractDimensionState3Dof* get3DofState();
      const agxPowerLine::detail::AbstractDimensionState3Dof* get3DofState() const;


      /**
      Stores internal data into stream.
      */
      virtual bool store(agxStream::StorageStream& str) const;

      /**
      Restores internal data from stream.
      */
      virtual bool restore(agxStream::StorageStream& in);


      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::PhysicalDimension);
      virtual void store(agxStream::OutputArchive& out) const override;
      virtual void restore(agxStream::InputArchive& in) override;
      void restore(agxStream::InputArchive& in, agx::RigidBody*& body, bool& externalBody);



      /**
      Returns the dimension type name of this PhysicalDimension instance. If
      the this is a RotationalDimension, then the dimension type name of
      RotationalDimension is returned.

      There are two types of names for a PhysicalDimension: the instance name
      and the type name. The instance name can be set by the user and the type
      name is an indentifier for a type of PhysicalDimension. For example,
      all RotationalDimensions have the same type name but may have different
      instance names.

      @return The actual dimension type name of this PhysicalDimension.

      @see getTypeName
      */
      virtual std::string getTypeName() const;


      /**
      returns the PhysicalDimension::Type of this dimension
      */
      agxPowerLine::PhysicalDimension::Type getType() const;


      /**
      Returns the dimension type name of PhysicalDimension. This is a static
      method and should be called as \p PhysicalDimension::getTypeName and not
      using . or -> operator on a reference or pointer.

      There are two types of names for a PhysicalDimension: the instance name
      and the type name. The instance name can be set by the user and the type
      name is an indentifier for a type of PhysicalDimension. For example,
      all RotationalDimensions have the same type name but may have different
      instance names.

      Each class inheriting from agxModel::Physical dimension HAS TO HAVE a
      static function that returns a unique name. The power line name singleton
      will sort and index the different names. The reason for this is that a
      user could create a physical dimension of his/hers own.
      The default dimension has name "UNDEFINED"

      @see getName
      @see getActualTypeName

      @return A string identifying the base PhysicalDimension type.
      */
      static std::string getStaticTypeName();

      static agxPowerLine::PhysicalDimension::Type getStaticType();

      /**
      \returns the total number of different types of physical dimensions in the simulation
      */
      static int getNumberOfPhysicalDimensions();


    protected:
      PhysicalDimension();

      PhysicalDimension(
          agxPowerLine::Unit* unit,
          agxPowerLine::PhysicalDimension::Type type,
          agxPowerLine::detail::DimensionState* state);

      virtual ~PhysicalDimension();

      /**
      Some serialization versions stored the dimension axis direction in classes
      deriving from PhysicalDimension. When loading such classes from such
      streams the restoring derived class need to pass the direction into the
      parent class, since the parent class is unable to read it from the stream.
      */
      bool restore(agxStream::StorageStream& in, const agx::Vec3* direction);

      friend class Unit;
      friend class Connector;

      void addNotification(agxPowerLine::PowerLine* powerLine);
      void removeNotification(agxPowerLine::PowerLine* powerLine);

      /**
      Called when the Unit owning this PhysicalDimension is added to a simulation.
      The body backing this PhysicalDimension is then added to the same simulation.
      */
      void addNotification(agxSDK::Simulation* simulation);

      /**
      Called when the Unit owning this PhysicalDimension is removed from a
      simulation.

      /// \todo May need to separate Unit being removed from PowerLine and Power-
      ///       Line being removed from Simulation.
      */
      void removeNotification(agxSDK::Simulation* simulation);

      /**
      Remove the PhysicalDimension from the simulation. May only be called if
      there are no connections to or from the dimension.
      */
      virtual void removeNotification(agxSDK::Simulation* simulation, agxUtil::ConstraintHolder* constraintHolder);

      bool removeInputConnection( Connection* connection );
      bool removeOutputConnection( Connection* connection );

      /**
      Add to singleton that controls the different types of physical dimensions.(PowerLineDimensionNameSingleton)
      */
      void addToListOfDimensions();

    protected:
      friend class agxPowerLine::detail::ConnectionOperations;
      agxPowerLine::ConnectionRefVector& getConnectionsWritable(agxPowerLine::Side side);

    protected:
      PhysicalDimension::Type m_type;
      ConnectionRefVector m_inputConnections;
      ConnectionRefVector m_outputConnections;
      Unit* m_unit;
      agxPowerLine::detail::PhysicalDimensionStateRef m_state;
      agx::Name m_name;
  };

  typedef agx::ref_ptr<PhysicalDimension> PhysicalDimensionRef;
  typedef agx::Vector<PhysicalDimensionRef> PhysicalDimensionRefVector;
  typedef agx::Vector<PhysicalDimension*> PhysicalDimensionPtrVector;
  typedef agx::SetVector<PhysicalDimension*> PhysicalDimensionPtrSetVector;
  typedef const PhysicalDimensionRef ConstPhysicalDimensionRef;
  typedef agx::Vector<ConstPhysicalDimensionRef> ConstPhysicalDimensionRefVector;


  AGXMODEL_EXPORT agx::RigidBody* createDimensionBody(const agx::Name& name);
}

#if defined(_MSC_VER)
#pragma warning( pop )
#endif


#endif // AGXMODEL_PHYSICAL_DIMENSION_H
