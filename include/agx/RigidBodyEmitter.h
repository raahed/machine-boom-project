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


#ifndef AGX_RIGIDBODY_EMITTER_H
#define AGX_RIGIDBODY_EMITTER_H

#include <agx/Emitter.h>
#include <agx/InternalData.h>
#include <agx/BitState.h>
#include <agx/ICloneable.h>
#include <agx/ObserverFrame.h>

namespace agxSDK
{
  class Simulation;
}


namespace agx
{
  AGX_DECLARE_POINTER_TYPES(RigidBodyEmitter);
  AGX_DECLARE_VECTOR_TYPES(RigidBodyEmitter);

  /**
  * Spawns new Rigid Bodies inside a volume given to the Emitter. The RigidBodyEmitter
  * requires a DistributionTable of RigidBodyEmitter::Distribution models in order to
  * create new Rigid Bodies.
  */
  class AGXPHYSICS_EXPORT RigidBodyEmitter : public agx::Emitter
  {
  private:
    /**
    Struct assigned to RigidBodies created from the RigidBodyEmitter or loaded from
    a snapshot. This is used to determine if a body has been emitted/loaded or not.
    */
    struct RigidBodyEmitterInternalData : public agx::ICloneable
    {
      enum EState
      {
        CREATED_BY_RIGIDBODYEMITTER = 1 << 0,
        CREATED_BY_SNAPSHOT_LOADING = 1 << 1,
        IS_BODYMODEL_TEMPLATE = 1 << 2
      };
      typedef agx::BitState< EState, agx::Int32 > State;

      RigidBodyEmitterInternalData() : state() {}

      State state;
    protected:
      virtual ~RigidBodyEmitterInternalData() {}

    private:
      virtual agx::ICloneableRef clone(agx::ICloneable* child) override
      {
        if (child == nullptr)
          child = new RigidBodyEmitterInternalData();

        child->as<RigidBodyEmitterInternalData>()->state = this->state;

        return agx::ICloneableRef(child);
      }
    };
  public:


    AGX_DECLARE_POINTER_TYPES( DistributionModel );

    typedef Event3<RigidBodyEmitter*, agx::RigidBody*, DistributionModel*> Event;
    Event emitEvent;

  public:

    /**
    * Default constructor for a RigidBodyEmitter.
    *\param quantity - the quantity that determine the unit of the emit rate.
    */
    RigidBodyEmitter(Quantity quantity = QUANTITY_COUNT);

    /**
    * Spawn Rigid Bodies inside a 3D bound based on a specified distribution table of rigid bodies.
    * \param bound - The specified bound to spawn in
    * \param sourceTable - The particle model distribution table used to spawn particles.
    * \param spacing - Spacing between the body center points that are generated.
    * \param jitterfactor - The factor of the distance that is used to randomize particle positions.
    */
    static agx::RigidBodyRefVector spawnBodiesInBound(const agx::Bound3& bound,
                                                       agx::Emitter::DistributionTable* sourceTable,
                                                       const agx::Vec3& spacing,
                                                       agx::Real jitterfactor = agx::Real(0));

    /**
    * \return true if the specified Rigid Body was created from an Emitter, false otherwise.
    */
    static bool rigidBodyIsEmitted( const agx::RigidBody * body );

    /**
    * \return true if the specified Rigid Body was loaded from a snapshot, false otherwise.
    */
    static bool rigidBodyIsLoaded( const agx::RigidBody * body );

    /**
    \return true if the specified RigidBody has the state bit enabled that signals
    if it is used a BodyModel template inside a RigidBodyEmitter distribution, false
    otherwise.
    */
    static bool rigidBodyIsTemplate(const agx::RigidBody* body);

    /**
    * Manually override setting that specifies if a RigidBody was created from an Emitter.
    * \param body - the specified body where the setting should apply.
    * \param isEmitted - true if the body should be registered as being created from
                         an Emitter, false otherwise.
    */
    static void setRigidBodyIsEmitted( agx::RigidBody * body, bool isEmitted );

    /**
    * Manually override setting that specifies if a RigidBody was loaded from an snapshot.
    * \param body - the specified body where the setting should apply.
    * \param isLoaded - true if the body should be registered as being created from
                         a snapshot, false otherwise.
    */
    static void setRigidBodyIsLoaded( agx::RigidBody * body, bool isLoaded );

    /**
    Set state bit in the RigidBodyEmitterData for the specifed RigidBody that
    signals if it is used as a BodyModel template inside a RigidBodyEmitter
    distribution.
    \param body - the specified body where the RigidBodyEmitterData should be modified.
    \param isTemplate - true if the state should be enabled, false otherwise.
    \note - this is used internally for disabling rendering of template bodies in
            RigidBodyEmitterdistributions as they are added to the simulation, in
            a disabled state, when the Emitter is added.
    */
    static void setRigidBodyIsTemplate(agx::RigidBody* body, bool isTemplate);

    /**
    \return a vector with all Rigid Bodies inside a simulation that was created from an emitter,
            checked via `rigidBodyIsEmitted` function.
    */
    static agx::RigidBodyPtrVector getEmittedBodiesInSimulation(agxSDK::Simulation* simulation);

    DOXYGEN_START_INTERNAL_BLOCK()

    static void storePersistentData(const agx::RigidBody* rb, agxStream::OutputArchive& out);

    static void restorePersistentData(agx::RigidBody* rb, agxStream::InputArchive& in);

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::RigidBodyEmitter);

  private:
    static RigidBodyEmitterInternalData* getRigidBodyEmitterData(const agx::RigidBody* obj);

    static RigidBodyEmitterInternalData* getOrCreateRigidBodyEmitterData(agx::RigidBody* obj);

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:

    /// Default destructor
    virtual ~RigidBodyEmitter();

    virtual void emitBody(const Vec3& position,
                          const Quat& rotation,
                          const Vec3& velocity,
                          Emitter::DistributionModel *model) override;
    virtual bool preEmit() override;
    virtual void postEmit() override;

    virtual void transfer(agxSDK::Simulation *simulation) override;

  private:
    // Creates and initializes the parameters.
    void init();



  private:
    agxSDK::Simulation *m_simulation;
  };

  /**
  A Distribution model for a RigidBody used in DistributionTables used in RigidBodyEmitter.
  */
  class AGXPHYSICS_EXPORT RigidBodyEmitter::DistributionModel : public Emitter::DistributionModel
  {
  public:
    /**
    * Default constructor
    */
    DistributionModel();

    /**
    Create a Rigid Body Distribution model for usage in the Emitter::DistributionTable class.
    \param bodyTemplate - the Rigid Body template that will be used to create new Rigid Bodies
                          from this model using the clone() function.
    \param probabilityWeight - the probability weight that will be assigned to the model. The weight
                               determines how much of the created quantity from a DistributionTable that should
                               belong to this model. This is based on the size of the weight relative
                               to the weights of other models in the DistributionTable.
    \param placementObserver - The ObserverFrame used for determining the placement
                               delta from the model body's center of mass during an emit event.
    */
    DistributionModel(agx::RigidBody *bodyTemplate,
                      agx::Real probabilityWeight,
                      agx::ObserverFrame* placementObserver=nullptr);

    /**
    \return The body template of the model that is used to create new RigidBodies.
    */
    agx::RigidBody *getBodyTemplate();

    /**
    \return the ObserverFrame used for determining the placement delta from
            the model body's center of mass during an emit event.
    */
    agx::ObserverFrame* getPlacementObserver() const;

    DOXYGEN_START_INTERNAL_BLOCK()

    AGXSTREAM_DECLARE_SERIALIZABLE(agx::RigidBodyEmitter::DistributionModel);

    DOXYGEN_END_INTERNAL_BLOCK()

  protected:
    virtual ~DistributionModel();
    virtual agx::Real calculateVolume() override;
    virtual agx::Real calculateMass() override;

  private:
    agx::RigidBodyRef           m_bodyTemplate;
    agx::ObserverFrameRef       m_placementObserver;
  };
}

#endif
