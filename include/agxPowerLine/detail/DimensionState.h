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


#ifndef AGX_POWERLINE_PHYSICAL_DIMENSION_STATE_H
#define AGX_POWERLINE_PHYSICAL_DIMENSION_STATE_H

#include <agx/Real.h>
#include <agx/Integer.h>
#include <agx/Jacobian.h>
#include <agx/RigidBody.h>
#include <agx/Referenced.h>
#include <agx/MergedBody.h>
#include <agx/Logger.h>

#include <agxPowerLine/DirectionReference.h>
#include <agxPowerLine/detail/DimensionStateStorage.h>

#include <agxModel/export.h>

#include <agxStream/StorageStream.h>


namespace agxPowerLine
{
  class SlotMapper;
  class PhysicalDimension;
  class PowerLine;
}



namespace agxPowerLine
{
  namespace detail
  {
    /**
    A DimensionState is the data carrier of a PhysicalDimension. It is reponsible
    for mapping data stored in the RigidBody into the values and types offered
    by the PhysicalDimension methods.

    DimensionState is pure virtual and a number of subclasses are used to describe
    various ways of performing the mapping. Each subclass define two properties:
    number of degrees of freedom and storage location. The storage location dictate
    where in the RigidBody that the dimension data is stored and can be either
    translational or rotational. The degrees of freedom is the number of slots
    the dimension state use in the RigidBody and can be either one or three. If
    one, then the dimension state owns one particular slot in the rigid body data
    and other dimension states may own other slots.

    Each dimension state has a direction that it either slides along or rotates
    around. For one dimensional dimension states this direction is always parallel
    to one of the principal axes, while for the three dimensional dimension states
    it can be any three dimensional vector. Further more, for three dimensional
    dimension states the direction can be specified in either local or world
    coordinates.
    */
    class AGXMODEL_EXPORT DimensionState : public agx::Referenced, public agxStream::Serializable
    {
      public:
        virtual agx::Real getValue() const = 0;
        virtual void setValue(agx::Real value) = 0;

        virtual agx::Real getGradient() const = 0;
        virtual void setGradient(agx::Real value) = 0;

        virtual agx::Real getSecondGradient() const = 0;

        virtual agx::Vec3 getLocalDirection() const = 0;
        virtual agx::Vec3 getWorldDirection() const = 0;
        virtual agx::Vec3 getDirection() const = 0;
        virtual agxPowerLine::DirectionReference getDirectionReference() const = 0;

        virtual agx::Real getMassProperty() const = 0;
        virtual bool setMassProperty(agx::Real massProperty) = 0;

        virtual void addLoad(agx::Real load) = 0;

        virtual void setVelocityDamping(float damping) = 0;
        virtual float getVelocityDamping() = 0;

        /**
        /return True if the body used by this dimension state was given by the user.
                False if it was created by the power line.
        */
        virtual bool isExternalBody() const = 0;

        /**
        \return True if this dimension state may be packed in a RigidBody with other dimension state.
        */
        bool getAllowsPacking() const;

        /**
        Reserve a body for this dimension state. After this call the body used
        by this dimension state will not be used by any other dimension state
        and future calls to \p getAllowsPacking will return false. The dimension
        state may move to a new RigidBody during reserve.
        */
        virtual void reserveBody() = 0;

        /**
        Gives access to a RigidBody that is known to be used by this dimension
        state only. Will return nullptr if no such guarantee can be given. Will
        always return a valid RigidBody if reserveBody has been called on this
        dimension state.
        */
        virtual agx::RigidBody* getReservedBody() = 0;
        virtual const agx::RigidBody* getReservedBody() const = 0;


        /**
        Returns the body currently used by this dimension state. If \p getAllowsPacking
        returns true then the body returned by \p getCurrentBody may change at any time.
        */
        agx::RigidBody* getCurrentBody();
        const agx::RigidBody* getCurrentBody() const;

        /**
        \return The PhysicalDimension that uses this dimension state for storage.
        */
        const agxPowerLine::PhysicalDimension* getDimension() const;

        virtual void calculateJacobian(agx::Jacobian6DOFElement* G, agx::Real ratio, size_t index) const = 0;

        virtual void addNotification(agxPowerLine::PowerLine* powerLine) = 0;
        virtual void removeNotification(agxPowerLine::PowerLine* powerLine) = 0;

        virtual void postUpdate(agx::Real timeStep) = 0;

        AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE(agxPowerLine::DimensionState)
        virtual void store(agxStream::OutputArchive& out) const override;
        virtual void restore(agxStream::InputArchive& in) override;

        virtual bool store(agxStream::StorageStream& out) const;
        virtual bool restore(agxStream::StorageStream& in);

      protected:
        DimensionState(agx::RigidBody* body, PhysicalDimension* dimension, bool allowPacking);

        virtual ~DimensionState();

        /**
        Mark is DimensionState as not a candidate for packing. Never call this
        method on a DimensionState that is currently packed since that would
        create an invalid state.

        \note Do not call this on a packed DimensionState.
        */
        void disallowPacking();

        agx::Vec3 getWorldPosition() const;

        agx::Vec3 getWorldVelocity() const;

        const agx::Frame* getBodyFrame() const;

        void setBody(agx::RigidBody* newBody);

      private:
        agx::RigidBodyRef m_body;
        agxPowerLine::PhysicalDimension* m_dimension;

        bool m_allowsPacking; ///<! True if body may be shared. May never transition false->true.
    };

    AGX_DECLARE_POINTER_TYPES(DimensionState);







    /**
    Base class for the two one dimensional dimension state classes. A one dimensional
    dimension state has a slot, and index in the range [0, 2], the specifies
    where the in the rigid body this dimension state is storing its data.
    */
    class AGXMODEL_EXPORT AbstractDimensionState1Dof : public DimensionState
    {
      public:
        virtual agx::UInt8 getSlot() const = 0;

      protected:
        AbstractDimensionState1Dof(agx::RigidBody* body, agxPowerLine::PhysicalDimension* dimension, bool allowPacking);
    };




    /**
    Templated one dimensional dimension state. The template type is the rigid
    body accessor (\p Translational or \p Rotational) that is to be used for
    accessing data in the RigidBody that currenly holds the dimension state data.
    */
    template<typename DimensionType>
    class AGXMODEL_EXPORT DimensionState1Dof : public AbstractDimensionState1Dof
    {
      public:
        DimensionState1Dof(agx::RigidBody* body, agxPowerLine::PhysicalDimension* dimension, agx::UInt8 slot);

        virtual agx::UInt8 getSlot() const override;

        virtual agx::Real getValue() const override;

        virtual void setValue(agx::Real value) override;

        virtual agx::Real getGradient() const override;

        virtual void setGradient(agx::Real gradient) override;

        virtual agx::Real getSecondGradient() const override;

        virtual agx::Vec3 getLocalDirection() const override;

        virtual agx::Vec3 getWorldDirection() const override;

        virtual agx::Vec3 getDirection() const override;

        virtual agx::Real getMassProperty() const override;

        virtual bool setMassProperty(agx::Real massProperty) override;

        virtual void addLoad(agx::Real load) override;

        virtual void setVelocityDamping(float damping) override;

        virtual float getVelocityDamping() override;

        virtual DirectionReference getDirectionReference() const override;

        virtual bool isExternalBody() const override;

        virtual void calculateJacobian(agx::Jacobian6DOFElement* G, agx::Real ratio, size_t index) const override;

        virtual void addNotification(agxPowerLine::PowerLine* powerLine) override;
        virtual void removeNotification(agxPowerLine::PowerLine* powerLine) override;


        /**
        Returns a RigidBody that is guaranteed to not be shared with any other
        DimensionState. Will return nullptr if such a guarantee cannot be
        made. Call \p reserveBody to ensure that this DimensionState does
        not share it's body with another DimensionState
        */
        virtual agx::RigidBody* getReservedBody() override;
        virtual const agx::RigidBody* getReservedBody() const override;

        /**
        Ensure that this DimensionState is the only state using this
        state's RigidBody. The DimensionState may create a new body in
        order to give that guarantee. After this call \p getReservedBody is
        guaranteed to not return nullptr and \p getAllowsPacking is guaranteed
        to return false.
        */
        virtual void reserveBody() override;

        /**
        Move this DimensionState to a new RigidBody that is guaranteed to not
        be shared with any other DimensionState. This does not alter the allow
        packing setting, so if the DimensionState allows packing then the
        DimensionState may later be packed into a body used by another
        DimensionState, and other DimensionState may be packed into the body
        just created.
        */
        void split();

        /**
        Make this dimension state use the given body for storage. All state
        stored in the current body is moved to the new body. It is important that
        the body/slot pair is not used by any other dimension state, and that
        the given body is not used for any other purpose than to store
        DimesionState1Dof data.

        The dimension state must allow packing for this operation to be legal.

        \param body - The body to move to.
        \param slot - The slot to inhabit in the new body.
        */
        void move(agx::RigidBody* body, agx::UInt8 slot);

        SlotMapper* getSlotMapper();

        virtual void postUpdate(agx::Real timeStep) override;

        virtual agxStream::StorageAgent* getStorageAgent() const override;
        static const char* getConstructClassId();
        friend class agxStream::DefStorageAgent<DimensionState1Dof<DimensionType>>;
        virtual const char* getClassName() const override;
        static agxStream::Serializable* create();
        static agxStream::Serializable* create(agxStream::InputArchive&);

        virtual void store(agxStream::OutputArchive& out) const override;
        virtual void restore(agxStream::InputArchive& in) override;

        virtual bool store(agxStream::StorageStream& out) const override;
        virtual bool restore(agxStream::StorageStream& in) override;

      private:
        friend class agxPowerLine::SlotMapper;


      protected:
        DimensionState1Dof(); // For serialization only.
        virtual ~DimensionState1Dof();
        void reserveImpl();
        void splitImpl();
        void moveImpl(agx::RigidBody* newBody, agx::UInt8 newSlot);

      private:
        agx::UInt8 m_slot;
        DimensionType m_storage;
    };



    /**
    Base class for the two three dimensional dimension state classes. A three
    dimensional dimension state has a direction that specifies how to convert
    beteen the three dimensional space in which the rigid body exists, and the
    one dimensional space that the dimension state exists.
    */
    class AGXMODEL_EXPORT AbstractDimensionState3Dof : public DimensionState
    {
      public:
        virtual void setLocalDirection(const agx::Vec3& direction) = 0;
        virtual void setWorldDirection(const agx::Vec3& direction) = 0;

      protected:
        AbstractDimensionState3Dof(agx::RigidBody* body, agxPowerLine::PhysicalDimension* dimension);
    };



    /**
    Templated three dimensional dimension state. The template type is the rigid
    body accessor (\p Translational or \p Rotational) that is to be used for
    accessing data in the RigidBody that currenly holds the dimension state data.
    */
    template<typename DimensionType>
    class AGXMODEL_EXPORT DimensionState3Dof : public AbstractDimensionState3Dof
    {
      public:
        DimensionState3Dof(
             agx::RigidBody* body,
             agxPowerLine::PhysicalDimension* dimension,
             const agx::Vec3& direction,
             DirectionReference directionReference);

        DimensionState3Dof(
            agx::RigidBody* body,
            agxPowerLine::PhysicalDimension* dimension,
            const agx::Vec3& direction,
            DirectionReference directionReference,
            bool externalBody);

        virtual agx::Real getValue() const override;

        virtual void setValue(agx::Real value) override;

        virtual agx::Real getGradient() const override;

        virtual void setGradient(agx::Real gradient) override;

        virtual agx::Real getSecondGradient() const override;

        virtual agx::Vec3 getLocalDirection() const override;

        virtual void setLocalDirection(const agx::Vec3& direction) override;

        virtual agx::Vec3 getWorldDirection() const override;

        virtual void setWorldDirection(const agx::Vec3& direction) override;

        virtual agx::Vec3 getDirection() const override;

        virtual agx::Real getMassProperty() const override;

        virtual bool setMassProperty(agx::Real massProperty) override;

        virtual void addLoad(agx::Real load) override;

        virtual void setVelocityDamping(float damping) override;
        virtual float getVelocityDamping() override;

        virtual DirectionReference getDirectionReference() const override;

        virtual bool isExternalBody() const override;

        virtual void reserveBody() override;
        virtual agx::RigidBody* getReservedBody() override;
        virtual const agx::RigidBody* getReservedBody() const override;

        virtual void calculateJacobian(agx::Jacobian6DOFElement* G, agx::Real ratio, size_t index) const override;

        virtual void addNotification(agxPowerLine::PowerLine* powerLine) override;
        virtual void removeNotification(agxPowerLine::PowerLine* powerLine) override;

        virtual void postUpdate(agx::Real /*timeStep*/) override;

        virtual bool store(agxStream::StorageStream& out) const override;
        virtual bool restore(agxStream::StorageStream& in) override;

        virtual void store(agxStream::OutputArchive& out) const override;
        virtual void restore(agxStream::InputArchive& in) override;

        virtual agxStream::StorageAgent* getStorageAgent() const override;
        static const char* getConstructClassId();
        friend class agxStream::DefStorageAgent<DimensionState1Dof<DimensionType>>;
        virtual const char* getClassName() const override;
        static agxStream::Serializable* create();
        static agxStream::Serializable* create(agxStream::InputArchive&);


      protected:
        virtual ~DimensionState3Dof();

      private:
        DimensionState3Dof();

        void calculateMergedBodyJacobian(
            agx::Jacobian6DOFElement* /*G*/,
            agx::Real /*ratio*/,
            size_t /*index*/,
            const agx::MergedBody* /*mergedBody*/) const;

        void printInvalidDirectionReference() const;

      private:
        agx::Vec3 m_direction;
        DirectionReference m_directionReference;
        DimensionType m_storage;
        bool m_externalBody; // Will this ever be false? When?
    };

    // Some human friendly names for the four possible combinations of degrees
    // of freedom and storage location in the rigid body.
    using Translational1DofState = DimensionState1Dof<Translational>;
    using Translational3DofState = DimensionState3Dof<Translational>;
    using Rotational1DofState = DimensionState1Dof<Rotational>;
    using Rotational3DofState = DimensionState3Dof<Rotational>;
  }
}

#endif
