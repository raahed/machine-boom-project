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


#ifndef AGXPOWERLINE_DETAIL_DIMENSION_STATE_STORAGE
#define AGXPOWERLINE_DETAIL_DIMENSION_STATE_STORAGE


#include <agx/Vec3.h>
#include <agx/Jacobian.h>

#include <agxModel/export.h>



namespace agx
{
  class RigidBody;
}

namespace agxStream
{
  class OutputArchive;
  class InputArchive;
  class StorageStream;
}


namespace agxPowerLine
{
  namespace detail
  {

    /**
    RigidBody accessor helper that knows how to extract rotational data from a
    RigidBody. A dimension state using one of these becomes a rotational dimension
    state. Supports both one dimensional and three dimensional accessing.
    */
    struct AGXMODEL_EXPORT Rotational
    {
        Rotational();

        agx::Vec3 getValue(const agx::RigidBody& body) const;

        agx::Real getValue(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getValue(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setValue(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real angle);

        void setValue(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real angle);

        agx::Vec3 getGradient(const agx::RigidBody& body) const;

        agx::Real getGradient(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getGradient(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setGradient(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real angularVelocity);

        void setGradient(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real angularVelocity);

        agx::Vec3 getSecondGradient(const agx::RigidBody& body) const;

        agx::Real getSecondGradient(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getSecondGradient(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        agx::Vec3 getMassProperty(const agx::RigidBody& body) const;

        agx::Real getMassProperty(const agx::RigidBody& body, agx::UInt8 slot) const;

        agx::Real getMassProperty(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        bool setMassProperty(agx::RigidBody& body, const agx::Vec3& inertia);

        bool setMassProperty(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real inertia);

        bool setMassProperty(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real inertia);

        agx::Vec3 getAddedLoad(const agx::RigidBody& body) const;

        agx::Real getAddedLoad(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getAddedLoad(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setAddedLoad(agx::RigidBody& body, agx::UInt8 slot, agx::Real load);

        void setAddedLoad(agx::RigidBody& body, const agx::Vec3& worldDirection, agx::Real load);

        void addLoad(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real load);

        void addLoad(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real load);

        agx::Vec3f getVelocityDamping(const agx::RigidBody& body) const;

        float getVelocityDamping(const agx::RigidBody& body, const agx::UInt8 slot) const;

        float getVelocityDamping(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setVelocityDamping(agx::RigidBody& body, const agx::UInt8 slot, const float damping);

        void setVelocityDamping(agx::RigidBody& body, const agx::Vec3& worldDirection, const float damping);

        void move(agx::RigidBody& oldBody, const agx::UInt8 oldSlot, agx::RigidBody& newBody, const agx::UInt8 newSlot);

        void writeJacobian(agx::Jacobian6DOFElement& G, const agx::Vec3& jacobian) const;

        void integrateValue(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real timeStep);

        void store(agxStream::OutputArchive& out) const;
        void restore(agxStream::InputArchive& in);
        void store(agxStream::StorageStream& out, const agx::RigidBody& body, const agx::UInt8 slot) const;
        void restore(agxStream::StorageStream& in, agx::RigidBody& body, const agx::UInt8 slot);
        void store(agxStream::StorageStream& out, const agx::RigidBody& body, const agx::Vec3& worldDirection) const;
        void restore(agxStream::StorageStream& in, agx::RigidBody& body, const agx::Vec3& worldDirection);

        static bool getAllowPacking();

      private:
        agx::Real m_angle;
    };


    /**
    RigidBody accessor helper that knows how to extract translational data
    from a RigidBody. A dimension state using one of these becomes a translational
    dimension state. Supports both one dimensional and three dimensional accessing.
    */
    struct AGXMODEL_EXPORT Translational
    {
        agx::Vec3 getValue(const agx::RigidBody& body) const;

        agx::Real getValue(const agx::RigidBody& body, agx::UInt8 slot) const;

        agx::Real getValue(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setValue(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real position);

        void setValue(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real position);

        agx::Vec3 getGradient(const agx::RigidBody& body) const;

        agx::Real getGradient(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getGradient(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setGradient(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real velocity);

        void setGradient(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real velocity);

        agx::Vec3 getSecondGradient(const agx::RigidBody& body) const;

        agx::Real getSecondGradient(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getSecondGradient(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        agx::Vec3 getMassProperty(const agx::RigidBody& body) const;

        agx::Real getMassProperty(const agx::RigidBody& body, agx::UInt8 slot) const;

        agx::Real getMassProperty(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        bool setMassProperty(agx::RigidBody& body, const agx::Vec3& mass);

        bool setMassProperty(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real mass);

        bool setMassProperty(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real mass);

        agx::Vec3 getAddedLoad(const agx::RigidBody& body) const;

        agx::Real getAddedLoad(const agx::RigidBody& body, const agx::UInt8 slot) const;

        agx::Real getAddedLoad(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setAddedLoad(agx::RigidBody& body, agx::UInt8 slot, agx::Real load);

        void setAddedLoad(agx::RigidBody& body, const agx::Vec3& worldDirection, agx::Real load);

        void addLoad(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real load);

        void addLoad(agx::RigidBody& body, const agx::Vec3& worldDirection, const agx::Real load);

        agx::Vec3f getVelocityDamping(const agx::RigidBody& body) const;

        float getVelocityDamping(const agx::RigidBody& body, const agx::UInt8 slot) const;

        float getVelocityDamping(const agx::RigidBody& body, const agx::Vec3& worldDirection) const;

        void setVelocityDamping(agx::RigidBody& body, const agx::UInt8 slot, const float damping);

        void setVelocityDamping(agx::RigidBody& body, const agx::Vec3& worldDirection, const float damping);

        void move(agx::RigidBody& oldBody, const agx::UInt8 oldSlot, agx::RigidBody& newBody, const agx::UInt8 newSlot);

        void writeJacobian(agx::Jacobian6DOFElement& G, const agx::Vec3& jacobian) const;

        void integrateValue(agx::RigidBody& body, const agx::UInt8 slot, const agx::Real timeStep);

        void store(agxStream::OutputArchive& out) const;
        void restore(agxStream::InputArchive& in);
        void store(agxStream::StorageStream& out, const agx::RigidBody& body, const agx::UInt8 slot) const;
        void restore(agxStream::StorageStream& in, agx::RigidBody& body, const agx::UInt8 slot);
        void store(agxStream::StorageStream& out, const agx::RigidBody& body, const agx::Vec3& worldDirection) const;
        void restore(agxStream::StorageStream& in, agx::RigidBody& body, const agx::Vec3& worldDirection);

        static bool getAllowPacking();
    };



  }
}


#endif
