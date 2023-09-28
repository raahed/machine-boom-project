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

#ifndef AGXPOWERLINE_DETAIL_PACKING_INFO
#define AGXPOWERLINE_DETAIL_PACKING_INFO

#include <agx/Integer.h>
#include <agx/Vector.h>

namespace agx
{
  class RigidBody;
}

namespace agxStream
{
  class InputArchive;
  class OutputArchive;
}

namespace agxPowerLine
{
  namespace detail
  {
    struct Rotational;
    template <typename DimensionType> class DimensionState1Dof;
    using Rotational1DofState = DimensionState1Dof<Rotational>;
  }
}

namespace agxPowerLine
{
  namespace detail
  {
    /**
    Information about which DimensionState1Dof is using which slot in a RigidBody.
    It is important that the information stored here is kept in sync with the
    body and slot stored in the DimensionState1Dof.

    Used by the slot mapper.
    */
    class PackingInfo
    {
    public:

      /**
      A default-constructed PackingInfo is empty, i.e., it tracks no PhysicalDimension.
      */
      PackingInfo();

      /**
      A packing info tracking the given dimension only. The given dimension's
      current slot is used to initialize the PackingInfo.
      */
      PackingInfo(Rotational1DofState* dimension);

      Rotational1DofState*& operator[](const agx::UInt8 slot);

      const Rotational1DofState* operator[](const agx::UInt8 slot) const;

      Rotational1DofState** begin();

      Rotational1DofState** end();

      bool contains(const Rotational1DofState* dimension) const;

      bool contains(const agx::RigidBody* body) const;

      bool isEmpty() const;

      bool isFull() const;

      int getSize() const;

      agx::UInt8 getFreeSlot() const;

      agx::RigidBody* getBody();

      void erase(const Rotational1DofState* dimension);

      void newSlot(Rotational1DofState* dimension);

      void assertValid() const;

      void store(agxStream::OutputArchive& out) const;
      void restore(agxStream::InputArchive& in);

    private:
      // Mapping between the dimension state and slots in a RigidBody.
      // All dimension states use the same body, and m_dimensions[i] use slot i
      // in the RigidBody.
      Rotational1DofState* m_dimensions[3];
    };

    using PackingInfos = agx::Vector<agxPowerLine::detail::PackingInfo>;
  }
}

#endif
