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


#ifndef AGXCABLE_CABLE_PROPERTIES_H
#define AGXCABLE_CABLE_PROPERTIES_H

#include <agxCable/export.h>
#include <agxCable/Direction.h>

#include <agxStream/Serializable.h>

namespace agxCable
{
  class Cable;

  AGX_DECLARE_POINTER_TYPES(CableProperties);

  /**
  Class controlling material properties of one or more cables.
  */
  class AGXCABLE_EXPORT CableProperties : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Set Young's modulus of the cable along the given direction. A high Young's
      modulus make the cable stiff and difficult to bend, twist or stretch.
      \param youngsModulus - New value of Young's modulus.
      \param direction - The type of deformation that the new Young's modulus should apply to.
      */
      void setYoungsModulus(agx::Real youngsModulus, agxCable::Direction direction);

      /**
      \return The Young's modulus of the cable in the given direction.
      */
      agx::Real getYoungsModulus(agxCable::Direction direction) const;

      /**
      Set the SPOOK damping of the constraints holding the cable together.
      \param damping - The new value for the damping.
      \param direction - The type of deformation that the new damping should apply to.
      */
      void setDamping(agx::Real damping, agxCable::Direction direction);

      /**
      \return The SPOOK damping of the cable in the given direction.
      */
      agx::Real getDamping(agxCable::Direction direction) const;

      void setPoissonsRatio(agx::Real poissonsRatio, agxCable::Direction direction);
      agx::Real getPoissonsRatio(agxCable::Direction direction) const;

    DOXYGEN_START_INTERNAL_BLOCK()
    public:
      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::CableProperties);


    protected:
      friend class agxCable::Cable;
      CableProperties();
      virtual ~CableProperties();

      bool isDirty() const;
      void clearDirtyFlag();
      void updateBulkProperties(Cable& cable) const;

    private:
      void restoreFromLinkedCable(agxStream::InputArchive& in);
      void restoreFromDeformable1D(agxStream::InputArchive& in);

    private:
      agx::Real m_youngsModulus[ALL_DIRECTIONS];
      agx::Real m_damping[ALL_DIRECTIONS];
      agx::Real m_poissonsRatio[ALL_DIRECTIONS];

      agx::Bool m_dirty;
    DOXYGEN_END_INTERNAL_BLOCK()
  };
}

#endif
