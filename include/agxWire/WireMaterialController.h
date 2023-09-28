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

#include <agx/agxPhysics_export.h>
#include <agx/Material.h>
#include <agxWire/WireDistanceCompositeConstraint.h>
#include <agxWire/WireNodeDensityController.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxWire
{
  class AGXPHYSICS_EXPORT WireMaterialController : public agx::Referenced
  {
    public:
      WireMaterialController( agx::Real resolutionPerUnitLength );

      /**
      Assign wire this material controller should work on.
      \param wire - new wire
      */
      void setWire( WireDistanceCompositeConstraint* wire );

      /**
      \return pointer to the wire this material controller is working on
      */
      const agxWire::WireDistanceCompositeConstraint* getWire() const;

      /**
      Updates all nodes (mass, friction coefficients etc) for the whole wire.
      */
      void updateMaterial();

      /**
      Updates all friction coefficients for all contact nodes.
      */
      void updateNodeMaterial();

      /**
      Sets the radius of the wire this controller is part of.
      \param radius - new radius
      */
      void setRadius( agx::Real radius);

      /**
      \return the radius
      */
      agx::Real getRadius() const;

      /**
      \return the cross section area
      */
      agx::Real getCrossSectionArea() const;

      /**
      The segment is assumed to be a cylinder. This volume ratio specifies the
      ratio between the cylinder of the given radius and the wanted geometry.
      So if the wanted geometry is a chain, then this ratio is around 0.1-0.2, which
      corresponds to the ratio of volume between a cylinder and a chain of the same radius.
      \param volumeRatio - volume ratio for this segment (default: 1)
      */
      void setVolumeRatio( agx::Real volumeRatio );

      /**
      \return the volume ratio used in this segment (default: 1)
      */
      agx::Real getVolumeRatio() const;

      /**
      Will set the global lumped node distance for the whole wire
      \param lumpDistance - The target length for the distance between two lumped nodes
      */
      void setGlobalLumpDistance( agx::Real lumpDistance );

      /**
      \return the global distance between two lumped nodes
      */
      agx::Real getGlobalLumpDistance() const;

      /**
      find if lumped node position is ok for the local resolution of the line
      */
      bool resolutionHighEnoughForPosition(const agx::Real restlengthToBfn,const agx::Real minDistanceToLump, const agx::Real eps = agx::Real(1E-6) ) const;

      /**
      \param distanceOnWireToStart - The start of the part of the wire we consider
      \param localLength  - The distance of the part of the line we consider
      \param foundRanges  - the ranges within local segment of length "localLength" where there could appear nodes
      */
      void findLocalHighResolutionRanges( const agx::Real distanceOnWireToStart,const agx::Real localLength, agx::Vector< WireNodeRange >& foundRanges );

      /**
      Get the mass per unit length,
      given wire radius, density of material and the volume ratio of the wire (if it is hollow somehow).
      */
      agx::Real getMassPerUnitLength() const;

      /**
      Calculates the mass of a wire segment with length \p segmentLength
      \param segmentLength - segment length
      \return the mass of the segment
      */
      agx::Real getMass( agx::Real segmentLength ) const;

      /**
      Updates node density given node list.
      */
      void updateWireNodeDensity();

      /**
      Clears internal structures in node density controller.
      */
      void clearWireNodeDensity();

      /**
      \return the node density controller
      */
      agxWire::WireNodeDensityController* getNodeDensityController() const;

      /**
      \return the material
      */
      agx::Material* getMaterial() const;

      /**
      Sets the material for the wire.
      \param material - new material (invalid if null)
      */
      void setMaterial( agx::Material* material );

      /**
      \return true if the wire is bend resistant (e.g., wire. Not bend resistant is e.g., chain)
      */
      bool getIsBendResistant() const;

      /**
      Set to true if the wire is bend resistant, i.e., rest state is straight (e.g., wire).
      If false, no bend constraints are created (e.g., chain).
      \param isBendResistant
      */
      void setIsBendResistant( bool isBendResistant );

      /**
      Calculates and sets the new mass and velocity for the 2 old lumps and the new one
      \param bfnAIt   - lumped node before bfnNew in node list
      \param bfnCIt   - lumped node after  bfnNew in node list
      \param bfnNewIt - new lumped node in node list
      \param halfMassFromA - the mass related to the part of the wire that is now closer to bfnNew than bfnA
      \param halfMassFromC - the mass related to the part of the wire that is now closer to bfnNew than bfnC
      */
      void changeMassAndVelocityByAddition( const NodeConstIterator bfnAIt,
                                            const NodeConstIterator bfnCIt,
                                            const NodeConstIterator bfnNewIt,
                                            agx::Real halfMassFromA,
                                            agx::Real halfMassFromC ) const;
      void changeMassAndVelocityByRemoval( const NodeConstIterator bfnAIt,
                                           const NodeConstIterator bfnCIt,
                                           const NodeConstIterator bfnRemoveIt,
                                           agx::Real halfMassToA,
                                           agx::Real halfMassToC ) const;

    private:
      NodeIterator updateNodeMaterial( NodeIterator it, const NodeConstIterator end ) const;

    private:
      agx::observer_ptr< WireDistanceCompositeConstraint> m_wire;
      agx::Real m_lumpNodeDistance;
      bool m_isBendResistant;
      agx::Real m_radius;
      agx::Real m_volumeRatio;
      agx::ref_ptr< WireNodeDensityController > m_wireNodeDensityController;
      agx::MaterialRef m_material;
  };

  AGX_FORCE_INLINE agx::Real WireMaterialController::getRadius() const
  {
    return m_radius;
  }

  AGX_FORCE_INLINE agx::Real WireMaterialController::getCrossSectionArea() const
  {
    return getRadius() * getRadius() * agx::PI;
  }

  AGX_FORCE_INLINE agx::Real WireMaterialController::getGlobalLumpDistance() const
  {
    return m_lumpNodeDistance;
  }

  AGX_FORCE_INLINE agx::Material* WireMaterialController::getMaterial() const
  {
    return m_material;
  }

  typedef agx::ref_ptr< WireMaterialController > WireMaterialControllerRef;
}

DOXYGEN_END_INTERNAL_BLOCK()
