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

#include <agxTerrain/Shovel.h>
#include <agxTerrain/Terrain.h>



namespace agxTerrain
{
  /**
  Utility namespace for calculating excavation forces in the terrain.
  */
  namespace TerrainContactForceCalculator
  {
    /**
   Given geometry contacts exists and the solver has solved them - calculates
   total contact force between the terrain soil particle aggregate and the given shovel. This represents the
   separation force that is required to move the excavated soil in the shovel active zone.
   \param terrain - The terrain to use for the computation
   \param shovel - interacting shovel
   \return the total contact force between given shovel and the soil particle aggregate in the terrain.
   */
    agx::Vec3 AGXTERRAIN_EXPORT getSeparationContactForce(const Terrain* terrain, const Shovel* shovel);

    /**
    Given geometry contacts exists and the solver has solved them - calculates total contact force between the terrain
    deformation soil aggregate and the given shovel. This represents the deformation force that is required to move
    soil via shovel deformation instead of excavation, i.e not excavation or digging.
    Examples of this would be side movement and backwards grading of the soil.
    \param terrain - The terrain to use for the computation
    \param shovel - interacting shovel
    \return the total contact force between given shovel and the deformer soil aggregates in the terrain.
    */
    agx::Vec3 AGXTERRAIN_EXPORT getDeformationContactForce(const Terrain* terrain, const Shovel* shovel);

    /**
    Calculates total contact force between the soil aggregate and the shovel associated with the specified excavation mode.
    \param terrain - The terrain to use for the computation
    \param shovel - interacting shovel
    \param excavationMode - the excavation mode that the specified soil aggregate belongs to
    \return the total contact force between given shovel and the soil aggregate specified by the excavation mode
    */
    agx::Vec3 AGXTERRAIN_EXPORT getExcavationModeContactForce(const Terrain* terrain, const Shovel* shovel, Shovel::ExcavationMode excavationMode);

    /**
    Given geometry contacts exists and the solver has solved them - calculates
    total shovel contact force between this terrain and the given shovel. This is the contact force
    that prevents the shovel from falling through the terrain when not in excavation mode, where contact
    feedback is generated from the soil aggregates.
    \note - This method returns regular contact forces ONLY when no soil aggregates are present to generate excavation feedback!
    \param terrain - The terrain to use for the computation
    \param shovel - interacting shovel
    \return the total non-excavation contact force between this terrain and the given shovel.
    */
    agx::Vec3 AGXTERRAIN_EXPORT getContactForce(const Terrain* terrain, const Shovel* shovel);

    /**
    Internal method
    Get the aggregate contact force with the terrain given an excavation mode and a shovel
    \param terrain - The terrain to use for the computation
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return the total force (N) acting on the aggregate in the terrain-aggregate contact
    */
    agx::Vec3 AGXTERRAIN_EXPORT getAggregateTerrainContactForce(const Terrain* terrain,  const Shovel* shovel, Shovel::ExcavationMode excavationMode);

    /**
    Internal method
    Get the aggregate normal force with the terrain given an excavation mode and a shovel
    \param terrain - The terrain to use for the computation
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return the total force (N) acting on the aggregate in the terrain-aggregate contact
    */
    agx::Vec3 AGXTERRAIN_EXPORT getAggregateTerrainNormalForce(const Terrain* terrain, const Shovel* shovel, Shovel::ExcavationMode excavationMode);

    /**
    Internal method
    Get the aggregate normal and shear contact force with the terrain given an excavation mode and a shovel
    \param terrain - The terrain to use for the computation
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return vector containing the geometry contacts between the specified aggregate and the terrain
    */
    agxCollide::GeometryContactPtrVector AGXTERRAIN_EXPORT getAggregateTerrainContacts(const Terrain* terrain, const Shovel* shovel, Shovel::ExcavationMode excavationMode);

    /**
    Internal method
    Get the shovel <-> aggregate contacts with the terrain given an excavation mode and a shovel.
    \param terrain - The terrain to use for the computation
    \param shovel - the specified shovel
    \param excavationMode - the excavation mode that the soil aggregate belongs too ( PRIMARY, DEFORM_BACK, DEFORM_RIGHT, DEFORM_LEFT )
    \return vector containing the geometry contacts between the specified shovel and the aggregate.
    */
    agxCollide::GeometryContactPtrVector AGXTERRAIN_EXPORT getShovelAggregateContacts( const Terrain* terrain, const Shovel* shovel, Shovel::ExcavationMode excavationMode );
  }
}