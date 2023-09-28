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

#include <agx/Plane.h>
#include <agx/Vec3.h>
#include <agx/Vector.h>

namespace agx
{
  class RigidBody;
}


namespace agxTerrain
{
  class Terrain;
  class TerrainToolCollection;

  /**
  Class used to calculate the contact depth in the soil aggregate <-> terrain contacts to mimic an elasto-plastic contact model.

  The contact depth in the in the model by using the current separation normal, the first aggregate contact point at the cutting edge and the
  plane spanned by the aggregate contact point. The movement of the contact point is stored in the depth model. The depth increases when
  that movement of the contact point projected in the separation direction moves against the plane spanned by the aggregate contact points,
  artificially creating soil compression. The depth increase to a maximum value specified by the user. The updated depth is used in the aggregate
  contacts to generate an elastic response. When the movement in the separation direction moves away from the contact plane ( contact separation ),
  the depth is decreased. Both the increase and the decrease of the depth can be adjusted according to scaling factors.
  */
  class AGXTERRAIN_EXPORT AggregateContactDepthModel
  {
  public:
    /**
    Default constructor
    */
    AggregateContactDepthModel();

    /**
    Update the depth in the model by using a reference contact point, separation normal of the excavation and the contact points of the
    terrain <-> aggregate contacts.
    \param referencePoint -
    \param separationNormal - The separation normal of the digging direction.
    \param contactPoints - The contact points in the aggregate <-> terrain contact.
    \param cos_contactAngleThreshold - cosine of the lower threshold angle between the separation normal
                                       and the contact plane normal where depth should start to accumulate.
    */
    void updateContactDepth( const agx::Vec3& referencePoint,
                             const agx::Vec3& separationNormal,
                             const agx::Vec3Vector& contactPoints,
                             agx::Real cos_contactAngleThreshold );

    /**
    Reset the contact history by erasing the current depth, contact plane and previous contact points.
    */
    void resetContactHistory();

    /**
    \return true if the contact model has a contact history, false otherwise.
    */
    bool hasContactHistory() const;

    /**
    \return the current depth (m) in the model.
    */
    agx::Real getCurrentDepth() const;

    /**
    \return the maximum depth (m) in the model. ( Default: 1.0 )
    */
    agx::Real getMaximumContactDepth() const;

    /**
    \return the depth decay factor of the model. This determines how fast the depth should decay when the movement
            in the separation direction moves _away_ from the contact plane ( contact separation ). ( Default: 2.0 )
    */
    agx::Real getDepthDecayFactor() const;

    /**
    \return the depth increase factor of the model. This determines how fast the depth should increase when the movement
            in the separation direction moves _towards_ the contact plane ( contact ). ( Default: 1.0 )
    */
    agx::Real getDepthIncreaseFactor() const;

    /**
    \return the aggregate contact area (m3) between the aggregate and the terrain, spanned by the contact points in that interface.
    \note - This is updated by the contact generator that the model belongs to.
    */
    agx::Real getAggregateContactArea() const;

    /**
    Update the current depth (m) of the model.
    \param currentDepth - the current depth that the model should have
    */
    void setCurrentDepth( agx::Real currentDepth );

    /**
    Set the depth decay factor of the model. This determines how fast the depth should decay when the movement
    in the separation direction moves _away_ from the contact plane ( contact separation ). ( Default: 2.0 )
    \param depthDecayFactor - the specified depth decay factor.
    */
    void setDepthDecayFactor(agx::Real depthDecayFactor);

    /**
    Set the depth increase factor of the model. This determines how fast the depth should increase when the movement
    in the separation direction moves _towards_ the contact plane ( contact ). ( Default: 2.0 )
    \param depthIncreaseFactor - the specified depth decay factor.
    */
    void setDepthIncreaseFactor(agx::Real depthIncreaseFactor);

    /**
    Set the maximum contact depth (m) in the model. ( Default: 1.0 )
    \param maximumDepth - The maximum depth of the model.
    */
    void setMaximumContactDepth( agx::Real maximumDepth  );

    /**
    Set the aggregate contact area (m3) in the model, spanned by the contact points in that interface.
    \note - This is updated by the contact generator that the model belongs to.
    \param area - the specified area to set to the model.
    */
    void setAggregateContactArea(agx::Real area);

    /**
    Set the color to use during debug rendering
    */
    void setDebugDrawColor(const agxRender::Color& color);

    /**
    Debug draw contact history
    */
    void DEBUG_DRAW_CONTACT_POINTS();

    /**
    Debug the depth meter, illustrating how far the current depth is from the maximum depth.
    */
    void DEBUG_DRAW_DEPTH_METER( const agx::Vec3& meterDirection );

    /**
    Debug the depth meter, illustrating how far the current depth is from the maximum depth.
    */
    void DEBUG_DRAW_CONTACT_PLANE( const agx::Vec3Vector& contactPoints );

    /**
    Trim the history of contact points
    */
    void trimPreviousContactPoints();

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    agx::Plane m_soilContactPlane;
    agx::Vec3  m_previousReferencePoint;
    agx::Real  m_maximumContactDepth;
    agx::Real  m_contactDepthDecayFactor;
    agx::Real  m_contactDepthIncreaseFactor;
    agx::Real  m_currentDepth;
    agx::Real  m_contactArea;
    bool       m_hasContactHistory;

    agx::Vec3Vector m_previousPoints;
    agxRender::Color m_debugColor;
  };

  /**
  Estimate the maximum force that can be generated from a soil aggregate in the contact between the aggregate and the terrain. This is based on the Mohr-Coulomb
  criteria for soil failure.
  \param collection - The specified collection.
  \param soilAggregateMass - the mass of the specified soil aggregate in the contact.
  \param contactArea - the contact area in the aggregate <-> terrain contact.
  \return the estimated maximum force that can be generated in the terrain <-> aggregate contact based on the Mohr-Coulomb condition for soil failure.
  */
  agx::Real AGXTERRAIN_EXPORT estimateMaximumAggregateForce( const TerrainToolCollection* collection,
                                                             agx::Real soilAggregateMass,
                                                             agx::Real contactArea,
                                                             agx::Real frictionAngle );

  /**
  Calculates the reference point that will be used in the contact model from specified contact points
  \param contactPoints - The contact points that is used to calculate the reference point used in the contact model
  \return the reference point to be used in the aggregate contact model
  */
  agx::Vec3 AGXTERRAIN_EXPORT calculateContactReferencePoint( const agx::Vec3Vector& contactPoints );
}
