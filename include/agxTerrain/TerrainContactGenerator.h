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

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#include <agxTerrain/export.h>
#include <agxTerrain/TerrainContact.h>
#include <agx/Referenced.h>
#include <agx/Vec3.h>
#include <agx/Vec2.h>
#include <agx/agx_vector_types.h>
#include <agx/Material.h>
#include <agx/HashSet.h>
#include <agx/Material.h>
#include <agx/Bound.h>
#include <agxCollide/Contacts.h>
#include <queue>

namespace agxSDK
{
  class Simulation;
}

namespace agxCollide
{
  class Geometry;
}

namespace agxTerrain
{
  class Terrain;

  typedef std::queue<agx::Vec2i> IndexQueue;

  AGX_DECLARE_POINTER_TYPES(TerrainContactGenerator);
  AGX_DECLARE_VECTOR_TYPES(TerrainContactGenerator);

  /**
  This class has two main responsibilities:

  - Generate TerrainContact objects between the terrain and externally colliding geometries. Primary used in compaction calculation.
  - Replace existing geometry contacts with contacts containing modified Young's Modulus depending on compaction of voxel in contact point.

  The generated TerrainContact objects contains information about approximative area and volume overlap which is used in compaction and deformation calculations.
  */
  class TerrainContactGenerator : public agx::Referenced
  {
  public:
    /**
    Default constructor
    */
    TerrainContactGenerator(Terrain* terrain);

    /**
    \internal

    Default constructor used in serialization.
    */
    TerrainContactGenerator();

    /**
    Do pre updates for the contact generator. This does the following:
    - Build Terrain contact objects from colliding objects
    - Generate new geometry contacts between colliders and the Terrain with updated Young's Modulus
    */
    void onPre(agxSDK::Simulation* simulation);

    /**
    Do post updates for the contact generator. This does the following:
    - Update the active TerrainContact objects with force data
    */
    void onPost(agxSDK::Simulation* simulation);

    /**
    Do cleanup when removed from the Simulation.
    */
    void onRemove(agxSDK::Simulation* simulation);

    /**
    \return the current generated Terrain contacts.
    */
    const TerrainContactRefVector& getTerrainContacts() const;

    /**
    \return the current generated Terrain contacts.
    */
    TerrainContactRefVector& getTerrainContacts();

  public:
    DOXYGEN_START_INTERNAL_BLOCK()
    AGXTERRAIN_STORE_RESTORE_INTERFACE;
    DOXYGEN_END_INTERNAL_BLOCK()

  protected:

    /**
    Build the TerrainContact objects used in compaction calculations
    */
    void buildTerrainContacts(agxSDK::Simulation* simulation);

    /**
    Update the TerrainContact objects with force data.
    */
    void updateTerrainContactsWithForceData();

    /**
    Replaces current geometry contacts with one new for each contact points containing a new Young's Modulus for each point depending on
    the compaction of the voxel.
    */
    void replaceTerrainGeometryContacts();

    /**
    Collect contacts between externally colliding geometries and the terrain (Does not include SoilParticleAggregateContacts).
    */
    void collectTerrainContacts(agxSDK::Simulation* simulation);

    /**
    Raycast in a terrain index in order to find intersection with a colliding geometry.
    \param collider - The colliding geometry to test ray intersection against.
    \param hfIndex - The index to raycast in.
    \param depth - The depth of the intersection of the geometry in the grid point.
    \return true if the line intersects with the collider below the height field.
    */
    bool raycastTerrainVertexToGeometry(const agxCollide::Geometry* collider, const agx::Vec2i& hfIndex, agx::Real& depth);

    /**
    Find the terrain grid indices intersecting with the colliding geometry via ray intersection tests.
    \param contacts - The geometry contacts associated with the contact between the collider and the terrain object
    \return table containing the terrain indices intersecting the geometry volume mapped to the depth of the intersection.
    */
    std::tuple<TerrainIndexToDepthTable, agx::Bound3> findIntersectingIndexInContact(const agxCollide::GeometryContactPtrVector& contacts);

    /**
    Get the current generated Terrain contacts.
    */
    TerrainContact* generateTerrainContact(const agxCollide::GeometryContactPtrVector& contacts, agxSDK::Simulation* simulation);

    /**
    Get the collider geometry in the geometry contact
    */
    const agxCollide::Geometry* getCollider(const agxCollide::GeometryContact* contact) const;

  protected:
    AGX_DECLARE_POINTER_TYPES(ContactReplacer);

    /**
    Internal class that is used to  create new geometry contacts between the terrain and the colliding objects. Generate new geometry
    contacts from each contact point in the old geometry contacts. These new geometry contacts have a modified Young's modulus based
    om the voxel compaction in the contact point position.
    */
    class ContactReplacer : public agx::Referenced
    {
    public:
      /**
      Default constructor
      */
      ContactReplacer(Terrain* terrain);

      /**
      Generate new geometry contacts from each contact point in the old geometry contacts. These new geometry contacts have a
      modified Young's modulus based om the voxel compaction in the contact point position.
      */
      agxCollide::LocalGeometryContactVector generateTerrainGeometryContacts(const agxCollide::GeometryContactPtrVector& geometryConacts);

      /**
      Create a new geometry contact from a contact point in an old geometry contact.
      */
      agxCollide::LocalGeometryContact createGeometryContactFromPoint(const agxCollide::GeometryContact* parentGeometryContact,
        const agxCollide::ContactPoint* cp,
        const agx::ContactMaterial* contactMaterial);

      /*
      Create a new geometry contact from a contact point in an old geometry contact.
      */
      agxCollide::LocalGeometryContact createGeometryContact(agxCollide::Geometry* g1, agxCollide::Geometry* g2,
        const agx::Vec3& point, const agx::Vec3f& normal, agx::Real depth, agx::Real pointArea, agx::ContactMaterial* cm);

      /**
      Generate a new contact material based on a new young's modulus computed from voxel compaction.
      */
      agx::ContactMaterial* createContactMaterial(const agx::ContactMaterial* originalCM, const agx::Real& newYoungs);

      /**
      Get modified Young's Modulus from depending on the compaction of the voxel of the specified position.
      \param position - The world position.
      \param existingContactMaterial - the existing contact material.
      \return The modified Young's Modulus from the position.
      */
      agx::Real getModifiedYoungsModuls(const agx::Vec3& position, const agx::ContactMaterial* existingContactMaterial) const;

      /**
      Empties the ContactMaterial pool.
      */
      void clearContactMaterialPool();

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~ContactReplacer();

    private:
      Terrain* m_terrain;
      agx::Vector<agx::ContactMaterialRef> m_temporaryContactMaterials;
      agx::Vector<agx::ContactMaterialRef> m_contactMaterialPool;
    };

  protected:
    /**
    Reference counted object - protected destructor.
    */
    virtual ~TerrainContactGenerator();

    void addNeighborIndex(
      const agx::Vec2i& index, std::queue<agx::Vec2i>& indicesToCheck, agx::HashSet<agx::Vec2i>& visitedIndices) const;
    void registerContactingGeometry(const agxCollide::Geometry* geometry, agxSDK::Simulation* simulation);
    void unRegisterContactingGeometry(const agxCollide::Geometry* geometry);
    void unRegisterGeometriesNotInContact(const agxCollide::GeometryHashSet& geometriesInContact);
    agx::Real getCurrentContactTime(const agxCollide::Geometry* geometry) const;
    bool isSoilAggregateContact(agxCollide::GeometryContact* gc) const;

  private:
    using GeometryContactTimeTable = agx::HashVector<const agxCollide::Geometry*, std::pair<agxCollide::GeometryConstObserver, agx::Real>>;

  private:
    Terrain* m_terrain;
    ContactReplacerRef m_terrainGeometryContactCreator;
    agxCollide::GeometryContactPtrVector m_currentContacts;
    GeometryContactTimeTable m_geometryToContactTime;
    TerrainContactRefVector m_currentTerrainContacts;
  };
}
