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

#ifndef AGX_CONTACTZONEMANAGER_H
#define AGX_CONTACTZONEMANAGER_H

#include <agx/Random.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/Physics/HierarchicalGrid/CellEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZone_InternalEntity.h>
#include <agx/Physics/HierarchicalGrid/ContactZoneDependencyEntity.h>
#include <agx/Physics/HierarchicalGrid/GridTierEntity.h>

#include "Util.h"
#include "Common.h"


namespace agx { namespace Physics { namespace HierarchicalGrid {

  // typedef agx::HashTable<agx::Vec3i, agx::Physics::HierarchicalGrid::ContactZonePtr, agx::HashFn< agx::Vec3i >, agxData::BufferProxyAllocator> ContactZoneTable;
  // typedef agx::HashTable<agx::UInt64, agx::Physics::HierarchicalGrid::ContactZoneDependencyPtr, agx::HashFn< agx::UInt64 >, agxData::BufferProxyAllocator> ZoneDependencyTable;

  class ContactZoneManager
  {
  public:
    ContactZoneManager(
      CellData& cell,
      ContactZoneData& zone,
      ContactZoneDependencyData& dependency,
      GridTierData& gridTier,
      // ContactZoneTable& zoneTable,
      // ZoneDependencyTable& zoneDependencyTable,
      UInt contactZoneAccumulationLevel,
      ContactZonePtr& iterationStartZone,
      ContactZonePtr& iterationEndZone
    );

    ~ContactZoneManager();

    ContactZoneInstance getContactZone(Index cellIndex1, Index cellIndex2);

    ContactZonePtr createInternalZone(Index cellIndex, Index tierIndex);
    ContactZonePtr createFaceZone(const Vec3i& id, Index tierIndex);
    ContactZonePtr createEdgeZone(const Vec3i& id, Index tierIndex);
    ContactZonePtr createCornerZone(const Vec3i& id, Index tierIndex);
    ContactZonePtr createBoundaryZone(const Vec3i& id, Index tierIndex);

    void removeZones(agxData::Array< ContactZonePtr >& emptyZones);


    // Automatically called by destructor
    void commitDependencyStorage();

  private:
    // void addDependency(ContactZonePtr parent, ContactZonePtr child);
    // void removeDependency(ContactZonePtr parent, ContactZonePtr child);
    // void addChildDependenciesToCorner(ContactZonePtr parent, const Vec3i& cornerId);
    // void addChildDependenciesToEdge(ContactZonePtr parent, const Vec3i& edgeId);
    // void addParentDependenciesToInternal(ContactZonePtr child, const Vec3i& internalId);
    // void addParentDependenciesToEdge(ContactZonePtr child, const Vec3i& edgeId);
    // void removeRedundantDependencies(ContactZonePtr zone);

    ContactZonePtr createZone(const Vec3i& id, ZoneCategory zoneType, Index tierIndex);

    // To calm the compiler
    ContactZoneManager& operator=(const ContactZoneManager&) { return *this; }

    Vec4f generateRandomColor();

  private:
    CellData& m_cell;
    ContactZoneData& m_zone;
    ContactZoneDependencyData& m_dependency;
    GridTierData& m_gridTier;
    // ContactZoneTable& m_zoneTable;
    // ZoneDependencyTable& m_zoneDependencyTable;
    UInt m_contactZoneAccumulationLevel;

    // The following two members are used only if CALCULATE_INCREMENTAL_DEPENDENCIES in
    // the .cpp file is none-zero, which it usually isn't.
    #ifdef __clang__
    # pragma clang diagnostic push
    # pragma clang diagnostic ignored "-Wunused-private-field"
    #endif
    ContactZonePtr& m_iterationStartZone;
    ContactZonePtr& m_iterationEndZone;
    #ifdef __clang__
    # pragma clang diagnostic pop
    #endif

    Vector<ContactZoneDependencyPtr> m_removedDependencies;

    UniformReal32Generator m_randomGenerator;
  };




}}}


#endif /* AGX_CONTACTZONEMANAGER_H */
