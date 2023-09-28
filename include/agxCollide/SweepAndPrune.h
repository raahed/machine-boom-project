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
#ifndef AGXCOLLIDE_SWEEPANDPRUNE_H
#define AGXCOLLIDE_SWEEPANDPRUNE_H

#include <agxCollide/Geometry.h>

#include <agx/Component.h>
#include <agx/Task.h>

#include <agx/Physics/BroadPhasePairEntity.h>
#include <queue>


#include <HashImplementationSwitcher.h>
#include <agx/LinearProbingHashTable.h>
#include <agx/LinearProbingHashSet.h>
#include <agx/QuadraticProbingHashTable.h>

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(SweepAndPrune);
  class AGXPHYSICS_EXPORT SweepAndPrune : public agx::Component
  {
    public:
      typedef agx::SymmetricPair< GeometryRef > SymmetricGeometryRefPair;
      typedef agx::SymmetricPair< Geometry* > SymmetricGeometryPtrPair;
      typedef agx::HashVector< SymmetricGeometryPtrPair, SymmetricGeometryRefPair > ExternallyReportedPairsContainer;

    public:
      /**
      Default constructor.
      */
      SweepAndPrune(Space *space);

      void add( Geometry* geometry );
      void remove( Geometry* geometry );
      void update();
      void commitNewOverlaps( agxData::EntityStorage* pairStorage, ExternallyReportedPairsContainer& newPairs );
      virtual bool testOverlap( agxCollide::GeometryPtrVector& testGeometries, agxCollide::GeometryPairVector& result, bool skip );

      /**
      Remove broad phase overlap given two geometries. It doesn't matter in which order one pass the geometries.
      \param geometry1 - first geometry
      \param geometry2 - second geometry
      */
      void removePair( const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2 );

      /**
      \return the broad phase pair between geometry1 and geometry2 if any - i.e., check so that the entity pointer is valid before use
      */
      agx::Physics::BroadPhasePairPtr getPair( const Geometry* geometry1, const Geometry* geometry2 ) const;

      agx::Task *createUpdateTask();
      void print(); //DEBUG


    public:
      class UpdateTask : public agx::SerialTask
      {
        public:
          UpdateTask(const agx::String& name, SweepAndPrune *sap);
          virtual void run() override;

        protected:
          virtual ~UpdateTask() {}

        private:
          SweepAndPrune *m_sap;

          agx::ArrayParameterRef m_boundingVolumes; //agx::Arg<agx::Bound3> m_boundingVolumes;
          agx::ArrayParameterRef m_geometries; //agx::Arg<agxData::EntityPtr> m_geometries;
          // agx::Arg<agx::Physics::RigidBodyPtr> m_bodies;
          // agx::Arg<agxCollide::BroadPhasePair> m_overlapPairs;
          agx::PointerT<agxData::EntityStorage> *m_pairStorage;
          agx::PointerT<agxData::EntityStorage> *m_separationStorage;
          // agx::Ptr<agxData::EntityStorage> m_cellBlockPairStorage;
          // agx::Arg<agx::Physics::GeometryPtr> m_pairGeometries1;
          // agx::Arg<agx::Physics::GeometryPtr> m_pairGeometries2;
      };

    protected:
      virtual ~SweepAndPrune( void );

    private:
      void insertNewGeometries();
      void encodeBounds();
      void tagRemovedBounds();
      void sortAxis(int axis);
      void removeGeometries();
      void reportOverlapsAndSeparations(agxData::EntityStorage *pairStorage, agxData::EntityStorage *separationStorage);

      void addPair( agx::UInt32 i1, agx::UInt32 i2 );
      void removePair( agx::UInt32 i1, agx::UInt32 i2 );

    public:
      // 8 bytes
      struct endpoint {
        agx::UInt32 m_index; // bounding box index | isMax
        agx::UInt32 m_value; // bounding box value encoded as uint
      };

      DOXYGEN_START_INTERNAL_BLOCK()
      // 28 or 32 bytes
      struct bbox {
        agx::UInt32 m_epi[2][3];
        Geometry* m_geometry;
      };
      DOXYGEN_END_INTERNAL_BLOCK()
    private:
      BoundingAABB m_totalAABB;

#if HASH_FOR_SWEEP == HASH_NEW
      using saptable = agx::LinearProbingHashSet<agx::UInt64>;
#elif HASH_FOR_SWEEP == HASH_OLD
      typedef agx::HashSet< agx::UInt64 > saptable;
#else
      #error
#endif

      class queuecmp {
        public:
          bool operator()(agx::UInt32 a, agx::UInt32 b) { return a > b; }
      };

      std::priority_queue<agx::UInt32, std::deque<agx::UInt32>, queuecmp> m_holes;

      typedef agx::SetVector< GeometryRef > GeometryRefSet;

      typedef std::pair<GeometryRef, agx::UInt32> RemovedGeometry;
      typedef agx::SetVector< RemovedGeometry > RemovedGeometrySet;


      saptable m_contactTable;
      saptable m_impactTable;
      saptable m_separationTable;


      void insertNewGeometry( Geometry* geom, agx::UInt32 axis, size_t insertIndex, size_t endpointIndex );
      bool boxOverlap( agx::UInt32 i1, agx::UInt32 i2, const agx::UInt32 axes[2] );

      GeometryRefSet m_addGeometries;
      RemovedGeometrySet m_removeGeometries;

      size_t m_numBoxes;

      size_t m_numBoxesAvailable;
      size_t m_numEndpointsAvailable;

      agx::Vector<bbox> m_boxes;

      agx::Vector<endpoint> m_endpoints[3];
      agx::TaskRef m_updateTask;
      Space *m_space;

#if HASH_FOR_SWEEP == HASH_NEW
      using ActivePairTable = agx::LinearProbingHashTable<agx::UInt64, agx::Physics::BroadPhasePairPtr>;
#elif HASH_FOR_SWEEP == HASH_OLD
      typedef agx::QuadraticProbingHashTable<agx::UInt64, agx::Physics::BroadPhasePairPtr> ActivePairTable;
#else
      #error
#endif
      ActivePairTable m_activePairs;

      agxData::EntityPtrVector m_pairRemovalList;
  };
}

#endif
