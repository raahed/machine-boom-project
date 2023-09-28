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

#ifndef AGXCOLLIDE_COLLISIONGROUPHANDLER_H
#define AGXCOLLIDE_COLLISIONGROUPHANDLER_H

#include <agx/HashSet.h>
#include <agx/AtomicValue.h>
#include <agx/Component.h>
#include <agx/Physics/CollisionGroupSetEntity.h>
#include <agx/Physics/CollisionGroupEntity.h>
#include <agxStream/Serializable.h>

namespace agxCollide
{
  AGX_DECLARE_POINTER_TYPES(CollisionGroupManager);
  class AGXPHYSICS_EXPORT CollisionGroupManager : public agx::Component, public virtual agxStream::Serializable
  {
  public:
    static CollisionGroupManager *defaultManager();

  public:
    // typedef agx::Vector< agx::UInt32 > GroupIdVector;
    typedef agx::Vector< agx::Physics::CollisionGroupPtr > GroupVector;
    typedef agx::Vector< agx::Physics::CollisionGroupSetPtr > GroupSetVector;
    typedef agx::HashSet< agx::UInt32 > GroupIdHash;
    typedef agx::HashSet< agx::Physics::CollisionGroupPtr > GroupHash;
    typedef agx::HashSet< agx::Physics::CollisionGroupSetPtr > GroupSetHash;

    typedef agx::HashSet<agx::UInt64> CollisionHash;

  public:
    CollisionGroupManager();

    /**
    Extend an existing group set by adding an additional group (can be nullptr pointer)
    \return The extended group set
    */
    agx::Physics::CollisionGroupSetPtr getExtendedSet(
      agx::Physics::CollisionGroupSetPtr set, agx::UInt32 groupId, bool unreferenceOldSet = true);

    /**
    Extend an existing group set by adding an additional group set
    \return The extended group set
    */
    agx::Physics::CollisionGroupSetPtr getExtendedSetFromAddedSet(
      agx::Physics::CollisionGroupSetPtr set, agx::Physics::CollisionGroupSetPtr set2, bool unreferenceOldSet = true);

    /**
    Reduce an existing group set by removing a group
    \return The reduced group set
    */
    agx::Physics::CollisionGroupSetPtr getReducedSet(
      agx::Physics::CollisionGroupSetPtr set, agx::UInt32 groupId, bool unreferenceOldSet = true);

    /**
    Get the matching set stored in another CollisionGroupManager
    NOTE: The old set must be manually unreferenced by calling unreferenceSet on the other manager
    \return The matching set in this manager
    */
    agx::Physics::CollisionGroupSetPtr getMatchingSet(agx::Physics::CollisionGroupSetPtr set);


    /**
    Unreference a set when it is no longer used.
    */
    void unreferenceSet(agx::Physics::CollisionGroupSetPtr set);

    /**
    Enable/disable collisions between two groups.
    */
    void setEnableCollisions(agx::UInt32 groupId1, agx::UInt32 groupId2, bool flag);

    /**
    \return true if the requested collision sets can collide. NOTE: Parameters are set IDs, not indices
    No mutex lock is used due to performance, assumes setEnableCollisions is not called at the same time
    */
    bool canSetsCollide(agx::UInt32 set1, agx::UInt32 set2) const;

    // Wrapper
    bool canSetsCollide(agx::Physics::CollisionGroupSetPtr set1, agx::Physics::CollisionGroupSetPtr set2) const;


    /**
    \return true if the requested collision groups can collide.
    No mutex lock is used due to performance, assumes setEnableCollisions is not called at the same time
    */
    bool canGroupsCollide(agx::UInt32 group1, agx::UInt32 group2) const;

    /**
    \return A unique group id.
    */
    agx::UInt32 generateUniqueId();

    static bool hasGroup(agx::Physics::CollisionGroupSetPtr set, agx::UInt32 groupId);
    static bool hasGroup(agx::Physics::CollisionGroupSetPtr set, const agx::Name& groupName);
    static bool verify(agx::Physics::CollisionGroupSetPtr set);

    const CollisionHash& getDisabledGroupCollisions() const;
    const CollisionHash& getDisabledSetCollisions() const;

    static void printSet(const agx::Physics::CollisionGroupSetPtr& set);

    // Not thread safe
    bool hasGroup(const agx::Name& groupName) const;
    bool hasGroup(agx::UInt32 groupId) const;

    // Not thread safe
    agx::UInt32 getGroupId(const agx::Name& groupName);
    agx::Physics::CollisionGroupPtr getGroup(const agx::Name& groupName);
    agx::Physics::CollisionGroupPtr getGroup(agx::UInt32 groupId);

    typedef agx::Vector<agx::SymmetricPair<agx::Physics::CollisionGroupPtr> > SymmetricCollisionGroupVector;

    /// \returns a vector of all pairs of disabled collision groups
    SymmetricCollisionGroupVector getDisabledCollisionGroupPairs() const;

    const agxData::EntityStorage* getCollisionGroupStorage() const;
    const agxData::EntityStorage* getCollisionGroupSetStorage() const;

    void internal_restore(agxStream::InputArchive& in);

    AGXSTREAM_DECLARE_SERIALIZABLE(agxCollide::CollisionGroupManager);

  protected:
    virtual ~CollisionGroupManager();

  private:
    friend class Space;
    agx::UInt32 generateNameId();
    void setUniqueIdCounter(agx::UInt32 id);
    void setNameIdCounter(agx::UInt32 id);
    agx::Physics::CollisionGroupPtr restoreGroup(const agx::Name& groupName, agx::UInt32 id);


  private:
    agx::Physics::CollisionGroupSetPtr getUnarySet(agx::UInt32 groupId);
    void internal_unreferenceSet(agx::Physics::CollisionGroupSetPtr set);
    bool internal_canCollide(const GroupIdHash& groups1, const GroupIdHash& groups2);
    void removeInactiveSets();
    void registerNewSet(agx::Physics::CollisionGroupSetPtr set);
    void reset();

    class GroupSetCache
    {
    public:
      class Bucket
      {
      public:
        enum State
        {
          INACTIVE,
          ACTIVE,
          VALID
        };

        agx::Physics::CollisionGroupSetPtr groupSet;

        AGX_FORCE_INLINE Bucket() : state(INACTIVE)
        {
        }

        AGX_FORCE_INLINE Bucket& operator=(const Bucket& b )
        {
          groupSet = b.groupSet;
          state = b.state;

          return *this;
        }

        AGX_FORCE_INLINE bool isActive() const
        { return state >= ACTIVE; }

        AGX_FORCE_INLINE bool isValid() const
        {
          // prefetch( &state, agx::NTA );
          return state == VALID;
        }

      private:
        friend class GroupSetCache;
        agx::UInt8 state;
      };


    public:
      GroupSetCache() :
        m_size(0), m_capacity(0), m_buckets(nullptr),
        m_maxProbeLength(0)
      {
        this->rebuild(0);
      }

      void clear()
      {
        deallocateBuckets();
        m_size = 0;
        m_capacity = 0;
        m_buckets = nullptr;
        m_maxProbeLength = 0;
        rebuild(0);
      }

      ~GroupSetCache()
      {
        deallocateBuckets();
      }


      inline agx::Physics::CollisionGroupSetPtr find(agx::Physics::CollisionGroupPtr group) const
      {
        agx::UInt32 bucketIndex = findBucket(&group, 1);
        return bucketIndex != agx::InvalidIndex ? m_buckets[bucketIndex].groupSet : agx::Physics::CollisionGroupSetPtr();
      }

      inline agx::Physics::CollisionGroupSetPtr findExtended(agx::Physics::CollisionGroupSetPtr set, agx::Physics::CollisionGroupPtr group) const
      {
        agxAssert(set && set.groups().size() + 1 < 1024);
        agx::Physics::CollisionGroupPtr localGroups[1024];
        size_t numGroups = 0;

        // Build extended list, insert group in already sorted vector
        const GroupVector& groups = set.groups();
        size_t i = 0;
        for (; i < groups.size() && groups[i].id() < group.id(); ++i)
          localGroups[numGroups++] = groups[i];

        localGroups[numGroups++] = group;

        for (; i < groups.size(); ++i)
          localGroups[numGroups++] = groups[i];


        agx::UInt32 bucketIndex = findBucket(localGroups, numGroups);
        return bucketIndex != agx::InvalidIndex ? m_buckets[bucketIndex].groupSet : agx::Physics::CollisionGroupSetPtr();
      }

      inline agx::Physics::CollisionGroupSetPtr findReduced(agx::Physics::CollisionGroupSetPtr set, agx::Physics::CollisionGroupPtr group) const
      {
        agxAssert(set && set.groups().size() - 1 < 1024);
        agx::Physics::CollisionGroupPtr localGroups[1024];
        size_t numGroups = 0;

        // Build reduced list
        const GroupVector& groups = set.groups();
        for (size_t i = 0; i < groups.size(); ++i)
        {
          if (groups[i].id() == group.id())
            continue;

          localGroups[numGroups++] = groups[i];
        }

        agx::UInt32 bucketIndex = findBucket(localGroups, numGroups);
        return bucketIndex != agx::InvalidIndex ? m_buckets[bucketIndex].groupSet : agx::Physics::CollisionGroupSetPtr();
      }

      inline agx::Physics::CollisionGroupSetPtr find(agx::Physics::CollisionGroupSetPtr set) const
      {
        if (!set)
          return agx::Physics::CollisionGroupSetPtr();

        const GroupVector& groups = set.groups();

        agx::UInt32 bucketIndex = findBucket(groups.begin(), groups.size());
        return bucketIndex != agx::InvalidIndex ? m_buckets[bucketIndex].groupSet : agx::Physics::CollisionGroupSetPtr();
      }

      void insert(agx::Physics::CollisionGroupSetPtr set)
      {
        agxAssert(set);
        const GroupVector& groups = set.groups();

        agx::UInt32 hashValue = calculateHash(groups.begin(), groups.size());

        Bucket *bucket;
        Bucket *targetBucket = 0;
        size_t offset = 0;
        // std::cout << "Insert " << cellIndex << " with id " << cellId << std::endl;

        /* Check if we should grow and rebuild hash table */
        if (m_size >= (size_t)((float)m_capacity * AGX_HASH_SET_GROW_THRESHOLD))
          this->rebuild((size_t)((float)(m_capacity+1) * AGX_HASH_SET_GROW_FACTOR));

        /* Iterate until target bucket is found */
        bool goon = true;
        do
        {
          bucket = &m_buckets[(hashValue + offset * offset) % m_capacity];

          /* Check if overwriting previous insertion with same key */
          if (bucket->isValid() && compareGroups(groups.begin(), groups.size(), bucket->groupSet))
          {
            targetBucket = bucket;
            break;
          }

          /* Tag first free slot */
          if (!bucket->isValid() && !targetBucket)
            targetBucket = bucket;

          /* If we have found a free bucket we can stop iterating when we are sure that the key was not previously inserted */
          if (targetBucket && offset >= m_maxProbeLength)
            break;

          /* Go to next bucket */
          offset++;

          /* Resize and rebuild table if it takes too long to find a free slot */
          if (offset > AGX_HASH_SET_MAX_PROBE_LENGTH)
          {
            this->rebuild((size_t)((float)m_capacity * AGX_HASH_SET_GROW_FACTOR));
            return this->insert(set);
          }
        } while (goon);

        agxAssert(targetBucket);

        if (offset > m_maxProbeLength)
          m_maxProbeLength = (agx::UInt16)offset;

        targetBucket->groupSet = set;

        /* Check if inserting a new value rather than overwriting an old entry */
        if (!targetBucket->isValid())
        {
          m_size++;
          targetBucket->state = Bucket::VALID;
        }
      }

      bool contains(agx::Physics::CollisionGroupSetPtr set) const
      {
        return find(set).isValid();
      }


      void erase(agx::Physics::CollisionGroupSetPtr set)
      {
        agxAssert(set);
        const GroupVector& groups = set.groups();

        agx::UInt32 bucketIndex = findBucket(groups.begin(), groups.size());
        agxAssert(bucketIndex != agx::InvalidIndex);

        Bucket *bucket = &m_buckets[bucketIndex];
        agxAssert(bucket->isValid() && compareGroups(groups.begin(), groups.size(), bucket->groupSet));

        m_size--;

        // Bucket does not become inactive since it might be in the middle of a probing chain
        bucket->state = Bucket::ACTIVE;
      }

    private:

      AGX_FORCE_INLINE agx::UInt32 findBucket(const agx::Physics::CollisionGroupPtr *localGroups, size_t numGroups) const
      {
        agxAssert(m_buckets);
        if (m_buckets) {

          agx::UInt32 hashValue = calculateHash(localGroups, numGroups);

          size_t index = hashValue % m_capacity;
          const Bucket *bucket = &m_buckets[index];

          /* First test is outside for-loop, hopefully getting better branch prediction */
          if (bucket->isValid() && compareGroups(localGroups, numGroups, bucket->groupSet))
            return (agx::UInt32)index;


          for (size_t offset = 1; bucket->isActive() && offset <= m_maxProbeLength; offset++)
          {
            index  = (hashValue + offset * offset) % m_capacity;
            bucket = &m_buckets[index];

            if (bucket->isValid() && compareGroups(localGroups, numGroups, bucket->groupSet))
              return (agx::UInt32)index;
          }
        }

        /* Not found */
        return agx::InvalidIndex;
      }

      AGX_FORCE_INLINE agx::UInt32 calculateHash(const agx::Physics::CollisionGroupPtr *localGroups, size_t numGroups) const
      {
        // Calculate hash
        agx::UInt32 hashValue = 0;
        for (size_t i = 0; i < numGroups; ++i)
          hashValue = agx::hash(hashValue, agx::hash(localGroups[i].id()));

        return hashValue;
      }


      AGX_FORCE_INLINE bool compareGroups(const agx::Physics::CollisionGroupPtr *localGroups, size_t numGroups, agx::Physics::CollisionGroupSetPtr set) const
      {
        agxAssert(set);
        const GroupVector& groups = set.groups();

        if (groups.size() != numGroups)
          return false;

        for (size_t i = 0; i < groups.size(); ++i)
        {
          if (groups[i].id() != localGroups[i].id())
            return false;
        }

        return true;
      }

      void rebuild(size_t size)
      {
        size = std::max( size, size_t(AGX_HASH_SET_MIN_SIZE) );
        //LOGGER_DEBUG() << "HashTable: Rebuilding and resizing from " << numBuckets() << " to " << agx::nextPrime((int)(numBuckets() * resizeFactor)) << " buckets.\n" << LOGGER_END();
        agxAssert(size >= m_size);
        size_t oldNumBuckets = m_capacity;
        Bucket *oldBuckets = m_buckets;

        m_capacity = agx::nextPrime((int)size);
        m_buckets = (Bucket *)m_allocator.allocateBytes(m_capacity * sizeof(Bucket));
        for (size_t i = 0; i < m_capacity; ++i)
          ::new (&m_buckets[i]) Bucket();

        m_size = 0;
        m_maxProbeLength = 0;

        for (size_t i = 0; i < oldNumBuckets; i++)
        {
          Bucket& bucket = oldBuckets[i];

          if (bucket.isValid())
            this->insert(bucket.groupSet);
        }

        m_allocator.deallocateBytes((void *)oldBuckets, oldNumBuckets * sizeof(Bucket));
      }

      void deallocateBuckets()
      {
        m_allocator.deallocateBytes(m_buckets, m_capacity * sizeof(Bucket));
      }

    private:
      size_t m_size;
      size_t m_capacity;
      Bucket *m_buckets;

      agx::UInt16 m_maxProbeLength;
      agx::ByteAllocator m_allocator;
    };


  private:
    agx::AtomicValue m_nameIdCounter;
    agx::AtomicValue m_uniqueIdCounter;
    agx::Mutex m_mutex;
    GroupSetCache m_groupSetCache;
    agxData::EntityStorageRef m_collisionGroupSetStorage;

    CollisionHash m_disabledSetCollisions;
    CollisionHash m_disabledGroupCollisions;

    agx::HashSet<agx::Physics::CollisionGroupSetPtr> m_inactiveSets;

    agxData::EntityStorageRef m_collisionGroupStorage;

    typedef agx::HashTable<agx::UInt32, agx::Physics::CollisionGroupPtr> CollisionGroupTable;
    CollisionGroupTable m_collisionGroupTable;

    typedef agx::HashVector<agx::Name, agx::UInt32> CollisionGroupNameTable;
    CollisionGroupNameTable m_collisionGroupNameTable;
  };


  AGXPHYSICS_EXPORT std::ostream& operator << ( std::ostream& output, const agx::Physics::CollisionGroupSetPtr& set );
  AGXPHYSICS_EXPORT std::ostream& operator << ( std::ostream& output, const agx::Physics::CollisionGroupPtr& group );




  /* Implementation */

  AGX_FORCE_INLINE bool CollisionGroupManager::canSetsCollide(agx::UInt32 set1, agx::UInt32 set2) const
  {
    // std::cout << "canSetsCollide " << set1 << " : " << set2 << " => " << !m_disabledSetCollisions.contains(agx::buildHashKey(set1, set2)) << std::endl;
    return !m_disabledSetCollisions.contains(agx::buildHashKey(set1, set2));
  }

  AGX_FORCE_INLINE bool CollisionGroupManager::canSetsCollide(agx::Physics::CollisionGroupSetPtr set1, agx::Physics::CollisionGroupSetPtr set2) const
  {
    agxAssert(set1);
    agxAssert(set2);
    agxAssert(set1.getStorage() == m_collisionGroupSetStorage);
    agxAssert(set2.getStorage() == m_collisionGroupSetStorage);
    return this->canSetsCollide((agx::UInt32)set1.getId(), (agx::UInt32)set2.getId());
  }

  AGX_FORCE_INLINE bool CollisionGroupManager::canGroupsCollide(agx::UInt32 group1, agx::UInt32 group2) const
  {
    return !m_disabledGroupCollisions.contains(agx::buildHashKey(group1, group2));
  }

  AGX_FORCE_INLINE const CollisionGroupManager::CollisionHash& CollisionGroupManager::getDisabledGroupCollisions() const { return m_disabledGroupCollisions; }
  AGX_FORCE_INLINE const CollisionGroupManager::CollisionHash& CollisionGroupManager::getDisabledSetCollisions() const { return m_disabledSetCollisions; }

}


#endif /* AGXCOLLIDE_COLLISIONGROUPHANDLER_H */
