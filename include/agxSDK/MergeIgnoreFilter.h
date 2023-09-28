/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once

#include <agx/HashSet.h>
#include <agxSDK/MergeIgnoreGroups.h>
#include <agxStream/OutputArchive.h>
#include <agxStream/InputArchive.h>


namespace agxSDK
{
  /**
  A collection of merge ignore group ID pairs for which merge should be
  rejected. The filter itself is owned and used by MergeSplitHandler and the
  merge ignore group IDs are held by MergeSplitProperties.
  */
  class AGXPHYSICS_EXPORT MergeIgnoreFilter
  {
    public:
      /**
      Add a rejection for the given group pair. The order the groups are passed
      does not matter.
      \param group1 - The first group.
      \param group2 - The second group.
      */
      void addRejection(agx::UInt32 group1, agx::UInt32 group2);

      /**
      Forget a previously added rejection pair. The order the groups are passed
      does not matter.
      \param group1 - The first group.
      \param group2 - The second group.
      */
      void removeRejection(agx::UInt32 group1, agx::UInt32 group2);


      /**
      Check if the given group pair is on the rejection list. The order the
      groups are passed does not matter.
      \param group1 - The first group.
      \param group2 - The second group.
      \return True if there is a rejection for the given pair. False otherwise.
      */
      agx::Bool rejects(agx::UInt32 group1, agx::UInt32 group2) const;

      /**
      Check if there is any pair in the cartesian product of the given group
      lists for which a rejection has been added.
      \param groups1 - The first set of groups.
      \param groups2 - The second set of groups.
      \return True if any rejected pair is found. False otherwise.
      */
      agx::Bool rejects(const agx::UInt32Vector& groups1, const agx::UInt32Vector& groups2) const;

      void store(agxStream::OutputArchive& archive) const;
      void restore(agxStream::InputArchive& archive);

    public:
      /**
      Compute the group ID that corresponds to the given group name.
      \param name - Name of the group to compute the group ID for.
      \return The group ID of the given group.
      */
      static agx::UInt32 nameToId(const agx::Name& name);

      /**
      Check if the given group ID corresponds to a named group.
      \param id - The group ID to check.
      \return True if the grou ID corresponds to a named group. False otherwise.
      */
      static agx::Bool isNameId(agx::UInt32 id);

      /**
      The maximum ID that may be used to create an unamed group. All group IDs
      created for named groups will be larger than this value;
      */
      static const agx::UInt32 MAX_UNAMED_GROUP_ID;


    private:
      // Set of pairs of UInt32 groups IDs, each pair stored with the smallest
      // group ID in the high half of the UInt64 and the higher group ID in the
      // low half.
      agx::HashSet<agx::UInt64> m_rejectPairs;
  };
}
