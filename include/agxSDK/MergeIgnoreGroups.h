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

#include <agx/agx_vector_types.h>


namespace agxStream
{
  class OutputArchive;
  class InputArchive;
}


namespace agxSDK
{
  /**
  A collection of merge ignore groups, some named and some not. Owned and
  maintained by MergeSplitProperties. Used by MergeSplitHandler to reject
  merges. The groups identified by ID is a subset of the groups identified by
  name. That is, adding a group by name will also add the corresponding group
  ID.
  */
  class AGXPHYSICS_EXPORT MergeIgnoreGroups
  {
    public:
      /**
      Add the given group ID. The ID must not correspond to a named group.
      \param id - The group ID to add.
      \return True if a new group was added. False if already part of the given group.
      */
      agx::Bool add(agx::UInt32 id);

      /**
      Add the given group name. The corresponding group ID will be added as well.
      \param name - The group name to add.
      \return True if a group was added. False if already part of the given group.
      */
      agx::Bool add(const agx::Name& name);

      /**
      Remove the given group ID. If the ID has a corresponding group name then
      the  name is removed as well. In that case it will be faster to pass the
      name to remove instead of the ID.
      \param id - The group ID to remove.
      \return True if the groups was removed. False otherwise.
      */
      agx::Bool remove(agx::UInt32 id);

      /**
      Remove the given group name. The corresponding group ID will be removed as
      well.
      \param name - The group name to remove.
      \return True if a group was removed. False otherwise.
      */
      agx::Bool remove(const agx::Name& name);

      /**
      \return Return group IDs for all groups, including named.
      */
      const agx::UInt32Vector& getIds() const;

      /**
      \return The named groups.
      */
      const agx::NameVector& getNames() const;

      void store(agxStream::OutputArchive& archive) const;
      void restore(agxStream::InputArchive& archive);

    private:
      agx::UInt32Vector m_ids;
      agx::NameVector m_names;
  };
}
