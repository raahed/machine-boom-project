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

#include <agx/Name.h>
#include <agx/agx_vector_types.h>

namespace agxCollide
{
  typedef agx::Vector<agx::Name> GroupNameVector;

  /**
  Collection of geometry group names and id's.
  */
  class AGXPHYSICS_EXPORT GroupIdCollection
  {
    public:
      /**
      Default constructor.
      */
      GroupIdCollection();

      /**
      \return group names
      */
      const GroupNameVector& getNames() const;

      /**
      \return group id's
      */
      const agx::UInt32Vector& getIds() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Add group name.
      */
      void add( const agx::Name& name );

      /**
      Add group id.
      */
      void add( agx::UInt32 id );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      GroupNameVector m_groupNames;
      agx::UInt32Vector m_groupIds;
  };
}
