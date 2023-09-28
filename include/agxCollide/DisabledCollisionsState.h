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

#include <agx/Vector.h>
#include <agx/Name.h>

namespace agxCollide
{
  class Geometry;

  struct NamePair
  {
    agx::Name first;
    agx::Name second;
  };
  typedef agx::Vector<NamePair> NamePairVector;

  struct GroupIdPair
  {
    agx::UInt32 first;
    agx::UInt32 second;
  };
  typedef agx::Vector<GroupIdPair> GroupIdPairVector;

  struct GeometryPtrPair
  {
    agxCollide::Geometry* first;
    agxCollide::Geometry* second;
  };
  typedef agx::Vector<GeometryPtrPair> GeometryPtrPairVector;

  /**
  Complete disabled collisions state in a simulation including disabled
  given name, group id (integer) and geometry pair.
  */
  class AGXPHYSICS_EXPORT DisabledCollisionsState
  {
    public:
      /**
      Default constructor.
      */
      DisabledCollisionsState();

      /**
      \return the name <-> name disabled pairs
      */
      const NamePairVector& getDisabledNames() const;

      /**
      \return the id <-> id disabled pairs
      */
      const GroupIdPairVector& getDisabledIds() const;

      /**
      \return the geometry <-> geometry disabled pairs
      */
      const GeometryPtrPairVector& getDisabledGeometryPairs() const;

    public:

    private:
      friend class Space;
      void addDisabled( const agx::Name& name1, const agx::Name& name2 );
      void addDisabled( agx::UInt32 id1, agx::UInt32 id2 );
      void addDisabled( agxCollide::Geometry* geometry1, agxCollide::Geometry* geometry2 );

    private:
      NamePairVector m_disabledNames;
      GroupIdPairVector m_disabledIds;
      GeometryPtrPairVector m_disabledGeometryPairs;
  };

#ifndef SWIG
  std::ostream& operator << ( std::ostream& output, const DisabledCollisionsState& state );
#endif
}
