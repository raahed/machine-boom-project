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

#include <agx/Component.h>

#include <agxStream/Serializable.h>

namespace agxCollide
{
  class Geometry;
}

namespace agxSDK
{
  class Simulation;

  /**
  agxTerrain::Terrain instance in a simulation, receiving step
  and contact callbacks in a controllable way to avoid interference
  with user step and contact callbacks.
  */
  class AGXPHYSICS_EXPORT TerrainInstance : public agx::Component, public agxStream::Serializable
  {
    public:
      /**
      \note It's undefined to modify this geometry instance.
      \return the geometry of this terrain
      */
      virtual agxCollide::Geometry* getGeometry() const;

      /**
      \return simulation this instance is part of, nullptr if not added to one
      */
      agxSDK::Simulation* getSimulation() const;

    public:
      DOXYGEN_START_INTERNAL_BLOCK()

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxSDK::TerrainInstance );
      DOXYGEN_END_INTERNAL_BLOCK()

    protected:
      /**
      Default constructor.
      */
      TerrainInstance();

      /**
      Reference counted object - protected destructor.
      */
      virtual ~TerrainInstance();

    protected:
      /**
      \internal

      Callback when this instance is added to a simulation.
      */
      virtual void addNotification();

      /**
      \internal

      Callback when this instance is removed from a simulation.
      */
      virtual void removeNotification();

      /**
      \internal

      Pre collide callback.
      */
      virtual void preCollide();

      /**
      \internal

      Pre solver callback.
      */
      virtual void pre();

      /**
      \internal

      Post solver callback.
      */
      virtual void post();

      /**
      \internal

      Callback to be executed at the end of the time step
      */
      virtual void last();

      /**
      \internal

      Run buildIslandsTask
      */
      virtual void runBuildIslandsTask();

    private:
      /**
      \internal

      Called from simulation when this instance is added or removed.
      */
      void setSimulation( agxSDK::Simulation* simulation );

    private:
      friend class Simulation;
      Simulation* m_simulation;
  };
}
