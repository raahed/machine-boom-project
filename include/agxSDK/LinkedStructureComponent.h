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

#include <agx/Referenced.h>

#include <agxStream/Serializable.h>

namespace agxSDK
{
  class Simulation;
  class LinkedStructure;

  AGX_DECLARE_POINTER_TYPES( LinkedStructureComponent );
  using LinkedStructureComponentContainer = agx::Vector<LinkedStructureComponentRef>;

  /**
  Base implementation of a component (e.g., specific functionality) in linked structures.
  Implementations will receive certain callbacks from the parent LinkedStructure object
  and simulation events, such as preCollide, pre, post and last step events.
  */
  class AGXPHYSICS_EXPORT LinkedStructureComponent : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      \return the given name of this component used to find this component from the linked
              structure this component is part of
      */
      const agx::Name& getName() const;

      /**
      Assign new name to this component. The name can be used to find this component from
      the linked structure this component is part of.
      \param name - new name of this component
      */
      void setName( const agx::Name& name );

      /**
      \return the simulation the linked structure has been added to - otherwise null
      */
      agxSDK::Simulation* getSimulation() const;

      /**
      \return the linked structure this component has been added to - otherwise null
      */
      agxSDK::LinkedStructure* getLinkedStructure() const;

    public:
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::LinkedStructureComponent );

    protected:
      // Callbacks and set simulation and linked structure.
      friend class LinkedStructure;

      /**
      Called when addNotification is executed when this linked
      structure is added to the simulation.
      */
      virtual void onAddNotification( agxSDK::Simulation* simulation );

      /**
      Called when this component is added to a linked structure.
      */
      virtual void onAddNotification( agxSDK::LinkedStructure* linkedStructure );

      /**
      Called when removeNotification is executed when this linked
      structure is removed from the simulation.
      */
      virtual void onRemoveNotification( agxSDK::Simulation* simulation );

      /**
      Called when this component is removed from a linked structure.
      */
      virtual void onRemoveNotification( agxSDK::LinkedStructure* linkedStructure );

      /**
      Called on simulation pre-collide step event - after user preCollide.
      */
      virtual void onPreCollideStep();

      /**
      Called on simulation pre step event - after user preStep.
      */
      virtual void onPreStep();

      /**
      Called on simulation post step event - after user postStep.
      */
      virtual void onPostStep();

      /**
      Called on simulation last step event - after user lastStep.
      */
      virtual void onLastStep();

    protected:
      /**
      Default, protected constructor. This object completely rely on a implementation
      so no point in just creating and adding an empty component to a linked structure.
      */
      LinkedStructureComponent( const agx::Name& name = "" );

      /**
      Reference counted object - protected destructor.
      */
      virtual ~LinkedStructureComponent();

    private:
      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Callback from linked structure when added to a simulation.
      \param simulation - simulation we're added to
      */
      void onAddToSimulation( agxSDK::Simulation* simulation );

      /**
      Callback from linked structure when removed from a simulation.
      \param simulation - simulation we're being removed from
      */
      void onRemoveFromSimulation( agxSDK::Simulation* simulation );

      /**
      Callback from linked structure when this component is added to or removed from the linked structure.
      \param linkedStructure - linked structure this component has been added to
      \param silent - false to trigger onAdd-event, true to not
      */
      void setLinkedStructure( agxSDK::LinkedStructure* linkedStructure, agx::Bool silent = false );
      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      agx::Name m_name;
      LinkedStructure* m_linkedStructure;
  };
}
