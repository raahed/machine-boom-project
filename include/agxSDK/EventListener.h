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

#ifndef AGXSDK_EVENTLISTENER_H
#define AGXSDK_EVENTLISTENER_H

#include <agxSDK/agxSDK.h>
#include <agx/Component.h>

#include <agxStream/Serializable.h>
#include <agx/SetVector.h>

namespace agxSDK
{

  class Simulation;

  AGX_DECLARE_POINTER_TYPES(EventListener);
  typedef agx::Vector<EventListener*> EventListenerPtrVector;
  typedef agx::SetVector< agx::ref_ptr<agxSDK::EventListener> > EventListenerRefSetVector;


  /**
  An EventListener can be associated to a Simulation and triggered upon various events.
  EventListener is a base class which is further specialized into StepEventListener, ContactEventListener and GuiEventListener
  */
  class AGXPHYSICS_EXPORT EventListener : public agx::Component, public agxStream::Serializable
  {
    public:

      /// Type of EventListener
      enum Type {
        UNDEFINED_TYPE,         /**< Undefined type */
        STEP_EVENT_LISTENER,    /**< Of type StepEventListener */
        CONTACT_EVENT_LISTENER, /**< Of type ContactEventListener */
        GUI_EVENT_LISTENER      /**< Of type GuiEventListener */
      };

      /**
      Specifies a bit mask which determines which event types that will activate this listener.
      The available masks are specified in derived classes
      */
      virtual void setMask(int f);

      /// \return the bitmask that specifies the event types for which a listener will be activated
      int getMask() const;

      /**
      Specify whether the listener should react at all at events.
      \param enable - If false, the listener will not be activated during any events
      */
      void setEnable( bool enable );

      /**
      \return true if the listener is enabled, that is can be activated.
      */
      bool isEnabled( ) const;

      /**
      \return a pointer to the simulation that triggered the listener
      */
      Simulation* getSimulation();

      /**
      \return a const pointer to the simulation that triggered the listener
      */
      const Simulation* getSimulation() const;

      /**
      Called when this listener is added to the simulation.
      (given that it not already is in the simulation).
      */
      virtual void addNotification() {}

      /**
      Called when this listener is removed from the simulation.
      */
      virtual void removeNotification() {}

      /// \return the type of this EventListener
      Type getType() const;

      /// \return the priority of this EventListener
      agx::UInt32 getPriority() const;

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::EventListener );
#endif

  private:
      /// Set the priority of this EventListener
      void setPriority( agx::UInt32 priority );

  protected:
      /**
      Default constructor
      \param type - Specifies the type of the EventListener
      */
      EventListener( Type type = UNDEFINED_TYPE );


      friend class Simulation;
      friend class EventManager;

      /// Set the associated simulation for this EventListener
      void setSimulation( Simulation* simulation );

      /// Destructor
      virtual ~EventListener();

      Type m_type;

      int m_mask;
      bool m_enabled;
      agx::UInt32 m_priority;
    private:

      Simulation* m_simulation;
  };

}

#endif
