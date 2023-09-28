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
#ifndef AGXSDK_EVENTMANAGER_H
#define AGXSDK_EVENTMANAGER_H

#include <agxSDK/agxSDK.h>

#include <agxSDK/ContactEventListener.h>
#include <agxSDK/StepEventListener.h>
#include <agxSDK/GuiEventListener.h>
#include <agxSDK/GuiEvent.h>

#include <map>

#include <agxCollide/Space.h>

DOXYGEN_START_INTERNAL_BLOCK()


namespace agxSDK
{

  AGX_DECLARE_POINTER_TYPES(EventManager);

  /*!
  This class manages events that occurs in the simulation.
  */
  class AGXPHYSICS_EXPORT EventManager : public agx::Referenced
  {
    public:
      /// Specifies the priority order for an event listener.
      enum ExecutePriority {
        LOWEST_PRIORITY = 0,
        DEFAULT_PRIORITY = 50,
        HIGHEST_PRIORITY = 100
      };

      /**
      Specification of what should be cleaned up from a simulation during a call to the cleanup() method.
      These flags can be or:ed together.
      */
      enum CleanupSelectionMask {
        STEP_LISTENERS = 0x1,                                           //!< Remove all step contact listeners
        CONTACT_LISTENERS = 0x2,                                        //!< Remove all contact contact listeners
        GUI_LISTENERS = 0x4,                                            //!< Remove all gui event listeners
        LISTENERS = STEP_LISTENERS | CONTACT_LISTENERS | GUI_LISTENERS, //!< Remove all listeners
        CLEANUP_ALL = LISTENERS
      };

      /// Constructor
      EventManager(Simulation* simulation);

      /**
      Add a listener that listens to PRE and POST simulation step events.
      \param listener - A pointer to the listener to be added.
      \param priority - An execution priority for this listener. Higher executed earlier in the list of added listeners.
      Range of priority is [LOWEST_PRIORITY, HIGHEST_PRIORITY]
      \return false if listener is already added or priority is outside of the valid range, otherwise true.
      */
      bool addStepEventListener(StepEventListener* listener, int priority = DEFAULT_PRIORITY );

      /**
      Add a contact event listener. A contact listener can be added multiple times so that it listens using different
      execute filters.
      \param listener - Pointer to the listener that will be activated during events.
      \param filter - A filter that specifies for which contacts the listener should be activated.
      \param priority - An execution priority for this listener. Higher executed earlier in the list of added listeners.
      Range of priority is [LOWEST_PRIORITY, HIGHEST_PRIORITY]
      \return false if listener priority is outside of the valid range, otherwise true.
      */
      bool addContactEventListener(ContactEventListener* listener, int priority = DEFAULT_PRIORITY );

      /**
      Add a gui event listener
      \param listener - Pointer to the listener that will be activated during events.
      \return false if listener is already added to simulation.
      */
      bool addGuiEventListener(GuiEventListener* listener, int priority = DEFAULT_PRIORITY );

      /**
      Default add dispatcher, will dispatch to the correct add*Eventlistener depending on the type of the added EventListener
      */
      bool addEventListener(EventListener* listener, int priority = DEFAULT_PRIORITY );

      /**
      When called, the registered listeners listening for IMPACT/CONTACT contact events will be activated (if they pass the filter test).
      */
      void triggerContactEventListeners(const agx::TimeStamp& timeStamp,
                                        const agxCollide::GeometryContactPtrVector& contacts,
                                        agx::Bool isPost);

      /**
      When called, the registered listeners listening for SEPARATION contact events will be activated (if they pass the filter test).
      */
      void triggerSeparationContactEventListeners (const agx::TimeStamp& timeStamp,
                                                   const agxCollide::SeparationPairVector& separations);

      /**
      Find and return the first matching named event listener
      \param name - name of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      const agxSDK::EventListener* getEventListener(const agx::String& name) const;

      /**
      Find and return the first matching named event listener
      \param name - name of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      agxSDK::EventListener* getEventListener(const agx::String& name);

      /**
      Find (linear search) and return the first event listener matching the uuid
      \param uuid - uuid of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      const agxSDK::EventListener* getEventListener(const agx::Uuid& name) const;

      /**
      Find (linear search) and return the first event listener matching the uuid
      \param uuid - uuid of the event listener to find
      \return pointer to the found event listener, null if not found
      */
      agxSDK::EventListener* getEventListener(const agx::Uuid& name);

      /**
      Activate listeners listening for PRE_COLLIDE simulation step events
      */
      void triggerPreCollideStepEvent( const agx::TimeStamp& timeStamp );

      /**
      Activate listeners listening for PRE simulation step events
      */
      void triggerPreStepEvent( const agx::TimeStamp& timeStamp );


      /**
      Activate listeners listening for POST simulation step events
      */
      void triggerPostStepEvent( const agx::TimeStamp& timeStamp );

      /**
      Activate listeners listening for LAST simulation step events
      */
      void triggerLastStepEvent( const agx::TimeStamp& timeStamp );

      /**
      Remove the specified listener.
      */
      bool removeEventListener(EventListener* listener);

      /**
      Execute a gui event.
      \return true if a listener handled the event, false otherwise.
      */
      bool triggerGuiEvent( const GuiEvent& guiEvent );

      /**
      Enable detailed report and timing of each listener.
      \param enable - true to enable, false to disable
      */
      void setEnableDetailedStatisticsReport( bool enable );

      /**
      \return true if detailed statistics reporting is enabled, otherwise false
      */
      bool getEnableDetailedStatisticsReport() const;

      template <typename T>
      class PrioHolder
      {
        public:
          PrioHolder(  ):  m_priority( 0 ) {}
          PrioHolder( T obj, int priority ) : m_obj( obj ), m_priority( priority ) {}
          inline int priority() const {
            return m_priority;
          }
          inline operator T () {
            return m_obj;
          }
          inline operator const T () const {
            return m_obj;
          }
          inline operator int () const {
            return m_priority;
          }
          inline T listener() {
            return m_obj;
          }
          inline const T listener() const {
            return m_obj;
          }

        protected:
          T m_obj;
          int m_priority;
      };

      /**
      Get a vector containing all the registered eventlisteners
      */
      void getEventListeners( EventListenerPtrVector& eventListeners ) const;



    protected:


      void cleanup( CleanupSelectionMask selection );

      friend class Simulation;

    protected:


      typedef std::map< uint32_t, PrioHolder<StepEventListenerRef> > StepEventListeners;
      typedef std::map< uint32_t, PrioHolder<ContactEventListenerRef> > ContactEventListeners;
      typedef std::map< uint32_t, PrioHolder<GuiEventListenerRef> > GuiEventListeners;

    private:

      bool checkPriorityForClass( EventListener* listener, int priority );
      typedef agx::HashTable<agx::String, bool> SystemClassNames;
      void initSystemClassNameTable();

      SystemClassNames m_systemClassNames;

      template <typename T1, typename DATA1>
      class ListenerQueueCache
      {
        public:

          typedef agx::Vector<std::pair<DATA1, int> > SortableVectorType;
          typedef agx::Vector<DATA1> QueueType;
          typedef typename QueueType::iterator iterator;

          ListenerQueueCache() : m_dirty( true ), m_needRebuild( true ) {}

          void dirty(  ) {
            m_dirty = true;
          }
          void setNeedRebuild( ) {
            m_needRebuild = true;
          }

          void clear() {
            m_queue.clear();
            m_needRebuild = true;
            m_dirty = true;
          }

          bool isDirty() const {
            return m_dirty;
          }
          bool needRebuild() const {
            return m_needRebuild;
          }

          QueueType& queue() {
            return m_queue;
          }

          inline iterator begin() {
            return m_queue.begin();
          }
          inline iterator end() {
            return m_queue.end();
          }

#ifdef __APPLE__
          void build( T1& container) {
            bool doSort = true;
#else
          void build( T1& container, bool doSort = true ) {
#endif

            if ( needRebuild() ) {
              clear();
              SortableVectorType sortVector;
              sortVector.reserve( container.size() );

              typename T1::iterator cit = container.begin();
              for ( size_t i = 0; cit != container.end(); ++cit, ++i ) {
                sortVector.push_back( std::make_pair(cit->second.listener().get(), cit->second.priority()) );
              }

              if (doSort)
                std::sort( sortVector.begin(), sortVector.end(), prioGreater<std::pair<DATA1, int> >() );

              m_queue.reserve( sortVector.size() );
              for (size_t i = 0; i < sortVector.size(); i++) {
                m_queue.push_back( sortVector[i].first );
              }

              m_dirty = false;
              m_needRebuild = false;
            }

          }
        private:

#ifndef SWIG
          // Class for sorting based on priority
          template<class T>
          struct prioGreater  {
            bool operator()(const T& _Left, const T& _Right) const {
              return (_Left.second > _Right.second);
            }
          };
#endif

          QueueType m_queue;
          bool m_dirty;
          bool m_needRebuild;

      };

      Simulation* m_simulation;

      StepEventListeners m_step_listeners;
      typedef ListenerQueueCache<StepEventListeners, StepEventListenerRef> StepEventListenerCache;
      StepEventListenerCache m_step_listener_cache;

      ContactEventListeners m_contact_listeners;
      typedef ListenerQueueCache<ContactEventListeners, ContactEventListenerRef> ContactListenerCache;
      ContactListenerCache m_contact_listener_cache;

      GuiEventListeners m_gui_listeners;
      typedef ListenerQueueCache<GuiEventListeners, GuiEventListenerRef > GuiEventListenerCache;
      GuiEventListenerCache m_gui_listener_cache;

      agx::Timer m_eventTimer;
      agx::Vector<bool> m_insideCallback;

      bool m_detailedStatisticsReport;

    protected:

      /// Destructor
      virtual ~EventManager();
  };
}
DOXYGEN_END_INTERNAL_BLOCK()

#endif
