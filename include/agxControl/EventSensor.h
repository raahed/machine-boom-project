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

#ifndef AGXCONTROL_EVENTSENSOR_H
#define AGXCONTROL_EVENTSENSOR_H
#include <agx/config.h>

#include <agx/agxPhysics_export.h>
#include <agxSDK/StepEventListener.h>
#include <agx/TimeStamp.h>
#include <agxCollide/Geometry.h>
#include <agxSDK/ContactEventListener.h>
#include <agx/ParticleContactSensor.h>
#include <agx/HashSet.h>
#include <agxSDK/CollisionGroupFilter.h>
#include <map>
#include <agxControl/SensorEvent.h>
#include <agx/Physics/ParticlePairContactEntity.h>

namespace agxControl
{

  class SensorEvent;

  AGX_DECLARE_POINTER_TYPES(EventSensor);
  AGX_DECLARE_VECTOR_TYPES(EventSensor);

  /**
  * This is a sensor class that can be attached to an geometry in the simulation.
  * When a collision is detected, according to the set rules of activation for the sensor,
  * it will start to execute SensorEvents in a prioritized order. These sensor events can either modify
  * the simulation or record data.
  */
  class CALLABLE AGXPHYSICS_EXPORT EventSensor : public agxSDK::StepEventListener
  {
    public:
      typedef agx::Vector < agx::Physics::ParticlePairContactPtr > ParticlePairContactVector;
      typedef agx::Vector < agxControl::SensorEvent * >            SensorEventVector;

    protected:
      // This is a static list that contains forbidden geometries. And exclusive sensor can mark
      // geometries as forbidden after they have been handled, which prevents other sensors form detecting them.
      // The same goes for particles.
      static agxCollide::GeometryPtrVector g_forbiddenGeometries;
      static agx::UInt32Vector g_forbiddenParticles;

    public:
      /// Defines the event execution states when the SensorEvents are executed during the timestep of the detected collision with the sensor
      enum TriggerEventStages
      {
        SENSOR_PRE_STEP, /**< Triggered in a PRE solve/integration step */
        SENSOR_POST_STEP /**< Triggered in a POST solve/integration step */
      };

     protected:

      AGX_DECLARE_POINTER_TYPES(GeometryContactEventListener);
      AGX_DECLARE_VECTOR_TYPES(GeometryContactEventListener) ;
      /*
      * This is an internal class handling the geometry contact triggers of the simulation
      */
      class GeometryContactEventListener : public agxSDK::ContactEventListener
      {
        public:

          GeometryContactEventListener(EventSensor* sensor);

          virtual KeepContactPolicy impact(const agx::TimeStamp&, agxCollide::GeometryContact*);
          virtual KeepContactPolicy contact( const agx::TimeStamp&, agxCollide::GeometryContact*);
          virtual void separation(const agx::TimeStamp& , agxCollide::GeometryPair&);

          friend class EventSensor;

        protected:
          virtual ~GeometryContactEventListener();

          EventSensor* m_sensor;
      };

      AGX_DECLARE_POINTER_TYPES(ParticleContactListener);
      AGX_DECLARE_VECTOR_TYPES(ParticleContactListener) ;

      /*
      * This is an internal class handling the geometry contact triggers of the simulation
      */
      class ParticleContactListener : public agx::ParticleContactSensor
      {
      public:

        ParticleContactListener(EventSensor* sensor);

        void setFilter(agxSDK::CollisionGroupFilter* filter);

        agxSDK::CollisionGroupFilter* getFilter();
        const agxSDK::CollisionGroupFilter* getFilter() const;

        virtual void contactCallback(agx::Physics::ParticleGeometryContactInstance contact,
          agx::Physics::ParticleData& particleData,
          agx::Physics::GeometryData& geometryData);

        friend class EventSensor;

      protected:
        virtual ~ParticleContactListener();

      protected:
        EventSensor* m_sensor;
        agxSDK::CollisionGroupFilter* m_filter;
      };

    public:

      /// Default constructor
      EventSensor();

      /**
      This constructor takes the priority of the internal contact sensors
      \param priority - Priority of the internal contact listeners
      */
      EventSensor(int priority);

      /**
      Constructor that takes the geometry to couple the sensor too.
      \param geometry The geometry that the sensor will register all the contacts from
      */
      EventSensor(agxCollide::Geometry* geometry);

      /**
      Constructor that takes the geometry to couple the sensor too. All collision on the geometry will be registered by the sensor.
      \param geometry The geometry that the sensor will register all the contacts from
      \param priority - Priority of the internal contact listeners
      */
      EventSensor(agxCollide::Geometry* geometry, int priority);

      /**
      Functions for adding and removing collision groups
      */
      void addGroup(const agx::Name& group);
      void addGroup(agx::UInt32 group);
      void removeGroup(const agx::Name& group);
      void removeGroup(agx::UInt32 group);
      void removeAllGroups();
      bool hasGroup(const agx::Name& group);
      bool hasGroup(agx::UInt32 group);

      /**
      Sets a sensor event to the sensor with a priority
      \param sEvent The SensorEvent
      \param priority The priority of the SensorEvent
      */
      bool addSensorEvent(SensorEvent* sEvent, int priority = 0);

      /**
      Removes a sensor event in the sensor
      \param sEvent The SensorEvent to be removed
      */
      bool removeSensorEvent(SensorEvent* sEvent);

      /**
      Checks if a sensor event already exists in the sensor
      \param sEvent The SensorEvent to be checked
      */
      bool containsSensorEvent(SensorEvent *sEvent);

      /**
      Returns the SensorEvents currently active in the Sensor in the
      order of execution priority
      */
      SensorEventVector getSensorEvents();

      /**
      Sets the geometry of the sensor
      */
      void setGeometry(agxCollide::Geometry* geometry);

      /**
      Get the geometry of the sensor
      */
      agxCollide::Geometry* getGeometry();
      agxCollide::Geometry* getGeometry() const;

      /**
      Set/get the trigger flag to control when the events will be executed during the time-step of contact detection
      */
      void setTriggerFlag(int flag);

      /**
      \return the trigger flag to control when the events will be executed during the time-step of contact detection
      */
      int getTriggerFlag();

      /**
      Set/get the collision mask for the sensor. This will determine what type of contacts that it will listen to
      */
      void setCollisionMask(int mask);

      /**
      \return the collision mask for the sensor. This will determine what type of contacts that it will listen to
      */
      int getCollisionMask();

      /**
      Check if the geometry is marked as forbidden by a sensor in this time step
      */
      bool isForbiddenGeometry(agxCollide::Geometry* geom);

      /**
      Check if the particle is marked as forbidden by a sensor in this time step
      */
      bool isForbiddenParticle(agx::UInt32 particle);

      /**
      Set the sensor to block the contact from other sensors.
      This will block the other sensors in the scene from detecting and
      responding to this contact under the rest of the time-step
      */
      void setEnableBlocker(bool isBlocker);

      /**
      \return true if sensor will block contact from other sensors
      */
      bool getEnableBlocker();

      /**
      Called before collision detection is performed in the simulation
      Implement this method in the derived class to get callbacks.
      \param time - the current simulation time
      */
      virtual void preCollide(const agx::TimeStamp& time) override;

      /**
      Called before a step is taken in the simulation
      Implement this method in the derived class to get callbacks.
      \param time - the current simulation time
      */
      virtual void pre(const agx::TimeStamp& time) override;

      /**
      Called after a step is taken in the simulation
      Implement this method in the derived class to get callbacks.
      \param time - the current simulation time
      */
      virtual void post(const agx::TimeStamp& time) override;

      /**
      This will execute when the sensor is inserted in the simulation. It will add its contactListeners
      for particles and geometries.
      */
      virtual void addNotification() override;

      /**
      This function will execute when the sensor is removed form the simulation. It removes the sensor's
      contact event listeners from the simulation.
      */
      virtual void removeNotification() override;

      /**
      * Returns the current particle-particle contacts in the sensor geometry
      */
      const ParticlePairContactVector& getParticlePairContactsInSensor();

      /*
      * Returns the particle id:s currently in contact with the sensor geometry
      */
      agx::UInt32Vector getContactingParticleIds();

      /*
      * Returns the particle id:s whose midpoint is currently _inside_ the sensor geometry
      */
      agx::UInt32Vector getParticlesInsideSensor();

      /*
      * Returns a vector containing RigidBodies inside the sensor volume
      */
      agx::RigidBodyPtrVector getRigidBodiesInsideSensor(bool onlyRigidBodyDEM = false);

      /*
      * Returns the number of particles currently contacting the sensor
      */
      agx::Real getNumParticlesInSensor() const;

      /*
      * Returns the number of particle-particle contacts in the sensor
      */
      agx::Real getNumParticleParticleContactsInSensor();

      /**
      * Returns a sensor event with the given uuid if there exists one in the sensor
      */
      SensorEvent* getSensorEvent(const agx::Uuid& uuid);

      /// Checks if the sensor keeps the only reference to its internal contact listener. Used in unittest.
      bool hasOnlyReference();

      AGXSTREAM_DECLARE_SERIALIZABLE( agxControl::EventSensor );

      DOXYGEN_START_INTERNAL_BLOCK()
      // Dummy method to allow EventSensor to pass makeCallable compilation.
      void setRealDummyMethod(agx::Real /*inVal*/) { }
      DOXYGEN_END_INTERNAL_BLOCK()

  public:

    template <typename T>
    class EventPrioHolder
    {
    public:
      EventPrioHolder(  ):  m_priority( 0 ) {}
      EventPrioHolder( T obj, int priority ) : m_obj( obj ), m_priority( priority ) {}
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
      inline T sensorEvent() {
        return m_obj;
      }
      inline const T sensorEvent() const {
        return m_obj;
      }


    protected:
      T m_obj;
      int m_priority;
    };

    typedef std::map<agx::Uuid, EventPrioHolder<SensorEventRef> > SensorEvents;
    SensorEvents m_sensorEvents;

    typedef agx::List<EventPrioHolder<SensorEventRef> > SensorEventList;
    SensorEventList m_sensorEventList;

    protected:

      /*
      * Placeholder for information regarding the geometry contacts
      */
      struct ContactInformation
      {
        ContactInformation(const agx::TimeStamp tin, agxCollide::GeometryContact* cdin)
          :t(tin),cd(cdin)
        {
        }

        const agx::TimeStamp t;
        agxCollide::GeometryContact* cd;
      private:
        ContactInformation& operator=(const ContactInformation&);
      };

      /**
      * Placeholder for information regarding the particle contacts
      */
      struct ParticleContactInformation
      {
        ParticleContactInformation(const agx::TimeStamp tIn,
        agx::Physics::ParticleGeometryContactInstance contactIn,
        agx::Physics::ParticleData& particleDataIn,
        agx::Physics::GeometryData& geometryDataIn)
        :t(tIn),contact(contactIn), particleData(particleDataIn), geometryData(geometryDataIn)
        {
        }
        const agx::TimeStamp t;
        agx::Physics::ParticleGeometryContactInstance contact;
        agx::Physics::ParticleData& particleData;
        agx::Physics::GeometryData& geometryData;
      private:
        ParticleContactInformation& operator=(const ParticleContactInformation&);
      };

      typedef agx::Vector<ParticleContactInformation> ParticleContactQueue;
      typedef agx::Vector<ContactInformation> GeometryContactQueue;

  protected:

      void init();

      void constructSortedEventList();

      SensorEvent* getSensorEventFromName(const agx::Name& name);

      bool contactIsValid(agxCollide::GeometryContact* cd);

      void clearContactInformation();

      void refreshParticlesInsideSensorData();

      void refreshRigidBodiesInsideSensorData();

      void updateSensorEvents(const agx::TimeStamp& t);

      void executeSensorEvents(const agx::TimeStamp& t);

      void triggerContactEvents();

      void triggerEvents();

      void triggerPostContactHandling();

      void triggerEventAddNotification();

      agx::RigidBody * getContactingRigidBody(agxCollide::GeometryContact* cd) const;

      void signalGeometryContact(const agx::TimeStamp& t, agxCollide::GeometryContact* cd);
      void signalParticleContact(const agx::TimeStamp& t,
        agx::Physics::ParticleGeometryContactInstance contact,
        agx::Physics::ParticleData& particleData,
        agx::Physics::GeometryData& geometryData);

      void triggerParticleEvents();

      void tryTriggerEventsOnContact(const agx::TimeStamp& t, agxCollide::GeometryContact* cd);
      void tryTriggerParticleEventsOnContact( const agx::TimeStamp& t,
        agx::Physics::ParticleGeometryContactInstance contact,
        agx::Physics::ParticleData& particleData,
        agx::Physics::GeometryData& geometryData);

      void triggerEventsOnContact(const agx::TimeStamp& t, agxCollide::GeometryContact* cd);
      void triggerParticleEventsOnContact( const agx::TimeStamp& t,
        agx::Physics::ParticleGeometryContactInstance contact,
        agx::Physics::ParticleData& particleData,
        agx::Physics::GeometryData& geometryData);

      bool particleContactIsValid( const agx::TimeStamp& t,
        agx::Physics::ParticleGeometryContactInstance contact,
        agx::Physics::ParticleData& particleData,
        agx::Physics::GeometryData& geometryData);

    protected:

      /// Destructor
      virtual ~EventSensor();

      /// Flag for when the events are supposed to be executed during the time-step
      int m_triggerFlag;

      /// Priority of the sensor over all the other sensors
      int m_priority;

      /// Get the collision mask for the sensor
      int m_collisionMask;

      /// The coupled geometry of the sensor
      agxCollide::Geometry* m_geometry;

      /// The collision group filter coupled with the ContactEventListener
      agxSDK::CollisionGroupFilterRef m_filter;

      ///Queues for the contacts
      GeometryContactQueue m_geomContactQueue;
      ParticleContactQueue m_particleContactQueue;

      /// Listeners that are coupled with the sensor
      GeometryContactEventListenerRef m_contactListener;
      ParticleContactListenerRef m_particleContactListener;

      /// Holds a set of the current particle id:s that are present in the sensor
      agx::HashSet<agx::UInt32> m_particlesInSensor;

      /// Holds a set of the current particle id:s whose midpoint is inside the sensor
      agx::HashSet<agx::UInt32> m_particlesInsideSensor;

      /// Holds pointers to the current rigid bodies (DEM Objects) in the sensor
      agx::HashSet<agx::RigidBody*> m_rigidBodiesContactingSensor;

      /// Holds pointers to the current rigid bodies (DEM Objects) where the midpoint is INSIDE the sensor
      agx::HashSet<agx::RigidBody*> m_rigidBodiesInsideSensor;

      /// True if the sensor should block the contact from other sensors in the scene under the same timestep
      bool m_isBlocker;

      /// Flags for doing when there is a collision for a particle or geometry
      bool m_particleCollision;
      bool m_geometryCollision;

      /// Vector that stores particle pair contacts. Will only be computed if the user asks for it (getParticlePairContactsInSensor)
      bool hasComputedParticlePairContactsInSensor;
      ParticlePairContactVector m_particlePairContacts;
  };
}

#endif /*AGXMODEL_EVENTSENSOR_H*/
