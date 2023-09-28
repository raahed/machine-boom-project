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

#ifndef AGXSDK_SIMULATIONSTATISTICSLISTENER_H
#define AGXSDK_SIMULATIONSTATISTICSLISTENER_H

#include <agxSDK/agxSDK.h>

#include <agxSDK/StepEventListener.h>
#include <agx/RigidBody.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/Shape.h>
#include <agxData/Value.h>
#include <agx/Vec3.h>

namespace agxStream
{
  class InputArchive;
  class OutputArchive;
}

namespace agxSDK
{
  class StatisticsData;


  /// Class for storing min/max values together with a ID for the associated RigidBody/Geometry
  template<typename T>
  class StatisticsListenerData {

  public:
    StatisticsListenerData(StatisticsData *parent, const agx::String& id) ;

    /// \return time of the event
    agx::Real getTime() const;

    /// \return an ID of the first geometry involved in the event, InvalidIndex if none.
    agx::UInt64 getGeometry1() const;

    /// \return an ID of the second geometry involved in the event, InvalidIndex if none.
    agx::UInt64 getGeometry2() const;

    /// \return an ID of the first rigid body involved in the event, InvalidIndex if none.
    agx::UInt64 getBody1() const;

    /// \return an ID of the second rigid body involved in the event, InvalidIndex if none.
    agx::UInt64 getBody2() const;

#ifndef SWIG

    /// will compare \p val with the previous value and store time, value, and bodies if \p val is larger than the previous
    T compareAndStoreMax( agx::Real time, T val, const agx::RigidBody* b1=nullptr, const agx::RigidBody* b2=nullptr);

    /// will compare \p val with the previous value and store time, value, and bodies if \p val is lower than the previous
    T compareAndStoreMin( agx::Real time, T val, const agx::RigidBody* b1, const agx::RigidBody* b2=nullptr);

    /// will compare \p val with the previous value and store time, value, and geometries if \p val is larger than the previous
    T compareAndStoreMax( agx::Real time, T val, const agxCollide::Geometry *geom1, const agxCollide::Geometry *geom2=nullptr);

    /// will compare \p val with the previous value and store time, value, and geometries if \p val is lower than the previous
    T compareAndStoreMin( agx::Real time, T val, const agxCollide::Geometry *geom1, const agxCollide::Geometry *geom2=nullptr);
#endif
    /// \return the stored value
    T get() const;

    /// Increment by one
    void inc( );

    /// Set the value explicitly without comparison
    void set( T val );

    /// Set the value explicitly without comparison including the time
    void set( agx::Real time, T val );

    /// reset the value to the given value
    void reset( T val=0 );

    /// Serialization store
    void store( agxStream::OutputArchive& out ) const;

    /// Serialization restore
    void restore( agxStream::InputArchive& in );

  private:

    void initData();

#ifndef SWIG
    StatisticsData *m_parent;
    agx::ComponentRef m_group;
    agx::String m_name;
    agx::ref_ptr<agxData::ValueT<agx::Real> > m_time;
    agx::ref_ptr<agxData::ValueT<T> > m_value;
    agx::ref_ptr<agxData::ValueT<agx::UInt64> > body1;
    agx::ref_ptr<agxData::ValueT<agx::UInt64> > body2;
    agx::ref_ptr<agxData::ValueT<agx::UInt64> > geometry1;
    agx::ref_ptr<agxData::ValueT<agx::UInt64> > geometry2;

#endif
  };

  typedef StatisticsListenerData<agx::Real> RealStatisticsListenerData;
  typedef StatisticsListenerData<agx::UInt32> UInt32StatisticsListenerData;

  /// Store all the various statistics of a simulation
  class AGXPHYSICS_EXPORT StatisticsData : public agx::Referenced
  {
  protected:
    agx::observer_ptr<agxSDK::Simulation> m_simulation;
    agx::ComponentRef m_dataRoot;

  public:

    /// Constructor
    StatisticsData( );


    /// Serialization store
    void store( agxStream::OutputArchive& out ) const;

    /// Serialization restore
    void restore( agxStream::InputArchive& in );

    void reset();

    void setSimulation( agxSDK::Simulation *sim );
    agxSDK::Simulation *getSimulation() { return m_simulation; }

    agx::Component *getDataRoot() {return m_dataRoot;}

    UInt32StatisticsListenerData numShapes;
    UInt32StatisticsListenerData numBodies;
    UInt32StatisticsListenerData numConstraints;
    UInt32StatisticsListenerData maxNumContactsPerTimestep;
    UInt32StatisticsListenerData numContacts;
    UInt32StatisticsListenerData totalNumContacts;
    UInt32StatisticsListenerData numTriangles;

    RealStatisticsListenerData maxLinearSpeed;
    RealStatisticsListenerData maxAngularSpeed;
    RealStatisticsListenerData maxContactSpeed;
    RealStatisticsListenerData maxPenetrationDepth;
    RealStatisticsListenerData minContactFeature;
    RealStatisticsListenerData maxContactSpeedFeatureRatio;
    RealStatisticsListenerData maxConstraintViolation;

    RealStatisticsListenerData maxFeatureSpeed;
    RealStatisticsListenerData maxFeature;
    RealStatisticsListenerData minFeature;


  protected:
    virtual ~StatisticsData();
  };

  typedef agx::ref_ptr<StatisticsData> StatisticsDataRef;


  /**
  This class will collect statistics and sizes of a simulation.
  Limitation: Currently it will only count the #shapes at the first time step.
  */
  class AGXPHYSICS_EXPORT SimulationStatisticsListener : public StepEventListener
  {

    public:
      SimulationStatisticsListener();

      /**
      Method that will given the current Simulation count number of bodies, constraints and collect
      number of triangles, shapes in the system.
      */
      void collectSystemData( agx::Real time = 0);

      /// \return the duration of the recorded simulation since last call to reset
      agx::Real getDuration() const;

      /// \return the total number of enabled bodies during the simulation
      agx::UInt32 getNumRigidBodies() const;

      /// \return the number of unique enabled constraints in total during the simulation (except contacts)
      agx::UInt32 getNumConstraints() const;

      /// \return the max number of contacts at one time step
      agx::UInt32 getMaxNumContactsPerTimeStep() const;

      /// \return the number of contacts in this time step
      agx::UInt32 getNumContacts() const;

      /// \return the max magnitude of linear speed of a RigidBody during the simulation
      RealStatisticsListenerData getMaxLinearSpeed() const;

      /// \return the max magnitude of rotational speed of a RigidBody during the simulation
      RealStatisticsListenerData getMaxAngularSpeed() const;

      /// \return the max magnitude of relative contact speed between two bodies
      RealStatisticsListenerData getMaxContactSpeed() const;

      /// \return the max penetration depth during a simulation
      RealStatisticsListenerData getMaxPenetrationDepth() const;

      RealStatisticsListenerData getMaxConstraintViolation() const;

      /// \return the smallest feature
      RealStatisticsListenerData getMinFeature() const;

      /// \return the smallest feature involved in a contact
      RealStatisticsListenerData getMinContactFeature() const;

      /// \return the largest ContactSpeed/featureSize during a contact
      RealStatisticsListenerData getMaxContactSpeedFeatureRatio() const;

      /// \return the largest feature
      RealStatisticsListenerData getMaxFeature() const;

      /// \return the approximately largest speed that a feature in the system has
      RealStatisticsListenerData getMaxFeatureSpeed() const;


      /// \return the total number of contacts during the whole simulation
      agx::UInt32 getTotalNumContacts() const;

      /// \return the total number of triangles
      agx::UInt32 getNumTriangles() const;

      /// \return the total number of shapes
      agx::UInt32 getNumShapes() const;

      /// Reset the statistics
      void reset();


      /// Get a report in string format
      agx::String getReport() const;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::SimulationStatisticsListener );

  protected:

      void updateContactData( agx::Real time );

      virtual void post(const agx::TimeStamp& ) override;
      virtual void pre(const agx::TimeStamp& ) override;

      virtual void addNotification() override;
      virtual void removeNotification() override;

      /// Destructor
      virtual ~SimulationStatisticsListener() {}

      bool calculateShapeFeatures(
        const agxCollide::Shape *shape, const agx::RigidBody *body,
        agx::Real& smallestFeature, agx::Real& largestFeature,
        agx::Vec3& largestFeaturePointRelBody, agx::Real& largestDistance);

      bool calculateFeatures(
        const agx::RigidBody* body, agx::Real& smallestFeature, const agxCollide::Geometry*& smallestFeatureGeometry,
        agx::Real& largestFeature, const agxCollide::Geometry*& largestFeatureGeometry, agx::Vec3& pointRelativeBody);


      bool m_firstPreCall;
      agx::Real m_startTime;
      agx::Real m_lastTime;
      agx::Real m_gravity;
      agx::Real m_avgTimeStep;
      agx::Real m_minTimeStep;
      agx::Real m_maxTimeStep;
      agx::Real m_sumTimeStep;
      agx::UInt32 m_numTimeSteps;

      StatisticsDataRef m_statisticsData;
  };

  typedef agx::ref_ptr<SimulationStatisticsListener> SimulationStatisticsListenerRef;


  template<typename T>
  StatisticsListenerData<T>::StatisticsListenerData(StatisticsData *parent, const agx::String& id) :
    m_parent(parent), m_name(id)
  {
    initData();
  }
  template<typename T>
  void StatisticsListenerData<T>::initData()
  {
    agx::Component *root = m_parent->getDataRoot();
    m_group = new agx::Component(m_name);
    root->addObject( m_group );

    m_value = new agxData::ValueT<T>("value");
    m_group->addObject( m_value );

    m_time = new agxData::ValueT<agx::Real>("time");
    m_group->addObject( m_time );

    body1 = new agxData::ValueT<agx::UInt64>("body1");
    m_group->addObject( body1 );
    body2 = new agxData::ValueT<agx::UInt64>("body2");
    m_group->addObject( body2 );
    geometry1 = new agxData::ValueT<agx::UInt64>("geometry1");
    m_group->addObject( geometry1 );
    geometry2 = new agxData::ValueT<agx::UInt64>("geometry2");
    m_group->addObject( geometry2 );
  }

  template<typename T>
  agx::Real StatisticsListenerData<T>::getTime() const { return m_time->get(); }

  template<typename T>
  agx::UInt64 StatisticsListenerData<T>::getGeometry1() const { return geometry1->get(); }
  template<typename T>
  agx::UInt64 StatisticsListenerData<T>::getGeometry2() const { return geometry2->get(); }
  template<typename T>
  agx::UInt64 StatisticsListenerData<T>::getBody1() const { return body1->get(); }
  template<typename T>
  agx::UInt64 StatisticsListenerData<T>::getBody2() const { return body2->get(); }



  template<typename T>
  T StatisticsListenerData<T>::get() const {
    return m_value->get();
  }

  /// Increment by one
  template<typename T>
  void StatisticsListenerData<T>::inc( ) {
    m_value->set(m_value->get()+1);
  }

  template<typename T>
  void StatisticsListenerData<T>::set( T val ) {
    m_value->set(val);
  }

  template<typename T>
  void StatisticsListenerData<T>::set( agx::Real time, T val ) {
    m_value->set(val);
    m_time->set( time );
  }

  template<typename T>
  void StatisticsListenerData<T>::reset( T val )
  {
    body1->set(agx::InvalidIndex);
    body2->set(agx::InvalidIndex);
    geometry1->set(agx::InvalidIndex);
    geometry2->set(agx::InvalidIndex);
    set(val);
  }

}

#endif
