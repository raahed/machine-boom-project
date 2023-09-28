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

#ifndef AGXMODEL_FRACTURESIMULATOR_H
#define AGXMODEL_FRACTURESIMULATOR_H

#include <agxModel/export.h>
#include <agx/Referenced.h>
#include <agxSDK/StepEventListener.h>
#include <agxSDK/ContactEventListener.h>
#include <agxCollide/VoronoiGenerator.h>
#include <agxCollide/Geometry.h>
#include <agx/Random.h>

namespace agxModel
{
  /**
  A FractureGenerator class for generating breaking and fracturing geometries in the simulation using 3D Voronoi tesselation.

  ATTENTION! __Concave Trimeshs NOT supported__. The FractureGenerator only supports Geometries containing
  only ONE shape.

  Geometries in the simulation can be added to the FractureGenerator to allow then to be breakable. Boxes, cylinders, spheres
  and convex meshes currently supports this feature. __Concave Trimesh are NOT supported atm__. During a fracture
  a geometry will be split and destroyed using a 3D Voronoi algorithm that created a number of convex fragments from the original geometry
  that should preserve the total volume. The points used in the the Voronoi fragmentation are randomly generated inside the
  shape of the geometry. The convex fragments are then added into the simulation as dynamic rigid bodies and can optionally be added to the
  FractureGenerator to allow recursive fracture.

  Fracturing occurs mainly when a 'breaker' geometry, which can be registered in the FractureGenerator, comes into contact with
  a breakable geometry. If global fracturing is enabled, ALL contacts have the possibility to generate a fracture in a breakable object.
  General ContactFilters can be added to the FractureListener to give more freedom in selecting what types of contact that should be able
  to generate a fracture in breakable objects.
  All these contacts will then be passed into a FractureAlgorithm which will decide if a contact should generate a fracture or not.
  There are some default Algorithm that are provided, nested in FractureGenerator class, such as:

  FractureAlgorithmBreakOnContact      - Will always return true. Every contact should trigger a fracture
  FractureAlgorithmBreakOnContactForce - Fracture will occur if contact force is above specified level
  FractureAlgorithmBreakOnStress       - Fracture will occur if approximated contact stress exceeds specified value

  FractureAlgorithms can be set into a FractureGenerator using setFractureAlgortihm().

  Example usage pattern:

  fracture = agxModel.FractureGenerator()
  sim.add(fracture);
  fracture.setFractureMinimumSize(.5);
  fracture.setNumFractureFragmentsInterval(3, 6);
  fracture.setAddFractureFragmentsToGenerator(True);
  fracture.addBreakble(breakable_geometry);
  fracture.addBreaker(breaker_geometry);
  fractureAlgorithm = agxModel.FractureGeneratorFractureAlgorithmBreakOnContact()
  fracture.setFractureAlgorithm(fractureAlgorithm)

  Optionally add fracture fragment creation callback using 'addNewBodyCallback'

  */
  AGX_DECLARE_POINTER_TYPES(FractureAlgorithm);
  AGX_DECLARE_POINTER_TYPES(FractureGenerator);
  AGX_DECLARE_VECTOR_TYPES(FractureGenerator);
  class AGXMODEL_EXPORT FractureGenerator : public agxSDK::StepEventListener
  {
  public:
    AGX_DECLARE_POINTER_TYPES(NewBodyCallbackWrapper);
    AGX_DECLARE_POINTER_TYPES(NewBodyCallbackWrapperLambda);

  public:
    typedef agx::HashVector<agxCollide::GeometryRef> GeometryHashContainer;
    typedef agx::HashVector<agxSDK::ExecuteFilterRef> ExecuteFilterHashContainer;
    typedef agx::Vector<agxCollide::Convex*> ConvexVector;
    typedef std::function<void(FractureGenerator*,agxCollide::Geometry*, agx::RigidBody*)> NewBodyCallback;

  public:
    /**
    Default constructor
    */
    FractureGenerator();

    /**
    Adds a geometry as a breakable object to the fracture listener
    \param geometry The geometry that should be included in the fracture listener
    \return Returns true if adding the geometry was successfully
    */
    bool addBreakable(agxCollide::Geometry* geometry);

    /**
    Adds all the geometries inside the rigid body as breakable objects
    \param body The rigid body that should include it's geometries as breakable
    \return Returns true if ALL the geometries in the rigid body was added to the fracture logic successfully
    */
    bool addBreakable(agx::RigidBody* body);

    /**
    Removes a geometry as a breakable object from the fracture logic
    \param geometry The geometry that should be remove from the fracture listener
    \return Returns true if removing the geometry was successfully
    */
    bool removeBreakable(agxCollide::Geometry* geometry);

    /**
    Removes all the geometries inside the rigid body as breakable objects in the fracture logic
    \param body The rigid body that should remove it's geometries as breakables
    \return Returns true if ALL the geometries in the rigid body was remove from the fracture logic successfully
    */
    bool removeBreakable(agx::RigidBody* body);

    /**
    Returns true/false if the specified geometry is registered as a breakable object in the fracture logic
    */
    bool isBreakable(const agxCollide::Geometry* geometry) const;

    /**
    Sets the minimum size that allows fracture of an object, derived from it's bounding box.
    \param minsize The minimum size of an object that should be affected by the fracture logic
    */
    void setFractureMinimumSize(agx::Real minsize);

    /**
    Set true/false if ALL contacts in the simulation should trigger possible fractures of breakable geometries.
    Default is only allowing contacts with specified breaker objects to trigger possible fracture.
    */
    void setGlobalFractureEnable(bool enable);

    /// Get true/false if ALL contacts in the simulation should trigger possible fractures of breakable geometries.
    bool getGlobalBreakageEnable() const;

    /// Set true/false if fragments generated by fracture should be included as breakable objects. ( Default = true )
    void setAddFractureFragmentsToGenerator(bool enable);

    /**
    Adds a geometry as a 'breaker' in the fracture logic. Contacts between registered breakable geometries
    and a 'breaker' geometry will trigger a possible fracture.
    \param geom The specified geom that will act as a 'breaker' in the fracture logic.
    \return Returns true if the geometry was successfully added as a 'breaker' to the fracture logic
    */
    bool addBreaker(agxCollide::Geometry* geom);

    /**
    Adds the specified rigid body's geometries as a 'breakers' in the fracture logic. This means contacts
    between registered breakable geometries and a 'breaker' geometry will trigger a possible fracture.
    \param body The specified rigid body that will add it's geometries as 'breakers' in the fracture logic.
    \return Returns true if the ALL the rigid body's geometries where successfully added to the fracture logic as 'breakers'
    */
    bool addBreaker(agx::RigidBody* body);

    /**
    Removes a geometry as a 'breaker' in the fracture logic.
    \param geom The specified geom that will be removed as a breaker
    \return Returns true if the geometry was successfully removed as a 'breaker' from the fracture logic
    */
    bool removeBreaker(agxCollide::Geometry* geom);

    /**
    Removes all the rigid body's geometries as a 'breakers' from the fracture logic.
    \param body The specified rigid body that will remove it's geometries from the simulation
    */
    bool removeBreaker(agx::RigidBody* body);

    /**
    Adds a contact filter that governs what types of contacts that should trigger a fracture. Objects in the contacts
    captured by the filter also need to be added to the fracture logic as breakable objects for fracture to happen.
    \param filter The filter that should be used to to decide which contacts should trigger a fracture
    \return True if the filter was successfully added, otherwise false.
    */
    bool addBreakerFilter(agxSDK::ExecuteFilter* filter);

    /**
    Sets the interval of possible fragments that can be created when a geometry is fractured.
    \param min the minimum number of fragments that can be created
    \param max The maximum number of fragments that can be created
    \return True if values where successfully set, false otherwise.
    */
    bool setNumFractureFragmentsInterval(agx::UInt min, agx::UInt max);

    /**
    Sets the FractureAlgorithm used in the fracture logic. The FractureAlgorithm's purpose is to ultimately decide if
    a contact slated for possible fracture, i.e between a breaker and a breakable geometry, should be allowed to fracture.
    */
    void setFractureAlgorithm(FractureAlgorithm* algorithm);

    /**
    * Set resolution for plane tessellation in Voronoi diagram of basic primitives such as Spheres, Cylinder and Capsules.
    */
    void setResolution(agx::Real resolution);

    /**
    * Get resolution for plane tessellation in Voronoi diagram of basic primitives such as Spheres, Cylinder and Capsules.
    */
    agx::Real getResolution() const;

    /**
    Adds a callback that will be called for every rigid body fragment created in a fracture event.
    This can be used to trigger render callbacks.
    */
    void setNewBodyCallback(NewBodyCallback callback);

    /**
    Adds a class instance that holds a callback that will be called for every rigid body fragment created in a fracture event.
    This can be used to trigger render callbacks.
    */
    void setNewBodyCallbackWrapper(NewBodyCallbackWrapper* newBodyCallback);

    /**
    Utility function to extract all FractureGeneratos from an simulation (and its sub-simulation).
    \param fractureGenerators - The vector where all the fractureGenerators found in the simulation will be placed
    \param simulation - The simulation where to find all fractureGenerators.
    \return true if fractureGenerators are added to the vector, otherwise false.
    */
    static bool extractFractureGenerators(
      agxModel::FractureGeneratorPtrVector& fractureGenerators, agxSDK::Simulation* simulation);

    /**
    Utility function to fracture an arbitrary shape with existing FractureGenerator settings.
    \param shape - the shape to fracture
    \return a vector containing all the convexes created from the fracturing of the specified shape
    */
    ConvexVector fractureShape( const agxCollide::Shape* shape );

    /**
    Called when this listener is added to the simulation.
    (given that it not already is in the simulation).
    */
    void addNotification() override;

    /**
    Called when this listener is removed from the simulation.
    */
    void removeNotification() override;

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
    * Base Abstract class for encapsulating a callback that is executed on new fragments that are created in the fracture generator
    * This class should be overridden to define how new fragments should be handled in the simulation.
    */
    class AGXMODEL_EXPORT NewBodyCallbackWrapper : public agx::Referenced
    {
    public:
      NewBodyCallbackWrapper(){}

      /**
      Function that will be called in the fracture generator for each new rigid body fragment that are created when
      a fracture occurs.
      \param fractureListener The fracture listener that called this function
      \param fractureGeomety The geometry that was fractured
      \param body One of the bodies that was created during a fracture
      */
      virtual void onNewBody(agxModel::FractureGenerator* fractureListener, agxCollide::Geometry* fractureGeomety, agx::RigidBody *body) = 0;

    protected:
      virtual ~NewBodyCallbackWrapper(){}
    };

    /**
    * Class for encapsulating a lambda that that will execute on new fragments created in the fracture generator.
    */
    class AGXMODEL_EXPORT NewBodyCallbackWrapperLambda : public FractureGenerator::NewBodyCallbackWrapper
    {
    public:
      NewBodyCallbackWrapperLambda(const NewBodyCallback& newBodyCallback) { m_newBodyCallback = newBodyCallback; }

      /**
      Function that will be called in the fracture generator for each new rigid body fragment that are created when
      a fracture occurs.
      \param fractureListener The fracture listener that called this function
      \param fractureGeomety The geometry that was fractured
      \param body One of the bodies that was created during a fracture
      */
      virtual void onNewBody(agxModel::FractureGenerator* fractureListener, agxCollide::Geometry* fractureGeomety, agx::RigidBody *body) override final
      {
        m_newBodyCallback(fractureListener, fractureGeomety, body);
      }

    protected:
      virtual ~NewBodyCallbackWrapperLambda() {}

    private:
      FractureGenerator::NewBodyCallback m_newBodyCallback;
    };

    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::FractureGenerator);

  protected:
    virtual ~FractureGenerator();

    void handleContacts(const agxCollide::GeometryContactPtrVector& contacts);

    ConvexVector fractureShape(const agxCollide::Shape* shape, size_t numPoints);

    void updateContactListener();

    bool shapeSupportsFracture(const agxCollide::Shape* shape) const;

    bool isBreaker(agxCollide::Geometry* breaker) const;

  private:
    bool addBreakerInternal(agxCollide::Geometry* geom);
    bool removeBreakerInternal(agxCollide::Geometry* geom);

  protected:
    AGX_DECLARE_POINTER_TYPES(FractureContactGatherer);
    /// FractureContactGatherer - Will gather relevant contact in the simulation, i.e. between breakers and breakable shapes.
    class FractureContactGatherer : public agxSDK::ContactEventListener
    {
    public:
      FractureContactGatherer();

      virtual KeepContactPolicy impact(const agx::TimeStamp& t, agxCollide::GeometryContact* cd) override;
      virtual KeepContactPolicy contact(const agx::TimeStamp& t, agxCollide::GeometryContact* cd) override;
      virtual void separation(const agx::TimeStamp& t, agxCollide::GeometryPair& pair) override;

      void clearBreakageContacts();
      agxCollide::GeometryContactPtrVector& getBreakageContacts();

    protected:
      virtual ~FractureContactGatherer();
      agxCollide::GeometryContactPtrVector m_breakageContacts;
    };

    /// Contact filter encompassing new geometries
    AGX_DECLARE_POINTER_TYPES(ContactFilter);
    class ContactFilter : public agxSDK::ExecuteFilter
    {
    public:
      virtual bool match(const agxCollide::GeometryContact& contact) const override;
      virtual bool match(const agxCollide::GeometryPair& geometryPair) const override;
      using agxSDK::ExecuteFilter::match;

      void addFilter(agxSDK::ExecuteFilter* filter);

    protected:
      virtual ~ContactFilter() {}

    private:
      agx::Vector<agxSDK::ExecuteFilterRef> m_filters;
    };

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  private:
    GeometryHashContainer           m_breakableGeometries;
    GeometryHashContainer           m_breakerGeometries;
    ExecuteFilterHashContainer      m_breakerCollisionFilters;
    bool                            m_enableGlobalFracture;
    bool                            m_addFractureFragmentsToGenerator;
    agx::Real                       m_minSizeToBreak;
    agx::RangeUInt                  m_numFragmentsRange;
    FractureAlgorithmRef            m_fractureAlgorithm;
    agx::UniformIntGenerator        m_pointGenerator;
    agxCollide::VoronoiGeneratorRef m_voronoiGenerator;
    FractureContactGathererRef      m_contactGatherer;
    NewBodyCallbackWrapperRef       m_newBodyCallbackWrapper;
  };

  /**
  * Class for encapsulating fracture conditions for generating fractures in a breakable object.
  */
  class AGXMODEL_EXPORT FractureAlgorithm : public agx::Referenced, public agxStream::Serializable
  {
    public:
      FractureAlgorithm(){};
      /**
      Function that is overridden by child classes to evaluate if a contact fulfills the condition
      for creating a fracture in the contact's breakable geometry.
      \param fractureListener The FractureGenerator object containing the fracture logic
      \param gc The geometry contact for the breakable object.
      \param geometryToFracture The geometry that should be fracture if the function returns true
      \return True or False if the given contact should create a fracture in the specified breakable geometry
      */
      virtual bool shouldFracture(agxModel::FractureGenerator* fractureListener,
        agxCollide::GeometryContact* gc,
        agxCollide::Geometry* geometryToFracture);

#ifndef SWIG
      DOXYGEN_START_INTERNAL_BLOCK()
      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::FractureAlgorithm );
      DOXYGEN_END_INTERNAL_BLOCK()
#endif //SWIG

  protected:
    virtual ~FractureAlgorithm(){};
  };

  /**
  * Triggers fracture directly on contact.
  */
  AGX_DECLARE_POINTER_TYPES(FractureAlgorithmBreakOnContact);
  class AGXMODEL_EXPORT FractureAlgorithmBreakOnContact : public FractureAlgorithm
  {
    public:
      FractureAlgorithmBreakOnContact() : FractureAlgorithm(){}

      /**
      Function that is overridden by child classes to evaluate if a contact fulfills the condition
      for creating a fracture in the contact's breakable geometry.
      \param fractureListener The FractureGenerator object containing the fracture logic
      \param gc The geometry contact for the breakable object.
      \param geometryToFracture The geometry that should be fracture if the function returns true
      \return True or False if the given contact should create a fracture in the specified breakable geometry
      */
      virtual bool shouldFracture(agxModel::FractureGenerator* fractureListener,
        agxCollide::GeometryContact* gc,
        agxCollide::Geometry* geometryToFracture) override;

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::FractureAlgorithmBreakOnContact);
#endif
  };

  /**
  * Triggers fracture depending on approximated contact force level
  */
  AGX_DECLARE_POINTER_TYPES(FractureAlgorithmBreakOnContactForce);
  class AGXMODEL_EXPORT FractureAlgorithmBreakOnContactForce : public FractureAlgorithm
  {
  public:
    FractureAlgorithmBreakOnContactForce(agx::Real contactForceToBreak);

    /**
    Function that is overridden by child classes to evaluate if a contact fulfills the condition
    for creating a fracture in the contact's breakable geometry.
    \param fractureListener The FractureGenerator object containing the fracture logic
    \param gc The geometry contact for the breakable object.
    \param geometryToFracture The geometry that should be fracture if the function returns true
    \return True or False if the given contact should create a fracture in the specified breakable geometry
    */
    virtual bool shouldFracture(agxModel::FractureGenerator* fractureListener,
      agxCollide::GeometryContact* gc,
      agxCollide::Geometry* geometryToFracture) override;

#ifndef SWIG
    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::FractureAlgorithmBreakOnContactForce);
#endif

  protected:
    FractureAlgorithmBreakOnContactForce();

    agx::Real m_contactForceToBreak;
  };

  /**
  * Triggers fracture depending on approximated contact stress level
  */
  AGX_DECLARE_POINTER_TYPES(FractureAlgorithmBreakOnStress);
  class AGXMODEL_EXPORT FractureAlgorithmBreakOnStress : public FractureAlgorithm
  {
  public:
    FractureAlgorithmBreakOnStress(agx::Real stressLevelToBreak);

    /**
    Function that is overridden by child classes to evaluate if a contact fulfills the condition
    for creating a fracture in the contact's breakable geometry.
    \param fractureListener The FractureGenerator object containing the fracture logic
    \param gc The geometry contact for the breakable object.
    \param geometryToFracture The geometry that should be fracture if the function returns true
    \return True or False if the given contact should create a fracture in the specified breakable geometry
    */
    virtual bool shouldFracture(agxModel::FractureGenerator* fractureListener,
      agxCollide::GeometryContact* gc,
      agxCollide::Geometry* geometryToFracture) override;

#ifndef SWIG
    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::FractureAlgorithmBreakOnStress);
#endif

  protected:
    FractureAlgorithmBreakOnStress();

    agx::Real m_stressLevelToBreak;
  };

  AGX_FORCE_INLINE void agxModel::FractureGenerator::setGlobalFractureEnable(bool enable)
  {
    m_enableGlobalFracture = enable;
  }

  AGX_FORCE_INLINE bool agxModel::FractureGenerator::getGlobalBreakageEnable() const
  {
    return m_enableGlobalFracture;
  }

  AGX_FORCE_INLINE void agxModel::FractureGenerator::setAddFractureFragmentsToGenerator(bool enable)
  {
    m_addFractureFragmentsToGenerator = enable;
  }

  AGX_FORCE_INLINE void agxModel::FractureGenerator::setFractureAlgorithm(agxModel::FractureAlgorithm* algorithm)
  {
    m_fractureAlgorithm = algorithm;
  }

  AGX_FORCE_INLINE bool agxModel::FractureGenerator::isBreakable(const agxCollide::Geometry* geometry) const
  {
    return m_breakableGeometries.find(const_cast<agxCollide::Geometry*>(geometry)) != m_breakableGeometries.end();
  }

  AGX_FORCE_INLINE agxCollide::GeometryContactPtrVector& agxModel::FractureGenerator::FractureContactGatherer::getBreakageContacts()
  {
    return m_breakageContacts;
  }

  AGX_FORCE_INLINE void agxModel::FractureGenerator::setNewBodyCallbackWrapper(
    agxModel::FractureGenerator::NewBodyCallbackWrapper* newBodyCallback)
  {
    m_newBodyCallbackWrapper = newBodyCallback;
  }
}



#endif // AGXMODEL_FRACTURESIMULATOR_H
