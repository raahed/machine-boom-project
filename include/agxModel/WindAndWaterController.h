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

#ifndef AGXMODEL_WINDANDWATERCONTROLLER_H
#define AGXMODEL_WINDANDWATERCONTROLLER_H

#include <agxModel/export.h>
#include <agxModel/AddedMassDb.h>
#include <agxModel/WindAndWaterParameters.h>
#include <agxModel/PressureFieldRenderer.h>

#include <agx/Vec6.h>

#include <agxCollide/Geometry.h>

#include <agxSDK/LinkedStructure.h>

#include <agxWire/Wire.h>

#include <mutex>


namespace agxModel {
  class WaterWrapper;
  AGX_DECLARE_POINTER_TYPES( WaterWrapper );
  typedef agx::ref_ptr< WindAndWaterController > WindAndWaterControllerRef;
  typedef agx::Vector< WindAndWaterController* > WindAndWaterControllerPtrVector;

  /**
  Controller that collects relevant data and executes algorithms (probably mostly aero- and hydrodynamics).
  */
  class AGXMODEL_EXPORT WindAndWaterController : public agxSDK::StepEventListener
  {
  public:
    AGX_DECLARE_POINTER_TYPES( Algorithm );

    /**
    Interface for generating water flow and wind.
    */
    class AGXMODEL_EXPORT WindAndWaterFlowGenerator : public agx::Referenced, public agxStream::Serializable
    {
    public:
      /**
      Default constructor disables serialization. Any implementation
      has to enable this explicitly.
      */
      WindAndWaterFlowGenerator();

      /**
      Calculate and return wind at a given position in world.
      \param worldPoint - point in world
      \return wind velocity in world coordinates
      */
      virtual agx::Vec3 getVelocity( const agx::Vec3& worldPoint ) const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( WindAndWaterController::WaterFlowGenerator )

    protected:
       virtual ~WindAndWaterFlowGenerator();
    };

    AGX_DECLARE_POINTER_TYPES( WaterFlowGenerator );

    /**
    Interface for generating water flow.
    */
    class AGXMODEL_EXPORT WaterFlowGenerator : public WindAndWaterFlowGenerator
    {
    public:
      /**
      Default constructor disables serialization. Any implementation
      has to enable this explicitly.
      */
      WaterFlowGenerator();

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE( WindAndWaterController::WaterFlowGenerator );
#endif
    protected:
       virtual ~WaterFlowGenerator();
    };

    AGX_DECLARE_POINTER_TYPES( WindGenerator );

    /**
    Interface for generating wind effects.
    */
    class AGXMODEL_EXPORT WindGenerator : public WindAndWaterFlowGenerator
    {
    public:
      /**
      Default constructor disables serialization. Any implementation
      has to enable this explicitly.
      */
      WindGenerator();

      /**
      Called before update of \p geometry.
      \param geometry - geometry to be updated
      */
      virtual void prepare( const agxCollide::Geometry* geometry );

#ifndef SWIG
      AGXSTREAM_DECLARE_SERIALIZABLE( WindAndWaterController::WindGenerator );
#endif
    protected:
       virtual ~WindGenerator();
    };


    /**
    Wraps wind related things.
    */
    class AGXMODEL_EXPORT WindWrapper
    {
    public:
      /**
      Default constructor.
      */
      WindWrapper( agx::Bool enable, const agxModel::WindAndWaterController::WindGenerator* windGenerator, agx::Real airDensity );

      /**
      \return density of the air
      */
      agx::Real getDensity() const;

      /**
      \return the world velocity of the air in a given word point
      */
      agx::Vec3 getVelocity( const agx::Vec3& worldPoint ) const;

      /**
      \return true if enabled for current geometry
      */
      agx::Bool getEnable() const;

    protected:
      agx::Real m_density;
      agx::Bool m_enable;
      const WindGenerator* m_windGenerator;
    };

    /**
    Data for different objects.
    */
    class AGXMODEL_EXPORT WindAndWaterData : public agx::Referenced
    {
    public:

      /**
      Constructor. Used by the derived classes ObjectData and WireLinkedStructureData.
      */
      WindAndWaterData( const agxModel::AerodynamicsParameters* aeroParams,
                        const agxModel::HydrodynamicsParameters* hydroParams );

      /**
      \return aerodynamic parameters for the shape
      */
      const agxModel::AerodynamicsParameters* getAerodynamicsParameters() const;

      /**
      \return hydrodynamic parameters for the shape
      */
      const agxModel::HydrodynamicsParameters* getHydrodynamicsParameters() const;

    protected:
      virtual ~WindAndWaterData();

      const AerodynamicsParameters* m_aeroParams;
      const HydrodynamicsParameters* m_hydroParams;
    };

    /**
    Static data for a shape.
    */
    class AGXMODEL_EXPORT ObjectData : public WindAndWaterData
    {
    public:
      /**
      Construct given shape, parameters, wind and water definitions and gravity.
      */
      ObjectData( const agxCollide::Shape* shape,
                  const agxModel::AerodynamicsParameters* aeroParams,
                  const agxModel::HydrodynamicsParameters* hydroParams,
                  const agxModel::WaterWrapper* water,
                  const agxModel::WindAndWaterController::WindWrapper* wind,
                  const agx::Vec3& gravity,
                  agxModel::PressureFieldRenderer* pressureFieldRenderer,
                  const agx::UInt numTriangles,
                  const agx::UInt startIndexT,
                  const agx::UInt startIndexV );

      /**
      \return start index for triangles of this object
      */
      agx::UInt getStartIndexT() const;

      /**
      \return number of triangles for this object
      */
      agx::UInt getNumTriangles() const;

      /**
      \return density of the water if belowSurface = true, density of air if belowSurface = false
      */
      agx::Real getDensity( agx::Bool belowSurface ) const;

      /**
      \return the density of the medium the triangle is currently in
      */
      agx::Real getDensity( const agxModel::TriangleData& tData ) const;

      /**
      \return the density of the medium the vertex is currently in
      */
      agx::Real getDensity( const agxModel::VertexData& vData ) const;

      /**
      \return the density of water
      */
      agx::Real getDensityWater() const;

      /**
      \return the density of air
      */
      agx::Real getDensityAir() const;

      /**
      \return the gravity at the position of the shape
      */
      const agx::Vec3& getGravity() const;

      /**
      \return the force (in world coordinate system) applied by the wind and water algorithms given they have been executed
      */
      const agx::Vec3& getForce() const;

      /**
      Access force vector (in world coordinate system).
      */
      agx::Vec3& force();

      /**
      \return the torque (in world coordinate system) applied by the wind and water algorithms given they have been executed
      */
      const agx::Vec3& getTorque() const;

      /**
      Access torque vector (in world coordinate system).
      */
      agx::Vec3& torque();

      /**
      \return added mass matrix in rigid body frame
      */
      const agx::Matrix3x3& getAddedMass() const;

      /**
      Access added mass matrix. Must be in body frame.
      */
      agx::Matrix3x3& addedMass();

      /**
      \return added inertia matrix in rigid body frame
      */
      const agx::Matrix3x3& getAddedInertia() const;

      /**
      Access added inertia matrix. Must be in body frame.
      */
      agx::Matrix3x3& addedInertia();

      /**
      Set to/is true if this data contains added mass and/or inertia.
      */
      agx::Bool& containsAddedMass();

      /**
      \return true if this data contains added mass and/or inertia.
      */
      const agx::Bool& containsAddedMass() const;

      /**
      \return the shape
      */
      const agxCollide::Shape* getShape() const;

      /**
      \return the geometry of the shape
      */
      const agxCollide::Geometry* getGeometry() const;

      /**
      \return the water wrapper
      */
      const WaterWrapper* getWater() const;

      /**
      \return the rigid body of the geometry
      */
      const agx::RigidBody* getRigidBody() const;

      /**
      This method is unsafe to use if this object is used during parallel execution.
      \return the rigid body of the geometry
      */
      agx::RigidBody* getRigidBodySerial();

      /**
      \return the pressure field renderer if present, otherwise null
      */
      agxModel::PressureFieldRenderer* getPressureFieldRenderer() const;

      /**
      Perform a mutex lock for this ObjectData.
      */
      void lock() const  { m_mutex.lock(); };

      /**
      Perform a mutex unlock for this ObjectData.
      */
      void unLock() const  { m_mutex.unlock(); };

    protected:
      virtual ~ObjectData();

    private:
      agx::Vec3                m_gravity;
      agx::Real                m_densityWater;
      agx::Real                m_densityAir;
      mutable agx::Vec3        m_force;
      mutable agx::Vec3        m_torque;
      agx::Matrix3x3           m_addedMass;
      agx::Matrix3x3           m_addedInertia;
      agx::Bool                m_containsAddedMass;
      const agxCollide::Shape* m_shape;
      PressureFieldRenderer*   m_pressureFieldRenderer;
      WaterWrapperConstRef     m_water;
      const agx::UInt          m_numTriangles;
      const agx::UInt          m_startIndexT;
      mutable std::mutex       m_mutex;
    };

    typedef agx::ref_ptr< ObjectData > ObjectDataRef;
    typedef agx::Vector< ObjectDataRef > ObjectDataContainer;

    struct WirePair
    {
      WirePair()
        : node1( nullptr ), node2( nullptr ), water( nullptr ), gravity( agx::Vec3() )
      {
      }
      WirePair( agxWire::Node* node1, agxWire::Node* node2, const agxCollide::Geometry* water, agx::Vec3 gravity )
        : node1( node1 ), node2( node2 ), water( water ), gravity( gravity )
      {
      }
      agxWire::Node* node1;
      agxWire::Node* node2;
      const agxCollide::Geometry* water;
      const agx::Vec3 gravity;

    private:
      /// \todo Use "= delete" here when Visual Studio supports it.
      WirePair& operator=( const WirePair& )
      {
        return *this;
      }
    };

    typedef agx::Vector< WirePair > WirePairContainer;

    struct Segment
    {
      Segment( agx::RigidBody* rb, agx::Vec3 point1, agx::Vec3 point2, agx::Vec3 velocity, agx::Vec3 velocityOfFluid,
               agx::Real fluidDensity, agx::Vec3 gravity, agx::Real volume )
        : rb( rb ), point1( point1 ), point2( point2 ), velocity( velocity ), velocityOfFluid( velocityOfFluid ),
          fluidDensity( fluidDensity ), gravity( gravity ), volume(volume)
      {
      }
      agx::RigidBodyRef rb;
      agx::Vec3 point1;
      agx::Vec3 point2;
      agx::Vec3 velocity;
      agx::Vec3 velocityOfFluid;
      agx::Real fluidDensity;
      agx::Vec3 gravity;
      agx::Real volume;
    };

    typedef agx::Vector< Segment > SegmentContainer;

    /**
    Data for a wire or a linked structure.
    */
    class AGXMODEL_EXPORT WireLinkedStructureData : public WindAndWaterData
    {
    public:
      /**
      Constructor.
      \param radius - Radius of the wire (a positive value).
      \param aeroParams - Aerodynamic parameters for this object. If there are no specific parameters for this object default parameters should be used.
      \param hydroParams - Hydrodynamic parameters for this object. If there are no specific parameters for this object default parameters should be used.
      */
      WireLinkedStructureData( agx::Real radius, const AerodynamicsParameters* aeroParams, const HydrodynamicsParameters* hydroParams );

      /**
      \return the radius of the wire
      */
      agx::Real getRadius();

      /**
      Adds a segment of the wire that is affected by hydrodynamics.
      */
      void addWaterSegment( const Segment& segment );

      /**
      Adds a segment of the wire that is affected by aerodynamics.
      */
      void addAirSegment( const Segment& segment );

      /**
      \return the segments of the wire that are affected by aerodynamics.
      */
      SegmentContainer& getAirSegments();

      /**
      \return the segments of the wire that are affected by hydrodynamics.
      */
      SegmentContainer& getWaterSegments();

    protected:
      virtual ~WireLinkedStructureData();


    private:
      SegmentContainer m_airSegments;
      SegmentContainer m_waterSegments;
      agx::Real m_radius;
    };

    typedef agx::ref_ptr< WireLinkedStructureData > WireLinkedStructureDataRef;
    typedef agx::Vector< WireLinkedStructureDataRef > WireLinkedStructureDataContainer;


    /**
    Data for rigid body with aero/hydro static/dynamic forces and added mass and inertia matrices.
    Note that the
    */
    struct AGXMODEL_EXPORT DynamicsData
    {
      DynamicsData() : worldForce(), worldTorque(), bodyFrameAddedMass( agx::Vec3() ), bodyFrameAddedInertia( agx::Vec3() ) {}

      /**
      Calculate the diagonal coefficients given the added mass/inertia matrices have
      been calculated given \p rb.
      \param rb - rigid body this dynamics data is assumed to belong to
      @param[out] addedMassDiagonalCoefficients - diagonal coefficients 'c', where M_final = (1 + c) * M_original
      @param[out] addedInertiaDiagonalCoefficients - diagonal coefficients 'c', where I_final = (1 + c) * I_original
      */
      void getAddedMassAndInertiaDiagonalCoefficients( const agx::RigidBody* rb, agx::Vec3& addedMassDiagonalCoefficients, agx::Vec3& addedInertiaDiagonalCoefficients ) const;

      agx::Vec3 worldForce;
      agx::Vec3 worldTorque;
      agx::Matrix3x3 bodyFrameAddedMass;
      agx::Matrix3x3 bodyFrameAddedInertia;
    };

    /**
    Hydro- and aerodynamic results for a wire.
    */
    struct AGXMODEL_EXPORT DynamicsWireData
    {
      agx::Vec3 waterForce;
      agx::Vec3 airForce;
      size_t numWaterSegments{0};
      size_t numAirSegments{0};
      agx::Real lengthWater{0.0};
      agx::Real lengthAir{0.0};
    };

    /**
    Hydro- and aerodynamic results for a linked structure.
    */
    struct AGXMODEL_EXPORT DynamicsLinkedStructureData : DynamicsWireData
    {
      DynamicsLinkedStructureData() : DynamicsWireData() {}
    };

  public:
    /**
    Construct wind and water controller.
    */
    WindAndWaterController();

    /**
    Flag a geometry as water. Overlap calculations will be made explicitly with other geometries
    where hydrodynamics is enabled.
    \param geometry - water geometry
    */
    virtual agx::Bool addWater( agxCollide::Geometry* geometry );

    /**
    Enable/disable a geometry for aerodynamics calculations.
    \param geometry - geometry to enable/disable for aerodynamics calculations
    \param enable - true to enable, false to disable
    \return true if the successful
    */
    virtual agx::Bool setEnableAerodynamics( agxCollide::Geometry* geometry, agx::Bool enable );

    /**
    Enable/disable aerodynamics for all objects in the simulation. If disabled, only objects that have
    aerodynamics enabled explicitly will be used in the aerodynamics calculation.
    \param enable - true to enable, false to disable.
    */
    virtual void setEnableAerodynamics( agx::Bool enable );

    /**
    Enable/disable all geometries in \p rb for aerodynamics calculations.
    \param rb - rigid body to enable/disable for aerodynamics calculations
    \param enable - true to enable, false to disable
    \return true if the successful
    */
    virtual agx::Bool setEnableAerodynamics( agx::RigidBody* rb, agx::Bool enable );

    /**
    Enable/disable a wire for aerodynamics calculations.
    \param wire - wire to enable/disable for aerodynamics calculations
    \param enable - true to enable, false to disable
    \return true if the successful
    */
    virtual agx::Bool setEnableAerodynamics( agxWire::Wire* wire, agx::Bool enable );

    /**
    Enable/disable a linked structure for aerodynamics calculations. Only
    works after the linked structure has been added to a simulation.
    \param linkedStructure - linked strucutre to enable/disable for aerodynamics calculations.
    \param enable - true to enable, false to disable
    \return true if successful
    */
    agx::Bool setEnableAerodynamics( agxSDK::LinkedStructure* linkedStructure, agx::Bool enable );

    /**
    Set the density of air.
    \param airDensity - The new air density. Default value is 1.2 kg per cubic meter.
    */
    void setAirDensity( agx::Real airDensity );

    /**
    Creates new or returns an already created parameter storage for a given shape.
    The parameters defines the interaction between the shape and air since there will be no contact material between them.
    */
    virtual agxModel::AerodynamicsParameters* getOrCreateAerodynamicsParameters( const agxCollide::Shape* shape ) const;

    /**
    Creates new or returns an already created parameter storage for a given shape.
    The parameters defines the interaction between the shape and water since there will be no contact material between them.
    */
    virtual agxModel::HydrodynamicsParameters* getOrCreateHydrodynamicsParameters( const agxCollide::Shape* shape ) const;

    /**
    Creates new or returns an already created parameter storage for a given wire.
    The parameters defines the interaction between the wire and air since there will be no contact material between them.
    */
    agxModel::AerodynamicsParameters* getOrCreateAerodynamicsParameters( const agxWire::Wire* wire ) const;

    /**
    Creates new or returns an already created parameter storage for a given wire.
    The parameters defines the interaction between the wire and water since there will be no contact material between them.
    */
    agxModel::HydrodynamicsParameters* getOrCreateHydrodynamicsParameters( const agxWire::Wire* wire ) const;

    /**
    Creates new or returns an already created parameter storage for a given
    linked structure.

    The parameters defines the interaction between the linked structure and
    air since there will be no contact material between them.
    */
    agxModel::AerodynamicsParameters* getOrCreateAerodynamicsParameters( const agxSDK::LinkedStructure* linkedStructure ) const;

    /**
    Creates new or returns an already created parameter storage for a given
    linked structure.

    The parameters defines the interaction between the linked structure and
    water since there will be no contact material between them.
    */
    agxModel::HydrodynamicsParameters* getOrCreateHydrodynamicsParameters( const agxSDK::LinkedStructure* linkedStructure ) const;

    /**
    Pressure field renderer for a given shape.
    */
    virtual agx::Bool registerPressureFieldRenderer( agxModel::PressureFieldRenderer* pressureFieldRenderer, const agxCollide::Shape* shape );

    /**
    Assign unique wind generator.
    \param windGenerator - new wind generator
    */
    virtual void setWindGenerator( agxModel::WindAndWaterController::WindGenerator* windGenerator );

    /**
    \return wind generator if present
    */
    virtual agxModel::WindAndWaterController::WindGenerator* getWindGenerator() const;

    /**
    Assign unique water flow generator to a water geometry. If \p flowGenerator
    is nullptr the previous flow generator will be removed.
    \param waterGeometry - the water geometry that is affected by the generator.
    \param flowGenerator - new water flow generator or nullptr to remove the previous one
    \return true if successfully assigned, otherwise false
    */
    virtual agx::Bool setWaterFlowGenerator( const agxCollide::Geometry* waterGeometry,
                                             agxModel::WindAndWaterController::WaterFlowGenerator* flowGenerator );

    /**
    \param waterGeometry - the water geometry that is affected by the generator.
    \return The water flow generator that is attached to the given water geometry,-
    */
    agxModel::WindAndWaterController::WaterFlowGenerator* getWaterFlowGenerator( const agxCollide::Geometry* waterGeometry ) const;

    /**
    Assign a water wrapper to a water geometry;
    \param geometry - the water geometry associated with the water wrapper
    \param waterWrapper - the wrapper to be attached to the given water geometry.
    */
    virtual agx::Bool setWaterWrapper( agxCollide::Geometry* geometry,
                                       agxModel::WaterWrapper* waterWrapper );
    /**
    \param waterGeometry - the water geometry.
    \return The water wrapper that is attached to the given water geometry,-
    */
    agxModel::WaterWrapper* getWaterWrapper( const agxCollide::Geometry* waterGeometry );

    /**
    Create a mesh from a primitive shape given a tessellation level. The same mesh is used for hydrodynamic calculations.
    \param shape - the shape that the mesh should be based on.
    \param tessellation - the desired tessellation of the mesh.
    \return the mesh that is used for hydro- and aerodynamic calculations. Returns nullptr if shape is unknown or a mesh.
    */
    static agxCollide::TrimeshRef createPrimitiveMesh( const agxCollide::Shape* shape,
                                                    agxModel::WindAndWaterParameters::ShapeTessellation tessellation );

    /**
    Calculates forces given water geometry (may be null) acting on a rigid body.
    \param waterGeometry - water geometry
    \param rb - rigid body to calculate (aero and hydro) interactions
    \param gravity - gravity at rb position
    \param space - if space is provided, group collision filters are checked as well (i.e., basically space->canCollide( waterGeoemtry, rbGeometry ))
    \param updateRenderers - if true, any shape with a pressure field renderer will be updated
    \return forces and added mass
    */
    virtual agxModel::WindAndWaterController::DynamicsData calculateDynamicsData(
      const agxCollide::Geometry* waterGeometry, const agx::RigidBody* rb, const agx::Vec3& gravity,
      const agxCollide::Space* space = nullptr, agx::Bool updateRenderers = false );

    /**
    Calculates forces for hydro- and aerodynamics acting on a wire.
    \param waterGeometry - water geometry
    \param wire - wire affected by aero- and hydrodynamics
    */
    agxModel::WindAndWaterController::DynamicsWireData calculateWireDynamicsData(
      const agxCollide::Geometry* waterGeometry, agxWire::Wire* wire );

    /**
    Calculates forces for hydro- and aerodynamics acting on a linked structure.
    \param waterGeometry - water geometry
    \param linkedStructure - linked structure affected by hydro- and aerodynamics.
    */
    agxModel::WindAndWaterController::DynamicsLinkedStructureData calculateLinkedStructureDynamicsData(
      const agxCollide::Geometry* waterGeometry, agxSDK::LinkedStructure* linkedStructure,
      const agxCollide::Space* space );

    /**
    Search the given simulation for a WindAndWaterController with the given Uuid.
    \param simulation - The simulation to search in.
    \param uuid - The Uuid of the WindAndWaterController to search for.
    \return A WindAndWaterController with Uuid \p uuid in simulation \p simulation, or nullptr.
    */
    static agxModel::WindAndWaterController* find( agxSDK::Simulation* simulation, const agx::Uuid& uuid );

    /**
    Find all WindAndWaterControllers with given name.
    \param simulation - The simulation to search in.
    \param name - The name of the WindAndWaterController.
    \return A vector of WindAndWaterControllers.
    */
    static agxModel::WindAndWaterControllerPtrVector findAll( const agxSDK::Simulation* simulation, const agx::Name& name );

    /**
    Finds all WindAndWaterControllers in the given simulation.
    \param simulation - The simulation to search in.
    \return A vector of WindAndWaterControllers.
    */
    static agxModel::WindAndWaterControllerPtrVector findAll( const agxSDK::Simulation* simulation );

    /**
    Internal method.
    */
    virtual void preCollide( const agx::TimeStamp& ) override;

    /**
    Internal method.
    */
    virtual void pre( const agx::TimeStamp& ) override;

    /**
    Internal method.
    */
    virtual void post( const agx::TimeStamp& ) override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::WindAndWaterController );

  protected:
    struct PressureFieldRendererData : public agx::Referenced
    {
      agxModel::PressureFieldRendererRef pressureField;
      agxCollide::ShapeConstObserver shape;

    protected:
      virtual ~PressureFieldRendererData() {}
    };
    typedef agx::ref_ptr< PressureFieldRendererData > PressureFieldRendererDataRef;

    struct AerodynamicsData : public agx::Referenced
    {
      AerodynamicsData( const agxCollide::Geometry* geometryIn ) : geometry( geometryIn ) {}
      agxCollide::GeometryConstObserver geometry;

    protected:
      virtual ~AerodynamicsData() {}
    };
    typedef agx::ref_ptr< AerodynamicsData > AerodynamicsDataRef;

    class UnitPrimitiveCache
    {
    public:
      UnitPrimitiveCache();
      ~UnitPrimitiveCache();

      const agxCollide::Mesh* getOrCreateMesh( const agxCollide::Shape* shape, agx::Vec3& scale,
                                               agxModel::WindAndWaterParameters::ShapeTessellation tessellationLevel ) const;

      static agx::Vec3 findScale( const agxCollide::Shape* shape );

      void garbageCollect();

    private:
      struct PrimitiveData : public agx::Referenced
      {
        PrimitiveData( const agxCollide::Shape* shapeIn, agxCollide::Mesh* meshIn )
          : shape( shapeIn ), mesh( meshIn )
        {
        }

        virtual void checkSize( agx::Real resolution ) = 0;

        agxCollide::ShapeConstObserver shape;
        agxCollide::MeshRef mesh;

      protected:
        virtual ~PrimitiveData() {}
      };

      struct CapsuleData : PrimitiveData
      {
        CapsuleData( const agxCollide::Shape* shapeIn, agxCollide::Mesh* meshIn, agx::Real radiusIn, agx::Real heightIn )
          : PrimitiveData( shapeIn, meshIn ), radius( radiusIn ), height( heightIn )
        {
        }

        virtual void checkSize( agx::Real resolution ) override;

        agx::Real radius;
        agx::Real height;

      protected:
        virtual ~CapsuleData() {}
      };

      struct ConeData : PrimitiveData
      {
        ConeData( const agxCollide::Shape* shapeIn, agxCollide::Mesh* meshIn , agx::Real topRadiusIn, agx::Real bottomRadiusIn, agx::Real heightIn )
          : PrimitiveData( shapeIn, meshIn ), topRadius( topRadiusIn ), bottomRadius( bottomRadiusIn ),
            height( heightIn )
        {
        }

        virtual void checkSize( agx::Real resolution ) override;

        agx::Real topRadius;
        agx::Real bottomRadius;
        agx::Real height;

      protected:
        virtual ~ConeData() {}
      };

      struct HollowConeData : PrimitiveData
      {
        HollowConeData( const agxCollide::Shape* shapeIn, agxCollide::Mesh* meshIn, agx::Real topOuterRadiusIn,
                        agx::Real bottomInnerRadiusIn, agx::Real heightIn,
                        agx::Real thicknessIn )
          : PrimitiveData( shapeIn, meshIn ), topOuterRadius( topOuterRadiusIn ),
            bottomInnerRadius( bottomInnerRadiusIn ), height( heightIn ), thickness( thicknessIn )
        {
        }

        virtual void checkSize( agx::Real resolution ) override;

        agx::Real topOuterRadius;
        agx::Real bottomInnerRadius;
        agx::Real height;
        agx::Real thickness;

      protected:
        virtual ~HollowConeData() {}
      };

      struct HollowCylinderData : PrimitiveData
      {
        HollowCylinderData( const agxCollide::Shape* shapeIn, agxCollide::Mesh* meshIn, agx::Real innerRadiusIn, agx::Real heightIn, agx::Real thicknessIn )
          : PrimitiveData( shapeIn, meshIn ), innerRadius( innerRadiusIn ), height( heightIn ),
            thickness( thicknessIn )
        {
        }

        virtual void checkSize( agx::Real resolution ) override;

        agx::Real innerRadius;
        agx::Real height;
        agx::Real thickness;

      protected:
        virtual ~HollowCylinderData() {}
      };

      using PrimitiveDataRef = agx::ref_ptr< PrimitiveData >;

      typedef agx::Vector< agxCollide::MeshRef > MeshContainer;
      using PrimitiveContainer = agx::HashTable< const agxCollide::Shape*, PrimitiveDataRef >;

    private:
      agxCollide::MeshRef createUnitShape( agxCollide::Shape::Type shapeType, agx::Real resolution = agx::Real( 1 ) ) const;
      const agxCollide::Mesh* getOrCreatePrimitive( const agxCollide::Shape* shape, agx::Real resolution = agx::Real( 1 ) ) const;

    private:
      mutable MeshContainer m_cache;
      mutable PrimitiveContainer m_primitives;

      mutable std::mutex meshCreationMutex;
    };

    typedef agx::Vector< agxModel::WindAndWaterController::AlgorithmRef > AlgorithmContainer;
    typedef agx::HashVector< const agxCollide::Shape*, AerodynamicsParametersRef > AerodynamicsParametersContainer;
    typedef agx::HashVector< const agxCollide::Shape*, HydrodynamicsParametersRef > HydrodynamicsParametersContainer;
    typedef agx::HashVector< const agxWire::Wire*, AerodynamicsParametersRef > AerodynamicsParametersWireContainer;
    typedef agx::HashVector< const agxWire::Wire*, HydrodynamicsParametersRef > HydrodynamicsParametersWireContainer;
    typedef agx::HashVector< const agxSDK::LinkedStructure*, AerodynamicsParametersRef > AerodynamicsParametersLinkedStructureContainer;
    typedef agx::HashVector< const agxSDK::LinkedStructure*, HydrodynamicsParametersRef > HydrodynamicsParametersLinkedStructureContainer;
    typedef agx::HashVector< const agxCollide::Shape*, PressureFieldRendererDataRef > PressureFieldRendererDataContainer;
    typedef agx::HashVector< const agxCollide::Geometry*, AerodynamicsDataRef > AerodynamicsDataContainer;
    typedef agx::HashVector< const agxCollide::Geometry*, WaterFlowGeneratorRef > WaterFlowGeneratorContainer;
    typedef agx::HashVector< const agxCollide::Geometry*, WaterWrapperRef > WaterWrapperContainer;
    typedef agx::Vector< agxCollide::GeometryConstObserver > GeometryConstObserverVector;
    typedef agx::Vector< agxWire::WireObserver > WireObserverVector;


  protected:
    /**
    Reference counted object, protected destructor.
    */
    virtual ~WindAndWaterController();

  protected:
    /**
    Configure the default algorithms.
    */
    void configureDefaultAlgorithms();

    /**
    Collects garbage (e.g., due to deleted shapes).
    */
    void garbageCollect();

    /**
    \return render data for shape if present
    */
    agxModel::WindAndWaterController::PressureFieldRendererData* getPressureFieldRenderData( const agxCollide::Shape* shape ) const;

    /**
    \return pressure field renderer if present
    */
    agxModel::PressureFieldRenderer* getPressureFieldRender( const agxCollide::Shape* shape ) const;

    /**
    Calculate object data (for each shape) in a geometry.
    */
    virtual agxModel::WindAndWaterController::ObjectDataContainer calculateObjectData(
      const agxCollide::Geometry* waterGeometry,
      const agxCollide::Geometry* geometry,
      const agx::Vec3& gravity,
      agx::Bool aeroEnabled,
      agx::Bool hydroEnabled,
      agx::Bool updateRenderers,
      agx::UInt& numProcessedTriangles,
      const agx::UIntVector startIndexT,
      const agx::UIntVector startIndexV,
      const agx::TimeStamp& t = 0 );

    /**
    Calculate wire data for all segments.
    */
    agxModel::WindAndWaterController::WireLinkedStructureData* calculateWireData(
      agxWire::Wire* wire, WirePairContainer wirePairs, WindAndWaterController::WindWrapper wind, agx::Bool aeroEnable,
      const agx::TimeStamp& t = 0 );

    /**
    Calculate linked structure  data for all segments.
    */
    agxModel::WindAndWaterController::WireLinkedStructureData* calculateLinkedStructureData(
      const agxCollide::Geometry* waterGeometry, const agxCollide::Geometry* geometry, const agx::Vec3& gravity,
      const AerodynamicsParameters* aeroParams, const HydrodynamicsParameters* hydroParams, agx::Bool aeroEnabled,
      agx::Bool hydroEnabled, const agx::TimeStamp& t = 0 );

    void getSizeOfGeometry( const agxCollide::Geometry* geometry,
                            agx::UIntVector& numTriangles,
                            agx::UIntVector& numVertices );

  protected:
    AlgorithmContainer                            m_algorithms;
    GeometryConstObserverVector                   m_waterGeometries;
    AerodynamicsDataContainer                     m_aeroGeometries;
    WireObserverVector                            m_aeroWires;
    AddedMassDbRef                                m_addedMassDb;
    AerodynamicsParametersRef                     m_globalAerodynamicsParameters;
    mutable AerodynamicsParametersContainer       m_aerodynamicsParameters;
    mutable AerodynamicsParametersWireContainer   m_aerodynamicsParametersWires;
    mutable AerodynamicsParametersLinkedStructureContainer  m_aerodynamicsParametersLinkedStructures;
    HydrodynamicsParametersRef                    m_globalHydrodynamicsParameters;
    mutable HydrodynamicsParametersContainer      m_hydrodynamicsParameters;
    mutable HydrodynamicsParametersWireContainer  m_hydrodynamicsParametersWires;
    mutable HydrodynamicsParametersLinkedStructureContainer m_hydrodynamicsParametersLinkedStructures;
    PressureFieldRendererDataContainer            m_pressureFieldRenderers;
    UnitPrimitiveCache                            m_unitPrimitiveCache;
    WindGeneratorRef                              m_windGenerator;
    WaterFlowGeneratorContainer                   m_flowGenerators;
    WaterWrapperContainer                         m_waterWrappers;
    agx::Real                                     m_airDensity;
    agx::Bool                                     m_aeroForAll;
    GlobalTriangleData::Container                 m_trianglePool;
    VertexData::Container                         m_vertexPool;
  };

  /**
  Interface for a WindAndWaterController algorithm. Callbacks during preCollide, pre
  and post state. Forces should be calculated in the updateInteraction call where
  all static data is present for a given shape.
  */
  class AGXMODEL_EXPORT WindAndWaterController::Algorithm : public agx::Referenced
  {
  public:
    typedef agxModel::WindAndWaterController::ObjectData ObjectData;

  public:
    /**
    Called during normal preCollide events.
    */
    virtual void preCollide( agxModel::WindAndWaterController* /*controller*/ ) {}

    /**
    Called before all data is collected and before updateInteraction is called.
    */
    virtual void pre( agxModel::WindAndWaterController* /*controller*/ ) {}

    /**
    Called during normal post.
    */
    virtual void post( agxModel::WindAndWaterController* /*controller*/ ) {}

    /**
    Called when it's time to update interaction.
    */
    virtual void updateInteraction( agxModel::WindAndWaterController::ObjectData& /*objectData*/, const GlobalTriangleData::Container& /*container*/ ) {}
  };

  AGX_DECLARE_POINTER_TYPES( ConstantWaterFlowGenerator );

  /**
  Class to generate constant water flow in a scene.
  */
  class AGXMODEL_EXPORT ConstantWaterFlowGenerator : public agxModel::WindAndWaterController::WaterFlowGenerator
  {
  public:
    /**
    Construct given flow velocity.
    \param velocity - velocity of the flow given in world coordinates
    */
    ConstantWaterFlowGenerator( const agx::Vec3& velocity );

    /**
    Assign new water flow velocity.
    \param velocity - new water flow velocity
    */
    void setVelocity( const agx::Vec3& velocity );

    /**
    \return currently used water flow velocity
    */
    const agx::Vec3& getVelocity() const;

  public:
    virtual agx::Vec3 getVelocity( const agx::Vec3& /*worldPosition*/ ) const override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::ConstantWaterFlowGenerator );

  protected:
    ConstantWaterFlowGenerator();
    virtual ~ConstantWaterFlowGenerator();

  private:
    agx::Vec3 m_velocity;
  };


  AGX_DECLARE_POINTER_TYPES( ConstantWindGenerator );

  /**
  Class to generate constant wind in a scene. Objects with aerodynamics enabled will be affected by this wind.
  */
  class AGXMODEL_EXPORT ConstantWindGenerator : public agxModel::WindAndWaterController::WindGenerator
  {
  public:
    /**
    Construct given wind velocity.
    \param velocity - velocity of the wind given in world coordinates
    */
    ConstantWindGenerator( const agx::Vec3& velocity );

    /**
    Assign new wind velocity.
    \param velocity - new wind velocity
    */
    void setVelocity( const agx::Vec3& velocity );

    /**
    \return currently used wind velocity
    */
    const agx::Vec3& getVelocity() const;

  public:
    virtual agx::Vec3 getVelocity( const agx::Vec3& /*worldPosition*/ ) const override;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::ConstantWindGenerator );

  protected:
    ConstantWindGenerator();
    virtual ~ConstantWindGenerator();

  private:
    agx::Vec3 m_velocity;
  };

  /**
  Water wrapper that can be inherited to create custom made wrappers. Note that the object needs to overlap the water
  geometry to get to this point. This means that the water surface always should be within the water geometry.
  */
  class AGXMODEL_EXPORT WaterWrapper : public agx::Referenced, public agxStream::Serializable
  {
    friend class WindAndWaterController;

  public:
    /**
    Default constructor.
    */
    WaterWrapper();

    /**
    Finds height from surface given point in world.
    \param worldPoint - point in world
    \param upVector - up (i.e., negative gravity direction in which the buoyancy force is acting), given in world coordinates
    \param t - Timestamp
    \return height from surface of the water geometry
    */
    virtual agx::Real findHeightFromSurface( const agx::Vec3& worldPoint, const agx::Vec3& upVector, const agx::TimeStamp& t ) const;

    /**
    \return true if valid (no internal checks are made)
    */
    virtual agx::Bool isValid() const;

    /**
    \return the density of the water
    */
    virtual agx::Real getDensity() const;

    /**
    \return the water velocity at a given world point
    */
    virtual agx::Vec3 getVelocity( const agx::Vec3& worldPoint ) const;

    /**
    \return the water geometry, nullptr if air
    */
    const agxCollide::Geometry* getGeometry() const;

    /**
    \return water flow generator if assigned to this water geometry - otherwise nullptr
    */
    const agxModel::WindAndWaterController::WaterFlowGenerator* getWaterFlowGenerator() const;

#ifndef SWIG
    AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::WaterWrapper );
#endif

  protected:
    virtual ~WaterWrapper() {}

    /**
    Projects a point in local coordinate system onto the surface in local up direction.
    \param localPoint - point in local frame
    \param localUpVector - up vector in local frame
    */
    agx::Vec3 project( const agx::Vec3& localPoint, const agx::Vec3& localUpVector ) const;

  protected:
    agxCollide::GeometryConstObserver                 m_geometry;
    const WindAndWaterController::WaterFlowGenerator* m_flowGenerator;
    const agxCollide::Shape*                          m_shape;
    agx::AffineMatrix4x4                              m_worldToShape;

  DOXYGEN_START_INTERNAL_BLOCK()
  private:
    // Only used by WindAndWaterController
    WaterWrapper( const agxCollide::Geometry* geometry );
    void setFlowGenerator( WindAndWaterController::WaterFlowGenerator* flowGenerator );
    void setWaterGeometry( agxCollide::Geometry* waterGeometry );
    void update();
  DOXYGEN_END_INTERNAL_BLOCK()
  };

  AGX_FORCE_INLINE const AerodynamicsParameters* WindAndWaterController::WindAndWaterData::getAerodynamicsParameters() const
  {
    return m_aeroParams;
  }

  AGX_FORCE_INLINE const HydrodynamicsParameters* WindAndWaterController::WindAndWaterData::getHydrodynamicsParameters() const
  {
    return m_hydroParams;
  }

  AGX_FORCE_INLINE agx::Vec3 ConstantWaterFlowGenerator::getVelocity( const agx::Vec3& /*worldPosition*/ ) const
  {
    return m_velocity;
  }

  AGX_FORCE_INLINE agx::Vec3 ConstantWindGenerator::getVelocity( const agx::Vec3& /*worldPosition*/ ) const
  {
    return m_velocity;
  }
}

#endif // AGXMODEL_WINDANDWATERCONTROLLER_H
