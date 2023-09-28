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

#ifndef AGXMODEL_TERRAIN_UTILS_H
#define AGXMODEL_TERRAIN_UTILS_H

#include <agxModel/export.h>


#include <agxSDK/StepEventListener.h>
#include <agxCollide/Geometry.h>
#include <agxCollide/HeightField.h>
#include <agxCollide/Contacts.h>
#include <agx/Material.h>

#include <agx/FrictionModel.h>

#include <agxSDK/Simulation.h>


#include <agxModel/TerrainDataInterface.h>
#include <agxModel/TerrainParticles.h>
#include <agxRender/RenderManager.h>

namespace agxModel
{

  class ImpactVolumeData;

  /**
  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  The height controller is used to control the heights.
  It is possible to change heights cheaper between the calls to startLocalHeightUpdate and endLocalHeightUpdate.
  */
  class AGXMODEL_EXPORT HeightController : public agx::Referenced, agxStream::Serializable
  {
    public:
      HeightController( agxCollide::HeightField* heightField )
        : m_localHeightUpdate(false), m_lockBorders(false), m_avalancheMassLoss(true), m_heightField(heightField)
      {}
      HeightController()
        : m_localHeightUpdate(false), m_lockBorders(false), m_avalancheMassLoss(true), m_heightField(nullptr)
      {}
      void setEnableLockedBorders( bool lock, bool avalancheMassLoss = true ) {
        m_lockBorders = lock;
        m_avalancheMassLoss = avalancheMassLoss;
      }
      bool getEnabledAvalancheMassLoss() const {
        return m_avalancheMassLoss;
      }
      bool getEnabledLockedBorders() const {
        return m_lockBorders;
      }
      void calculateMassProperties() {
        m_heightField->calculateMassProperties();
      }
      void setHeight( size_t x, size_t y, agx::Real height );
      AGX_FORCE_INLINE size_t getLocalX( size_t globalX ) {
        return std::max(globalX, m_minX) - m_minX;
      }
      AGX_FORCE_INLINE size_t getLocalY( size_t globalY ) {
        return std::max(globalY, m_minY) - m_minY;
      }
      AGX_FORCE_INLINE size_t getLocalVectorIndex( size_t globalX, size_t globalY ) {
        return (m_maxX - m_minX + 1) * getLocalY(globalY) + getLocalX(globalX);
      }
      agx::Real getHeight( size_t x, size_t y ) const;
      void startLocalHeightUpdate( size_t minX = 0, size_t maxX = 0, size_t minY = 0, size_t maxY = 0);
      void endLocalHeightUpdate();

      AGX_FORCE_INLINE bool getLocalHeightUpdate() const {
        return m_localHeightUpdate;
      }
      AGX_FORCE_INLINE const agxCollide::HeightField* getHeightField() const {
        return m_heightField;
      }

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::HeightController);

    protected:

    private:
      bool m_localHeightUpdate;
      bool m_lockBorders;
      bool m_avalancheMassLoss;
      agxCollide::HeightFieldRef m_heightField;
      agx::RealVector m_tempHeights;
      size_t m_minX;
      size_t m_maxX;
      size_t m_minY;
      size_t m_maxY;
  };
  typedef agx::ref_ptr<HeightController> HeightControllerRef;

  class TerrainDebugRenderer : public agxRender::Renderable
  {
    public:
      TerrainDebugRenderer( agxModel::Terrain* terrain );
      void render( agxRender::RenderManager* mgr );

    protected:
      virtual ~TerrainDebugRenderer();

      Terrain* m_terrain;
      agx::observer_ptr<agxRender::HeightFieldProxy> m_proxy;
      agx::RealVector m_heights;
  };

  /**
  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  Making it possible to have arbitrary resolution of the particle grid.
  */
  class ParticleCoordinateSystem
  {
    public:

      ParticleCoordinateSystem( )
        : m_particleResolutionX(1), m_particleResolutionY(1), m_scale(1), m_heightFieldToParticleCoordinateMultiplier(1)
      {}

      /**
      If the particle radius is changed, we must update our resolution for the particle coordinate system
      */
      void updateResolution( const agx::Real& particleRadius, const size_t& heightFieldResolutionX, const size_t& heightFieldResolutionY, const agx::Real& scale  );

      enum DirectionFCC { VERTEX_POS, POS_X_NEG_Z, POS_Y_NEG_Z, NEG_X_NEG_Z, NEG_Y_NEG_Z, POS_X_POS_Y, NEG_X_POS_Y, NEG_X_NEG_Y, POS_X_NEG_Y, POS_X_POS_Z, POS_Y_POS_Z, NEG_X_POS_Z, NEG_Y_POS_Z };
      //middle vertex,--------- the four below-------ccw--------------,-----the four on same height as vertex----ccw---,----the four above------ccw----------------------

      /**
      Calculate height of height field fcc position, given x and y derivatives and the height of the middle vertex
      */
      agx::Real findFccHeight( agx::Real heightAtVertexPos, agx::Real dX, agx::Real dY, DirectionFCC fccLocalPos );

      /**
      Calculate the height of a particle position
      */
      agx::Real findParticleCoordinateHeight(const size_t pcsX, const size_t pcsY, Terrain* terrain );


      /**
      Translates a position in the imaginary FCC 3 dimensional grid to a world position
      Returns positions in the Particle Coordinate system
      */
      agx::Vec3 getHeightFieldPositionFromFCCIndex( size_t fccIndex, const agx::Real minZ, size_t& pcsX, size_t& pcsY );

      /**
      To find what hf vertex position
      For example needed when we need to know which material a particle should get
      */
      void getParentHeightFieldVertex( const size_t& pcsX, const size_t& pcsY, size_t& hfX, size_t& hfY );

      /**
      Given one vertex in the height field (that has changed height for example) we get all surrounding
      particle positions for those 6 triangles that the vertex is found in.
      */
      void collectCloseParticleCoordinates( const size_t& hfX, const size_t& hfY, agx::Vec2iVector& particleCoordinates );

      /**
      Given one vertex in the height field (that has changed height for example) we get all surrounding
      particle positions for those 6 triangles that the vertex is found in.
      Also we get the height of the heightfield at that position
      */
      void collectCloseParticleLowerXYCoordinatesWithHeight( const agxCollide::HeightField* hf, const size_t& hfX, const size_t& hfY,
          Vec2iHeightDataPairVector& particleCoordinatesWithHeight, Terrain* terrain);

      /**
      hfX, hfY are index in height field. Fcc lattice has double resolution, and height information
      dir is if we refer to the same position as height field vertex or any of the 8 surrounding position
      */
      size_t indexForFCCLattice( size_t pcsX, size_t pcsY, size_t pcsZ, DirectionFCC dir );

      /**
      Given the changed height (depth) of a height field vertex, we get the height change at the particle coordinate system position.
      */
      agx::Real scaleDepth( const agx::Real& depth, const size_t pcsX, const size_t pcsY, const agx::Vec2& hf2Dpos );

      /**
      Get 2D position on the height field in length units. Given particle coordinate system position.
      dX and dY are if we are looking for position on fcc position, not on a particle system vertex.
      */
      agx::Vec2 getHeightFieldPosition( const size_t pcsX, const size_t pcsY, const size_t dX = 0, const size_t dY = 0 );

    private:
      size_t m_particleResolutionX;
      size_t m_particleResolutionY;

      agx::Real m_scale;

      agx::Real m_heightFieldToParticleCoordinateMultiplier;
  };

  ///  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  class AGXMODEL_EXPORT ContactGenerator : public agxSDK::StepEventListener
  {
    public:
      ContactGenerator( ) {
        setMask( agxSDK::StepEventListener::PRE_STEP | agxSDK::StepEventListener::POST_STEP );
      }

      void post(const agx::TimeStamp& ) {
        m_localContacts.clear(agx::Container::MAINTAIN_BUFFER);
      }

      void pre( const agx::TimeStamp& /*t*/ ) {
        this->getSimulation()->getSpace()->addGeometryContacts(m_localContacts);
      }

      bool createGeometryContact(agxCollide::Geometry* g1, agxCollide::Geometry* g2, agx::Vec3Vector points, agx::Vec3Vector normals, agx::RealVector depths, agx::Real pointArea, agx::ContactMaterial* cm = nullptr);


    private:
      agxCollide::LocalGeometryContactVector m_localContacts;

  };

  typedef agx::ref_ptr<ContactGenerator> ContactGeneratorRef;

  AGX_DECLARE_POINTER_TYPES(TerrainIterativeProjectedConeFrictionModel);

  typedef agx::Vector<std::pair<agx::RigidBodyObserver, agx::Vec3>> RigidBodyObserverVec3PairVector;

  ///  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  class AGXMODEL_EXPORT TerrainIterativeProjectedConeFrictionModel : public agx::IterativeProjectedConeFriction
  {
  public:
    /**
    Default constructor.

    INITIALIZE and SCALE_BOUNDS if solveType != SPLIT.
    For split solves we only get project callbacks during iterations.
    DIRECT_AND_ITERATIVE ScaleBoxFriction::update gets calls
    from the NLMCP and this::project during iterations.
    */
    TerrainIterativeProjectedConeFrictionModel(agx::FrictionModel::SolveType solveType = agx::FrictionModel::SPLIT);

    /**
    Calculates friction plane given normal. Only valid if \p normal, \p ret_u and \p ret_v are orthogonal.
    \param geometry1 - first geometry in the geometry contact
    \param geometry2 - the other geometry in the geometry contact
    \param point - contact point in world coordinate system
    \param normal - contact normal in world coordinate system
    \param depth - penetration depth
    \param ret_u - returned, primary direction in friction plane
    \param ret_v - returned, secondary direction in the friction plane
    */
    virtual void calculateTangentPlane(const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2, const agx::Vec3& point, const agx::Vec3& normal, const agx::Real depth, agx::Vec3& ret_u, agx::Vec3& ret_v) const override;

    void addBody(agx::RigidBody* body, const agx::Vec3& localFrictionDir);

    AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::TerrainIterativeProjectedConeFrictionModel);

  protected:

    RigidBodyObserverVec3PairVector m_bodiesAndDir;
    virtual ~TerrainIterativeProjectedConeFrictionModel();
  };


  ///  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  class DynamicHeightFieldSimulation : public agxSDK::Simulation
  {
    public:
      DynamicHeightFieldSimulation( );

      ParticleCoordinateSystem& getParticleCoordinateSystem() {
        return m_dynamicHeightFieldParticleCoordinateSystem;
      }

      ContactGenerator* getContactGenerator() {
        return m_contactGenerator;
      }

      void setParticleRadius( agx::Real radius ) {
        m_particleRadius = radius;
      }

      agx::Real getParticleRadius() {
        return m_particleRadius;
      }

      agx::UInt32 getParticleUniqueId() const {
        return m_uniqueParticleId;
      }

    protected:
      agx::UInt32 m_uniqueParticleId;
      ParticleCoordinateSystem m_dynamicHeightFieldParticleCoordinateSystem;
      ContactGeneratorRef m_contactGenerator;
      agx::Real m_particleRadius;
      bool m_useDistanceJoint;
  };

  typedef agx::ref_ptr<DynamicHeightFieldSimulation> DynamicHeightFieldSimulationRef;


  /**
  \deprecated This class is deprecated and will be removed in a future version of AGX. \sa agxTerrain::Terrain
  Class for storing a geometry contact and information about all its contact points
  */
  class GeometryContactInfo : public agx::Referenced
  {
    public:
      GeometryContactInfo( agxCollide::GeometryContact* gc, size_t index )
        : m_geometryContact(gc), m_index(index) {
        m_geometry1 = gc->geometry(0);
        m_geometry2 = gc->geometry(1);
      }

      bool isValid() const {
        return m_geometry1->getEnable() && m_geometry2->getEnable() && (m_geometry1->getRigidBody() == nullptr || m_geometry1->getRigidBody()->getEnable() ) && (m_geometry2->getRigidBody() == nullptr || m_geometry2->getRigidBody()->getEnable());
      }

      void add( ImpactVolumeData* ivd ) {
        m_impactVolumeDataVector.push_back(ivd);
      }

      const agxCollide::Geometry* getGeometry1() const { return m_geometry1; }
      agxCollide::Geometry* getGeometry1() { return m_geometry1; }
      const agxCollide::Geometry* getGeometry2() const { return m_geometry2; }
      agxCollide::Geometry* getGeometry2() { return m_geometry2; }

      size_t getIndex() {
        return m_index;
      }

      void clear() {
        m_impactVolumeDataVector.clear();
      }

      agxCollide::GeometryContact*     getGeometryContact() {
        return m_geometryContact;
      }
      const ImpactVolumeDataPtrVector& getImpactVolumeDataVector() {
        return m_impactVolumeDataVector;
      }
    protected:
      virtual ~GeometryContactInfo( ) {
        clear();
      }

    private:
      ImpactVolumeDataPtrVector m_impactVolumeDataVector;
      agxCollide::GeometryContact* m_geometryContact;
      agxCollide::GeometryRef m_geometry1;
      agxCollide::GeometryRef m_geometry2;
      size_t m_index;

  };

  typedef agx::ref_ptr<GeometryContactInfo> GeometryContactInfoRef;
  typedef agx::Vector< GeometryContactInfoRef > GeometryContactInfoRefVector;

  AGX_FORCE_INLINE agx::Real HeightController::getHeight( size_t x, size_t y ) const
  {
    if ( m_heightField == nullptr )
      return agx::Real(0);
    size_t vectorIndex = m_heightField->getResolutionX() * y + x;
    agxAssert( !getLocalHeightUpdate() || m_tempHeights.size() > vectorIndex);

    if ( getLocalHeightUpdate() && m_tempHeights[vectorIndex] != -agx::Infinity )
      return m_tempHeights[vectorIndex];

    return m_heightField->getHeight(x, y);
  }

  AGX_FORCE_INLINE void HeightController::setHeight( size_t x, size_t y, agx::Real height )
  {
    if ( m_heightField == nullptr )
      return;

    if ( m_lockBorders && ( x == 0 || y == 0 || x + 1 >= m_heightField->getResolutionX() || y + 1 >= m_heightField->getResolutionY() ) )
      return;
    if ( getLocalHeightUpdate() )
      m_tempHeights[getLocalVectorIndex(x, y)] = height;
    else
      m_heightField->setHeight(x, y, height);
  }

  class ContactPointData
  {
  public:
    ContactPointData(ImpactVolumeData* vd,Terrain* terrain);

    ImpactVolumeData* ivd;
    size_t idx;
    size_t x;
    size_t y;
    const agx::Material* terrainMat;
    const agx::Material* particleMat;

  private:
    void initialize(Terrain* terrain);


  };

  class DeformingGeometryContactProperties
  {
  public:

    DeformingGeometryContactProperties(agxCollide::Geometry* otherGeometry);

    void extractGeometryProperties(Terrain* terrain);

    //Values found before analyzing contact points
    agx::observer_ptr<agxCollide::Geometry> collidedGeometry;
    agx::Real maxDepthForward;
    agx::Vec3 shovelForwardW;
    agx::Real shovelLength;
    agx::Bool isDeformer;
    agx::Bool useGravityDepth;
    agx::Bool isShovel;
    agx::Bool isShovelMovingForward;
    agx::Vec3 shovelExtraContactPosition;
    agx::Vec3 shovelNormalW;
    agx::Real shovelNormalLength;
    agx::Vec3 gPos;

    // values updated during contact point analysis
    agx::Real maxDepth;
    agx::Real maxAdhesion;
    agx::Real maxDeformabiliy;
    agx::Vec3 meanContactPositionW;
    agx::Real totalYoungsModulus;
    agx::Real maxYoungs;
    agx::Real shovelContactLength;
    agx::Real minLengthContactToShovelTip;
    agx::Real minYoungsModulusTerrain;

  };

  class VolumeOverlap : public agx::Vec3
  {
    public:
      VolumeOverlap();
      enum Type
      {
        TOTAL = 0,
        COMPRESSED,
        MOVED
      };

  };

}

#endif
