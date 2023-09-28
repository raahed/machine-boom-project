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

#ifndef AGXRENDER_RENDERMANAGER_H
#define AGXRENDER_RENDERMANAGER_H

#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/HashVector.h>
#include <agx/AffineMatrix4x4.h>
#include <agxCollide/Space.h>

#include <agxRender/RenderProxy.h>
#include <agxRender/Renderable.h>
#include <agxRender/Graph.h>

namespace agxSDK
{
  class Simulation;
  class StatisticsEntries;
}

namespace agxRender
{

  class RenderProxyFactory;
  class RenderProxy;
  class SpaceListener;
  class SphereProxy;
  class BoxProxy;
  class LineProxy;
  class HeightFieldProxy;
  class CylinderProxy;
  class HollowCylinderProxy;
  class ConeProxy;
  class TruncatedConeProxy;
  class HollowTruncatedConeProxy;
  class CapsuleProxy;
  class TextProxy;
  class PlaneProxy;
  class GraphRenderer;
  class Graph;

  /// Flags for specifying what is enabled in the debug rendering system
  enum Flags {
    RENDER_GEOMETRIES = 1,                      /**< Render collision geometries */
    RENDER_BOUNDING_VOLUMES = 1 << 1,           /**< Render bounding volumes for geometries */
    RENDER_GEOMETRY_CONTACTS = 1 << 2,          /**< Render geometry contacts */
    RENDER_PARTICLE_CONTACTS = 1 << 3,          /**< Render particle contacts */
    RENDER_PARTICLE_GEOMETRY_CONTACTS = 1 << 4, /**< Render particle-geometry contacts */
    RENDER_CONSTRAINTS = 1 << 5,                /**< Render rigid body constraint attachments */
    RENDER_BODIES = 1 << 6,                     /**< Render rigid body mass centers */
    RENDER_PARTICLE_SYSTEMS = 1 << 7,           /**< Render particle systems */
    RENDER_PARTICLE_EMITTERS = 1 << 8,          /**< Render particle emitters */
    RENDER_TEXT = 1 << 9,                       /**< Render text */
    RENDER_AABB_TREES = 1 << 10,                /**< Render AABB trees*/
    RENDER_SENSORS = 1 << 11,                   /**< Render geometries which are sensors */
    RENDER_MESH = 1 << 12,                      /**< Render trimesh/height field */
    RENDER_NICE_CONTACTS = 1 << 13,             /**< Render contacts with high fidelity */
    RENDER_STATISTICS = 1 << 14,                /**< Render statistics text/graphs onto screen */
    RENDER_RENDERABLES = 1 << 15,               /**< Execute render for all registered Renderables */
    RENDER_BODY_PARTITION = 1 << 16,            /**< Color for a RigidBody is selected based on which partition it belongs to */
    RENDER_BATCH_CONTACTS = 1 << 17,            /**< Render contacts as one batch call for all contacts */
    RENDER_BATCH_BODIES = 1 << 18,              /**< Batch render bodies. */
    RENDER_BATCH_WIRES = 1 << 19,               /**< Wire content batch rendered. */
    RENDER_OBSERVER_FRAMES = 1 << 20,           /**< Render Observer frames. */
    RENDER_CONTACTS =
      RENDER_GEOMETRY_CONTACTS | RENDER_PARTICLE_CONTACTS | RENDER_PARTICLE_GEOMETRY_CONTACTS, /**< Render all types of contacts*/
    RENDER_DEFAULT =
      RENDER_GEOMETRIES |
      RENDER_BODIES |
      RENDER_CONTACTS |
      RENDER_CONSTRAINTS |
      RENDER_RENDERABLES |
      RENDER_PARTICLE_SYSTEMS |
      RENDER_PARTICLE_EMITTERS |
      RENDER_SENSORS |
      RENDER_MESH |
      RENDER_OBSERVER_FRAMES /**< The default rendering mode */
  };

  AGX_DECLARE_POINTER_TYPES( RenderManager );

  /**
  Class for managing the rendering of geometries, shapes, rigid bodies, constraints etc.
  The state (what to be rendered can be set using methods enableFlags/disableFlags/setFlags.
  The RenderManager needs a RenderProxyFactory, implemented according to the design pattern 'Factory'. It will
  be responsible for creating the renderable objects which the RenderManager requests.
  The implementation of the RenderProxyFactory is up to the user, depending on rendering system.

  The RenderManager has a cache of previously created RenderProxy, so it will try to reuse these between frames.
  */
  class AGXPHYSICS_EXPORT RenderManager : public agx::Referenced
  {
    public:

      /**
      Constructor
      \param sim - The associated simulation for which this RenderManager exists.
      */
      RenderManager( agxSDK::Simulation* sim );

      /**
      Updates a new state for the rendering, acquires renderable objects for shapes, text etc.
      Will dispatch create calls to the associated RenderProxyFactory.
      */
      void update();

      /**
      Associate a new factory for creating RenderProxy's on request from the RenderManager
      \param factory - The new render proxy factory
      */
      void setProxyFactory ( RenderProxyFactory* factory );

      /**
      \return a pointer to the associated RenderProxyFactory
      */
      RenderProxyFactory* getProxyFactory();

      /**
      Enable/disable the whole rendering system. Rendering of shapes, constraints and Renderables is disabled.
      \param flag - if true, the rendering is enabled, otherwise disabled
      */
      void setEnable( bool flag );

      /**
      \return true if rendering is enabled.
      */
      bool getEnable( ) const;

      /**
      \return true if rendering is enabled.
      */
      bool isEnabled() const;

      /**
      Loop over all proxies and set alpha value to the specified value, also sets the new default alpha value.
      \param alpha - the new alpha value
      */
      void setAlpha( float alpha );

      /**
      \return the overall, default alpha value
      */
      float getAlpha( ) const;

      /**
      Disable all rendering flags specified in \p flags
      \param flags - The rendering flags to be disabled
      */
      void disableFlags( unsigned int flags );

      /**
      Enable all rendering flags specified in \p flags
      \param flags - The rendering flags to be enabled
      */
      void enableFlags( unsigned int flags );

      /**
      \return true if all rendering flags specified in \p flags are enabled.
      */
      bool isEnabledFlags( unsigned int flags ) const;

      /**
      Set the render flags.
      */
      void setFlags( unsigned int flags );

      /**
      Fires event given eventType for all proxies.
      \param eventType - event type to fire
      */
      void updateAllProxies( agxRender::RenderProxy::EventType eventType );

      /**
      Create and return a new proxy sphere from the Proxy factory
      \param radius - The radius of the new sphere
      \return a pointer to a new SphereProxy with specified radius
      */
      SphereProxy*    acquireSphere( float radius );

      /**
      Create and return a new proxy boxy from the Proxy factory
      \param halfExtents - The size of the box
      \return a pointer to a new BoxProxy with specified size
      */
      BoxProxy*       acquireBox( const agx::Vec3& halfExtents );

      /**
      Create and return a new proxy Line from the Proxy factory
      \param p1, p2 - Start, end points in WORLD coordinate system
      \return a pointer to a new LineProxy
      */
      LineProxy*      acquireLine( const agx::Vec3& p1, const agx::Vec3& p2  );

      /**
      Create and return a new text sphere from the Proxy factory
      \param text - The text
      \param pos - Position of the text. Currently only x,y is used.
      \return a pointer to a new TextProxy
      */
      TextProxy*      acquireText( const agx::String& text, const agx::Vec3& pos );

      /**
      \param normal - The normal of a plane
      \param distance - The scalar part of the plane
      \return a pointer to a new PlaneProxy
      */
      PlaneProxy*     acquirePlane( const agx::Vec3& normal, agx::Real distance );

      /**
      Create and return a new proxy Cylinder from the Proxy factory
      \param radius - The radius of a cylinder
      \param height - The height of a cylinder
      \return a pointer to a new CylinderProxy
      */
      CylinderProxy*  acquireCylinder( float radius, float height );

      /**
      Create and return a new proxy Cylinder from the Proxy factory
      \param begin - The start point in WORLD coordinate system for a cylinder
      \param end - The end point in WORLD coordinate system for a cylinder
      \param radius - The radius of a cylinder
      \return a pointer to a new CylinderProxy
      */
      CylinderProxy*  acquireCylinder( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      /**
      Create and return a new proxy HollowCylinder from the Proxy factory
      \param radius - The radius of a cylinder
      \param height - The height of a cylinder
      \return a pointer to a new HollowCylinderProxy
      */
      HollowCylinderProxy*  acquireHollowCylinder( float radius, float height, float thickness );


      /**
      Create and return a new proxy Cone from the Proxy factory
      \param p1 - The start point in WORLD coordinate system for a cone
      \param p2 - The end point in WORLD coordinate system for a cone
      \param radius - The top radius of a cone
      \return a pointer to a new ConeProxy
      */
      ConeProxy*      acquireCone( const agx::Vec3& p1, const agx::Vec3& p2, float radius );

      /**
      Create and return a new proxy TruncatedCone from the Proxy factory
      \param p1 - The start point in WORLD coordinate system for a cone
      \param p2 - The end point in WORLD coordinate system for a cone
      \param topRadius - The top radius of a cone (at p1)
      \param bottomRadius - The bottom radius for the cone (at p2)
      \return a pointer to a new ConeProxy
      */
      TruncatedConeProxy* acquireTruncatedCone( const agx::Vec3& p1, const agx::Vec3& p2, float topRadius, float bottomRadius );

      /**
      Create and return a new proxy TruncatedCone from the Proxy factory
      \param p1 - The start point in WORLD coordinate system for a cone
      \param p2 - The end point in WORLD coordinate system for a cone
      \param topRadius - The top radius of a cone (at p1)
      \param bottomRadius - The bottom radius for the cone (at p2)
      \param thickness - The wall thickness
      \return a pointer to a new ConeProxy
      */
      HollowTruncatedConeProxy* acquireHollowTruncatedCone( const agx::Vec3& p1, const agx::Vec3& p2, float topRadius, float bottomRadius, float thickness );

      /**
      Create and return a new proxy Capsule from the Proxy factory
      \param radius - The radius of a capsule
      \param height - The height of a capsule
      \return a pointer to a new CapsuleProxy
      */
      CapsuleProxy*  acquireCapsule( float radius, float height );

      /**
      Create and return a new proxy Capsule from the Proxy factory
      \param begin - The start point in WORLD coordinate system for a capsule
      \param end - The end point in WORLD coordinate system for a capsule
      \param radius - The radius of a capsule
      \return a pointer to a new CapsuleProxy
      */
      CapsuleProxy*  acquireCapsule( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      /**
      Create and return a new proxy Contact from the Proxy factory
      \param contacts - Vector with all contacts that should be rendered
      \param scale    - Scale that can be used for scaling the size of the rendered contacts
      \return a pointer to a new ContactsProxy
      */
      ContactsProxy*  acquireContacts( const agxCollide::GeometryContactPtrVector& contacts, float scale );

      /**
      Create and return a new proxy rigid body batch from the Proxy factory
      \param enabledBodies - container with enabled bodies
      \param scale - scale that can be used for scaling the size of the rendered bodies
      \return a pointer to a new rigid body batch render proxy
      */
      RigidBodyBatchRenderProxy* acquireRigidBodies( const agx::RigidBodyPtrSetVector* enabledBodies, float scale );

      /**
      Create and return a new proxy wire from the proxy factory.
      \param radius - global radius of this wire
      \param color - color of this wire
      */
      WireRenderProxy* acquireWire( float radius, const agx::Vec3& color );

      /**
      Clear all acquired RenderProxy. Will call atRemove for each Proxy that is removed.
      \param clearText - true to clear text proxies (Default: true)
      \param clearGlobalProxies - true to clear global proxies that are not part of a cache (Default: true)
      */
      void clear( bool clearText = true, bool clearGlobalProxies = true );

      /**
      Add custom render object.
      \param renderable - custom render object derived from the base class Renderable.
      */
      void addRenderable( Renderable* renderable );

      /**
      \return true if \p renderable is added
      */
      bool hasRenderable( Renderable* renderable ) const;

      /**
      Remove custom render object \p renderable.
      \return true if remove of the object were successful
      */
      bool removeRenderable( Renderable* renderable );

      void acquireGeometry( const agxCollide::Geometry* geometry, const agx::Vec3& color );

      /**
      Try to locate a RenderProxy which is associated (already added) for the specified shape
      \param shape - A pointer to a shape, for which a RenderProxy should be found.
      \return a pointer to a RenderProxy associated to the specified shape. nullptr if no matching proxy is found.
      */
      RenderProxy* findShapeProxy( const agxCollide::Shape* shape );

      /**
      Set the new default render mode for all RenderProxies activated
      after this call.
      */
      void setDefaultRenderMode( RenderProxy::RenderMode mode );

      /**
      Set the scale of the debug rendering of contacts and center of mass, it will not affect already created/rendered objects, only new,
      if you want the scale to affect all objects, call setEnable(false), setEnable(true)
      Default is 1.0 which corresponds to a rendering which suits objects of size 1m or larger...

      */
      void setScaleFactor( float scaleFactor = 1.0f );

      /**
      \return the scale factor used for rendering contacts and center of mass.
      */
      float getScaleFactor() const;

    public:
      typedef std::pair< const agx::ParticleSystem*, agxRender::RenderProxyRef > ConstParticleSystemPtrRenderProxyPtrPair;
      typedef agx::Vector< ConstParticleSystemPtrRenderProxyPtrPair > ParticleSystemProxyContainer;

    protected:

      /**
      Based on the number of used RenderProxy (for each type), reduce the cache down to that size. Remove any outstanding ones.
      */
      void flush();

      friend class SpaceListener;
      friend class agxCollide::Space;
      friend class agxSDK::Simulation;
      friend class agxSDK::StatisticsEntries;


      /**
      Associate a new space with this RenderManager, used by agxSDK::Simulation
      */
      void setSpace( agxCollide::Space* space );

      /**
      \return a pointer to the Graph, used by agxSDK::Simulation
      */
      Graph* getGraph();

      void add( const agx::ParticleSystem* particleSystem );
      void remove( const agx::ParticleSystem* particleSystem );

      /**
      Add a geometry (and its shapes) to the set of active shape proxies.
      \param geometry - The geometry which should be rendered
      \param color - the color which will be used for the geometry. The default value x=-1, is to indicate that its the RenderManager which should set the color
      \param addToProxies - If true (default), it will be added to the Shape proxies and will be updated when shape transforms. If not, it will be rendered only once.
      */
      void add( const agxCollide::Geometry* geometry, const agx::Vec3& color = agx::Vec3( -1, 0, 0 ), bool addToProxies = true );

      /**
      Add a shape  to the set of active shape proxies.
      \param shape - The shape which should be rendered
      \param geometry - The Geometry for the shape
      \param color - the color which will be used for the geometry. The default value x=-1, is to indicate that its the RenderManager which should set the color
      \param addToProxies - If true (default), it will be added to the Shape proxies and will be updated when shape transforms. If not, it will be rendered only once.
      */
      RenderProxy* add( agxCollide::Shape* shape, const agxCollide::Geometry* geometry, const agx::Vec3& color = agx::Vec3( -1, 0, 0 ), bool addToProxies = true );

      /**
      Remove a Shape from the set of shapes that should be rendered
      \param shape - The shape that should be removed
      */
      void remove( const agxCollide::Shape* shape );

      /**
      Remove a Geometry (and its shapes) from the set of shapes that should be rendered
      \param geometry - The Geometry that should be removed
      */
      void remove( const agxCollide::Geometry* geometry );

      /// Render bounding volume for the \p geom
      void renderBoundingVolume( const agxCollide::Geometry* geometry );

      /// Render AABB tree for the specified \p proxy
      void renderAABBTrees( RenderProxy* proxy );

      /// Go through space and add all geometries to the set of shapes that should be rendered
      void addAllGeometries();

      /// Update only text proxies
      void updateText();

      /// Update the graphs
      void updateGraphs();

      /// Update all AABB (for geometries) proxies (boxes)
      void updateAABBProxies( );

      /// Update all observer frames
      void updateObserverFrames();

      /// Render an ObserverFrame
      void renderObserverFrame(const agx::ObserverFrame* observerFrame);

      /// Go through all bodies and acquire proxies for them
      void updateBodyProxies( agxData::EntityStorage* islandStorage = nullptr );

      /// Update (transform and color) for all shapes
      void updateShapeProxies(agxData::EntityStorage* islandStorage = nullptr );

      /// Update particleSystems
      void updateParticleSystemProxies();

      /// Call render for all Renderables
      void updateRenderables();

      /// acquire proxies for contacts
      void updateContactProxies();


      void updateParticleContactProxies();
      void updateParticleGeometryContactProxies();

      /// Update all constraints (call render)
      void updateConstraintProxies();

      /// Remove all shape proxies
      void clearShapeProxies();

      /// Update size, color and transform
      void updateProxy( RenderProxy* proxy, agxData::EntityStorage* islandStorage = nullptr );

      /// Destructor
      virtual ~RenderManager();

      // Attributes
      typedef agx::HashVector<agx::UInt32, agx::ref_ptr<RenderProxy> > RenderProxyHashVector;
      RenderProxyHashVector m_proxies;
      agx::ref_ptr<RenderProxyFactory> m_factory;
      bool m_enable;
      float m_alpha;
      agxSDK::Simulation* m_simulation;
      agxRender::GraphRef m_graph;

      float m_ContactsProxySize;
      float m_scaleFactor;

      ParticleSystemProxyContainer m_particleSystems;

      // Abstract base class for a ProxyCache
      struct CacheBase {
        CacheBase( RenderManager* mgr ) : m_mgr( mgr ) {
          m_mgr->m_cacheVector.push_back( this );
        }

        virtual void clear() = 0;
        virtual void flush() = 0;
        virtual void setAlpha( float alpha ) = 0;
        RenderManager* m_mgr;

        virtual ~CacheBase() {}
      };


      /**
      Class for storing RenderProxies that are currently in use.
      */
      template <typename T>
      class PrimitiveCache : public CacheBase
      {
        public:

          typedef agx::Vector<agx::ref_ptr<T> > ProxyVector;
          typedef typename ProxyVector::const_iterator const_iterator;
          typedef typename ProxyVector::iterator iterator;

          /// Constructor
          PrimitiveCache( RenderManager* mgr ) : CacheBase( mgr ), m_index( 0 ) {}

          /**
          \return a new RenderProxy (of type T*)
          */
          T* acquire();

          /**
          Reduce size of pooled RenderProxy down to the number of used ones.
          */
          void flush() {
            for ( size_t i = m_index; i < m_pool.size(); i++ )
              m_pool[i]->callOnChange( RenderProxy::REMOVE );

            m_pool.resize( std::min( m_index, m_pool.size() ) );
            m_index = 0;
          }

          /**
          Set alpha value for all proxies
          \param alpha - The new alpha value (0..1)
          */
          void setAlpha( float alpha ) {
            for ( size_t i = 0; i < m_pool.size(); i++ )
              m_pool[i]->setAlpha( alpha );
          }

          /**
          Reduce number of proxies to 0. Clear all data.
          */
          void clear() {
            for ( typename ProxyVector::iterator it = m_pool.begin(); it != m_pool.end(); ++it )
              ( *it )->callOnChange( RenderProxy::REMOVE );
            m_pool.clear( agx::Container::SHRINK_BUFFER );
            m_index = 0;
          }

          /// Destructor: calls clear()
          virtual ~PrimitiveCache() {
            clear();
          }

          iterator begin() { return m_pool.begin(); }
          const_iterator begin() const { return m_pool.begin(); }

          iterator end() { return m_pool.end(); }
          const_iterator end() const { return m_pool.end(); }

        protected:

          ProxyVector m_pool;
          size_t m_index;
      };

      /**
      Class for storing RenderProxies that are currently in use.
      */
      template <typename T, typename P1>
      class PrimitiveCache1 : public PrimitiveCache<T>
      {
        public:

          /// Constructor
          PrimitiveCache1( RenderManager* mgr ) : PrimitiveCache<T>( mgr ) {}

          /**
          \return a new RenderProxy (of type T*)
          */
          T* acquire( const P1& param );

        private:

          T* acquire() { return nullptr; }
      };

      typedef agx::Vector<CacheBase*> CacheVector;
      CacheVector m_cacheVector;

      PrimitiveCache<TextProxy> m_textCache;
      PrimitiveCache<PlaneProxy> m_planeCache;
      PrimitiveCache<CapsuleProxy> m_capsuleCache;
      PrimitiveCache<CylinderProxy> m_cylinderCache;
      PrimitiveCache<HollowCylinderProxy> m_hollowCylinderCache;
      PrimitiveCache<ConeProxy> m_coneCache;
      PrimitiveCache<TruncatedConeProxy> m_truncatedConeCache;
      PrimitiveCache<HollowTruncatedConeProxy> m_hollowTruncatedConeCache;
      PrimitiveCache1<SphereProxy, float> m_sphereCache;
      PrimitiveCache1<BoxProxy, agx::Vec3> m_boxCache;
      PrimitiveCache<LineProxy> m_lineCache;
      PrimitiveCache1<ContactsProxy, agxCollide::GeometryContactPtrVector> m_contactsCache;
      PrimitiveCache1<RigidBodyBatchRenderProxy, const agx::RigidBodyPtrSetVector*> m_rbCache;

      agx::ref_ptr<SpaceListener> m_listener;

      agx::UInt32 m_flags;
      typedef agx::Vector<RenderableRef> RenderableRefVector;
      RenderableRefVector m_renderables;
      typedef agx::Vector< agx::ref_ptr< agxRender::RenderProxy > > RenderProxyRefContainer;
      RenderProxyRefContainer m_globalProxies; /**< E.g., wires that hold their proxy (i.e., not doing acquire to fetch their render proxy). */
      RenderProxy::RenderMode m_defaultRenderMode;

      agxData::BufferRef m_particleContactRadiusBuffer;
      agxData::BufferRef m_particleGeometryContactRadiusBuffer;
      agxData::ValueRef m_particleContactBound;
      agxData::ValueRef m_particleGeometryContactBound;
      RenderProxyRef m_particleContactRenderProxy;
      RenderProxyRef m_particleGeometryContactRenderProxy;

  };


  AGX_DECLARE_POINTER_TYPES( RenderProxyFactory );
  /**
  Abstract class which is responsible for returning a pointer to a specified RenderProxy type.
  This class is to be implemented by the user, depending on the rendering system.
  The implementation of this factory might have some own cache of Proxies, or not.

  This class is also responsible for creating a class derived from agxRender::Graph::GraphRenderer.
  */
  class AGXPHYSICS_EXPORT RenderProxyFactory : public agx::Referenced
  {
    public:
      RenderProxyFactory( );

      /// Interface for creating and returning a SphereProxy
      virtual SphereProxy* createSphere( float radius ) = 0;

      /// Interface for creating and returning a BoxProxy
      virtual BoxProxy* createBox( const agx::Vec3& halfExtents ) = 0;

      /// Interface for creating and returning a LineProxy
      virtual LineProxy* createLine( const agx::Vec3& p1, const agx::Vec3& p2 ) = 0;

      /// Interface for creating and returning a CylinderProxy
      virtual CylinderProxy* createCylinder( float radius, float height ) = 0;

      /// Interface for creating and returning a HollowCylinderProxy
      virtual HollowCylinderProxy* createHollowCylinder( float radius, float height, float thickness ) = 0;

      /// Interface for creating and returning a ConeProxy
      virtual ConeProxy* createCone( float radius, float height ) = 0;

      /// Interface for creating and returning a TruncatedConeProxy
      virtual TruncatedConeProxy* createTruncatedCone( float topRadius, float bottomRadius, float height ) = 0;

      /// Interface for creating and returning a HollowTruncatedConeProxy
      virtual HollowTruncatedConeProxy* createHollowTruncatedCone( float topRadius, float bottomRadius, float height, float thickness ) = 0;

      /// Interface for creating and returning a CapsuleProxy
      virtual CapsuleProxy* createCapsule( float radius, float height ) = 0;

      /// Interface for creating and returning a WireShapeProxy
      virtual WireShapeProxy* createWireShape( float radius, float height,
          const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1 ) = 0;

      /// Interface for creating and returning TextProxy
      virtual TextProxy* createText( const agx::String& text, const agx::Vec3& pos ) = 0;

      /// Interface for creating and returning PlaneProxy
      virtual PlaneProxy* createPlane( const agx::Vec3& normal, agx::Real distance ) = 0;

      /// Interface for creating and returning HeightfieldProxy
      virtual HeightFieldProxy* createHeightfield( agxCollide::HeightField* hf ) = 0;

      /// Interface for creating and returning ContactsProxy
      virtual ContactsProxy* createContacts( const agxCollide::GeometryContactPtrVector& ) = 0;

      /// Interface for creating and returning batch rendering of rigid bodies (cm).
      virtual RigidBodyBatchRenderProxy* createRigidBodies( const agx::RigidBodyPtrSetVector* /*enabledBodies*/ ) = 0;

      virtual WireRenderProxy* createWire( float radius, const agx::Vec3& color ) = 0;

      virtual RenderProxy* createSphereBatchRenderer(
        agxData::Buffer *positions,
        agxData::Buffer *rotations,
        agxData::Buffer *radii,
        agxData::Buffer *colors,
        agxData::Buffer *enableRendering,
        agxData::Value *bound,
        agx::Component *context ) = 0;

      /// Interface for creating and returning TrimeshProxy
      virtual TrimeshProxy* createTrimesh( agxCollide::Trimesh* mesh ) = 0;

      /// Create a SphereProxy without arguments
      SphereProxy* create( float radius );

      /// Create a BoxProxy without arguments
      BoxProxy* create( const agx::Vec3& halfExtents );

      /// Create a LineProxy without arguments
      LineProxy* create( const LineProxy* );

      /// Create a CylinderProxy without arguments
      CylinderProxy* create( const CylinderProxy* );

      /// Create a CylinderProxy without arguments
      HollowCylinderProxy* create( const HollowCylinderProxy* );

      /// Create a ConeProxy without arguments
      ConeProxy* create( const ConeProxy* );

      /// Create a TruncatedConeProxy without arguments
      TruncatedConeProxy* create( const TruncatedConeProxy* );

      /// Create a HollowTruncatedConeProxy without arguments
      HollowTruncatedConeProxy* create( const HollowTruncatedConeProxy* );

      /// Create a CapsuleProxy without arguments
      CapsuleProxy* create( const CapsuleProxy* );

      /// Create a TextProxy without arguments
      TextProxy* create( const TextProxy* );

      /// Create a PlaneProxy without arguments
      PlaneProxy* create( const PlaneProxy* );

      /// Create a ContactsProxy without arguments
      ContactsProxy* create( const agxCollide::GeometryContactPtrVector& contacts );

      /// Create a rigid body batch render proxy
      RigidBodyBatchRenderProxy* create( const agx::RigidBodyPtrSetVector* enabledBodies );

      /**
      This method should create and return an implementation of a GraphRenderer, a class that can render the statistics graphs
      as described in the class Graph::GraphRenderer.
       \return a pointer to the GraphRenderer
      */
      virtual Graph::GraphRenderer* getGraphRenderer() = 0;

      /**
      Set the default RenderMode that will be used when a new RenderProxy is created.
      \param mode - New RenderMode
      */
      void setDefaultRenderMode( RenderProxy::RenderMode mode );

      /**
      \return The RenderMode used when new RenderProxy's are created
      */
      RenderProxy::RenderMode getDefaultRenderMode( ) const;

      void setEnable( bool flag );
      bool getEnable( ) const;

    protected:
      friend class RenderManager;
      void setRenderManager( RenderManager* mgr ) {
        m_mgr = mgr;
      }
      virtual ~RenderProxyFactory();
      RenderManager* m_mgr;
      RenderProxy::RenderMode m_defaultRenderMode;
      bool m_enabled;
  };

  inline void RenderProxyFactory::setEnable( bool flag )
  {
    m_enabled = flag;
  }
  inline bool RenderProxyFactory::getEnable( ) const
  {
    return m_enabled;
  }

  /// A listener that will be responsible for calling RenderManager every time a Shape is added/removed from space
  class AGXPHYSICS_EXPORT SpaceListener : public agxCollide::SpaceListener
  {
    public:

      /// Constructor
      SpaceListener( RenderManager* mgr );

      agxCollide::Space* getSpace();
    protected:

      /// Destructor
      virtual ~SpaceListener();

      /// Called when this listener is added to a Space
      void addNotification( agxCollide::Space* space );

      /// Called when a Geometry is added to the associated Space
      void addGeometryNotification( agxCollide::Space* space, agxCollide::Geometry* geometry );
      /// Called when a Geometry is removed to the associated Space
      void removeGeometryNotification( agxCollide::Space* space, agxCollide::Geometry* geometry );
      /// Called when a Shape is added to a Geometry which is associated to Space
      void addShapeNotification( agxCollide::Space* space, agxCollide::Geometry* geometry, agxCollide::Shape* shape );
      /// Called when a Shape is removed from a Geometry which is associated to Space
      void removeShapeNotification( agxCollide::Space* space, agxCollide::Geometry* geometry, agxCollide::Shape* shape );

      agx::observer_ptr<RenderManager> m_mgr;
      agx::observer_ptr<agxCollide::Space> m_space;
  };




  // Implementation

  inline bool RenderManager::getEnable( ) const
  {
    return m_enable;
  }


  inline float RenderManager::getAlpha( ) const
  {
    return m_alpha;
  }


  inline bool RenderManager::isEnabled() const
  {
    return getEnable();
  }


  /// Create a SphereProxy without arguments
  inline SphereProxy* RenderProxyFactory::create( float radius )
  {
    return createSphere( radius );
  }
  /// Create a BoxProxy without arguments
  inline BoxProxy* RenderProxyFactory::create( const agx::Vec3& halfExtents )
  {
    return createBox( halfExtents );
  }
  /// Create a LineProxy without arguments
  inline LineProxy* RenderProxyFactory::create( const LineProxy* )
  {
    return createLine( agx::Vec3( agx::Real( 0 ) ), agx::Vec3( agx::Real( 1 ) ) );
  }
  /// Create a CylinderProxy without arguments
  inline CylinderProxy* RenderProxyFactory::create( const CylinderProxy* )
  {
    return createCylinder( 1, 1 );
  }
  /// Create a HollowCylinderProxy without arguments
  inline HollowCylinderProxy* RenderProxyFactory::create( const HollowCylinderProxy* )
  {
    return createHollowCylinder( 1, 1, 0.1f );
  }

  /// Create a ConeProxy without arguments
  inline ConeProxy* RenderProxyFactory::create( const ConeProxy* )
  {
    return createCone( 1, 1 );
  }
  /// Create a TruncatedConeProxy without arguments
  inline TruncatedConeProxy* RenderProxyFactory::create( const TruncatedConeProxy* )
  {
    return createTruncatedCone( 1, 1, 1 );
  }
  /// Create a TruncatedConeProxy without arguments
  inline HollowTruncatedConeProxy* RenderProxyFactory::create( const HollowTruncatedConeProxy* )
  {
    return createHollowTruncatedCone( 1, 1, 1, 0.1f );
  }

  /// Create a CapsuleProxy without arguments
  inline CapsuleProxy* RenderProxyFactory::create( const CapsuleProxy* )
  {
    return createCapsule( 1, 1 );
  }
  /// Create a TextProxy without arguments
  inline TextProxy* RenderProxyFactory::create( const TextProxy* )
  {
    return createText( "", agx::Vec3( 0, 0, 0 ) );
  }
  /// Create a PlaneProxy without arguments
  inline PlaneProxy* RenderProxyFactory::create( const PlaneProxy* )
  {
    return createPlane( agx::Vec3( 0, 0, 1 ), 0 );
  }

  /// Create a ContactsProxy without arguments
  inline ContactsProxy* RenderProxyFactory::create( const agxCollide::GeometryContactPtrVector& contacts )
  {
    return createContacts( contacts );
  }

  inline RigidBodyBatchRenderProxy* RenderProxyFactory::create( const agx::RigidBodyPtrSetVector* enabledBodies )
  {
    return createRigidBodies( enabledBodies );
  }

  inline RenderProxy::RenderMode RenderProxyFactory::getDefaultRenderMode() const
  {
    return m_defaultRenderMode;
  }

  template< typename T>
  T* RenderManager::PrimitiveCache<T>::acquire( void )
  {
    if ( m_index < m_pool.size() ) {
      T* proxy = m_pool[m_index++];

      // Reset color, alpha and enabled state
      proxy->reset();
      return proxy;
    } else {
      const void* p = nullptr;
      const T* px = static_cast<const T*>( p );
      T* proxy = m_mgr->getProxyFactory()->create( px);
      proxy->setEnable( true );
      m_pool.push_back( proxy );
      m_index = m_pool.size();
      return proxy;
    }
  }

  template< typename T, typename P1>
  T* RenderManager::PrimitiveCache1<T, P1>::acquire( const P1& param )
  {
    if ( this->m_index < this->m_pool.size() ) {
      T* proxy = this->m_pool[this->m_index++];
      // Reset color, alpha and enabled state
      proxy->reset();
      return proxy;
    } else {
      T* proxy = this->m_mgr->getProxyFactory()->create( param );
      proxy->setEnable( true );
      this->m_pool.push_back( proxy );
      this->m_index = this->m_pool.size();
      return proxy;
    }
  }
}

#endif
