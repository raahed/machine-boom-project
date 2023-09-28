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

#ifndef AGXRENDER_RENDERPROXY_H
#define AGXRENDER_RENDERPROXY_H


#include <agx/agxPhysics_export.h>
#include <agx/DynamicsSystem.h>

namespace agxCollide
{
  class Shape;
  class Geometry;
  class Sphere;
  class Box;
  class HeightField;
  class Line;
  class Cylinder;
  class HollowCylinder;
  class Capsule;
  class Cone;
  class HollowCone;
  class Trimesh;
  class Plane;
  class WireShape;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxRender
{


  enum PrimitiveType {
    PRIMITIVE_BOX,
    PRIMITIVE_CAPSULE,
    PRIMITIVE_CYLINDER,
    PRIMITIVE_CONE,
    PRIMITIVE_LINE,
    PRIMITIVE_PLANE,
    PRIMITIVE_SPHERE,
    PRIMITIVE_TRIMESH,
    PRIMITIVE_HEIGHT_FIELD,
    PRIMITIVE_TRIANGLE,
    PRIMITIVE_SPHERE_BATCH,
    PRIMITIVE_TEXT,
    PRIMITIVE_CONTACTS,
    PRIMITIVE_HOLLOW_CYLINDER,
    PRIMITIVE_TRUNCATED_CONE,
    PRIMITIVE_HOLLOW_TRUNCATED_CONE,
    NUM_PRIMITIVES
  };

  AGX_DECLARE_POINTER_TYPES( RenderProxy );

  /**
  Abstract base class which encapsulated information for rendering a specific shape.
  */
  class AGXPHYSICS_EXPORT RenderProxy : public agx::Referenced
  {
    public:

      /// Enum for specifying a type of event which might change a RenderProxy
      enum EventType {
        ENABLE,         /**< A RenderProxy is enabled/disabled */
        TRANSFORM,      /**< The transformation for a proxy is updated */
        ALPHA,          /**< The alpha value for a proxy is updated */
        COLOR,          /**< The color value for a proxy is updated */
        SHAPE,          /**< The Shape has been modified a rebuild is needed */
        REMOVE,         /**< The Shape has been removed from space, need to be removed from rendering */
        RENDERMODE      /**< The RenderMode has been changed */
      };

      enum RenderMode {
        NOT_SET = 0x0,
        SOLID = 0x1,
        WIREFRAME = 0x2
      };

      /// Constructor
      RenderProxy( PrimitiveType type, agxCollide::Shape *shape = nullptr );

      /// \return the type of this RenderProxy
      PrimitiveType getType() const;

      /// Set the transform of this RenderProxy, if overridden, set the m_transform and call onUpdate(TRANSFORM)
      virtual void setTransform( const agx::AffineMatrix4x4& transform );

      /**
      Set the color of this proxy. If overridden, store color in m_color and call onUpdate(COLOR)
      */
      virtual void setColor( const agx::Vec3& color );

      /**
      \return the current color
      */
      virtual agx::Vec3 getColor() const;


      /**
      Set the alpha value of this proxy. If overridden, store color in m_alpha and call onUpdate(ALPHA)
      */
      virtual void setAlpha( float alpha );

      /// \return the current alpha value for this proxy
      virtual float getAlpha() const;

      /**
      Called when a Shape has been changed (for example radius on a sphere is changed).
      */
      virtual void updateShape() {}


      /// \return the current transform for this proxy
      const agx::AffineMatrix4x4& getTransform() const;

      /**
      This value tells us that if the proxy is synchronized with the associated shape.
      When a shape change any of its attributes, it will increment its own number. If it does not
      match this, it will trigger onChange(SHAPE).
      \return the current modified count for this Proxy.
      */
      agx::UInt32 getModifiedCount() const;


      const agxCollide::Shape *getShape() const;
      const agxCollide::Geometry *getGeometry() const;

      /// Compare the modified count with its associated shape
      bool compare() const;

      /**
      Explicitly fires event of \p type.
      \param type - type of event to fire to this render proxy
      */
      void callOnChange( EventType type );

      /**
      Toggle this proxy for rendering on/off.
      Will trigger onUpdate(ENABLE)
      */
      void setEnable( bool flag );

      /// \return true if this proxy is enabled
      bool getEnable() const;

      /// \return true if this proxy is enabled
      inline bool isEnabled() const { return getEnable(); }

      /**
      Set the render mode for this Proxy.
      Will trigger onUpdate(RENDERMODE)
      */
      void setRenderMode( RenderMode mode );

      /// \return the current render mode for this Proxy
      RenderMode getRenderMode( ) const;

      /// Will reset color, alpha and enabled state back to default state (as when initially created).
      virtual void reset();

      /**
      \return true if the rigid body should be rendered - otherwise false
      */
      static agx::Bool shouldRender( const agx::RigidBody* rb );

    protected:
      /**
      This method should be implemented so that it extracts size parameters from an associated shape and store it in the Proxy parameters (radius/length etc).
      */
      virtual void  synchronizeParameters() {}

      /**
      Pure virtual method that should response on the different events.
      This method is (indirectly via callOnChange) called from most set* methods.
      Should really never be called direct from anywhere else but from callOnChange.
      */
      virtual void onChange( EventType /*type*/ ) {}

      friend class RenderManager;
      void setModifiedCount( agx::UInt32 c );
      /// Set the shape for this proxy
      void setShape( agxCollide::Shape* shape, const agxCollide::Geometry *geometry );

      /// Destructor
      virtual ~RenderProxy();

      agx::observer_ptr<agxCollide::Shape> m_shape;
      agx::AffineMatrix4x4 m_transform;

    private:
      PrimitiveType m_type;
      const agxCollide::Geometry* m_geometry;
      agx::UInt32 m_modifiedCount;
      float m_alpha;
      agx::Vec3 m_color;
      bool m_enable;
      agx::Int32 m_renderMode;
  };

  /// Subclass that implements a TextProxy
  class AGXPHYSICS_EXPORT TextProxy : public RenderProxy
  {
    public:

      TextProxy() : RenderProxy( PRIMITIVE_TEXT ) {}
      TextProxy( const agx::String& text, const agx::Vec3& pos );

      void set( const agx::String& text, const agx::Vec3& pos ) {
        m_text = text;
        m_transform.setTranslate( pos[0], pos[1], pos[2] );
        callOnChange( SHAPE );
      }
      agx::String getText() const {
        return m_text;
      }
      agx::Vec3 getPosition() const {
        return m_transform.getTranslate();
      }

    protected:
      virtual ~TextProxy() {}
      agx::String m_text;
  };

  /// Subclass that implements a PlaneProxy
  class AGXPHYSICS_EXPORT PlaneProxy : public RenderProxy
  {
    public:

      PlaneProxy() : RenderProxy( PRIMITIVE_PLANE ) {}
      PlaneProxy( const agx::Vec3& normal, agx::Real distance );

      void set( const agx::Vec3& normal, agx::Real distance ) {
        m_normal = normal;
        m_distance = distance;
        callOnChange( SHAPE );
      }

      /// \return the shape pointer cast to a Plane
      const agxCollide::Plane *getCastShape() const;

      agx::Vec3 getNormal() const {
        return m_normal;
      }
      agx::Real getDistance() const {
        return m_distance;
      }

    protected:

      void  synchronizeParameters();

      virtual ~PlaneProxy() {}
      agx::Real m_distance;
      agx::Vec3 m_normal;
  };

  /// Subclass that implements a SphereProxy
  class AGXPHYSICS_EXPORT SphereProxy : public RenderProxy
  {
    public:

      SphereProxy() : RenderProxy( PRIMITIVE_SPHERE ), m_radius( 0 ) {}
      SphereProxy( float radius );

      void set( float radius ) {
        m_radius = radius;
        callOnChange( SHAPE );
      }

      float getRadius() const {
        return m_radius;
      }

      /// \return the shape pointer cast to a Sphere
      const agxCollide::Sphere *getCastShape() const;

    protected:

      void synchronizeParameters();

      virtual ~SphereProxy() {}
      float m_radius;
  };

  /// Subclass that implements a LineProxy
  class AGXPHYSICS_EXPORT LineProxy : public RenderProxy
  {
    public:

      LineProxy() : RenderProxy( PRIMITIVE_LINE ) {}
      LineProxy( const agx::Vec3& p1, const agx::Vec3& p2  );

      void set( const agx::Vec3& p1, const agx::Vec3& p2  ) {
        m_p1 = p1;
        m_p2 = p2;
        callOnChange( SHAPE );
      }
      agx::Vec3 getP1() const {
        return m_p1;
      }
      agx::Vec3 getP2() const {
        return m_p2;
      }

      /// \return the shape pointer cast to a Line
      const agxCollide::Line *getCastShape() const;

    protected:

      void synchronizeParameters();

      virtual ~LineProxy() {}
      agx::Vec3 m_p1;
      agx::Vec3 m_p2;
  };

  /// Subclass that implements a CylinderProxy
  class AGXPHYSICS_EXPORT CylinderProxy : public RenderProxy
  {
    public:

      CylinderProxy() : RenderProxy( PRIMITIVE_CYLINDER ) {}
      CylinderProxy( float radius, float height );
      CylinderProxy( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      void set( float radius, float height ) {
        m_radius = radius;
        m_height = height;
        callOnChange( SHAPE );
      }
      void set( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      /// \return radius of this proxy
      float getRadius() const {
        return m_radius;
      }

      /// \return height of this proxy
      float getHeight() const {
        return m_height;
      }

      /// \return the shape pointer cast to a Cylinder
      const agxCollide::Cylinder *getCastShape() const;

    protected:

      void synchronizeParameters();

      virtual ~CylinderProxy() {}
      float m_height;
      float m_radius;
  };


  /// Subclass that implements a CylinderProxy
  class AGXPHYSICS_EXPORT HollowCylinderProxy : public RenderProxy
  {
    public:

      HollowCylinderProxy() : RenderProxy( PRIMITIVE_HOLLOW_CYLINDER ) {}
      HollowCylinderProxy( float radius, float height, float thickness );
      HollowCylinderProxy( const agx::Vec3& begin, const agx::Vec3& end, float radius, float thickness );

      void set( float radius, float height, float thickness ) {
        m_radius = radius;
        m_height = height;
        m_thickness = thickness;
        callOnChange( SHAPE );
      }

      void set( const agx::Vec3& begin, const agx::Vec3& end, float radius, float thickness );

      /// \return radius of this proxy
      float getRadius() const {
        return m_radius;
      }

      /// \return height of this proxy
      float getHeight() const {
        return m_height;
      }

      /// \return thickness for this proxy
      float getThickness() const {
        return m_thickness;
      }

      /// \return the shape pointer cast to a Cylinder
      const agxCollide::HollowCylinder *getCastShape() const;

    protected:
      void synchronizeParameters();

      virtual ~HollowCylinderProxy() {}
      float m_height;
      float m_radius;
      float m_thickness;
  };



  /// Subclass that implements a ConeProxy
  class AGXPHYSICS_EXPORT ConeProxy : public RenderProxy
  {
    public:

      ConeProxy() : RenderProxy( PRIMITIVE_CONE ) {}
      ConeProxy( float radius, float height );
      ConeProxy( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      void set( float radius, float height ) {
        m_radius = radius;
        m_height = height;
        callOnChange( SHAPE );
      }

      void set( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      /// \return radius of this proxy
      float getRadius() const {
        return m_radius;
      }

      /// \return height of this proxy
      float getHeight() const {
        return m_height;
      }


      /// \return the shape pointer cast to a Cylinder
      const agxCollide::Cone *getCastShape() const;
    protected:

      void synchronizeParameters();

      virtual ~ConeProxy() {}
      float m_radius;
      float m_height;
  };

  /// Subclass that implements a TruncatedConeProxy
  class AGXPHYSICS_EXPORT TruncatedConeProxy : public RenderProxy
  {
    public:

      TruncatedConeProxy() : RenderProxy( PRIMITIVE_TRUNCATED_CONE ) {}
      TruncatedConeProxy( float topRadius, float bottomRadius, float height );
      TruncatedConeProxy( const agx::Vec3& begin, const agx::Vec3& end, float topRadius, float bottomRadius );

      void set( float topRadius, float bottomRadius, float height ) {
        m_topRadius    = topRadius;
        m_bottomRadius = bottomRadius;
        m_height = height;
        callOnChange( SHAPE );
      }

      void set( const agx::Vec3& top, const agx::Vec3& bottom, float topRadius, float bottomRadius );

      /// \return top radius of this proxy
      float getTopRadius() const {
        return m_topRadius;
      }

      /// \return bottom radius for this proxy
      float getBottomRadius() const {
        return m_bottomRadius;
      }

      /// \return height of this proxy
      float getHeight() const {
        return m_height;
      }


      /// \return the shape pointer cast to a Cylinder
      const agxCollide::Cone *getCastShape() const;
    protected:

      void synchronizeParameters();

      virtual ~TruncatedConeProxy() {}
      float m_topRadius;
      float m_bottomRadius;
      float m_height;
  };

  /// Subclass that implements a HollowTruncatedConeProxy
  class AGXPHYSICS_EXPORT HollowTruncatedConeProxy : public RenderProxy
  {
    public:

      HollowTruncatedConeProxy() : RenderProxy( PRIMITIVE_HOLLOW_TRUNCATED_CONE ) {}
      HollowTruncatedConeProxy( float topRadius, float bottomRadius, float height, float thickness );
      HollowTruncatedConeProxy( const agx::Vec3& begin, const agx::Vec3& end, float topRadius, float bottomRadius, float thickness );

      void set( float topRadius, float bottomRadius, float height, float thickness ) {
        m_topRadius    = topRadius;
        m_bottomRadius = bottomRadius;
        m_height = height;
        m_thickness = thickness;
        callOnChange( SHAPE );
      }

      void set( const agx::Vec3& top, const agx::Vec3& bottom, float topRadius, float bottomRadius, float thickness );

      /// \return top radius of this proxy
      float getTopRadius() const {
        return m_topRadius;
      }

      /// \return bottom radius for this proxy
      float getBottomRadius() const {
        return m_bottomRadius;
      }

      /// \return height of this proxy
      float getHeight() const {
        return m_height;
      }

      /// \return thicknes for this proxy
      float getThickness() const {
        return m_thickness;
      }


      /// \return the shape pointer cast to a Cylinder
      const agxCollide::HollowCone *getCastShape() const;
    protected:

      void synchronizeParameters();

      virtual ~HollowTruncatedConeProxy() {}
      float m_topRadius;
      float m_bottomRadius;
      float m_height;
      float m_thickness;
  };


  /// Subclass that implements a CapsuleProxy
  class AGXPHYSICS_EXPORT CapsuleProxy : public RenderProxy
  {
    public:

      CapsuleProxy() : RenderProxy( PRIMITIVE_LINE ) {}
      CapsuleProxy( float radius, float height );
      CapsuleProxy( const agx::Vec3& begin, const agx::Vec3& end, float radius );

      void set( float radius, float height ) {
        m_radius = radius;
        m_height = height;
        callOnChange( SHAPE );
      }

      void set( const agx::Vec3& begin, const agx::Vec3& end, float radius );


      /// \return radius of this proxy
      float getRadius() const {
        return m_radius;
      }

      /// \return height of this proxy
      float getHeight() const {
        return m_height;
      }

      /// \return the shape pointer cast to a Capsule
      const agxCollide::Capsule *getCastShape() const;

    protected:

      void synchronizeParameters();

      virtual ~CapsuleProxy() {}
      float m_height;
      float m_radius;
  };


  /// Subclass that implements a WireShapeProxy
  class AGXPHYSICS_EXPORT WireShapeProxy : public RenderProxy
  {
  public:

    WireShapeProxy() : RenderProxy( PRIMITIVE_LINE ) {}
    WireShapeProxy( float radius, float height,
      const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1 );
    WireShapeProxy( const agx::Vec3& begin, const agx::Vec3& end,
      const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1, float radius );

    void set( float radius, float height) {
      m_radius = radius;
      m_height = height;
      callOnChange( SHAPE );
    }

    void set( float radius, float height,
      const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1)
    {
      m_radius = radius;
      m_height = height;
      m_previousEndPoint0 = previousEndPoint0;
      m_previousEndPoint1 = previousEndPoint1;
      callOnChange( SHAPE );
    }

    void set( const agx::Vec3& begin, const agx::Vec3& end,
      const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1, float radius );


    /// \return radius of this proxy
    float getRadius() const {
      return m_radius;
    }

    /// \return height of this proxy
    float getHeight() const {
      return m_height;
    }

    /// \return previous end point of this proxy
    agx::Vec3 getPreviousEndPoint0() const
    {
      return m_previousEndPoint0;
    }


    /// \return previous end point of this proxy
    agx::Vec3 getPreviousEndPoint1() const
    {
      return m_previousEndPoint1;
    }


    /// \return the shape pointer cast to a WireShape
    const agxCollide::WireShape *getCastShape() const;

  protected:
    void synchronizeParameters();

    virtual ~WireShapeProxy() {}

    agx::Vec3 m_previousEndPoint0;
    agx::Vec3 m_previousEndPoint1;
    float m_height;
    float m_radius;
  };


  /// Subclass that implements a BoxProxy
  class AGXPHYSICS_EXPORT BoxProxy : public RenderProxy
  {
    public:

      BoxProxy() : RenderProxy( PRIMITIVE_BOX ) {}
      BoxProxy( const agx::Vec3& halfExtents );

      /// Set the size of the box, will trigger onChange(SHAPE)
      void set( const agx::Vec3& halfExtents ) {
        m_halfExtents = halfExtents;
        callOnChange( SHAPE );
      }

      agx::Vec3 getHalfExtents() const {
        return m_halfExtents;
      }

      /// \return the shape pointer cast to a Box
      const agxCollide::Box *getCastShape() const;

    protected:

      void synchronizeParameters();


      virtual ~BoxProxy() {}

      agx::Vec3 m_halfExtents;
  };

  /// Subclass that implements a HeightfieldProxy
  class AGXPHYSICS_EXPORT HeightFieldProxy : public RenderProxy
  {
    public:

      HeightFieldProxy() : RenderProxy( PRIMITIVE_HEIGHT_FIELD ) {}

      HeightFieldProxy( agxCollide::HeightField *hf );

      /**
      Set specified heights of a Heightfield
      \param modifiedIndices - The indices which should receive a new height value
      \param heights - Array containing heights for the indices
      */
      void setHeights(const agx::Vec2iVector& modifiedIndices, const agx::RealVector& heights);

      /// \return the shape pointer cast to a Heightfield
      const agxCollide::HeightField *getCastShape() const;

    protected:

      /**
      Override this method to support changing a selection of indices in the heightfield.
      updateShape on the other hand will rebuild the heightfield completely.
      */
      virtual void set( const agx::Vec2iVector& modifiedIndices, const agx::RealVector& heights )=0;

      /// Destructor
      virtual ~HeightFieldProxy() {}

      HeightFieldProxy& operator=( const HeightFieldProxy& ) {
        return *this;
      }
  };

  /// Subclass that implements a TrimeshProxy
  class AGXPHYSICS_EXPORT TrimeshProxy : public RenderProxy
  {
    public:

      TrimeshProxy() : RenderProxy( PRIMITIVE_TRIMESH ) {}

      TrimeshProxy( agxCollide::Trimesh *mesh );

      void set( agxCollide::Trimesh *mesh );

      const agxCollide::Trimesh *getCastShape() const;

    protected:
      virtual ~TrimeshProxy() {}
      TrimeshProxy& operator=( const TrimeshProxy& ) {
        return *this;
      }
  };

#if 0
  class AGXPHYSICS_EXPORT SphereSpriteBatchRenderProxy : public RenderProxy
  {
    public:
      SphereSpriteBatchRenderProxy() : RenderProxy( PRIMITIVE_SPHERE_BATCH )
      {
        this->setColor(agx::Vec3(agx::Real(0.9)));
      }

      // SphereSpriteBatchRenderProxy( agxData::Buffer *positions, agxData::Buffer *radii, agxData::Value *bound )
      //   : RenderProxy( PRIMITIVE_SPHERE_BATCH ), m_positions(positions), m_radii(radii), m_bound(bound) {}

      #if 0
      // void set( agxData::Array<agx::Vec3> positions, agxData::Array<agx::Real> radii, agx::Bound3 bound )
      void set( agxData::Buffer *positions, agxData::Buffer *radii, agxData::Value *bound )
      {
        m_positions = positions;
        m_radii = radii;
        m_bound = bound;
        callOnChange( SHAPE );
      }
      #endif

    protected:
      virtual ~SphereSpriteBatchRenderProxy() {}

    protected:
      // agxData::Array< agx::Vec3 > m_positions;
      // agxData::Array< agx::Real > m_radii;
      // agx::Bound3 m_bound;
      // agxData::BufferRef m_positions;
      // agxData::BufferRef m_radii;
      // agxData::ValueRef m_bound;
  };

  typedef agx::ref_ptr< SphereSpriteBatchRenderProxy > SphereSpriteBatchRenderProxyRef;


  class AGXPHYSICS_EXPORT SphereMeshBatchRenderProxy : public RenderProxy
  {
    public:
      SphereMeshBatchRenderProxy( const agxData::Array< agx::Vec3 >* buffer )
        : RenderProxy( PRIMITIVE_SPHERE ), m_buffer( buffer ) {}

      void set( const agxData::Array< agx::Vec3 >* buffer, float /*radius*/ )
      {
        m_buffer = buffer;
        callOnChange( SHAPE );
      }

    protected:
      virtual ~SphereMeshBatchRenderProxy() {}

    protected:
      const agxData::Array< agx::Vec3 >* m_buffer;
  };

  typedef agx::ref_ptr< SphereMeshBatchRenderProxy > SphereMeshBatchRenderProxyRef;
#endif

  /**
  Class holding contacts that makes it possible to batch render these objects.
  */
  class AGXPHYSICS_EXPORT ContactsProxy : public RenderProxy
  {
    public:
      ContactsProxy( const agxCollide::GeometryContactPtrVector& contacts );

      void set( const agxCollide::GeometryContactPtrVector& contacts, float scale )
      {
        m_contacts = &contacts;
        m_scale = scale;
        callOnChange( SHAPE );
      }

      const agxCollide::GeometryContactPtrVector& getContacts() const { return *m_contacts; }

    protected:
      ContactsProxy() : RenderProxy( PRIMITIVE_CONTACTS ), m_contacts(nullptr), m_scale(1) {}

      virtual ~ContactsProxy() {}
      const agxCollide::GeometryContactPtrVector* m_contacts;
      float m_scale;
  };

  /**
  Class holding rigid bodies that makes it possible to batch render these objects.
  */
  class AGXPHYSICS_EXPORT RigidBodyBatchRenderProxy : public RenderProxy
  {
    public:
      typedef agx::Vector< const agx::RigidBody* > Container;

    public:
      RigidBodyBatchRenderProxy()
        : RenderProxy( PRIMITIVE_SPHERE ), m_bodies(), m_scale( 1.f ) {}

      void set( const agx::RigidBodyPtrSetVector* enabledBodies, float scale )
      {
        m_bodies.clear();
        if ( enabledBodies != nullptr ) {
          m_bodies.reserve( enabledBodies->size() );
          for ( agx::UInt i = 0; i < enabledBodies->size(); ++i ) {
            const agx::RigidBody* rb = enabledBodies->at( i );
            if ( RenderProxy::shouldRender( rb ) )
              m_bodies.push_back( rb );
          }
        }
        m_scale = scale;
        callOnChange( SHAPE );
      }

    protected:
      virtual ~RigidBodyBatchRenderProxy() {}

    protected:
      agxRender::RigidBodyBatchRenderProxy::Container m_bodies;
      float                                           m_scale;
  };

  class AGXPHYSICS_EXPORT WireRenderProxy : public RenderProxy
  {
    public:
      struct SegmentDef
      {
        SegmentDef( const agx::Vec3& sp, const agx::Vec3& ep, float r )
          : startPoint( sp ), endPoint( ep ), radius( r ) {}

        const SegmentDef* operator->() const { return this; }

        agx::AffineMatrix4x4 getTransform() const;
        agx::Vec3f getScale() const;

        agx::Vec3 startPoint;
        agx::Vec3 endPoint;
        float radius;
      };

      struct SphereDef
      {
        SphereDef( const agx::Vec3& p )
          : point( p ) {}

        const SphereDef* operator->() const { return this; }
        agx::AffineMatrix4x4 getTransform() const;

        agx::Vec3 point;
      };

      typedef agx::VectorPOD< SegmentDef > SegmentDefContainer;
      typedef agx::VectorPOD< SphereDef >  SphereDefContainer;

    public:
      WireRenderProxy( float radius, const agx::Vec3& color );

      /**
      Called when the wire is added to the simulation.
      */
      virtual void onAddNotification() {}

      /**
      Called when the wire is removed from the simulation.
      */
      virtual void onRemoveNotification() {}

      /**
      Prepare buffers. Will clear previous data.
      */
      void prepare( agx::UInt numSegmentEstimate, agx::UInt numEdgesEstimate, agx::UInt numNodesEstimate );

      /**
      Commit the current state to the renderer.
      */
      void commit();

      /**
      Add segment with global radius and color.
      */
      void addSegment( const agx::Vec3& startPoint, const agx::Vec3& endPoint, const agx::Vec3& prevStartPoint, const agx::Vec3& prevEndPoint );

      /**
      Slow, non-batch-rendered, segment.
      */
      void addSegment( const agx::Vec3& startPoint, const agx::Vec3& endPoint, float radius, const agx::Vec3& color, agxRender::RenderManager* manager ) const;

      /**
      Edges are always red and same radius as others.
      */
      void addEdge( const agx::Vec3& startPoint, const agx::Vec3& endPoint );

      /**
      Node with default color (lumped node).
      */
      void addNode( const agx::Vec3& point );

      /**
      Node with different color and/or size.
      */
      void addNode( const agx::Vec3& point, float radius, const agx::Vec3& color, agxRender::RenderManager* manager ) const;

    protected:
      virtual ~WireRenderProxy() {}

    protected:
      float               m_radius;
      SegmentDefContainer m_segments;
      SegmentDefContainer m_edges;
      SphereDefContainer  m_nodes;
  };

  typedef agx::ref_ptr< WireRenderProxy > WireRenderProxyRef;

  // -------- Implementation ----------------

  inline PrimitiveType RenderProxy::getType() const { return m_type; }

  inline void RenderProxy::setTransform( const agx::AffineMatrix4x4& transform ) {
    m_transform = transform;
    callOnChange( TRANSFORM );
  }
  inline const agx::AffineMatrix4x4& RenderProxy::getTransform() const {
    return m_transform;
  }

  inline void RenderProxy::setColor( const agx::Vec3& color ) {
    if (!agx::equivalent(m_color, color)) {
      m_color = color;
      callOnChange( COLOR );
    }
  }

  inline agx::Vec3 RenderProxy::getColor() const {
    return m_color;
  }

  inline void RenderProxy::setAlpha( float alpha ) {

    if (!agx::equivalent(m_alpha, alpha)) {
      m_alpha = alpha;
      callOnChange( ALPHA );
    }
  }

  inline float RenderProxy::getAlpha() const {
    return m_alpha;
  }

  inline void RenderProxy::setModifiedCount( agx::UInt32 c ) {
    m_modifiedCount = c;
  }

  inline agx::UInt32 RenderProxy::getModifiedCount() const {
    return m_modifiedCount;
  }

  inline void RenderProxy::setShape( agxCollide::Shape* shape, const agxCollide::Geometry *geometry ) {
    m_shape = shape;
    m_geometry = geometry;
    setModifiedCount(agx::UInt32(-1));
    //callOnChange( SHAPE );
  }

  inline const agxCollide::Shape* RenderProxy::getShape() const {
    return m_shape;
  }

  inline const agxCollide::Geometry* RenderProxy::getGeometry() const {
    return m_geometry;
  }

  inline void RenderProxy::setEnable( bool flag ) {
    if (m_enable != flag)
    {
      m_enable = flag;
      callOnChange( ENABLE );
    }
  }

  inline bool RenderProxy::getEnable() const {
    return m_enable;
  }

  inline void RenderProxy::setRenderMode( RenderProxy::RenderMode mode ) {
    if ( mode != m_renderMode ) {
      m_renderMode = ( RenderMode )mode;
      callOnChange( RENDERMODE );
    }
  }

  inline RenderProxy::RenderMode RenderProxy::getRenderMode( ) const {
    return ( RenderMode )m_renderMode;
  }

  inline void HeightFieldProxy::setHeights(const agx::Vec2iVector& modifiedIndices, const agx::RealVector& heights)
  {
    agxAssert(modifiedIndices.size() == heights.size() );

    set( modifiedIndices, heights );

    // Sync modified count
    setModifiedCount( getShape()->getModifiedCount() );
  }

  inline agx::AffineMatrix4x4 WireRenderProxy::SegmentDef::getTransform() const
  {
    return agx::AffineMatrix4x4::rotate( agx::Vec3::Z_AXIS(), (endPoint - startPoint).normal() ) * agx::AffineMatrix4x4::translate( (startPoint + endPoint) * agx::Real( 0.5 ) );
  }

  inline agx::Vec3f WireRenderProxy::SegmentDef::getScale() const
  {
    return agx::Vec3f( radius, radius, (float)startPoint.distance( endPoint ) );
  }

  inline agx::AffineMatrix4x4 WireRenderProxy::SphereDef::getTransform() const
  {
    return agx::AffineMatrix4x4::translate( point );
  }
}

#endif
