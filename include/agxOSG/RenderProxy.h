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

#ifndef AGXOSG_RENDERPROXY_H
#define AGXOSG_RENDERPROXY_H

#include <agxOSG/export.h>
#include <agxRender/RenderProxy.h>
#include <agxRender/RenderManager.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/ref_ptr>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/AutoTransform>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxOSG/GraphRenderer.h>
#include <agxOSG/RenderText.h>
#include <agxOSG/utils.h>
#include <agxOSG/PointSpriteDrawable.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4355 ) // warning C4355: 'this' : used in base member initializer list
#endif

namespace osg
{
  class Material;
  class ShapeDrawable;
}

namespace osgText
{
  class Text;
}

namespace agxOSG
{

  class SphereProxy;
  class LineProxy;

  AGX_DECLARE_POINTER_TYPES(RenderProxyFactory);

  /**
  Implementation of the abstract class from the agxRender namespace, this class is responsible
  for creating RenderProxy of various types and render them as efficiently as possible in OpenSceneGraph.

  There are several node-roots:

  - getTextNode() returns the root of the Text, which is projected into 2D and
  rendered without lights.

  - getRootNode() - Parent of both: getSolidRootNode() and getWireFrameRootNode()
  */
  class AGXOSG_EXPORT RenderProxyFactory : public agxRender::RenderProxyFactory
  {
    public:
      /**
      Assign new default detail ratio for the rendered geometries.
      */
      static void  setDefaultDetailRatio( float detailRatio );

      /**
      \return the currently used detail ratio
      */
      static float getDefaultDetailRatio();

    public:
      RenderProxyFactory();

      /**
      \return the implementation of a GraphRenderer for osg.
      */
      agxRender::Graph::GraphRenderer *getGraphRenderer() override;


      /**
      \param radius - The radius of the new sphere
      \return a pointer to a new SphereProxy with specified radius
      */
      agxRender::SphereProxy* createSphere( float radius ) override;

      /**
      \param text - The text
      \param pos - Position of the text. Currently only x,y is used.
      \return a pointer to a new TextProxy
      */
      agxRender::TextProxy* createText( const agx::String& text, const agx::Vec3& pos ) override;

      /**
      \param p1, p2 - Start, end points in WORLD coordinate system
      \return a pointer to a new LineProxy
      */
      agxRender::LineProxy* createLine( const agx::Vec3& p1, const agx::Vec3& p2 ) override;

      /**
      \param radius - The radius of a cylinder
      \param height - The height of a cylinder
      \return a pointer to a new CylinderProxy
      */
      agxRender::CylinderProxy* createCylinder( float radius, float height ) override;

      /**
      \param radius - The radius for the a cylinder
      \param height - The height for the cylinder
      \param thickness - The thickness for the cylinder
      */
      agxRender::HollowCylinderProxy* createHollowCylinder( float radius, float height, float thickness ) override;

      /**
      \param radius - The radius of a capsule
      \param height - The height of a capsule
      \return a pointer to a new CapsuleProxy
      */
      agxRender::CapsuleProxy* createCapsule( float radius, float height ) override;


      /**
      \param radius - The radius of a capsule
      \param height - The height of a capsule
      \param previousEndPoint0 - The previous lower end point of the WireShape.
      \param previousEndPoint1 - The previous upper end point of the WireShape.
      \return a pointer to a new WireShapeProxy
      */
      agxRender::WireShapeProxy* createWireShape( float radius, float height,
        const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1) override;


      /**
      \param halfExtents - The size of the box
      \return a pointer to a new BoxProxy with specified size
      */
      agxRender::BoxProxy* createBox( const agx::Vec3& halfExtents ) override;

      /**
      Create and return a RenderProxy for a Heightfield.
      \param hf - The heightfield for which a HeightfieldProxy will be created.
      */
      agxRender::HeightFieldProxy* createHeightfield( agxCollide::HeightField* hf ) override;

      /**
      Create and return a RenderProxy for a Trimesh.
      \param mesh - The mesh shape for which a TrimeshProxy will be created.
      */
      agxRender::TrimeshProxy* createTrimesh( agxCollide::Trimesh* mesh ) override;

      /**
      \param radius - The radius of the cone
      \param height - The height of the cone
      \return a pointer to a new ConeProxy
      */
      agxRender::ConeProxy *createCone( float radius, float height ) override;

      /**
      \param topRadius - The top radius of the cone
      \param bottomRadius - The bottom radius of the cone
      \param height - The height of the cone
      \return a pointer to a new ConeProxy
      */
      agxRender::TruncatedConeProxy *createTruncatedCone( float topRadius, float bottomRadius, float height ) override;

      /**
      \param topRadius - The top radius of the cone
      \param bottomRadius - The bottom radius of the cone
      \param height - The height of the cone
      \param thickness - The thickness of the cone
      */
      agxRender::HollowTruncatedConeProxy* createHollowTruncatedCone( float topRadius, float bottomRadius, float height, float thickness ) override;

      /**
      \param normal - The normal of a plane
      \param distance - The scalar part of the plane
      \return a pointer to a new PlaneProxy
      */
      agxRender::PlaneProxy *createPlane( const agx::Vec3& normal, agx::Real distance ) override;


      /**
      \param contacts - Vector with all contacts that should be visualized
      \return a pointer to a new ContactsProxy
      */
      agxRender::ContactsProxy* createContacts( const agxCollide::GeometryContactPtrVector& contacts ) override;

      agxRender::RigidBodyBatchRenderProxy* createRigidBodies( const agx::RigidBodyPtrSetVector* enabledBodies ) override;

      agxRender::WireRenderProxy* createWire( float radius, const agx::Vec3& color ) override;


      void setSphereBatchRenderModeSprites(bool flag);
      agxRender::RenderProxy* createSphereBatchRenderer(
        agxData::Buffer* positions, agxData::Buffer* rotations, agxData::Buffer* radii, agxData::Buffer* colors,
        agxData::Buffer* enableRendering, agxData::Value* bound, agx::Component* context) override;

      /**
      The text node is not child of the node returned in getRootNode(), hence this node MUST
      be added separately.
      \return the node under which all the Text element will go
      */
      osg::Group *getTextNode() { return m_textParent.get(); }

      /**

      \return the node under which the solid and wire frame nodes lies.
      */
      osg::Node *getRootNode() { return m_parent; }

      /**
      \return the parent of all solid renderable objects
      */
      osg::Group *getSolidRootNode() { return m_solidParent; }

      /**
      \return the parent of all wireframe renderable objects
      */
      osg::Group *getWireframeRootNode() { return m_wireframeParent; }

      /**
      Set a transformation which will be applied to all transformations for RenderProxy's.
      This can be used to transform all objects to the center of the earth instead of trying to render on the surface of the earth (6000km).
      This transform is by default I.
      If you for example do m.setTranslate(-10,0,0) all RenderProxy's will be rendered translated -10 units in X.
      If a RenderManager is present, RenderManager::update() will be called
      */
      void setGlobalTransform( const agx::AffineMatrix4x4& m );

      /**
      \return the current global transformation
      */
      const agx::AffineMatrix4x4& getGlobalTransform() const { return m_invTransform; }

      /**
      Set the RenderMode for a specified node
      */
      void setRenderMode( agxRender::RenderProxy *proxy, osg::Node *node, agxRender::RenderProxy::RenderMode mode );

    public:
      /// Class for storing OSG specific data for each RenderProxy
      template<typename T>
      struct ShapeData {
        ShapeData() : material(nullptr) {}
        ShapeData(const ShapeData& other)
        {
          shape = other.shape;
          geode = other.geode;
          material = other.material;
          stateSet = other.stateSet;
        }

        osg::ref_ptr<T> shape;
        osg::ref_ptr<osg::Geode> geode;
        osg::Material *material;
        osg::ref_ptr<osg::StateSet> stateSet;
      };


      /**
      Add a child to the root, default getDefaultRenderMode() will be used for determining if
      the node should go into wireframe and or solid parent.
      */
      void addChild( osg::Node * node, agxRender::RenderProxy::RenderMode mode );
      void addChild( osg::Node * node, agxRender::RenderProxy *proxy );

    protected:
      /// Destructor
      virtual ~RenderProxyFactory();

      osg::ref_ptr<osg::Group> m_solidParent;
      osg::ref_ptr<osg::Group> m_wireframeParent;
      osg::ref_ptr<osg::MatrixTransform> m_parent;
      osg::ref_ptr<osg::Group> m_textParent;
      agx::ref_ptr<agxOSG::GraphRenderer> m_graphRenderer;
      agx::AffineMatrix4x4 m_invTransform;
      bool m_sphereBatchRenderModeSprites;
  };



  /// Class for handling methods related to OSG such as setColor, setTransform etc.
  template<typename T>
  class OSGData
  {
    public:
      OSGData( const RenderProxyFactory::ShapeData<T>& data, agxOSG::RenderProxyFactory* factory, agxRender::RenderProxy* proxy )
        : m_data(data), m_factory(factory), m_proxy(proxy)
      {
        m_transformNode = new osg::MatrixTransform;

        m_transformNode->addChild( m_data.geode );

        // Tell osg not to remove this transformation under optimization pass
        m_transformNode->setDataVariance( osg::Object::DYNAMIC );

        m_data.stateSet = m_transformNode->getOrCreateStateSet();

        m_data.material = new osg::Material;
        m_data.stateSet->setAttributeAndModes(m_data.material,osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
        setColorOSG( agx::Vec3( 1, 0, 0 ), 1.f );
      }

      OSGData( agxOSG::RenderProxyFactory* factory, agxRender::RenderProxy* proxy )
        : m_data(), m_factory( factory ), m_proxy( proxy )
      {
        m_transformNode = new osg::MatrixTransform();
        m_transformNode->setDataVariance( osg::Object::DYNAMIC );

        m_data.stateSet = m_transformNode->getOrCreateStateSet();
        m_data.material = new osg::Material;
        m_data.stateSet->setAttributeAndModes( m_data.material, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED );
        setColorOSG( agx::Vec3( 1, 0, 0 ), 1.f );
      }

      /// Transform a node
      void setTransformOSG(const agx::AffineMatrix4x4& transform)
      {
        /// Take the global transform into account
        agx::AffineMatrix4x4 m = transform * m_factory->getGlobalTransform();
        m_transformNode->setMatrix( AGX_MAT4X4_TO_OSG( m ) );
      }

      /// Enable/disable rendering
      void setEnableOSG( bool flag )
      {
        if (!flag) { // Remove node from all its parents
          while(m_transformNode->getNumParents())
          {
            m_transformNode->getParent(0)->removeChild( m_transformNode.get() );
          }

        }
        else //Add it for rendering again.
          m_factory->addChild( m_transformNode.get(), m_proxy );
      }

      /// Set the alpha value
      void setAlphaOSG( float transparency )
      {
        m_data.material->setTransparency(osg::Material::FRONT_AND_BACK, 1-transparency);
      }

      /// Set the color and alpha value
      void setColorOSG(const agx::Vec3& color, float alpha)
      {
        osg::Vec4 col = m_data.material->getDiffuse(osg::Material::FRONT_AND_BACK);

        if ( agx::equivalent( (float)col[0], (float)color[0] ) &&
             agx::equivalent( (float)col[1], (float)color[1] ) &&
             agx::equivalent( (float)col[2], (float)color[2] ) )
          return;

        osg::Vec4 osgCol( (float)color[0], (float)color[1], (float)color[2], alpha );
        m_data.material->setDiffuse(osg::Material::FRONT_AND_BACK, osgCol);

        for ( unsigned int i = 0; m_data.geode != nullptr && i < m_data.geode->getNumDrawables(); ++i ) {
          osg::ShapeDrawable* drawable = dynamic_cast< osg::ShapeDrawable* >( m_data.geode->getDrawable( i ) );
          if ( drawable )
            drawable->setColor( osgCol );
        }
      }

      /// Remove the node from all its parents
      void onRemoveOSG()
      {
        osg::ref_ptr<osg::MatrixTransform> r = m_transformNode;
        while(m_transformNode->getNumParents())
        {
          m_transformNode->getParent(0)->removeChild( m_transformNode.get() );
        }
      }

      /// Change the render mode
      void setRenderModeOSG( agxRender::RenderProxy::RenderMode mode )
      {
        m_factory->setRenderMode( m_proxy, m_transformNode, mode );
      }

      ///\ return the node for this RenderProxy
      osg::MatrixTransform* getNode() { return m_transformNode.get(); }

    protected:
      osg::ref_ptr<osg::MatrixTransform>  m_transformNode;
      RenderProxyFactory::ShapeData<T>    m_data;
      agxOSG::RenderProxyFactory*         m_factory;
      agx::AffineMatrix4x4                m_invTransform;
      agxRender::RenderProxy*             m_proxy;
  };

  /// Macro for adding some methods for each specialization of RenderProxy
#define ADD_COMMON_METHODS()                                        \
  void onChange( RenderProxy::EventType type ) override {           \
    switch( type ) {                                                \
      case (RenderProxy::ENABLE):                                   \
        this->setEnableOSG(this->getEnable());                      \
        break;                                                      \
      case (RenderProxy::ALPHA):                                    \
        this->setAlphaOSG(this->getAlpha());                        \
        break;                                                      \
      case (RenderProxy::TRANSFORM):                                \
        this->setTransformOSG(this->getTransform());                \
        break;                                                      \
      case (RenderProxy::COLOR):                                    \
        this->setColorOSG(this->getColor(), this->getAlpha());      \
        break;                                                      \
      case (RenderProxy::SHAPE):                                    \
        break;                                                      \
      case (RenderProxy::REMOVE):                                   \
        this->onRemoveOSG();                                        \
        break;                                                      \
      case (RenderProxy::RENDERMODE):                               \
        this->setRenderModeOSG(this->getRenderMode());              \
        break;                                                      \
    }                                                               \
  }                                                                 \
  virtual void updateShape() override

  /// Implementation of osg-based SphereProxy
  class AGXOSG_EXPORT SphereProxy : public agxRender::SphereProxy, public OSGData<osg::Sphere>
  {
    public:
      SphereProxy(float radius, RenderProxyFactory::ShapeData<osg::Sphere> sphere, RenderProxyFactory* factory);

    protected:
      virtual ~SphereProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based TextProxy
  class AGXOSG_EXPORT TextProxy : public agxRender::TextProxy, public OSGData<agxOSG::TextGeometry>
  {
    public:
      TextProxy( const agx::String& text, const agx::Vec3& pos, RenderProxyFactory::ShapeData<agxOSG::TextGeometry> data, RenderProxyFactory* factory );

      void setTransform(const agx::AffineMatrix4x4& transform) override;

    protected:
      virtual ~TextProxy() {}

      ADD_COMMON_METHODS();
  };


  /// Implementation of osg-based BoxProxy
  class AGXOSG_EXPORT BoxProxy : public agxRender::BoxProxy, public OSGData<osg::Box>
  {
    public:
      BoxProxy( const agx::Vec3& halfExtents, RenderProxyFactory::ShapeData<osg::Box> box, RenderProxyFactory* factory );

    protected:
      virtual ~BoxProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based LineProxy
  class AGXOSG_EXPORT LineProxy : public agxRender::LineProxy, public OSGData<osg::Geometry>
  {
    public:
      LineProxy( const agx::Vec3& p1, const agx::Vec3& p2, RenderProxyFactory::ShapeData<osg::Geometry> line, RenderProxyFactory* factory );

    protected:
      virtual ~LineProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based CylinderProxy
  class AGXOSG_EXPORT CylinderProxy : public agxRender::CylinderProxy, public OSGData<osg::Cylinder>
  {
    public:
      CylinderProxy( float radius, float height, RenderProxyFactory::ShapeData<osg::Cylinder> data, RenderProxyFactory* factory );

    protected:
      virtual ~CylinderProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based HollowCylinderProxy
  class AGXOSG_EXPORT HollowCylinderProxy: public agxRender::HollowCylinderProxy, public OSGData<osg::Geometry>
  {
    public:
      HollowCylinderProxy( float radius, float height, float thickness, RenderProxyFactory::ShapeData<osg::Geometry> data, RenderProxyFactory* factory );

    protected:
      virtual ~HollowCylinderProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based ConeProxy
  class AGXOSG_EXPORT ConeProxy : public agxRender::ConeProxy, public OSGData<osg::Cone>
  {
    public:
      ConeProxy( float radius, float height, RenderProxyFactory::ShapeData<osg::Cone> data, RenderProxyFactory* factory );

    protected:
      virtual ~ConeProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based TruncatedConeProxy
  class AGXOSG_EXPORT TruncatedConeProxy : public agxRender::TruncatedConeProxy, public OSGData<osg::Geometry>
  {
    public:
      TruncatedConeProxy( float topRadius, float bottomRadius, float height, RenderProxyFactory::ShapeData<osg::Geometry> data, RenderProxyFactory* factory );

    protected:
      virtual ~TruncatedConeProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based HollowTruncatedConeProxy
  class AGXOSG_EXPORT HollowTruncatedConeProxy : public agxRender::HollowTruncatedConeProxy, public OSGData<osg::Geometry>
  {
    public:
      HollowTruncatedConeProxy( float topRadius, float bottomRadius, float height, float thickness,
                                RenderProxyFactory::ShapeData<osg::Geometry> data, RenderProxyFactory* factory );

    protected:
      virtual ~HollowTruncatedConeProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based PlaneProxy
  class AGXOSG_EXPORT PlaneProxy : public agxRender::PlaneProxy, public OSGData<osg::Geometry>
  {
    public:
      PlaneProxy( const agx::Vec3& normal, agx::Real distance, RenderProxyFactory::ShapeData<osg::Geometry> data, RenderProxyFactory* factory );

    protected:
      virtual ~PlaneProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based CapsuleProxy
  class AGXOSG_EXPORT CapsuleProxy : public agxRender::CapsuleProxy, public OSGData<osg::Capsule>
  {
    public:
      CapsuleProxy( float radius, float height, RenderProxyFactory::ShapeData<osg::Capsule> data, RenderProxyFactory* factory );

    protected:
      virtual ~CapsuleProxy() {}

      ADD_COMMON_METHODS();
  };


  /// Implementation of osg-based WireShapeProxy
  class AGXOSG_EXPORT WireShapeProxy : public agxRender::WireShapeProxy, public OSGData<osg::CompositeShape>
  {
    public:
      WireShapeProxy( float radius, float height, const agx::Vec3& previousEndPoint0, const agx::Vec3& previousEndPoint1,
        RenderProxyFactory::ShapeData<osg::CompositeShape> data, RenderProxyFactory* factory );

    protected:
      virtual ~WireShapeProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based HeightfieldProxy
  class AGXOSG_EXPORT HeightFieldProxy : public agxRender::HeightFieldProxy, public OSGData<osg::Geometry>
  {
    public:
      HeightFieldProxy( agxCollide::HeightField* hf, RenderProxyFactory::ShapeData<osg::Geometry> hfData, RenderProxyFactory* factory );

      void set( const agx::Vec2iVector& modifiedIndices, const agx::RealVector& heights ) override;

    protected:
      virtual ~HeightFieldProxy() {}

      ADD_COMMON_METHODS();
  };

  /// Implementation of osg-based TrimeshProxy
  class AGXOSG_EXPORT TrimeshProxy : public agxRender::TrimeshProxy, public OSGData<osg::TriangleMesh>
  {
    public:
      TrimeshProxy( agxCollide::Trimesh* mesh, RenderProxyFactory::ShapeData<osg::TriangleMesh> data, RenderProxyFactory* factory );

    protected:
      virtual ~TrimeshProxy() {}

      ADD_COMMON_METHODS();
  };

  /**
  Base class for batch rendering of any shape or shape composite. This
  class currently handles osg::MatrixTransform and osg::AutoTransform.
  */
  template< typename TransformType >
  class ShapeBatchRenderer : public OSGData< osg::Group >
  {
    public:
      /**
      Clear the root transform node from all child transforms.
      */
      void clear();

      /**
      Assign transform scale. Default: 1.
      \param scale - new scale
      */
      void setScale( float scale );

      /**
      \return current scale
      */
      float getScale() const;

    protected:
      /**
      Construct given single shape.
      \param shape - only instantiated shape
      \param factory - the render proxy factory
      \param proxy - render proxy (often the subclass)
      */
      ShapeBatchRenderer( osg::Shape* shape, agxOSG::RenderProxyFactory* factory, agxRender::RenderProxy* proxy );

      /**
      Construct given geode.
      \param geode - geode with one or many shapes
      \param factory - the render proxy factory
      \param proxy - render proxy (often the subclass)
      */
      ShapeBatchRenderer( osg::Geode* geode, agxOSG::RenderProxyFactory* factory, agxRender::RenderProxy* proxy );

      /**
      Destructor.
      */
      virtual ~ShapeBatchRenderer() { clear(); }

      /**
      Prepare the root transform with the current amount of child nodes.
      \param numObjects - number of child transforms
      */
      void prepare( size_t numObjects );

      /**
      Including scale transform.
      */
      void prepareWithScale( size_t numObjects );

      /**
      Assign transform (given agx type) to a supported osg transform.
      \param obj - transform object to be assigned
      \param transform - transform to a assign
    \param scale - scale factor
      */
      void setTransform( TransformType* obj, const agx::AffineMatrix4x4& transform, const agx::Vec3f& scale );

    protected:
      osg::ref_ptr< osg::Geode > m_geode;
      float m_scale;
  };

  /**
  Class that handles batch rendering of objects in forward iterator compliant
  container. Default ContainerType::value_type& val -> val->getTransform() is
  used (i.e., typically reference counted or observers to objects with method
  getTransform()). Template specialization has to be implemented to handle
  other types... at all.

  E.g., for ContainerType = agxCollide::GeometryContactPtrVector:
  void findTransform( const agxCollide::GeometryContactPtrVector::value_type& )
  has to be implemented.
  */
  template< typename ContainerType, typename TransformType >
  class ContainerShapeBatchRenderer : public ShapeBatchRenderer< TransformType >
  {
    public:
      /**
      Construct given shape composite and compatible container.
      \param geode - geode with one or many shapes
      \param container - compatible container
      \param factory - render proxy factory
      \param proxy - render proxy
      */
      ContainerShapeBatchRenderer( osg::Geode* geode, const ContainerType* container, agxOSG::RenderProxyFactory* factory, agxRender::RenderProxy* proxy )
        : ShapeBatchRenderer< TransformType >( geode, factory, proxy ), m_container( container ) {}

      /**
      Destructor.
      */
      virtual ~ContainerShapeBatchRenderer() { this->clear(); }

      /**
      Reset container pointer and removes all child transforms.
      */
      void clear();

      /**
      Updates transforms.
      */
      void update( bool addScaleTransform = false );

      /**
      Assign compatible container.
      */
      void setContainer( const ContainerType* container );

    protected:
      /**
      Find transform (agx::AffineMatrix4x4) given container value type.
      \param val - container value
      \return current transform
      */
      agx::AffineMatrix4x4 findTransform( const typename ContainerType::value_type& val ) const;

      /**
      Find scale (agx::Vec3) given container value type. Default assigned scale
      returned.
      */
      agx::Vec3f findScale( const typename ContainerType::value_type& val ) const;

    protected:
      const ContainerType* m_container;
  };

  #if 0
  class SphereSpriteBatchRenderer : public ContainerShapeBatchRenderer< agxData::Array< agx::Vec3 >, osg::MatrixTransform >, public agxRender::SphereSpriteBatchRenderProxy
  {
    public:
      SphereSpriteBatchRenderer( osg::Geode* geode, const agxData::Array< agx::Vec3 >* container, agxOSG::RenderProxyFactory* factory )
        : ContainerShapeBatchRenderer< agxData::Array< agx::Vec3 >, osg::MatrixTransform >( geode, container, factory, this ),
          agxRender::SphereSpriteBatchRenderProxy( container ) {}

    protected:
      virtual ~SphereSpriteBatchRenderer() {}

      ADD_COMMON_METHODS()
      {
        this->m_container = m_buffer;
        this->update();
      }

      virtual void reset() override
      {
        // I don't think we can or could do anything during reset.
      }
  };
  #endif

  /**
  Implementation of batch rendering of rigid body center of mass position.
  */
  template< typename ContainerType, typename TransformType >
  class RigidBodyBatchRenderProxy : public ContainerShapeBatchRenderer< ContainerType, TransformType >, public agxRender::RigidBodyBatchRenderProxy
  {
    public:
      RigidBodyBatchRenderProxy( osg::Geode* geode, const ContainerType* container, agxOSG::RenderProxyFactory* factory )
        : ContainerShapeBatchRenderer< ContainerType, TransformType >( geode, container, factory, this ) {}

    protected:
      virtual ~RigidBodyBatchRenderProxy() {}

      // This is virtual void updateShape().
      ADD_COMMON_METHODS()
      {
        this->m_container = &m_bodies;
        ShapeBatchRenderer<TransformType>::setScale( agxRender::RigidBodyBatchRenderProxy::m_scale );

        // Will update root transform given a global transform.
        agxRender::RenderProxy::setTransform( agx::AffineMatrix4x4() );

        this->update();
      }

      virtual void reset() override
      {
        // I don't think we can or could do anything during reset.
      }
  };

  /**
  Implementation of batch rendering of contact points.
  */
  class ContactProxy : public ContainerShapeBatchRenderer< agxCollide::GeometryContactPtrVector, osg::AutoTransform >, public agxRender::ContactsProxy
  {
    public:
      ContactProxy( osg::Geode* geode, const agxCollide::GeometryContactPtrVector* contacts, agxOSG::RenderProxyFactory* factory );

    protected:
      virtual ~ContactProxy() {}

      ADD_COMMON_METHODS();

      virtual void reset() override
      {
        // I don't think we can or could do anything during reset.
      }
  };

  class WireRenderProxy : public agxRender::WireRenderProxy
  {
    public:
      WireRenderProxy( float radius, const agx::Vec3& color, osg::Geode* segmentGeode, osg::Geode* edgeGeode, osg::Geode* nodeGeode, agxOSG::RenderProxyFactory* factory );

      osg::MatrixTransform* getSegmentTransform();
      osg::MatrixTransform* getEdgeTransform();
      osg::MatrixTransform* getSphereTransform();

    protected:
      virtual ~WireRenderProxy();

      virtual void onAddNotification() override;
      virtual void onRemoveNotification() override;

      void setTransformOSG( const agx::AffineMatrix4x4& transform );
      void setEnableOSG( bool flag );
      void setAlphaOSG( float transparency );
      void setColorOSG(const agx::Vec3& color, float alpha);
      void onRemoveOSG();
      void setRenderModeOSG( agxRender::RenderProxy::RenderMode mode );

      ADD_COMMON_METHODS();

      virtual void reset() override
      {
        // I don't think we can or could do anything during reset.
      }

    protected:
      ContainerShapeBatchRenderer< agxRender::WireRenderProxy::SegmentDefContainer, osg::MatrixTransform > m_segmentBatch;
      ContainerShapeBatchRenderer< agxRender::WireRenderProxy::SegmentDefContainer, osg::MatrixTransform > m_edgeBatch;
      ContainerShapeBatchRenderer< agxRender::WireRenderProxy::SphereDefContainer,  osg::MatrixTransform > m_sphereBatch;

    private:
      // Not defined to call this method on this object.
      osg::MatrixTransform* getNode() { return nullptr; }
  };

  template< typename TransformType >
  void ShapeBatchRenderer<TransformType>::clear()
  {
    if ( m_transformNode )
      m_transformNode->removeChild( 0, m_transformNode->getNumChildren() );
  }

  template< typename TransformType >
  void ShapeBatchRenderer<TransformType>::setScale( float scale )
  {
    m_scale = scale;
  }

  template< typename TransformType >
  float ShapeBatchRenderer<TransformType>::getScale() const
  {
    return m_scale;
  }

  template< typename ContainerType, typename TransformType >
  void ContainerShapeBatchRenderer< ContainerType, TransformType >::clear()
  {
    ShapeBatchRenderer< TransformType >::clear();
    m_container = nullptr;
  }

  template< typename ContainerType, typename TransformType >
  void ContainerShapeBatchRenderer< ContainerType, TransformType >::update( bool addScaleTransform /* = false */ )
  {
    if ( this->m_container == nullptr )
      return;

    if ( addScaleTransform )
      this->prepareWithScale( this->m_container->size() );
    else
      this->prepare( this->m_container->size() );

    unsigned int counter = 0;
    for ( typename ContainerType::const_iterator i = m_container->begin(); i != m_container->end(); ++counter, ++i ) {
      agxAssert( counter < this->m_transformNode->getNumChildren() );
      TransformType* transform = static_cast< TransformType* >( this->m_transformNode->getChild( counter ) );
      this->setTransform( transform, this->findTransform( *i ), this->findScale( *i ) );
    }
  }

  template< typename ContainerType, typename TransformType >
  void ContainerShapeBatchRenderer< ContainerType, TransformType >::setContainer( const ContainerType* container )
  {
    m_container = container;
  }

  template< typename ContainerType, typename TransformType >
  agx::AffineMatrix4x4 ContainerShapeBatchRenderer< ContainerType, TransformType >::findTransform( const typename ContainerType::value_type& val ) const
  {
    return val->getTransform();
  }

  template<>
  inline agx::AffineMatrix4x4 ContainerShapeBatchRenderer< agxCollide::GeometryContactPtrVector, osg::AutoTransform >::findTransform( const agxCollide::GeometryContactPtrVector::value_type& /*contact*/ ) const
  {
    return agx::AffineMatrix4x4();
  }

  template<>
  inline agx::AffineMatrix4x4 ContainerShapeBatchRenderer< agx::RigidBodyPtrSetVector, osg::AutoTransform >::findTransform( const agx::RigidBodyPtrSetVector::value_type& rbRefPtr ) const
  {
    return rbRefPtr->getCmTransform();
  }

  template<>
  inline agx::AffineMatrix4x4 ContainerShapeBatchRenderer< agxRender::RigidBodyBatchRenderProxy::Container, osg::AutoTransform >::findTransform( const agxRender::RigidBodyBatchRenderProxy::Container::value_type& rbPtr ) const
  {
    return rbPtr->getCmTransform();
  }

  template<>
  inline agx::AffineMatrix4x4 ContainerShapeBatchRenderer< agxData::Array< agx::Vec3 >, osg::MatrixTransform >::findTransform( const agxData::Array< agx::Vec3 >::value_type& p ) const
  {
    return agx::AffineMatrix4x4::translate( p );
  }

  template< typename ContainerType, typename TransformType >
  agx::Vec3f ContainerShapeBatchRenderer< ContainerType, TransformType >::findScale( const typename ContainerType::value_type& /*val*/ ) const
  {
    return agx::Vec3f( ShapeBatchRenderer<TransformType>::m_scale, ShapeBatchRenderer<TransformType>::m_scale, ShapeBatchRenderer<TransformType>::m_scale );
  }

  template<>
  inline agx::Vec3f ContainerShapeBatchRenderer< agxRender::WireRenderProxy::SegmentDefContainer, osg::MatrixTransform >::findScale( const agxRender::WireRenderProxy::SegmentDefContainer::value_type& val ) const
  {
    return val->getScale();
  }
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
