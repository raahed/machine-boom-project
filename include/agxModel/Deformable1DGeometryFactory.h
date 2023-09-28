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

#ifndef AGXMODEL_DEFORMABLE1DGEOMETRYFACTORY_H
#define AGXMODEL_DEFORMABLE1DGEOMETRYFACTORY_H

#include <agxModel/Deformable1DNode.h>

#include <agxCollide/Sphere.h>

#include <agxStream/OutputArchive.h>
#include <agxStream/InputArchive.h>

namespace agxModel
{
  AGX_DECLARE_POINTER_TYPES( Deformable1DGeometryFactory );
  AGX_DECLARE_POINTER_TYPES( Deformable1DBoxGeometryFactory );
  AGX_DECLARE_POINTER_TYPES( Deformable1DCapsuleGeometryFactory );
  AGX_DECLARE_POINTER_TYPES( Deformable1DCylinderGeometryFactory );
  AGX_DECLARE_POINTER_TYPES( Deformable1DSphereGeometryFactory );

  /**
  Geometry factory is an object that spawns geometries for a Deformable1D object.
  It's also possible to ask this object for dimensions given a certain node/position
  along the object.
  */
  class AGXMODEL_EXPORT Deformable1DGeometryFactory : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Create geometry for given node. The length of the node is given in the node.
      \param node - node to create geometry for
      \return new geometry for the given node
      */
      virtual agxCollide::GeometryRef createGeometry(const agxModel::Deformable1DNode* node) = 0;

      /**
      \return estimated or known width at the given node
      */
      virtual agx::Real getWidth(const agxModel::Deformable1DNode* node) const = 0;

      /**
      \return estimated or known height at the given node
      */
      virtual agx::Real getHeight(const agxModel::Deformable1DNode* node) const = 0;

      /**
      Characteristic overlapping distance, i.e., the distance the shape extends
      the shape definition. E.g., a capsule is 'radius' longer on each side.
      */
      virtual agx::Real getOverlappingDistance(const agxModel::Deformable1DNode* /*node*/) const;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agxModel::Deformable1DGeometryFactory );

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DGeometryFactory();
  };

  /**
  Geometry factor for boxes of given width and height.
  */
  class AGXMODEL_EXPORT Deformable1DBoxGeometryFactory : public agxModel::Deformable1DGeometryFactory
  {
    public:
      /**
      Construct given width and height.
      \param width - width of the Deformable1D
      \param height - height of the Deformable1D
      */
      Deformable1DBoxGeometryFactory( agx::Real width, agx::Real height );

    public:
      /**
      \return geometry with a box shape
      */
      virtual agxCollide::GeometryRef createGeometry(const agxModel::Deformable1DNode* node) override;

      /**
      \return the width of the boxes
      */
      virtual agx::Real getWidth(const agxModel::Deformable1DNode* node) const override;

      /**
      \return the height of the boxes
      */
      virtual agx::Real getHeight(const agxModel::Deformable1DNode* node) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Deformable1DBoxGeometryFactory );

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1DBoxGeometryFactory();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DBoxGeometryFactory();

    protected:
      agx::Real m_width;
      agx::Real m_height;
  };

  /**
  General geometry factory for shapes where width = height = radius.
  */
  template< typename T >
  class Deformable1DRadiusGeometryFactory : public agxModel::Deformable1DGeometryFactory
  {
    public:
      /**
      Constraint given radius = width = height.
      \param radius - radius of the Deformable1D object
      */
      Deformable1DRadiusGeometryFactory( agx::Real radius );

      /**
      \return the radius
      */
      agx::Real getRadius() const;

    public:
      /**
      Create a geometry with given shape for \p node.
      \param node - node the geometry should belong to
      \return geometry for the node
      */
      virtual agxCollide::GeometryRef createGeometry(const agxModel::Deformable1DNode* node) override;

      /**
      \return the width (radius)
      */
      virtual agx::Real getWidth(const agxModel::Deformable1DNode* node) const override;

      /**
      \return the height (radius)
      */
      virtual agx::Real getHeight(const agxModel::Deformable1DNode* node) const override;

      void store( agxStream::OutputArchive& out ) const override;
      void restore( agxStream::InputArchive& in ) override;

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1DRadiusGeometryFactory();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DRadiusGeometryFactory();

    protected:
      agx::Real m_radius;
  };

#ifdef SWIG
  %template(Deformable1DRadiusGeometryFactoryCapsule)  agxModel::Deformable1DRadiusGeometryFactory<agxCollide::Capsule>;
  %template(Deformable1DRadiusGeometryFactoryCylinder) agxModel::Deformable1DRadiusGeometryFactory<agxCollide::Cylinder>;
  %template(Deformable1DRadiusGeometryFactorySphere)   agxModel::Deformable1DRadiusGeometryFactory<agxCollide::Sphere>;
#endif

  /**
  Geometry factory that creates Capsules with given radius.
  */
  class AGXMODEL_EXPORT Deformable1DCapsuleGeometryFactory : public agxModel::Deformable1DRadiusGeometryFactory< agxCollide::Capsule >
  {
    public:
      /**
      Construct given radius.
      \param radius - capsule radius
      */
      Deformable1DCapsuleGeometryFactory( agx::Real radius );

    public:
      /**
      \return the overlapping distance given node
      */
      virtual agx::Real getOverlappingDistance(const agxModel::Deformable1DNode* node) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Deformable1DCapsuleGeometryFactory );

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1DCapsuleGeometryFactory();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DCapsuleGeometryFactory();
  };

  /**
  Geometry factory that creates Cylinders with given radius.
  */
  class AGXMODEL_EXPORT Deformable1DCylinderGeometryFactory : public agxModel::Deformable1DRadiusGeometryFactory< agxCollide::Cylinder >
  {
    public:
      /**
      Construct given radius.
      \param radius - cylinder radius
      */
      Deformable1DCylinderGeometryFactory( agx::Real radius );

      AGXSTREAM_DECLARE_SERIALIZABLE( agxModel::Deformable1DCylinderGeometryFactory );

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1DCylinderGeometryFactory();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DCylinderGeometryFactory();
  };

  /**
  Geometry factory that creates Spheres with given radius.
  */
  class AGXMODEL_EXPORT Deformable1DSphereGeometryFactory : public agxModel::Deformable1DRadiusGeometryFactory< agxCollide::Sphere >
  {
    public:
      /**
      Construct given radius.
      \param radius - sphere radius
      */
      Deformable1DSphereGeometryFactory( agx::Real radius );

    public:
      /**
      \return the overlapping distance given node
      */
      virtual agx::Real getOverlappingDistance(const agxModel::Deformable1DNode* node) const override;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::Deformable1DSphereGeometryFactory);

    protected:
      /**
      Default constructor for serialization.
      */
      Deformable1DSphereGeometryFactory();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DSphereGeometryFactory();
  };

  template< typename T >
  Deformable1DRadiusGeometryFactory< T >::Deformable1DRadiusGeometryFactory()
  {
  }

  template< typename T >
  Deformable1DRadiusGeometryFactory< T >::Deformable1DRadiusGeometryFactory( agx::Real radius )
    : m_radius( radius )
  {
  }

  template< typename T >
  Deformable1DRadiusGeometryFactory< T >::~Deformable1DRadiusGeometryFactory()
  {
  }

  template< typename T >
  agx::Real Deformable1DRadiusGeometryFactory< T >::getRadius() const
  {
    return m_radius;
  }

  template< typename T >
  agx::Real Deformable1DRadiusGeometryFactory< T >::getWidth(const agxModel::Deformable1DNode* /*node*/) const
  {
    return agx::Real( 2 ) * getRadius();
  }

  template< typename T >
  agx::Real Deformable1DRadiusGeometryFactory< T >::getHeight(const agxModel::Deformable1DNode* /*node*/) const
  {
    return agx::Real( 2 ) * getRadius();
  }

  template<>
  inline agxCollide::GeometryRef Deformable1DRadiusGeometryFactory< agxCollide::Sphere >::createGeometry(const agxModel::Deformable1DNode* node)
  {
    if ( node == nullptr )
      return agxCollide::GeometryRef( nullptr );

    // Maybe input?
    const agx::Real offset = getRadius();

    const agx::UInt numSpheres = std::max( agx::UInt( node->getLength() / offset + agx::Real( 0.5 ) ), agx::UInt( 1 ) );
    const agx::Real dl = node->getLength() / agx::Real( numSpheres );
    agxCollide::GeometryRef geometry = new agxCollide::Geometry();
    agx::Vec3 pos( 0, 0, 0 );
    agx::Vec3 dir = agx::Vec3::Z_AXIS();
    for ( agx::UInt i = 0; i < numSpheres; ++i ) {
      geometry->add( new agxCollide::Sphere( getRadius() ), agx::AffineMatrix4x4::translate( pos ) );
      pos += dl * dir;
    }

    return geometry;
  }

  template< typename T >
  agxCollide::GeometryRef Deformable1DRadiusGeometryFactory< T >::createGeometry(const agxModel::Deformable1DNode* node)
  {
    if ( node == nullptr )
      return agxCollide::GeometryRef( nullptr );

    agxCollide::GeometryRef geometry = new agxCollide::Geometry( new T( getRadius(), node->getLength() ) );
    geometry->setLocalTransform( agx::AffineMatrix4x4::rotate( agx::Vec3::Y_AXIS(), agx::Vec3::Z_AXIS() ) *
                                 agx::AffineMatrix4x4::translate( agx::Vec3::Z_AXIS() * agx::Real( 0.5 ) * node->getLength() ) );

    return geometry;
  }

  template< typename T >
  void Deformable1DRadiusGeometryFactory< T >::store( agxStream::OutputArchive& out ) const
  {
    out << agxStream::out( "radius", m_radius );
  }

  template< typename T >
  void Deformable1DRadiusGeometryFactory< T >::restore( agxStream::InputArchive& in )
  {
    in >> agxStream::in( "radius", m_radius );
  }
}

#endif // AGXMODEL_DEFORMABLE1DGEOMETRYFACTORY_H
