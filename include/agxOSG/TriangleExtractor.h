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

#ifndef AGXOSG_TRIANGLEEXTRACTOR_H
#define AGXOSG_TRIANGLEEXTRACTOR_H


#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/NodeVisitor>
#include <osg/Matrix>
#include <osg/Referenced>
#include <osg/TriangleFunctor>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Vector.h>
#include <agx/agx_vector_types.h>


#ifdef _MSC_VER
// Disable warning about assignment operator could not be generated due to reference members.
# pragma warning( push )
# pragma warning( disable: 4512 )
#endif

DOXYGEN_START_INTERNAL_BLOCK()


namespace osg {
  class Geometry;
  class Geode;
  class MatrixTransform;
}
DOXYGEN_END_INTERNAL_BLOCK()

namespace agxOSG
{

  class TriangleExtractor;

  ///This class is a base class that implements a functor, with a method triangle that is executed for each triangle visited

  /*!
  This class is used as a functor class, with its method triangle executed per triangle.
  Each triangles vertices's is supplied by the TriangleExtractor class. THe vertices will be in
  world coordinates. That is all the transformations will be accumulated during the traversal.
  */
  class AGXOSG_EXPORT TriangleExtractOperatorBase
  {

    public:
      friend class TriangleExtractor;

      /// Constructor
      TriangleExtractOperatorBase() : m_num_triangles( 0 ) { }

      virtual ~TriangleExtractOperatorBase() {}

#if   OSG_VERSION_GREATER_OR_EQUAL(3,6,4)
      void operator () (const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3);
#else
      void operator () (const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3, bool);
#endif

    protected:


      /// Stores the accumulated matrix
      void setMatrix( const osg::Matrix& matrix ) {
        m_matrix = matrix;
      }
      void setMatrix() {
        m_matrix.identity();
      }

      /// Returns the number of triangles processed
      unsigned int numTriangles() {
        return m_num_triangles;
      }

      /// Resets the number of triangles back to zero
      void reset() {
        m_num_triangles = 0;
      }

      /*!
      This method is a pure virtual method that has to be inherited.
      This method will be called for each triangle. The vertices are given in
      world coordinates using the accumulated matrix.
      */
      virtual void triangle( const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3 ) = 0;

    private:
      unsigned int m_num_triangles;
      osg::Matrix m_matrix;
  };

  //
  inline void TriangleExtractOperatorBase::operator () ( const osg::Vec3& v1,
      const osg::Vec3& v2,
      const osg::Vec3& v3
#if   OSG_VERSION_GREATER_OR_EQUAL(3,6,4)
#else
      , bool
#endif
    )
  {
    if ( v1 == v2 || v2 == v3 || v1 == v3 ) return;

    // Calculate the vertices world coordinates.
    osg::Vec3 v1n, v2n , v3n;
    v1n = m_matrix.preMult( v1 );
    v2n = m_matrix.preMult( v2 );
    v3n = m_matrix.preMult( v3 );

    // Execute the inherited method to process the vertices
    triangle( v1n, v2n, v3n );
    m_num_triangles++;
  }


  /// TriangleExtractOperator is the method inherit to create a operator that will be executed per triangle.
  typedef osg::TriangleFunctor<TriangleExtractOperatorBase> TriangleExtractOperator;


  /// This class traverses a subgraph, accumulates transformations and execute a functor (TriangleExtractOperator per triangle.
  /*!

  The functor which is a descendant from TriangleExtractOperator has a method triangle, which will be executed per triangle.
  The triangle will have the coordinates in world coordinates, that is the matrix which is accumulated during the
  path through the scene graph to the triangle.
  */
  class AGXOSG_EXPORT TriangleExtractor : public osg::NodeVisitor
  {
    public:

      /*!
      Constructor.
      \param op - Is the functor operator which triangle method will be executed for each triangle found during traversal.
      */
      TriangleExtractor( TriangleExtractOperator &op );

      /// Destructor
      virtual ~TriangleExtractor() {}

      /*! Begins the triangle extraction on the subgraph node.
      \param node - The subgraph from which the triangles will be extracted.
      */
      void extract( osg::Node& node );

      /*! Begins the triangle extraction on the subgraph node which is a geode
      \param node - The subgraph from which the triangles will be extracted.
      */
      void extract( osg::Geode& node ) {
        apply( node );
      }

      /*! Begins the triangle extraction on the subgraph node.
      \param node - The subgraph from which the triangles will be extracted.
      */
      void extract( osg::MatrixTransform& node ) {
        apply( node );
      }
    protected:

      virtual void apply( osg::Node& );
      virtual void apply( osg::Geode& node );
      virtual void apply( osg::Billboard& node );

      virtual void apply( osg::Group& node );
      virtual void apply( osg::MatrixTransform& node );
      virtual void apply( osg::Switch& node );
      virtual void apply( osg::LOD& node );

      void apply( osg::Geometry& geom );
      using osg::NodeVisitor::apply;

      void pushMatrix( const osg::Matrix& matrix );
      void popMatrix();

      TriangleExtractOperator & m_tri_op;

    private:
      typedef agx::Vector<osg::ref_ptr<osg::RefMatrix> > MatrixStack;

      MatrixStack m_matrix_stack;
  };

  /**
  Store Vertices and indices as a vector in this class instance
  */
  class MeshExtractor : public agxOSG::TriangleExtractOperator
  {
  public:

    MeshExtractor() {}
    virtual void triangle( const osg::Vec3& v0, const osg::Vec3& v1, const osg::Vec3& v2 );

    agx::Vec3Vector* getVertices() {return &m_vertices;}
    agx::UInt32Vector* getIndices() {return &m_indices;}

    virtual ~MeshExtractor( ) {}

  private:
    agx::Vec3Vector m_vertices;
    agx::UInt32Vector m_indices;
  };

  /**
    Store vertices and indices in vectors by reference
  */
  class MeshExtractorReference : public agxOSG::TriangleExtractOperator
  {
  public:

    MeshExtractorReference(agx::Vec3Vector& vertices, agx::UInt32Vector& indices) : m_vertices(vertices), m_indices(indices)
    {}

    virtual void triangle(const osg::Vec3& v0, const osg::Vec3& v1, const osg::Vec3& v2);

    agx::Vec3Vector* getVertices() { return &m_vertices; }
    agx::UInt32Vector* getIndices() { return &m_indices; }

    virtual ~MeshExtractorReference() {}

  private:
    agx::Vec3Vector& m_vertices;
    agx::UInt32Vector& m_indices;
  };

  inline void MeshExtractor::triangle( const osg::Vec3& v0, const osg::Vec3& v1, const osg::Vec3& v2 )
  {
    m_vertices.push_back( OSG_VEC3_TO_AGX(v0) );
    m_vertices.push_back( OSG_VEC3_TO_AGX(v1) );
    m_vertices.push_back( OSG_VEC3_TO_AGX(v2) );
    m_indices.push_back((agx::UInt32) m_indices.size());
    m_indices.push_back((agx::UInt32) m_indices.size());
    m_indices.push_back((agx::UInt32)m_indices.size());
  }

  inline void MeshExtractorReference::triangle(const osg::Vec3& v0, const osg::Vec3& v1, const osg::Vec3& v2)
  {
    m_vertices.push_back(OSG_VEC3_TO_AGX(v0));
    m_vertices.push_back(OSG_VEC3_TO_AGX(v1));
    m_vertices.push_back(OSG_VEC3_TO_AGX(v2));
    m_indices.push_back((agx::UInt32)m_indices.size());
    m_indices.push_back((agx::UInt32)m_indices.size());
    m_indices.push_back((agx::UInt32)m_indices.size());
  }

} // namespace agxOSG

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
