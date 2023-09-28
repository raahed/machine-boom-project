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

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

/*
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2007 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com


This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#pragma once


#include <agx/agx_vector_types.h>

#include <agxCollide/BoundingAABB.h>
#include <agx/agxPhysics_export.h>

#include <external/GIMPACT/MiddlePhaseBoundingAABB.h>
#include <external/GIMPACT/PrimitiveIndexPair.h>
#include <agxCollide/LineClipping.h>

#include <agxStream/Serializable.h>

#include <agxData/LocalVector.h>

#include <iterator>

namespace agxCollide
{
  class MeshData;
  class CollisionMeshData;
  class HeightField;

  /// A small class for storing an AABB and its index.
  class AabbTreeNodeData
  {
  public:
    MiddlePhaseBoundingAABB bound;
    int primitiveData;
  };

  typedef agx::Vector<AabbTreeNodeData> AabbTreeNodeDataVector;


  AGXPHYSICS_EXPORT bool triangleBoxOverlap( agx::Vec3 boxcenter, agx::Vec3 boxhalfsize, const agx::Vec3 triverts[3]);


  AGX_DECLARE_POINTER_TYPES(AabbTree);
  /**
  A static, binary, axis aligned bounding box tree.
  Deformable in a way that nodebounds can be updated, but the tree structure
  remains constant.
  */
  class AGXPHYSICS_EXPORT AabbTree : public agx::Referenced, public agxStream::Serializable
  {
    friend class HeightField;
    friend class Trimesh;

  public:
    /**
    Create AabbTree from triangle mesh data.
    The MeshData should not contain more vertices or triangles than what can be
    represented by an int.
    */
    AabbTree( const agxCollide::MeshData* meshData );

    /// Create AabbTree from an array of bounding boxes and indices to data
    AabbTree( const AabbTreeNodeDataVector& primitiveBoxes );

    AabbTree();

    /// \retval number of nodes (both leaf and internal) in tree
    size_t getNodeCount() const;

    /// \retval The bounding volume of the whole tree
    BoundingAABB getBoundingAABB() const;

    /// \retval The bounding volume of the indicated box.
    const MiddlePhaseBoundingAABB& getNodeBound( int nodeIndex ) const;

    /// \retval Indexed node is leaf.
    bool isLeafNode( int nodeIndex ) const;

    /// \retval Indexed node's data.
    int getNodePrimitiveData( int nodeIndex ) const;

    /// \retval Left child node.
    int getLeftNode( int nodeIndex ) const;

    /// \retval Right child node.
    int getRightNode( int nodeIndex ) const;

    /// Set node bounding box for the indexed node. Changes will propagate up the tree.
    void updateNodeBound( int nodeIndex, const MiddlePhaseBoundingAABB& newBound );

    /**
    Finds collision between this AabbTree and another one.
    \param other  The other AabbTree.
    \param collisionPairs  A vector of index pairs to the primitives whose
                           bounding boxes overlap.
    \param transformOwn  This tree's transformation matrix in a common frame.
    \param transformOther  The other tree's transformation matrix in a common frame.
    */
    void findCollision(
      const AabbTree* other,
      PrimitiveIndexPairVector& collisionPairs,
      const agx::AffineMatrix4x4& transformOwn,
      const agx::AffineMatrix4x4& transformOther,
      const agxCollide::CollisionMeshData* mesh0 = nullptr,
      const agxCollide::CollisionMeshData* mesh1 = nullptr
      ) const;

    /**
    Finds collision between this AabbTree and a BoundingAABB.
    \param boundingAABB  The BoundingAABB.
    \param collisionData  A vector of index pairs to the primitives whose
                           bounding boxes overlap.
    \param transformOwn  This tree's transformation matrix in a common frame.
    \param transformOther  The bounding AABB's transformation matrix in a common frame.
    */
    void findCollision(
      const BoundingAABB* boundingAABB,
      agx::UInt32Vector& collisionData,
      const agx::AffineMatrix4x4& transformOwn,
      const agx::AffineMatrix4x4& transformOther ) const;


    template <typename UInt32Container>
    void findCollision(
      const BoundingAABB* boundingAABB,
      std::back_insert_iterator<UInt32Container> collisionData,
      const agx::AffineMatrix4x4& transformOwn,
      const agx::AffineMatrix4x4& transformOther) const;

    /**
    Finds collision between this AabbTree and a Sphere.
    \param radius  The sphere's radius.
    \param collisionData  A vector of indices to the primitives whose
                           bounding boxes overlap.
    \param sphereTranslate  The sphere's translation in AabbTree's coordinate system.
    */
    void findCollisionWithSphere(
      agx::Real radius,
      agxData::LocalVector<agx::UInt32>& collisionData,
      const agx::Vec3& sphereTranslate ) const;

    /**
    Finds collision between this AabbTree and a Capsule.
    Allows for false positives.
    \param radius  The capsule's radius.
    \param capsulePoint0 One capsule axis end point in AabbTree's coordinates system.
    \param capsulePoint1 The other capsule axis end point in AabbTree's coordinates system.
    \param collisionData  A vector of indices to the primitives whose
                           bounding boxes overlap.
    */
    void findCollisionWithCapsule(
      agx::Real radius,
      const agx::Vec3& capsulePoint0,
      const agx::Vec3& capsulePoint1,
      agx::UInt32Vector& collisionData ) const;


    /**
    Finds collision between this AabbTree and a Cylinder.
    Allows for false positives.
    \param radius  The cylinder's radius.
    \param height  The cylinder's height.
    \param cylinderPoint0 One cylinder axis end point in AabbTree's coordinates system.
    \param cylinderPoint1 The other cylinder axis end point in AabbTree's coordinates system.
    \param collisionData  A vector of indices to the primitives whose
                           bounding boxes overlap.
    */
    void findCollisionWithCylinder(
      agx::Real radius,
      agx::Real height,
      const agx::Vec3& cylinderPoint0,
      const agx::Vec3& cylinderPoint1,
      agx::UInt32Vector& collisionData ) const;

    /**
    Finds collision between this AabbTree and a line segment.
    \param segmentStart The line segment's start point given in the AabbTree's frame.
    \param segmentEnd The line segment's end point given in the AabbTree's frame.
    \param collisionData  A vector of indices to the primitives whose
                           bounding boxes overlap.
    */
    void findCollisionWithLineSegment(
      const agx::Vec3& segmentStart,
      const agx::Vec3& segmentEnd,
      agx::UInt32Vector& collisionData ) const;

    /**
    Finds collision between this AabbTree and a line segment.
    \param segmentStart The line segment's start point given in the AabbTree's frame.
    \param segmentEnd The line segment's end point given in the AabbTree's frame.
    \param collisionData  A vector of indices to the primitives whose
                           bounding boxes overlap.
    */
    void findCollisionWithLineSegment(
      const agx::Vec3& segmentStart,
      const agx::Vec3& segmentEnd,
      agxData::LocalVector<agx::UInt32>& collisionData) const;

    /// Creates a deep copy of all data. Caller takes ownership of pointer.
    AabbTree* deepCopy() const;

    AGXSTREAM_DECLARE_SERIALIZABLE( agxCollide::AabbTree );


  protected:
    virtual ~AabbTree();

    void buildTree(const AabbTreeNodeDataVector& primitiveBoxes);

    /// Set node bounding box for the indexed node. Changes will not propagate.
    void setNodeBound( int nodeIndex, const MiddlePhaseBoundingAABB& newBound );

    int splitAndSort(
      const AabbTreeNodeDataVector& primitiveBoxes,
      agx::UInt32Vector& indices,
      int startIndex,
      int endIndex );

    void buildSubTree(
      const AabbTreeNodeDataVector& primitiveBoxes,
      agx::UInt32Vector& indices,
      int startIndex,
      int endIndex,
      int& nextNodeIndex,
      int parentIndex );

    void setNodePrimitiveData( int nodeIndex, int data );

    void setNodeRightChildIndex( int nodeIndex, int rightChildIndex );

    void setNodeParent( int nodeIndex, int parentIndex );

  protected:
    agx::Vector<MiddlePhaseBoundingAABB> m_bounds;
    agx::Int32Vector m_rightChildOrData; // Data if >= 0, -right child if < 0.
    agx::Int32Vector m_parents;
  };

  typedef agx::ref_ptr<const AabbTree> ConstAabbTreeRef;

  void AGXPHYSICS_EXPORT createAabbTreeNodeDataVectorFromMesh(
    const agxCollide::MeshData* meshData,
    AabbTreeNodeDataVector& primitiveBoxes );



  // implementations




  template <typename UInt32Container>
  void AabbTree::findCollision(
    const BoundingAABB* boundingAABB,
    std::back_insert_iterator<UInt32Container> collisionData,
    const agx::AffineMatrix4x4& transformTree,
    const agx::AffineMatrix4x4& transformBox) const
  {
    if (getNodeCount() == 0)
      return;

    const BoxBoxTransformCache transformCacheBoxToTree( transformBox, transformTree );

    const MiddlePhaseBoundingAABB box = *boundingAABB;

    const agx::Vec3 transformedHalfExtents0 = agx::Vec3(box.getHalfExtents()) * transformCacheBoxToTree.absoluteRotate0To1;
    const agx::Vec3 transformedCenter0 = box.getCenter() * transformCacheBoxToTree.rotate0To1 + transformCacheBoxToTree.translate0To1;

    agxData::LocalVector<agx::UInt32> nodes;
    nodes.reserve(2 * agx::log2(this->m_bounds.size())); // Reserve memory for the depth of the tree.
    nodes.push_back(0);
    while (nodes.size()) {
      agx::UInt32 node = nodes.back();
      nodes.pop_back();
      while (box.collideBoxBox(this->getNodeBound(node), transformCacheBoxToTree,
                               transformedHalfExtents0, transformedCenter0))
      {
        if (this->isLeafNode( node )) {
          // collision result
          *collisionData = this->getNodePrimitiveData( node );
          ++collisionData;
          break;
        }
        else {
          // collide left recursive
          nodes.push_back(this->getRightNode( node ) );
          node = this->getLeftNode( node );
        }
      }
    }
  }

  AGX_FORCE_INLINE size_t AabbTree::getNodeCount() const
  {
    return m_bounds.size();
  }


  AGX_FORCE_INLINE AabbTree* AabbTree::deepCopy() const
  {
    AabbTree* otherTree = new AabbTree();
    otherTree->m_bounds = this->m_bounds;
    otherTree->m_rightChildOrData = this->m_rightChildOrData;
    otherTree->m_parents = this->m_parents;
    return otherTree;
  }


  AGX_FORCE_INLINE BoundingAABB AabbTree::getBoundingAABB() const
  {
    if (m_bounds.size() > 0) {
      const MiddlePhaseBoundingAABB& totalbox = getNodeBound(0);
      agx::Vec3 min, max;
      totalbox.getMinMax( min, max );
      return agxCollide::BoundingAABB( min, max );
    }
    else
      return agxCollide::BoundingAABB();
  }


  AGX_FORCE_INLINE const MiddlePhaseBoundingAABB& AabbTree::getNodeBound( int nodeIndex ) const
  {
    return m_bounds[nodeIndex];
  }


  AGX_FORCE_INLINE bool AabbTree::isLeafNode( int nodeIndex ) const
  {
    return m_rightChildOrData[nodeIndex] >= 0;
  }


  AGX_FORCE_INLINE int AabbTree::getNodePrimitiveData( int nodeIndex ) const
  {
    return m_rightChildOrData[nodeIndex];
  }


  AGX_FORCE_INLINE int AabbTree::getLeftNode( int nodeIndex ) const
  {
    return nodeIndex + 1;
  }


  AGX_FORCE_INLINE int AabbTree::getRightNode( int nodeIndex ) const
  {
    return -m_rightChildOrData[nodeIndex];
  }


  AGX_FORCE_INLINE void AabbTree::setNodeBound( int nodeIndex, const MiddlePhaseBoundingAABB& newBound )
  {
    m_bounds[nodeIndex] = newBound;
  }


  AGX_FORCE_INLINE void AabbTree::setNodePrimitiveData( int nodeIndex, int data )
  {
    m_rightChildOrData[nodeIndex] = data;
  }


  AGX_FORCE_INLINE void AabbTree::setNodeRightChildIndex( int nodeIndex, int rightChildIndex )
  {
    m_rightChildOrData[nodeIndex] = -rightChildIndex;
  }


  AGX_FORCE_INLINE void AabbTree::setNodeParent( int nodeIndex, int parentIndex )
  {
    m_parents[nodeIndex] = parentIndex;
  }



}

