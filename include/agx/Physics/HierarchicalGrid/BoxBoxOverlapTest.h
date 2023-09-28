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

#ifndef AGX_PHYSICS_HIERARCHICALGRID_BOXBOXOVERLAP_TEST_H
#define AGX_PHYSICS_HIERARCHICALGRID_BOXBOXOVERLAP_TEST_H


#include <agxCollide/Box.h>
#include <agxCollide/BasicPrimitiveTests.h>
#include <agxCollide/LineClipping.h>


/*

This file is pretty much a copy of the box-box collider, but with the actual
contact generation removed, and instead returns a bool telling if any contacts
would have been created by the full box-box collider. I have not been able to
fully convince myself that it produces the correct result in all cases, see
the \todo near the end of this file.

*/


namespace
{

  AGX_FORCE_INLINE bool calculateDepthEdgeProv ( const agx::Vec3& he1, const agx::Vec3& he2, 
    const agx::Matrix3x3& trans2To1Abs, 
    const agx::AffineMatrix4x4& trans2To1,
    const agx::Vec3& midP2In1, 
    const int i, const int iIncr1, const int iIncr2,
    const int j, const int jIncr1, const int jIncr2,
    bool& testEdges, agx::Real& depth )
  {
    const agx::Real length2 = trans2To1Abs(j,iIncr1) * trans2To1Abs(j,iIncr1) + trans2To1Abs(j,iIncr2) * trans2To1Abs(j,iIncr2); // squared length, no square root yet, will do this later
    if (agx::equalsZero( length2, agx::Real(1.0e-8) )) {
      // should not get here, numerical issues? Drop this axis, since it's nearly parallel. 
      testEdges = false;
      return true;
    }
    depth = (he1[iIncr1] * trans2To1Abs(j, iIncr2) +  he1[iIncr2] * trans2To1Abs(j, iIncr1) +
      he2[jIncr1] * trans2To1Abs(jIncr2, i) + he2[jIncr2] * trans2To1Abs(jIncr1, i) -
      std::abs( midP2In1[iIncr2] * trans2To1(j, iIncr1) - midP2In1[iIncr1] * trans2To1(j, iIncr2)) );

    return (depth >= 0); // return false if we found a separating axis
  }



  AGX_FORCE_INLINE void calculateDepthEdge ( const agx::AffineMatrix4x4& trans2To1, 
    const int i, const int iIncr1, const int iIncr2, const int j, 
    const agx::Real depth, int& depthIndex, agx::Real& minDepth) 
  {
    // Now we calculate the true length.
    // Delaying the square root until when it is needed.
    const agx::Real length2 = trans2To1(j, iIncr1) * trans2To1(j, iIncr1) + trans2To1(j, iIncr2) * trans2To1(j, iIncr2);     
    // Avoiding the division by multiplying both sides of the inequality with a positive number.   
    if (depth * depth < minDepth * minDepth * length2) {
      const int dim = 3 * i + j;
      minDepth = depth / std::sqrt(length2);
      depthIndex = 6 + dim;
    } 
  }





  bool boxBoxOverlapTest
  (
    const agx::Vec3& he1, const agx::Vec3& he2,
    const agx::AffineMatrix4x4& trans1Inv, const agx::AffineMatrix4x4& trans2Inv
  )
  {
    const agx::Real epsilon = 2 * (he1[0] + he1[1] + he1[2] + he2[0] + he2[1] + he2[2])  * agx::Real(agx::RealEpsilon); // epsilon for numerical comparisons, adapted to problem's size

    const agx::AffineMatrix4x4 trans2To1 = trans2Inv.inverse() * trans1Inv;
    const agx::Vec3 midP2In1 = trans2Inv.getInvTranslate() * trans1Inv; 

    agx::Matrix3x3 trans2To1Abs(trans2To1 );
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        trans2To1Abs( i, j ) = std::abs( trans2To1Abs( i, j ) );

    // Looking for separating axes.

    //
    // First, test against box1's three coordinate axes.
    //

    agx::Real minDepth = agx::Real(agx::RealMax);  
    int depthIndex = 0;
    agx::StackArray<agx::Real, 6> depthsFace;

    // Test against first box   
    for (unsigned i = 0; i < 3; ++i) {
      const agx::Real depth = he1[i] + he2 *  agx::Vec3( trans2To1Abs( 0, i ), trans2To1Abs( 1, i ), trans2To1Abs( 2, i )) - std::abs( midP2In1[i] );   
      if (depth < 0) // found a separating axis, exit!      
        return false;
      depthsFace.push_back(depth);
    }


    // 
    // Check if we are in a parallel case by looking at the rotation part of transformation matrix from shape2's frame to shape1's (trans2To1).
    // We count the number of almost-1 diagonal elements std::abs values.
    // If we have 1, we have 1 parallelism. This means we are in a 2D-case with additional testing required in the dimension they are parallel in.
    // This means testing in 5 directions, but 6 won't hurt either.
    // If we have 3 (since 2 are mathematically not possible, only due to numerical errors), the boxes have identical orientation (apart from 90 degree rotations).
    // It is enough to test for the first box's three axes.
    //

    int numParallels = 0;
    agx::Real maxRotError2 = 0;
    for (int i = 0; i < 3; ++i) {
      agx::Real length2[2] = {0, 0};    
      for (int j = 0; j < 3; ++j) {
        length2[0] += trans2To1(i, j) * trans2To1(i, j);
        length2[1] += trans2To1(j, i) * trans2To1(j, i);
      }
      for (int j = 0; j < 2; ++j) {
        if (std::abs(length2[j] - 1) > maxRotError2)
          maxRotError2 = std::abs(length2[j] - 1);
      }
    }

    agx::Real rotationError = epsilon;
    if (maxRotError2 > rotationError)
      rotationError = std::sqrt( maxRotError2 );
    for (int i = 0; i < 3; ++i) {
      if (trans2To1Abs( i, i ) > 1.0 - 0.0001 + rotationError) { // almost parallel to coordinate axis
        numParallels++;
      }
    }

    //
    // Then, test the up to 12 other axes for separation, and exit if found. If no separation is found, record the least overlap depth and the corresponding axis.
    // The tests are taken from Christer Ericson, agx::Real Time Collision Detection, 2005, p.101-106.  
    //
    const agx::Vec3 midP1In2 = trans1Inv.getInvTranslate() * trans2Inv;

    if (numParallels < 2) {
      // Test against second box     
      for (unsigned i = 0; i < 3; ++i) {
        const agx::Real depth = he2[i] + he1 * agx::Vec3( trans2To1Abs( i, 0 ), trans2To1Abs( i, 1 ), trans2To1Abs( i, 2 )) - std::abs( midP1In2[i] );         
        if (depth < 0) // found a separating axis, exit!
          return false; 
        depthsFace.push_back(depth);
      }
    }

    for (unsigned int i = 0; i < depthsFace.size(); ++i) {
      if (depthsFace[i] < minDepth) {
        minDepth = depthsFace[i];
        depthIndex = i;
      }
    }
    // int depthIndexFace = depthIndex;  
    // agx::Real minDepthFace = minDepth;


    if (numParallels < 1) {
      // test against 9 cross products
      // The test given in Christer Ericson's book does not give the true length, we have to normalize.
      // Since the matrix has column-and row norm 1 in all columns and rows, we can use the same normalizing factor
      // since the column and row defined by removing element i,j have the same norm.
      bool testEdges = true;        

      agx::Real depths[3][3];
      memset(depths, 0, sizeof(agx::Real)*9 );

      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 0, 1, 2, 0, 1, 2, testEdges, depths[0][0] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 0, 1, 2, 1, 2, 0, testEdges, depths[0][1] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 0, 1, 2, 2, 0, 1, testEdges, depths[0][2] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 1, 2, 0, 0, 1, 2, testEdges, depths[1][0] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 1, 2, 0, 1, 2, 0, testEdges, depths[1][1] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 1, 2, 0, 2, 0, 1, testEdges, depths[1][2] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 2, 0, 1, 0, 1, 2, testEdges, depths[2][0] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 2, 0, 1, 1, 2, 0, testEdges, depths[2][1] )) return false;
      if (!calculateDepthEdgeProv( he1, he2, trans2To1Abs, trans2To1, midP2In1, 2, 0, 1, 2, 0, 1, testEdges, depths[2][2] )) return false;


      // Make normalization step only now, since overlap testing does not need a normalized depth calculation.
      // This means that in case the cross products would yield a separating axis, the effort for exact depth computation
      // (9 square roots and 9 divisions) would have been wasted.
      // Do it now instead since we know now that we are in overlap.
      // Also, since normalization can only increase the length of a vector (created from a vector with length one by setting one component to zero), 
      // do not normalize if the unnormalized length is already larger than the minimum depth.
      if (testEdges) {
        calculateDepthEdge( trans2To1, 0, 1, 2, 0, depths[0][0], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 0, 1, 2, 1, depths[0][1], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 0, 1, 2, 2, depths[0][2], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 1, 2, 0, 0, depths[1][0], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 1, 2, 0, 1, depths[1][1], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 1, 2, 0, 2, depths[1][2], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 2, 0, 1, 0, depths[2][0], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 2, 0, 1, 1, depths[2][1], depthIndex, minDepth );
        calculateDepthEdge( trans2To1, 2, 0, 1, 2, depths[2][2], depthIndex, minDepth );
      }
    }

    /// \todo Not sure if this is safe. Can solveFaceCaseAGX produce zero contact points?
    return true;

    #if 0
    // 
    // Since the test has not exited yet, we must have an overlap.
    // We have information about the depth (but will recalculate it for each created contact point individually), 
    // and know how to calculate the contact normal.
    // For edge-edge contacts, we will create 1 contact point, for face contacts, 1 to 8.
    //

    if (depthIndex > 5) {
      if (createEdgeContact) {
        ShapeCollider::addContactPoint( result, (closestPoints.pointOn1 + closestPoints.pointOn2) * agx::Real(0.5), (trans1.transform3x3( normalIn1 )), minDepth  );      
      }
      else {
        // create face contact
        depthIndex = depthIndexFace;
        if (depthIndex < 3)
          solveFaceCaseAgX( depthIndex, -1, signIn1, trans1, trans2To1, trans1To2, he1, he2, epsilon, result );    
        else
          solveFaceCaseAgX( depthIndex - 3, 1, signIn2, trans2, trans1To2, trans2To1, he2, he1, epsilon, result );     
      }
    }
    else {
      // create face contact
      depthIndex = depthIndexFace;      
      if (depthIndex < 3)
        solveFaceCaseAgX( depthIndex, -1, sign(midP2In1[depthIndex]), trans1, trans2To1, trans1To2, he1, he2, epsilon, result );    
      else
        solveFaceCaseAgX( depthIndex - 3, 1, sign(midP1In2[depthIndex - 3]), trans2, trans1To2, trans2To1, he2, he1, epsilon, result );     
    }
    #endif
  }


} // Anonymous namespace.

#endif

