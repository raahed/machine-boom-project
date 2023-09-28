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
#pragma once


#include <agxCollide/Contacts.h>
#include <agxCollide/ShapeCollider.h>

#include <agxData/LocalVector.h>

namespace agxCollide {

  /**
  Class for contact reduction.
  Contact reduction is important for solver performance in case of many contacts.

  The contact reducer reduces contacts based on binning in 6 dimensions.
  No contact points are directly removed; their 'enabled'-flag is set to 'disabled'.
  Contacts are defined by each contacts position and normal, in a grid based manner.
  The purpose is to retain the widest span in 6 dimensions.
    This approach does not take depth directly into consideration;
  however, if a contact point with larger depth is removed, the one remaining
  in its bin will be assigned the larger depth.
    The approach is based on a discretization of the contact set, and it is not
  rotation invariant, which means that the reduction is not perfect,
  and that sometimes desired contacts might get disabled, too.
    Since this contact reduction approach is not rotation invariant,
  it might be desirable to give it additional information in order
  to obtain the same results in different frames.
  If no frame is given (null pointer), the contact will be calculated
  at the translational midrange of all contact points in world coordinates.
    The number of bins per dimension should be between 1 and
    10 (and will be clamped to this range).
  This value is equal for all dimensions.
  After reduction, each bin could possibly contain one contact.
  Setting binsPerDimension to 2 could thus theoretically yield 2^6 = 64 contacts,
  setting it to 3 could result in up to 3^6 = 729 contacts and so on;
  however, in real-world problems these numbers will be much lower,
  since most often, contact points are similar enough to end up in the
  same bin.
  */
  class AGXPHYSICS_EXPORT ContactReducer {
    public:

      /**
      Reduces contacts based on binning in 6 dimensions.
      See class description for more details.
      \param contactPoints A vector containing pointers to all contact points
             that define the set which should be reduced.
             The enabled()-flag of the points will be modified, as might be depth().
      \param binsPerDimension How many bins should be given per dimension.
             The value should be between 1 and 10 (and will be clamped to this range).
             This value is equal for all dimensions.
      \param objectFrame Frame where the contact reduction should take place.
             If no frame is given (null pointer), the contact will be calculated
             at the translational midrange of all contact points in world coordinates.
      \param positionTolerance Threshold deciding when differences in positions
             should still put into different bins.
      \return Number of disabled contact points.
      */
      static size_t reduce(
        agx::Vector<agxCollide::LocalContactPoint*>& contactPoints,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0E-6));

      /**
      Reduces contacts based on binning in 6 dimensions.
      See class description for more details.
      \param contactPoints A vector containing pointers to all contact points
             that define the set which should be reduced.
             The enabled()-flag of the points will be modified, as might be depth().
      \param binsPerDimension How many bins should be given per dimension.
             The value should be between 1 and 10 (and will be clamped to this range).
             This value is equal for all dimensions.
      \param objectFrame Frame where the contact reduction should take place.
             If no frame is given (null pointer), the contact will be calculated
             at the translational midrange of all contact points in world coordinates.
      \param positionTolerance Threshold deciding when differences in positions
             should still put into different bins.
      \return Number of disabled contact points.
      */
      static size_t reduce(
        agx::Vector<agxCollide::ContactPoint*>& contactPoints,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0E-6));

      /**
      Reduces contacts based on binning in 6 dimensions.
      See class description for more details.
      \param contactPoints A vector containing pointers to all contact points
             that define the set which should be reduced.
             The enabled()-flag of the points will be modified, as might be depth().
      \param binsPerDimension How many bins should be given per dimension.
             The value should be between 1 and 10 (and will be clamped to this range).
             This value is equal for all dimensions.
      \param objectFrame Frame where the contact reduction should take place.
             If no frame is given (null pointer), the contact will be calculated
             at the translational midrange of all contact points in world coordinates.
      \param positionTolerance Threshold deciding when differences in positions
             should still put into different bins.
      \return Number of disabled contact points.
      */

      static size_t reduce(
        agxData::LocalVector<agxCollide::LocalContactPoint*>& contactPoints,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0E-6));


      static size_t reduce(
        agxData::LocalVector<agxCollide::LocalContactPoint>& contactPoints,
        agx::IndexRangeT<size_t> range,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0e-6));


      /**
      Reduces contacts based on binning in 6 dimensions.
      See class description for more details.
      \param contactPoints A vector containing pointers to all contact points
             that define the set which should be reduced.
             The enabled()-flag of the points will be modified, as might be depth().
      \param binsPerDimension How many bins should be given per dimension.
             The value should be between 1 and 10 (and will be clamped to this range).
             This value is equal for all dimensions.
      \param objectFrame Frame where the contact reduction should take place.
             If no frame is given (null pointer), the contact will be calculated
             at the translational midrange of all contact points in world coordinates.
      \param positionTolerance Threshold deciding when differences in positions
             should still put into different bins.
      \return Number of disabled contact points.
      */
      static size_t reduce(
        agx::Vector<agxCollide::ContactPoint>& contactPoints,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0E-6));

      /**
      Reduces contacts based on binning in 6 dimensions.
      See class description for more details.
      \param contactPoints A vector containing pointers to all contact points
             that define the set which should be reduced.
             The enabled()-flag of the points will be modified, as might be depth().
      \param binsPerDimension How many bins should be given per dimension.
             The value should be between 1 and 10 (and will be clamped to this range).
             This value is equal for all dimensions.
      \param objectFrame Frame where the contact reduction should take place.
             If no frame is given (null pointer), the contact will be calculated
             at the translational midrange of all contact points in world coordinates.
      \param positionTolerance Threshold deciding when differences in positions
             should still put into different bins.
      \return Number of disabled contact points.
      */
      static size_t reduce(
        agxData::Array<agxCollide::ContactPoint>& contactPoints,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0E-6));

      /**
      Reduces contacts based on binning in 6 dimensions.
      See class description for more details.
      \param contactPoints A vector containing pointers to all contact points
             that define the set which should be reduced.
             The enabled()-flag of the points will be modified, as might be depth().
      \param binsPerDimension How many bins should be given per dimension.
             The value should be between 1 and 10 (and will be clamped to this range).
             This value is equal for all dimensions.
      \param objectFrame Frame where the contact reduction should take place.
             If no frame is given (null pointer), the contact will be calculated
             at the translational midrange of all contact points in world coordinates.
      \param positionTolerance Threshold deciding when differences in positions
             should still put into different bins.
      \return Number of disabled contact points.
      */
      static size_t reduce(
        agx::VectorPOD<agxCollide::LocalContactPoint>& contactPoints,
        size_t binsPerDimension,
        const agx::Frame* objectFrame = nullptr,
        agx::Real positionTolerance = agx::Real(1.0E-6));
  };
}

