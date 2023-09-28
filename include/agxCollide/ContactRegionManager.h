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

#ifndef AGXCOLLIDE_CONTACT_REGION_MANAGER_H
#define AGXCOLLIDE_CONTACT_REGION_MANAGER_H

#include <agx/agx_vector_types.h>

#include <agx/agxPhysics_export.h>
#include <agxCollide/RegressionPlane.h>
#include <agxData/LocalVector.h>

namespace agxCollide {

  /// Class for merging contact regions and computing their normals.
  class AGXPHYSICS_EXPORT ContactRegionManager {

  public:

    ContactRegionManager();

    /**
    * Marks two contact regions for having to be merged.
    *
    * \param contactRegionIndex0 Index of first contact region
    * \param contactRegionIndex1 Index of other contact region
    */
    void markContactRegionsForMerge(
      size_t contactRegionIndex0,
      size_t contactRegionIndex1);

    /**
    * Merges contact regions after they have been marked for merging.
    *
    * \param contactRegions Contact region for each triangle contact
    * \param nrContactRegions Gives number of contact regions before merge
    *
    * \retval Number of found contact regions after merge
    */
    size_t finalizeMergeContactRegions(
      agxData::LocalVector<agx::Int32>& contactRegions,
      size_t nrContactRegions);

    /**
    * Computes regression planes for several contact regions at once.
    *
    * \param regionPerContact Contact region for each triangle contact
    * \param points All contact points for all triangle contacts.
    * \param nrPointsPerContact How many points does each triangle contact have?
    * \param nrContactRegions The number of contact regions.
    * \param regressionPlanes The result: The regression planes will be written here
    *        Previous content will be overwritten.
    */
    static void computeRegressionPlanes (
      /// \todo Consider a span/view-like type here.
      const agxData::LocalVector<agx::Int32>& regionPerContact,
      const agxData::LocalVector<agx::Vec3>& points,
      const agxData::LocalVector<agx::UInt32>& nrPointsPerContact,
      const agxData::LocalVector<agx::UInt32>& nrPointsBeforeEachContact,
      size_t nrContactRegions,
      agx::Vector<RegressionPlane>& regressionPlanes);
  private:
    agxData::LocalVector<agx::UInt32> m_toMerge;
  };

}

#endif // AGXCOLLIDE_CONTACT_REGION_MANAGER_H

