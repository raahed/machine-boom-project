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


#ifndef AGXPOWERLINE_DETAIL_SLOT_MAPPTER_UTILITIES_H
#define AGXPOWERLINE_DETAIL_SLOT_MAPPTER_UTILITIES_H

#include <agxPowerLine/detail/PackingInfo.h>
#include <agxPowerLine/detail/DimensionState.h>


namespace agx
{
  class RigidBody;
}



namespace agxSDK
{
  class Simulation;
}



namespace agxPowerLine
{
  namespace detail
  {

    /*
    This is a collection of helper functions used by the slot mapper and its
    tests.
    */

    PackingInfos::iterator findPackingInfo(
       const agx::RigidBody* body,
       PackingInfos& packingInfos);


    PackingInfos::iterator findPackingInfo(
       const Rotational1DofState* dimension,
       PackingInfos& packingInfos);


    std::tuple<PackingInfos::iterator, PackingInfos::iterator> searchPackingInfos(
       Rotational1DofState* dimension,
       PackingInfos& fullPackings,
       PackingInfos& nonfullPackings);


    std::tuple<PackingInfos::iterator, PackingInfos::iterator> searchPackingInfos(
       agx::RigidBody* body,
       PackingInfos& fullPackings,
       PackingInfos& nonfullPackings);



    std::tuple<PackingInfos::iterator, PackingInfos::iterator> findPackingInfos(
       Rotational1DofState* dimension,
       PackingInfos& fullPackings,
       PackingInfos& nonfullPackings);


    std::tuple<PackingInfos::iterator, PackingInfos::iterator> findPackingInfos(
       agx::RigidBody* body,
       PackingInfos& fullPackings,
       PackingInfos& nonfullPackings);


    void addTo(
       agxSDK::Simulation* simulation,
       agx::RigidBody* body);


    void removeFrom(agxSDK::Simulation* simulation, agx::RigidBody* body);


    void removeFromFull(
       PackingInfos& fullPackings,
       PackingInfos::iterator fullIt,
       Rotational1DofState* dimension,
       PackingInfos& nonfullPackings);


    void removeFromNonfull(
       PackingInfos& nonfullPackings,
       PackingInfos::iterator nonfullIt,
       Rotational1DofState* dimension,
       agx::RigidBody* oldBody,
       agxSDK::Simulation* simulation);
  }
}

#endif
