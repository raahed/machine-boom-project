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

#include <agx/InternalData.h>

namespace agxSDK
{
  class LinkedStructure;
  class LinkedSegment;

  /**
  Connection back from bodies, constraints etc. to a LinkedStructure instance.
  */
  struct LinkedStructureObjectData : public agx::Referenced
  {
    /**
    \return new or already created instance of the InternalData::LINKED_STRUCTURE slot in
            the given object
    */
    template<typename T, typename ObjT, typename... ArgsT>
    static agx::ref_ptr<T> getOrCreate( const ObjT& obj, ArgsT&&... args )
    {
      return agx::InternalData::getOrCreate<T>( obj, agx::InternalData::LINKED_STRUCTURE, std::forward<ArgsT>( args )... );
    }

    /**
    \return InternalData::LINKED_STRUCTURE slot instance if created, otherwise null
    */
    template<typename T, typename ObjT>
    static T* get( const ObjT& obj )
    {
      return agx::InternalData::get<T>( obj, agx::InternalData::LINKED_STRUCTURE );
    }

    LinkedStructureObjectData()
      : linkedStructure( nullptr ), segment( nullptr )
    {
    }

    LinkedStructureObjectData( LinkedSegment* segment )
      : linkedStructure( nullptr ), segment( segment )
    {
    }

    LinkedStructure* linkedStructure;
    LinkedSegment* segment;

    protected:
      virtual ~LinkedStructureObjectData()
      {
      }
  };
}