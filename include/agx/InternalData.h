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

#ifndef AGX_INTERNALDATA_H
#define AGX_INTERNALDATA_H

#include <agx/Referenced.h>
#include <agx/agxPhysics_export.h>

namespace agx
{
  /**
  Internal data for any object, ObjT, with methods:
    agx::Referenced* ObjT::getInternalData() const;
    void ObjT::setInternalData( agx::Referenced* data );

  For now, add type of object to the Source enum.
  */
  class AGXPHYSICS_EXPORT InternalData : public agx::Referenced
  {
    public:
      /**
      Source of internal data. I.e., the object/algorithm that's
      handling the data.
      */
      enum Source
      {
        WIRE,               /**< Maps geometries and rigid bodies back to wires and wire nodes. */
        MERGED_BODY,        /**< Merged states, neighbors etc.. */
        MERGE_SPLIT,        /**< AMOR related data. */
        DEFORMABLE1D,       /**< Deformable1d related data. */
        RIGID_BODY_EMITTER, /**< Rigid Body data. */
        LINKED_STRUCTURE,   /**< Tracked vehicle related data. */
        ENERGY,             /**< Energy related data. */
        NUM_SOURCES
      };

    public:
      /**
      Get already created data or create new instance of the internal data.
      \note \p obj has to be a valid pointer.
      \param obj - object to receive data from (or create if first call)
      \param source - source object/algorithm for the data
      \return instance of data assigned to \p obj
      */
      template<typename DataT, typename ObjT, typename... Args>
      static DataT* getOrCreate( ObjT obj, Source source, Args&&... args );

      /**
      Get already created data of the internal data. Returns null if \p obj is null or
      if no data has been assigned.
      \param obj - object to receive data from (or create if first call)
      \param source - source object/algorithm for the data
      \return instance of data assigned to \p obj
      */
      template<typename DataT, typename ObjT>
      static DataT* get( const ObjT& obj, Source source );

      /**
      Assign data to source.
      \param obj - object to assign internal data to
      \param data - data to assign (may be null)
      \param source - source object/algorithm for the data
      */
      template<typename ObjT>
      static void set( ObjT obj, agx::Referenced* data, Source source );

      /**
      \return true if source data exist
      */
      template<typename ObjT>
      static agx::Bool hasData( const ObjT& obj, Source source );

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~InternalData() {}

    private:
      /**
      \return already create instance of creates new internal data
      */
      template<typename ObjT>
      static InternalData* getOrCreateInternal( ObjT obj );

      /**
      \return already create instance of the internal data, otherwise null
      */
      template<typename ObjT>
      static InternalData* getInternal( const ObjT& obj );

    private:
      agx::ref_ptr<agx::Referenced> m_data[ NUM_SOURCES ];
  };

  template<typename DataT, typename ObjT, typename... Args>
  DataT* InternalData::getOrCreate( ObjT obj, InternalData::Source source, Args&&... args )
  {
    agxAssert( obj != nullptr );

    InternalData* iData = getOrCreateInternal( obj );
    if ( iData->m_data[ source ] == nullptr )
      iData->m_data[ source ] = new DataT( std::forward<Args>( args )... );

    return iData->m_data[ source ]->as<DataT>();
  }

  template<typename DataT, typename ObjT>
  DataT* InternalData::get( const ObjT& obj, InternalData::Source source )
  {
    InternalData* iData = getInternal( obj );
    return iData != nullptr && iData->m_data[ source ] != nullptr ? iData->m_data[ source ]->as<DataT>() : nullptr;
  }

  template<typename ObjT>
  void InternalData::set( ObjT obj, Referenced* data, InternalData::Source source )
  {
    if ( obj == nullptr )
      return;

    InternalData* iData = getInternal( obj );
    if ( iData == nullptr && data == nullptr )
      return;

    iData = getOrCreateInternal( obj );
    iData->m_data[ source ] = data;
  }

  template<typename ObjT>
  agx::Bool InternalData::hasData( const ObjT& obj, InternalData::Source source )
  {
    const InternalData* iData = getInternal( obj );
    return iData != nullptr && iData->m_data[ source ] != nullptr;
  }

  template<typename ObjT>
  InternalData* InternalData::getOrCreateInternal( ObjT obj )
  {
    agx::Referenced* data = obj->getInternalData();
    if ( data == nullptr ) {
      data = new InternalData();
      obj->setInternalData( data );
    }

    return data->as<InternalData>();
  }

  template<typename ObjT>
  InternalData* InternalData::getInternal( const ObjT& obj )
  {
    return obj != nullptr && obj->getInternalData() != nullptr ? obj->getInternalData()->template as<InternalData>() : nullptr;
  }
}

#endif
