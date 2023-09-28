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

#ifndef AGXMODEL_ADDEDMASSDB_H
#define AGXMODEL_ADDEDMASSDB_H

#include <agxModel/export.h>
#include <agxModel/WindAndWaterUtils.h>

#include <agx/Vec6.h>

#include <agxCollide/Mesh.h>

#include <agxStream/StorageStream.h>
#include <agxStream/OutputArchive.h>
#include <agxStream/InputArchive.h>

namespace agxModel
{
  class WindAndWaterParameters;

  /**
  Storage that holds data needed for added mass calculations.
  */
  class AGXMODEL_EXPORT AddedMassStorage : public agx::Referenced
  {
    public:
      typedef agxModel::dense::Vector Phi;
      typedef agx::Vector< Phi > PhiContainer;
      typedef agx::VectorPOD< agx::Vec6 > NormalTangentContainer;
      typedef agx::UInt32 Identifier;

    public:
      /**
      Creates an identifier given name (normally path + filename) used to identify added mass
      storages.
      */
      static agxModel::AddedMassStorage::Identifier createIdentifier( const agx::String& name );

      /**
      \return true if the identifier is valid
      */
      static agx::Bool isValid( agxModel::AddedMassStorage::Identifier id );

    public:
      /**
      Report from creation, store and restore.
      */
      struct AGXMODEL_EXPORT Report
      {
        Report( agx::Bool validIn, agx::Real executionTimeIn ) : valid( validIn ), executionTime( executionTimeIn ) {}
        agx::Bool valid;
        agx::Real executionTime;
      };

    public:
      /**
      Create/calculate data given mesh and center of mass position given in mesh coordinate frame.
      */
      agxModel::AddedMassStorage::Report create( const agxCollide::Mesh* mesh, const agx::Vec3& localCenterOfMassPosition );

      /**
      Store data to stream.
      */
      agxModel::AddedMassStorage::Report store( agxStream::StorageStream& str ) const;

      /**
      Restore data from stream.
      */
      agxModel::AddedMassStorage::Report restore( agxStream::StorageStream& str );

      /**
      Mostly for debug purposes. Check if two storages are identical.
      */
      agx::Bool equals( const agxModel::AddedMassStorage* other, agx::Real threshold = agx::Real( 0 ) ) const;

      /**
      \return the flow data, empty if not loaded
      */
      const agxModel::AddedMassStorage::PhiContainer& getFlowData() const;

      /**
      \return normal-tangent data, empty if not loaded
      */
      const agxModel::AddedMassStorage::NormalTangentContainer& getNormalTangentData() const;

      /**
      \return the identifier of this storage
      */
      agxModel::AddedMassStorage::Identifier getIdentifier() const;

    protected:
      friend class AddedMassDb;
      /**
      Construct given identifier.
      */
      AddedMassStorage( agxModel::AddedMassStorage::Identifier identifier );

      /**
      Reference counted object, protected destructor.
      */
      virtual ~AddedMassStorage();

      /**
      Allocate internal structure given the number of elements (triangles).
      */
      void allocate( agx::UInt numElements );

    private:
      PhiContainer m_phi;
      NormalTangentContainer m_normalsTangents;
      Identifier m_id;
  };

  typedef agx::ref_ptr< AddedMassStorage > AddedMassStorageRef;

  /**
  Holder/manager of added mass storages.
  */
  class AGXMODEL_EXPORT AddedMassDb : public agx::Referenced
  {
    public:
      /**
      Creates new storage given shape. It's recommended to cache these storages so consider
      using getOrCreateStorage that handles read and write of files containing this data.
      \sa getOrCreateStorage
      \param parameters - parameters associated to a shape
      \param identifier - unique identifier
      \return added mass storage if successful, otherwise null
      */
      static agxModel::AddedMassStorageRef createStorage( const agxModel::WindAndWaterParameters* parameters,
                                                          agxModel::AddedMassStorage::Identifier identifier );

    public:
      /**
      Default constructor.
      */
      AddedMassDb();

      /**
      \return an already loaded storage, returns null if not loaded
      \sa getOrCreateStorage
      */
      const agxModel::AddedMassStorage* getStorage( agxModel::AddedMassStorage::Identifier id ) const;

      /**
      Returns already loaded storage or loads existing file. If \p parameters is valid and the file doesn't exist
      a new file will be created.
      \param filename - filename including path to file where to write or read data
      \param parameters - parameters associated to a shape (if null, only existing data files will be loaded, i.e., fails if file doesn't exist)
      \return added mass storage for shape in \p parameters
      */
      agxModel::AddedMassStorage* getOrCreateStorage( const agx::String& filename, const agxModel::WindAndWaterParameters* parameters );

      /**
      Reloads a file and updates the storage identifier if it exist. If the file doesn't exist null is returned.
      \sa getOrCreateStorage
      */
      agxModel::AddedMassStorage* load( const agx::String& filename );

      /**
      Saves storage to file.
      \sa getOrCreateStorage
      */
      agx::Bool save( agxModel::AddedMassStorage* storage, const agx::String& filename );

      /**
      Internal method.
      */
      virtual void store( agxStream::OutputArchive& out ) const;

      /**
      Internal method.
      */
      virtual void restore( agxStream::InputArchive& in );

    protected:
      typedef agx::HashVector< AddedMassStorage::Identifier, AddedMassStorageRef > AddedMassStorageContainer;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~AddedMassDb();

    private:
      AddedMassStorageContainer m_storage;
  };

  typedef agx::ref_ptr< AddedMassDb > AddedMassDbRef;

  AGXMODEL_EXPORT agx::Vec3 getCenter( const agxCollide::Mesh::Triangle& triangle );
}

#endif // AGXMODEL_ADDEDMASSDB_H
