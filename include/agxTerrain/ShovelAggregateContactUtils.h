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

#include <agxTerrain/export.h>
#include <agx/Material.h>
#include <agxTerrain/TerrainMaterial.h>
#include <agxTerrain/Shovel.h>



namespace agxTerrain
{
  AGX_DECLARE_POINTER_TYPES( ShovelAggregateContactMaterialContainer );
  AGX_DECLARE_VECTOR_TYPES( ShovelAggregateContactMaterialContainer );

  /**
  Configure the contact material that is to be used in shovel-aggregate contacts. The contact material properties
  are set from the contact material between the shovel geometries and the terrain geometry by default.
  The user can override it by specifying an explicit contact material that is used in the different kinds of
  excavation modes. See 'setShovelAggregateContactMaterial' in agxTerrain::Terrain.

  \param shovelAggregateCM - the shovel-aggregate contact material that should be configured.
  \param contactArea - The area of the shovel-aggregate contact for the specified excavation
                       mode. Used in scaling of Young's modulus.
  \param mode - the excavation mode that is used to search if an explicit contact material has be set for the contact.
                If so, it is used instead of the shovel-terrain contact material.
  \param collection - The specified collection that has the shovel object in the contact.
  \return true if the shovelAggregateCM was configured successfully, false otherwise.
  */
  AGXTERRAIN_EXPORT bool setupShovelAggregateContactMaterial( agx::ContactMaterial* shovelAggregateCM,
                                                              agx::Real contactArea,
                                                              Shovel::ExcavationMode mode,
                                                              TerrainToolCollection* collection );

  /**
  Set the shovel-aggregate contact material properties from a source contact material that should
  either be the regular shovel-terrain contact material or a user-specified contact material for a
  given excavation mode.
  \param shovelAggregateCM - the shovel-aggregate contact material where properties will be set.
  \param contactArea - The area of the shovel-aggregate contact for the specified
                       excavation mode. Used in scaling of Young's modulus.
  \param sourceMaterial - the base material whose properties will be used in the setup.
  \param terrainMaterial - the TerrainMaterial of the active terrain where the shovel-aggregate contact exists.
  */
  AGXTERRAIN_EXPORT void setShovelAggregateContactMaterialProperties( agx::ContactMaterial* shovelAggregateCM,
                                                                      agx::Real contactArea,
                                                                      const agx::ContactMaterial* sourceMaterial,
                                                                      const TerrainMaterial* terrainMaterial );

  /**
  Checks if the specified geometry is valid for creating aggregate contacts against.
  \return true if the specified geometry can be used for aggregate contacts.
  */
  AGXTERRAIN_EXPORT bool isValidContactGeometry( agxCollide::Geometry* geometry );

  /**
  Get first contact valid geometry in the shovel rigid body for contact. The geometry must be enabled
  and also not a sensor.
  \param shovelBody - The rigid body of the Shovel that will be checked.
  \return the first valid geometry for aggregate contacts available in the shovel body. If no one exists, return nullptr.
  */
  AGXTERRAIN_EXPORT agxCollide::Geometry* getFirstContactValidGeometry( const agx::RigidBody* shovelBody );

  /**
  Find the agx::Material on the first contact valid contact geometry in the specified shovel.
  \param shovel - the specified shovel.
  \return the agx::Material found on the first contact valid geometry in the specified shovel body.
          Returns nullptr if no contact valid geometry is found.
  */
  AGXTERRAIN_EXPORT agx::Material* findShovelMaterial( agxTerrain::Shovel* shovel );

  /**
  Simple container class for storing and handling explicit contact material for shovel-aggregate contacts,
  indexed by specified excavation mode.
  */
  class AGXTERRAIN_EXPORT ShovelAggregateContactMaterialContainer : public agx::Referenced, public virtual agxStream::Serializable
  {
  public:
    ShovelAggregateContactMaterialContainer()
    {
      // Initialize with 4 since we have 4 different aggregate contacts
      m_contactMaterials.resize( 4, nullptr );
    }

    ~ShovelAggregateContactMaterialContainer()
    {
    }

    bool setContactMaterial( Shovel::ExcavationMode excavationMode,
                             agx::ContactMaterial* contactMaterial )
    {
      agx::UInt32 materialIndex = static_cast< agx::UInt32 >( excavationMode );
      if ( materialIndex < m_contactMaterials.size() )
      {
        m_contactMaterials[ materialIndex ] = contactMaterial;
        return true;
      }
      else
      {
        return false;
      }
    }

    agx::ContactMaterial* getContactMaterial( Shovel::ExcavationMode excavationMode )
    {
      agx::UInt32 materialIndex = static_cast< agx::UInt32 >( excavationMode );
      if ( materialIndex < m_contactMaterials.size() )
      {
        return m_contactMaterials[ static_cast< agx::UInt32 >( excavationMode ) ];
      }
      else
      {
        return nullptr;
      }
    }

    DOXYGEN_START_INTERNAL_BLOCK()
    AGXSTREAM_DECLARE_SERIALIZABLE( agxTerrain::ShovelAggregateContactMaterialContainer );
    DOXYGEN_END_INTERNAL_BLOCK()

  private:
    agx::Vector< agx::ContactMaterialRef > m_contactMaterials;
  };
}


