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

#include <agxSDK/agxSDK.h>

#include <agx/Material.h>
#include <agx/QuadraticProbingHashTable.h>

#include <agxCollide/Contacts.h>

#include <agxData/EntityStorage.h>

namespace agx
{
  typedef agx::Vector<const agx::ContactMaterial*> ContactMaterialPtrVector;
  typedef agx::Vector<const agx::Material*> MaterialPtrVector;
}

namespace agxSDK
{
  typedef agx::SymmetricPair<const agx::Material*> MaterialPair;
  typedef agx::ContactMaterial* (*ContactMaterialFunction)(const agx::Material* m1, const agx::Material* m2);

  typedef agx::SymmetricPair< const agx::Material* > SymmetricMaterialPair;
  typedef agx::HashTable< agx::String, agx::MaterialRef > StringMaterialRefTable;
  typedef agx::QuadraticProbingHashTable< SymmetricMaterialPair, agx::ContactMaterialRef > MaterialSPairContactMaterialRefTable;

  AGX_DECLARE_POINTER_TYPES( MaterialManager );

  /**
  Simulation unique manager that handles logics around and parameters in
  contact materials given two agx::Materials.
  */
  class AGXPHYSICS_EXPORT MaterialManager : public agx::Component
  {
    public:
      /**
      \return the class model
      */
      static agx::Model* ClassModel();

      /**
      Static method to calculate the parameters in a contact material given the two
      materials assigned to it.
      \param contactMaterial - contact material to fill with values (getMaterial1() and getMaterial2() will be used)
      */
      static void calculateContactMaterial( agx::ContactMaterial* contactMaterial );

    public:
      /**
      Default constructor.
      */
      MaterialManager();

      /**
      Add material to this manager.
      If a material with the same name already exists, it will return false.
      \param material - the material to add
      \return true if successful - otherwise false
      */
      bool add( agx::Material* material );

      /**
      Add explicit contact material to this manager. This method adds all geometry materials in
      the contact material that aren't present in this manager.

      If an explicit contact material with the same two materials already exists,
      the old explicit contact material will be overwritten.

      The two materials defining this contact material can never trigger a recalculation of the
      values stored in this contact material. The ones specified will always be used.
      \param contactMaterial - the contact material to add
      \return true if successful - otherwise false
      */
      bool add( agx::ContactMaterial* contactMaterial );

      /**
      Remove material from this manager.
      \param material - material to remove
      \return true if successful - otherwise false
      */
      bool remove( agx::Material* material );

      /**
      Remove explicit contact material from this manager.
      \param contactMaterial - contact material to remove
      \return true if successful - otherwise false
      */
      bool remove( agx::ContactMaterial* contactMaterial );

      /**
      Returns or creates an explicit contact material. If \p material1 or \p material2 are changed after this contact material
      has been created, no recalculations of the contact material will be made.
      \param material1 - first material, if nullptr the default material will be used
      \param material2 - second material, if nullptr the default material will be used
      \return contact material between the two materials
      */
      agx::ContactMaterial* getOrCreateContactMaterial( const agx::Material* material1, const agx::Material* material2 );

      /**
      Returns an implicit or explicit contact material if it exists. When the contact material can be implicit, it's not
      defined to change any parameters in the contact material. I.e., this is the current version of the contact material.
      \param material1 - first material, if nullptr the default material will be used
      \param material2 - second material, if nullptr the default material will be used
      \return contact material between material1 and material2 if present - otherwise nullptr
      */
      const agx::ContactMaterial* getContactMaterial( const agx::Material* material1, const agx::Material* material2 ) const;

      /**
      Returns an implicit or explicit contact material if it exists. When the contact material can be implicit, it's not
      defined to change any parameters in the contact material. I.e., this is the current version of the contact material.
      \param material1 - first material, if nullptr the default material will be used
      \param material2 - second material, if nullptr the default material will be used
      \return contact material between material1 and material2 if present - otherwise nullptr
      */
      const agx::ContactMaterial* getContactMaterial( agx::Physics::MaterialPtr material1, agx::Physics::MaterialPtr material2 ) const;

      /**
      \return material given name, if a material with that name is present, otherwise nullptr
      */
      agx::Material* getMaterial( const agx::String& name );

      /**
      \return material given name, if a material with that name is present, otherwise nullptr
      */
      const agx::Material* getMaterial( const agx::String& name ) const;

      /**
      \return material given uuid, if a material with that uuid is present, otherwise 0
      */
      agx::Material* getMaterial(const agx::Uuid& uuid);

      /**
      \return the default material
      */
      const agx::Material* getDefaultMaterial() const;

      /**
      \return all materials with unique name that have been handled by this manager
      */
      const StringMaterialRefTable& getMaterials() const;

      /**
      This method will create and return new vector with all materials.
      This is raw pointers to materials, so if they are unreferenced (for example if
      someone removes it from the simulation), the pointer might be invalidated
      \return all materials with unique name that have been handled by this manager
      */
      agx::MaterialPtrVector getMaterialVector() const;

      /**
      \return all contact materials created or added to this manager
      */
      const MaterialSPairContactMaterialRefTable& getContactMaterials() const;

      /**
      This method will create and return new vector with all ContactMaterials.
      This is raw pointers to ContactMaterial objects, so if they are unreferenced (for example if
      someone removes it from the simulation), the pointer might be invalidated
      \return all ContactMaterials
      */
      agx::ContactMaterialPtrVector getContactMaterialVector() const;

      /**
      \return true if this manager has seen at least one material with contact reduction enabled
      */
      bool hasMaterialsWithContactReduction() const;

      /**
      Assigns contact material to a local geometry contact. This contact material
      can be either implicit or explicit.
      \note It's undefined to change properties of an implicit contact material.
      \param localGeometryContact - local geometry contact to assign contact material to
      \return true if a contact material was assigned, otherwise false
      */
      bool setContactMaterial( agxCollide::LocalGeometryContact* localGeometryContact );

      /**
      Not part of the public API. This method can create implicit contact materials if no explicit is present.
      */
      agx::ContactMaterial* getOrCreateContactMaterial( const agxCollide::GeometryContact* geometryContact );

      /**
      \return data storage for contact materials
      */
      agxData::EntityStorage* getContactMaterialStorage();

      DOXYGEN_START_INTERNAL_BLOCK()
      /**
      Will definitely return a contact material.
      If no explicit contact material has been defined, an implicit contact material is generated.
      */
      agx::ContactMaterial* getContactMaterialOrCreateImplicit(const agx::Material* material1, const agx::Material* material2);

      /**
      Will definitely return a contact material.
      If no explicit contact material has been defined, an implicit contact material is generated.
      */
      agx::ContactMaterial* getContactMaterialOrCreateImplicit(const agxCollide::Geometry* geometry1, const agxCollide::Geometry* geometry2);
      DOXYGEN_END_INTERNAL_BLOCK()

    public:
      /**
      Internal class for store/restore of MaterialManager.
      */
      class MaterialManagerSerializer : public agxStream::Serializable
      {
        public:
          MaterialManagerSerializer( const MaterialManager* mgr )
            : m_mgr( mgr ) {}
          AGXSTREAM_DECLARE_SERIALIZABLE( agxSDK::MaterialManager::MaterialManagerSerializer );

          const MaterialManager* get() const { return m_mgr; }
          void set( MaterialManager* mgr ) { m_mgr = mgr; }

          typedef agx::Vector<agx::ref_ptr<agx::ContactMaterial> > ContactMaterialVector;
          ContactMaterialVector m_contactMaterials;

        protected:
          MaterialManagerSerializer() : m_mgr( nullptr ) {}
          const MaterialManager* m_mgr;
      };

      /**
      Saves necessary internal structures.
      */
      void store( agxStream::OutputArchive& out ) const;

      /**
      Restores necessary internal structures.
      */
      void restore( const MaterialManagerSerializer& mgr );

    protected:
      /**
      Destructor. Reference counted so protected.
      */
      virtual ~MaterialManager();

      /**
      \return symmetric pair given contact material
      */
      SymmetricMaterialPair getSymmetricPair( const agx::ContactMaterial* contactMaterial ) const;

      /**
      \return symmetric pair given two materials
      */
      SymmetricMaterialPair getSymmetricPair( const agx::Material* material1, const agx::Material* material2 ) const;

      friend class Simulation;
      friend class SimulationController;

      /**
      Clears all internal structures.
      */
      void clear();

      /**
      Updates implicit contact materials containing at least one dirty material.
      */
      void resetDirtyMaterials() const;

      /**
      For any geometry contact where the contact material is zero, this method performs getOrCreateContactMaterial
      and assigns it. Note that this method doesn't updates dirty implicit materials, it only makes sure
      GeometryContact::m_material != nullptr. This method should be called after narrow phase update.
      */
      void initializeNewGeometryContacts( const agxCollide::GeometryContactPtrVector& geometryContacts );

      /**
      Updates implicit, dirty, materials that are currently used.
      */
      void updateContactMaterials( const agxCollide::GeometryContactPtrVector& geometryContacts );

      /**
      Try to load the Library ContactMaterial for the pair (material1, material2) and fill in the settings
      in the provided contactMaterial.
      Return true if successful.
      */
      bool loadLibraryContactMaterial( const agx::Material* material1, const agx::Material* material2, agx::ContactMaterial* contactMaterial );

    private:
      StringMaterialRefTable m_materials;
      MaterialSPairContactMaterialRefTable m_contactMaterials;

      bool m_hasMaterialsWithContactReduction;
      agxData::EntityStorageRef m_contactMaterialStorage;

      agx::Mutex m_mutex;
  };

  inline const agx::Material* MaterialManager::getDefaultMaterial() const
  {
    return agx::Material::getDefaultMaterial();
  }

  inline agxData::EntityStorage* MaterialManager::getContactMaterialStorage()
  {
    return m_contactMaterialStorage;
  }

  inline const agx::ContactMaterial* MaterialManager::getContactMaterial( agx::Physics::MaterialPtr material1, agx::Physics::MaterialPtr material2 ) const
  {
    return getContactMaterial( material1 ? material1.model() : getDefaultMaterial(), material2 ? material2.model() : getDefaultMaterial() );
  }

  AGX_FORCE_INLINE SymmetricMaterialPair MaterialManager::getSymmetricPair( const agx::ContactMaterial* contactMaterial ) const
  {
    agxAssert( contactMaterial );
    const agx::Material* m1 = contactMaterial ? contactMaterial->getMaterial1() : nullptr;
    const agx::Material* m2 = contactMaterial ? contactMaterial->getMaterial2() : nullptr;
    return SymmetricMaterialPair( m1 ? m1 : getDefaultMaterial(), m2 ? m2 : getDefaultMaterial() );
  }

  AGX_FORCE_INLINE SymmetricMaterialPair MaterialManager::getSymmetricPair( const agx::Material* material1, const agx::Material* material2 ) const
  {
    return SymmetricMaterialPair( material1 ? material1 : getDefaultMaterial(), material2 ? material2 : getDefaultMaterial() );
  }
}

