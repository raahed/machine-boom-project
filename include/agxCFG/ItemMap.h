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

#ifndef AGXCFG_ITEMMAP_H
#define AGXCFG_ITEMMAP_H


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names

// Disable the warning about
# pragma warning( disable: 4275 ) //  warning C4275: non dll-interface class

#endif

#include <agx/macros.h>
DOXYGEN_START_INTERNAL_BLOCK()

#include <string>
#include <map>

#include <agx/agx.h>
#include <agxCFG/ConfigValue.h>
#include <agxCFG/utils.h>
#include <agx/agxPhysics_export.h>

namespace agxCFG
{

  typedef std::multimap<std::string, ConfigValueObserver> ItemMap_t;
  typedef ItemMap_t::iterator ItemMapIterator;


  /**
  ItemMap is a class for storing items read from setupfile.
  Derived from multimap, i.e. it can contain multiple of occurrences with the
  same key */
  class AGXPHYSICS_EXPORT ItemMapError
  {
    public:
      ///
      ItemMapError( std::string m ) : message( m ) {}
      ///
      std::string message;

  };



/// A map for storing Items, derived from multimap
  class AGXPHYSICS_EXPORT ItemMap : public agx::Referenced //ItemMap_t
  {
    public:

      typedef ItemMap_t::iterator iterator;
      typedef ItemMap_t::const_iterator const_iterator;
      typedef ItemMap_t::reverse_iterator reverse_iterator;
      typedef ItemMap_t::value_type value_type;
      typedef std::pair<const_iterator, const_iterator> ConstPairRange;
      typedef std::pair<iterator, iterator> PairRange;
      typedef ItemMap_t::key_type key_type;


      ///
      ItemMap& operator= ( ConfigValue* );

      inline const_iterator end() const {
        return m_map.end();
      }
      inline const_iterator begin() const {
        return m_map.begin();
      }

      inline iterator end()  {
        return m_map.end();
      }
      inline iterator begin()  {
        return m_map.begin();
      }
      inline reverse_iterator rbegin() {
        return m_map.rbegin();
      }
      inline reverse_iterator rend() {
        return m_map.rend();
      }

      AGXPHYSICS_EXPORT friend std::ostream& operator <<( std::ostream& os, const ItemMap& item_map );
      AGXPHYSICS_EXPORT friend std::ostream& operator << ( std::ostream& os, const ItemMapIterator& ii );

      /*!
        Locates a literal in the item map named name. It should be a float, integer or an expression
        \return true if name is found as the right type, otherwise false
      */
      bool getLiteral( const std::string& name, agx::Real& value );

      inline iterator insert( const value_type& t ) {
        return m_map.insert( t );
      }
      inline bool empty( )  const {
        return m_map.empty();
      }
      inline void clear( ) {
        m_map.clear();
      }
      inline void erase( iterator it ) {
        m_map.erase( it );
      }
      ConstPairRange equal_range( const key_type& key ) const {
        return m_map.equal_range( key );
      }
      PairRange equal_range( const key_type& key )  {
        return m_map.equal_range( key );
      }

      /// Returns the first matching item
      ItemMapIterator find( const std::string& key ) {
        return m_map.find( key );
      }


      /** Makes a two level search, starts to search in this map after key.
          If there was a match it continues to the second level map (if the stored value
          is a struct. It searches in the 2:nd level map after sndKey.
          If there was a match it continues to check the value (if it was a string value) for a match. */
      ItemMapIterator find( const std::string& key, const std::string& sndKey, const std::string& sndValue );

      /** Replaces the item pointed by p with the value.
         If the values are not of the same type a ItemMapError is thrown  */
      ItemMapIterator replace( ItemMapIterator p, ConfigValue* value );

      /** Replaces the item identified with key with the value.
         If the values are not of the same type a ItemMapError is thrown  */
      ItemMapIterator replace( const std::string& key, ConfigValue* value ) {
        return insert_item( key, value );
      };


      /// Inserts a new item, replaces the first occurrence if it already exists.
      ItemMapIterator insert_item( const std::string& key, ConfigValue* item );

    protected:
      ///
      virtual ~ItemMap();


      ItemMap_t m_map;
  };

  typedef agx::ref_ptr<ItemMap> ItemMapRef;

}
#ifdef _MSC_VER
# pragma warning(pop)
#endif

DOXYGEN_END_INTERNAL_BLOCK()

#endif

