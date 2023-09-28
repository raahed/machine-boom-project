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

#ifndef AGX_PROPERTY_H
#define AGX_PROPERTY_H


#include <agx/agxPhysics_export.h>
#include <agx/Referenced.h>
#include <agx/ref_ptr.h>

#include <agx/stdint.h>
#include <typeinfo>
#include <stdexcept>
#include <agxStream/Serializable.h>
#include <agx/Vec3.h>
#include <agx/Vec2.h>

#define PROPERTY_THROW_IF_TRUE(X, MESSAGE) if (X) { agxThrow std::runtime_error(MESSAGE); }
#include <agx/HashTable.h>
#include <string>
#include <agx/HashSet.h>

namespace agx {

#define PROPERTY_TYPE(type, Type) \
public: \
  \
  typedef agx::HashTable<std::string,type > Type ## Map;\
  \
protected: \
  \
  Type ## Map m_ ## Type ## _map;\
  \
public:\
  \
  /**\
  Get a map with all the properties \
  \return a reference to the map of properties.\
  */\
  Type ## Map& getPropertyMap ## Type() { return m_ ## Type ## _map; };\
  \
  /**\
  Get the number of properties in the map. \
   \
  \return number of added properties\
*/\
  size_t getNumberOfProperties ## Type() const{ return m_ ## Type ## _map.size(); };\
  \
  void registerPropertyType ## Type()\
  {\
  m_types.insert( #Type );\
};\
  \
  /**\
  Add a named property with a value\
  \
  \return a reference to the value just added \
  */\
  type& addProperty ## Type( const std::string& name,\
  type value = type() )\
  {\
  m_ ## Type ## _map[name] = value;\
  return m_ ## Type ## _map[name];\
};\
  /**\
  Remove a named property \
  \
  \return True if the removal was successful\
*/\
  bool removeProperty ## Type( const std::string& name)\
  {\
  Type ## Map::iterator p = m_ ## Type ## _map.find( name );\
  if(p == m_ ## Type ## _map.end()) return false; else  { m_ ## Type ## _map.erase(p); return true; } \
}\
  \
  /**\
  Set the value of a named property. \
  Will Throw an exception if the named property doest exist. \
  \return a reference to the value\
*/\
  type& setProperty ## Type( const std::string& name, type value)\
  {\
  Type ## Map::iterator p = m_ ## Type ## _map.find( name );\
  PROPERTY_THROW_IF_TRUE( p == m_ ## Type ## _map.end(), "Property named "+name+" is missing" );\
  p->second = value;\
  return p->second;\
};\
  \
  /**\
  Check if a named property exists. \
  \return true if the named property of a specified type exist \
*/\
  bool hasProperty ## Type( const std::string& name) const \
  {return m_ ## Type ## _map.find(name)!=m_ ## Type ## _map.end();};\
  \
  /**\
  Get a reference to the named property value\
  Will Throw an exception if the named property doest exist. \
  \return a reference to the value\
  */\
  type getProperty ## Type( const std::string& name ) const \
  {\
  Type ## Map::const_iterator p = m_ ## Type ## _map.find( name );\
  PROPERTY_THROW_IF_TRUE( p == m_ ## Type ## _map.end(), "Property " + name + " of type " + #Type + " is missing!" );\
  return p->second;\
};\
  type& getProperty ## Type( const std::string& name ) \
{\
  Type ## Map::iterator p = m_ ## Type ## _map.find( name );\
  PROPERTY_THROW_IF_TRUE( p == m_ ## Type ## _map.end(), "Property " + name + " of type " + #Type + " is missing!" );\
  return p->second;\
};\
  /**\
  Get the value of a named property. \
  \param value - Where the value of the property will be put.\
  \return false if the named property doesn't exist.\
  */\
  bool getProperty ## Type( const std::string& name, type & value) const \
  { \
  if (hasProperty ## Type( name )) { \
  value = getProperty ## Type( name ); \
  return true; \
  } \
  return false; \
}\
  \
  \
  void copyProperties ## Type( const PropertyContainer* pc ) {\
  if ( pc->m_ ## Type ## _map.size() == 0 ) return;\
  for ( Type ## Map::const_iterator i = pc->m_ ## Type ## _map.begin();\
  i != pc->m_ ## Type ## _map.end(); i++ ) {\
  m_ ## Type ## _map[i->first] = i->second;\
}\
}\




  /**
  Class that is a container of named properties.
  A PropertyContainer can hold named properties of various types (bool, int, float, double, string).
  At any time these properties can be queried for their value, or updated.
  */
  class AGXPHYSICS_EXPORT PropertyContainer : public agx::Referenced, public agxStream::Serializable
  {
  public:

    PropertyContainer() {
      registerPropertyTypeInt();
      registerPropertyTypeFloat();
      registerPropertyTypeDouble();
      registerPropertyTypeReal();
      registerPropertyTypeBool();
      registerPropertyTypeString();
      registerPropertyTypeLong();
      registerPropertyTypeVec2();
      registerPropertyTypeVec3();
    };

    virtual ~PropertyContainer() {}


    void copyProperties( const PropertyContainer* pc ) {
      copyPropertiesInt( pc );
      copyPropertiesFloat( pc );
      copyPropertiesDouble( pc );
      copyPropertiesReal( pc );
      copyPropertiesBool( pc );
      copyPropertiesString( pc );
      copyPropertiesLong( pc );
      copyPropertiesVec2( pc );
      copyPropertiesVec3( pc );
    };

    /// Return true if a property of type name is present in the container.
    bool hasPropertyType( const std::string& name ) const {
      return m_types.find( name ) != m_types.end();
    };

    /// Return the number of types that are registered.
    size_t size(  ) const {
      return m_types.size();
    };


    PROPERTY_TYPE( int64_t, Long )

    PROPERTY_TYPE( int, Int )

    PROPERTY_TYPE( float, Float )

    PROPERTY_TYPE( double, Double )

    PROPERTY_TYPE( agx::Real, Real )

    PROPERTY_TYPE( bool, Bool )

    PROPERTY_TYPE( std::string, String )

    PROPERTY_TYPE( Vec2, Vec2 )

    PROPERTY_TYPE( Vec3, Vec3 )


    /// Remove all properties from propertycontainer
    void clear()
    {
      m_Long_map.clear();
      m_Int_map.clear();
      m_Float_map.clear();
      m_Double_map.clear();
      m_Real_map.clear();
      m_Bool_map.clear();
      m_String_map.clear();
      m_Vec2_map.clear();
      m_Vec3_map.clear();
    }

    AGXSTREAM_DECLARE_SERIALIZABLE( agx::PropertyContainer );

  protected:
    typedef agx::HashSet<std::string> TypeSet;
    TypeSet m_types;

  };

  typedef agx::ref_ptr<PropertyContainer> PropertyContainerRef;

#define SAVE_PROPERTY_CONTAINER( Out, TypeName, Type ) \
  {\
    TypeName ## Map::const_iterator it =  m_ ## TypeName ## _map.begin(); \
    Out.beginSection("PropertyContainer");\
    Out.addAttribute("size", m_ ## TypeName ## _map.size() );\
    for(; it != m_ ## TypeName ## _map.end(); ++it) {\
      Type TypeName ## value; \
      std::string name = it->first; \
      TypeName ## value = it->second; \
      Out << agxStream::out( "name", name );\
      Out << agxStream::out( "value", TypeName ## value );\
    }\
    Out.endSection("PropertyContainer");\
  }

/*
  define SAVE_PROPERTY_CONTAINER_NON_OPERATOR( Out, TypeName, Type ) \
   {\
     TypeName ## Map::const_iterator it =  m_ ## TypeName ## _map.begin(); \
     Out << agxStream::out( "size", m_ ## TypeName ## _map.size() );\
     for(; it != m_ ## TypeName ## _map.end(); ++it) {\
       Type TypeName ## value; \
       std::string name = it->first; \
       TypeName ## value = it->second; \
       Out << agxStream::out( "name", \
       TypeName ## value.save( Out ); \
     }\
   }
*/

#define RESTORE_PROPERTY_CONTAINER( In, TypeName, Type ) \
  {\
    size_t size; std::string name;\
    Type value;\
    \
    m_ ## TypeName ## _map.clear(); \
    In.beginSection("PropertyContainer");\
    In.getAttribute("size", size);   \
    for(size_t i=0; i < size ; ++i) {\
      In >> agxStream::in( "name", name );\
      In >> agxStream::in( "value", value );\
      addProperty ## TypeName (name, value);\
    }\
    In.endSection("PropertyContainer");\
  }

/*
 define RESTORE_PROPERTY_CONTAINER_NON_OPERATOR( In, TypeName, Type ) \
   {\
     size_t size; std::string name;\
     Type value;\
     \
     m_ ## TypeName ## _map.clear(); \
     In >> agxStream::in( "size", size );\
     for(size_t i=0; i < size ; ++i) {\
       In >> agxStream::in( name );\
       value.restore( In );\
       addProperty ## TypeName (name, value);\
     }\
   }
*/

} // namespace agx


#endif // AGX_PROPERTY_H
