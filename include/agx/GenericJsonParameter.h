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

#include <agx/Logger.h>

#include <agx/Json.h>
#include <agx/String.h>

namespace agx
{

  /**
  A template class that can be used to read/write a json property and set/get the value in another object.
  */
  template<typename T>
  class GenericJsonParameter {
    public:

      using PropertyGetBool = std::function< bool( const T* ) >;
      using PropertyGetReal = std::function< agx::Real( const T* ) >;
      using PropertyGetVec3 = std::function< agx::Vec3( const T* ) >;

      using PropertySetBool = std::function< void( bool, T* ) >;
      using PropertySetReal = std::function< void( agx::Real, T* ) >;
      using PropertySetVec3 = std::function< void( agx::Vec3, T* ) >;


      /**
      Type of parameter that should be handled.
      */
      enum ParameterType {
        UNDEFINED_TYPE,
        SCALAR_BOOL,
        SCALAR_REAL,
        ARRAY_VEC3
      };


      GenericJsonParameter( const agx::String& name, PropertySetReal setter, PropertyGetReal getter )
        : m_type(SCALAR_REAL), m_name(name), m_getReal(getter), m_setReal(setter)
      {
      }

      GenericJsonParameter( const agx::String& name, const agx::String& group, PropertySetReal setter, PropertyGetReal getter )
        : m_type(SCALAR_REAL), m_name(name), m_group(group), m_getReal(getter), m_setReal(setter)
      {
      }

      GenericJsonParameter( const agx::String& name, PropertySetVec3 setter, PropertyGetVec3 getter, unsigned int numValues )
        : m_type(ARRAY_VEC3), m_name(name), m_numValues(numValues), m_getVec3(getter), m_setVec3(setter)
      {
      }

      /**
      Is value marked as "default"?
      */
      bool isDefault( const agxJson::Value& value ) const
      {
        bool def = false;
        if ( value.isString() ) {
          def = agx::String(value.asString()).lower() == "default";
        }
        return def;
      }

      /**
      Read a bool from a json value.
      */
      bool parseBool( const agxJson::Value& value ) const
      {
        bool result = false;

        if ( value.isBool() )
        {
          result = value.asBool();
        }
        else if ( value.isString() ) {
          result = agx::String(value.asString()).lower() == "true";
        }

        return result;
      }

      /**
      Read a real number from a json value
      */
      agx::Real parseReal( const agxJson::Value& value ) const
      {
        if ( value.isDouble() ) {
          return value.asDouble();
        }
        else if ( value.isString() ) {

          if ( value.asString() == "inf" ) {
            return agx::Infinity;
          }
          else if ( value.asString() == "-inf" ) {
            return -agx::Infinity;
          }
        }

        LOGGER_WARNING() << "Unknown value \"" << value.asString() << "\" in json data for " <<
          getParameterPath() << LOGGER_ENDL();

        return 0;
      }


      /**
      Parse a json node and produce a Vec3.
      The result is passed in by reference and should already contain the default values.
      Possible json include:
      - "var" = [1]
      - "var" = [1,2,3]
      - "var" = ["default", 2,3]
      - "var" = "default"
      */
      void parseVec3( const agxJson::Value& value, agx::Vec3& result ) const
      {

        if ( value.isArray() ) {
          if ( value.size() != m_numValues )
            LOGGER_WARNING() << "Array with values for " << m_name << " does not have expected length "
              << m_numValues << "\n" << LOGGER_END();

          agxJson::Value::ArrayIndex maxIndex = std::min(value.size(), std::min(m_numValues, 3u));

          for ( agxJson::Value::ArrayIndex i = 0; i < maxIndex; i++) {
            agxJson::Value valueNode = value[i];

            if ( !isDefault(valueNode) ) {
              result[i] = parseReal( valueNode );
            }
          }

        }
        else if ( value.isDouble() ) {
          // If the value is a single number, then we return a Vec3 with said value
          result = agx::Vec3( value.asDouble() );
        }
        else if ( value.isString() ) {
          if ( !isDefault(value) )
            result = agx::Vec3( parseReal( value ) );
        }
        else {
          LOGGER_WARNING() << "Unknown value \"" << value.asString() << "\" in json data for " <<
            getParameterPath() << LOGGER_ENDL();
        }
      }



      /**
      Reads a generic property from a json structure.
      */
      bool readProperty(const agxJson::Value& root, T* material )
      {
        agxJson::Value object;

        if ( m_group != "" ) {
          if ( !root.hasMember( m_group.c_str() ) ) {
            LOGGER_WARNING() << "Json file is missing object " << m_group << LOGGER_ENDL();
            return false;
           }
           object = root[m_group];
        }
        else {
          object = root;
        }

        if ( !object.hasMember( m_name.c_str() ) ) {
          LOGGER_WARNING() << "Json object is missing " << m_name << LOGGER_ENDL();
          return false;
        }

        agxJson::Value valueNode = object[m_name];

        // Resolved, the item is found as "valueNode" in the json data.
        // There can still be warnings from the parsing
        m_resolved = true;

        if ( m_type == ARRAY_VEC3 ) {
          // Need to read previous values in case of partial use of default,
          // e.g. [1, "default", 2]
          agx::Vec3 v = m_getVec3( material );

          parseVec3( object[m_name], v );

          m_setVec3( v, material );
        }
        else if ( m_type == SCALAR_REAL) {
          if ( !isDefault(valueNode)) {
            agx::Real r = parseReal( valueNode );
            m_setReal( r, material );
          }
        }
        else if ( m_type == SCALAR_BOOL ) {
          if ( !isDefault(valueNode)) {
            bool b = parseBool( valueNode );
            m_setBool( b, material );
          }
        }
        else {
          LOGGER_WARNING() << "Unknown parameter type for " << getParameterPath() << LOGGER_ENDL();
          return false;
        }

        return true;
      }


      /**
      Writes a real value to a json node
      */
      void writeValueReal( agxJson::Value& node, agx::Real value )
      {
        if ( agx::equivalent(value, agx::Infinity)) {
          node = "inf";
        }
        else if ( agx::equivalent(value, -agx::Infinity)) {
          node = "-inf";
        }
        else {
          node = value;
        }
      }

      /**
      Writes a bool value to a json node
      */
      void writeValueBool( agxJson::Value& node, bool value )
      {
        node = (value ? "true" : "false" );
      }


      /**
      Write a short array, at most 3 elements
      */
      void writeValueVec3( agxJson::Value& node, agx::Vec3 value )
      {
        agxJson::Value::ArrayIndex maxIndex = std::min(m_numValues, 3u);
        node.resize( maxIndex );

        for ( agxJson::Value::ArrayIndex i = 0; i < maxIndex; ++i) {
          node[i] = value[i];
        }
      }


      /**
      Store a generic property to json.
      If group objects are used, then those has to be created in advance.
      */
      bool writeProperty(agxJson::Value& root, const T* material )
      {
        agxJson::Value* outputNode = &root;

        if ( m_group.length() > 0 ) {

          if ( !root.hasMember( m_group ) ) {
            return false;
          }

          outputNode = &root[ m_group ];
        }


        if (m_type == SCALAR_REAL ) {
          writeValueReal( (*outputNode)[ m_name ], m_getReal( material ) );
        }
        else if ( m_type == SCALAR_BOOL ) {
          writeValueBool( (*outputNode)[ m_name ], m_getBool( material ) );
        }
        else if ( m_type == ARRAY_VEC3 ) {
          writeValueVec3( (*outputNode)[ m_name ], m_getVec3( material ) );
        }
        else {
          LOGGER_WARNING() << "Unknown parameter type for " << getParameterPath() << LOGGER_ENDL();
        }

        m_resolved = outputNode->hasMember( m_name );

        return m_resolved;
      }


      /**
      Status flag if parameter could be read/written
      */
      bool isResolved() const
      {
        return m_resolved;
      }

      /**
      Text representation of where in the json structure this parameter lives
      */
      agx::String getParameterPath() const
      {
        if ( m_group.length() > 0 ) {
          return agx::String::format("%s.%s", m_group.c_str(), m_name.c_str());
        }

        return m_name;
      }


    private:
      ParameterType m_type { UNDEFINED_TYPE };
      agx::String   m_name;
      agx::String   m_group;

      unsigned int m_numValues { 0 };

      PropertyGetBool m_getBool;
      PropertySetBool m_setBool;

      PropertyGetReal m_getReal;
      PropertySetReal m_setReal;

      PropertyGetVec3 m_getVec3;
      PropertySetVec3 m_setVec3;

      bool m_resolved { false };
  };

}

