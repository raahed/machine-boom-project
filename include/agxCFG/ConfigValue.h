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

#ifndef AGXCG_CONFIGVALUE_H
#define AGXCG_CONFIGVALUE_H

#include <agx/macros.h>
DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif



#include <string>
#include <iostream>
#include <agx/agxPhysics_export.h>
#include <agx/agx.h>
#include <agx/Referenced.h>

namespace agxCFG
{


/// Base class for Items stored in setup file
  class AGXPHYSICS_EXPORT ConfigValue : public agx::Referenced
  {
    public:

      /// Defines the type of the item
      enum ValueType { VALUE_NONE = 0x0,
                       VALUE_INT = 0x1,
                       VALUE_FLOAT = 0x2,
                       VALUE_STRING = 0x4,
                       VALUE_FLOAT_ARRAY = 0x8,
                       VALUE_STRUCT = 0x10,
                       VALUE_EXPRESSION = 0x20,
                       VALUE_STRING_ARRAY = 0x40,
                       VALUE_NUMBER = VALUE_INT | VALUE_FLOAT | VALUE_EXPRESSION,
                       VALUE_ANY = VALUE_NUMBER | VALUE_STRING | VALUE_FLOAT_ARRAY | VALUE_STRING_ARRAY
                     };


      ConfigValue( ValueType type ) : m_type( type ), m_isTemplate( false ), m_isLocal(false) {}

      virtual ConfigValue* copy( void ) {
        throw "ConfigValue::Copy() Never get here";
      };

      ///

      void setTemplate( bool flag ) {
        m_isTemplate = flag;
      }
      bool isTemplate( ) const {
        return m_isTemplate;
      }

      void setLocal( bool flag ) {
        m_isLocal = flag;
      }
      bool isLocal( ) const {
        return m_isLocal;
      }

      /// Recalculates value if it is an expression (or a struct containing an expression
      virtual void update() {}
      ///
      virtual std::ostream& put( std::ostream& os )const {
        os << "Never get here" << std::endl;
        return os;
      }

      ///
      friend std::ostream& operator <<( std::ostream& os, const ConfigValue& vo ) {
        vo.put( os );
        return os;
      }

      bool getLiteralValue( double& value );
      bool getLiteralValue( float& value );
      bool getStringValue( std::string& value );

      /// Returns the type of the stored item
      ValueType type( void ) const {
        return m_type;
      };

      virtual const char* description() const = 0;

      virtual ~ConfigValue() {}
    protected:


      ///
      ValueType m_type;
      bool m_isTemplate;
      bool m_isLocal;

  };

  typedef agx::observer_ptr<ConfigValue> ConfigValueObserver;
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

DOXYGEN_END_INTERNAL_BLOCK()

#endif


