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

#ifndef AGXCFG_ITEM_H
#define AGXCFG_ITEM_H

#include <agx/macros.h>
DOXYGEN_START_INTERNAL_BLOCK()

# include <agxCFG/ItemMap.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif


#include <string>
#include <agx/agx.h>
#include <agx/agx_vector_types.h>
#include <agx/Vector.h>
#include <agxCFG/ConfigValue.h>
#include <agxCFG/utils.h>
#include <sstream>


#include <agxCFG/ExpressionSolver.h>
#include <agx/agxPhysics_export.h>
#include <agxCFG/ItemMap.h>


namespace agxCFG
{


  extern const int s_indentationLevel;


  struct  SeparatorRepeat {
    SeparatorRepeat( const char* separator = " ", int level = 0 ) : m_numSeparators( level ), m_pSeparator( separator ) {}
    int m_numSeparators;
    const char* m_pSeparator;
  };



  inline  std::ostream& operator<<( std::ostream& os, const SeparatorRepeat& sep )
  {
    int n = 2 * sep.m_numSeparators;
    for ( int i = 0; i < n; i++ )
      os << sep.m_pSeparator;

    return os;
  }

  class Indentation : public SeparatorRepeat
  {
    public:
      Indentation( int level = 0 ) : SeparatorRepeat( " ", level )  { }

      AGX_FORCE_INLINE void set( int level )  {
        m_numSeparators = level;
      }
      AGX_FORCE_INLINE operator int() {
        return m_numSeparators;
      }

      std::ostream& operator<<( std::ostream& out ) {
        out << *this;
        return out;
      }

      AGX_FORCE_INLINE Indentation& operator --( int ) {
        if ( m_numSeparators > 0 )
          m_numSeparators--;
        return *this;
      }

      AGX_FORCE_INLINE Indentation& operator ++( int ) {
        m_numSeparators++;
        return *this;
      }
  };


  typedef agx::RealVector FloatVector;

  class StringItem;
  typedef agx::Vector<StringItem> StringVector;

  class StructMap;
  class Array;
  class Expression;


///
  class AGXPHYSICS_EXPORT Array /*: public Vector<ConfigValue *> */
  {
    public:
      typedef agx::Vector<ConfigValueObserver> ConfigVector;
      typedef ConfigVector::const_iterator const_iterator;
      typedef ConfigVector::iterator iterator;
      typedef ConfigVector::value_type value_type;
    public:
      Array() {}
      Array( const FloatVector& vec );
      Array( const agx::Vector<std::string>& vec );

      Array( const Array& copy ) {
        if ( &copy == this )
          return;

        *this = copy;
      }

      Array& operator=( const Array& copy );
      Array& operator=( const FloatVector& copy );
      Array& operator=( const agx::Vector<std::string>& copy );

      AGX_FORCE_INLINE iterator begin() {
        return m_vec.begin();
      }
      AGX_FORCE_INLINE iterator end() {
        return m_vec.end();
      }

      AGX_FORCE_INLINE const_iterator begin() const {
        return m_vec.begin();
      }
      AGX_FORCE_INLINE const_iterator end() const {
        return m_vec.end();
      }

      void clear( bool delete_flag = false );

      AGX_FORCE_INLINE size_t size() {
        return m_vec.size();
      }
      AGX_FORCE_INLINE iterator erase( iterator it ) {
        return m_vec.erase( it );
      }

      AGX_FORCE_INLINE void push_back( value_type v ) {
        m_vec.push_back( v );
      }

      AGX_FORCE_INLINE void update() {}

      ~Array();
      agx::Real operator[]( unsigned int i );

      void ith( unsigned int i, std::string& s_val );
//  void push_back(ConfigValue *v) { m_vec.push_back(v); }
//  unsigned int size() {return m_vec.size(); }
      void getFloatVector( agx::RealVector& vec );
      void getStringVector( agx::Vector<std::string>& vec );
    private:
      ConfigVector m_vec;
  };
///





  class AGXPHYSICS_EXPORT Expression
  {
    public:
      Expression( const std::string& expression, const ExpressionSolver::ExpressionType result ) :
        m_expression( expression ), m_result( result ), m_valid( true ) {}


      Expression( const std::string& expression );

      Expression() : m_expression( "" ), m_result( 0 ), m_valid( true ) {}

      std::string getExpression() const {
        return m_expression;
      }

      bool isValid() const {
        return m_valid;
      }

      ExpressionSolver::ExpressionType getVal() const {
        return m_result;
      }

      // recursively recalculates all sub elements that are expressions
      virtual void update() {}
      virtual ~Expression() {}

    private:
      std::string m_expression;
      ExpressionSolver::ExpressionType m_result;
      bool m_valid;
  };


/// Class for storing items, derived from ConfigValue
  template<class T> class Item : public ConfigValue
  {
    public:

      /// Constructor, sets the value and the type of the Item
      Item( T val, ValueType type ) : ConfigValue( type ) {
        _val = val;
      };

      /// Writes an item to ostream
      virtual std::ostream& put ( std::ostream& os ) const {/* os.flags(std::ios_base::showpoint); */
        os << _val;
        return os;
      };

      virtual ConfigValue* copy( void ) {
        std::cerr << "Never get here" << std::endl;
        return 0;
      }; //std::cerr << _val << std::endl; Item<T> *ptr = new Item<T>(_val, type()); return ptr; };

      /// Returns a reference to the stored value
      AGX_FORCE_INLINE T& getVal( void )  {
        return _val;
      };

      AGX_FORCE_INLINE bool getVal( int& i ) {
        if ( type() == ConfigValue::VALUE_INT )  {
          i = ( int )_val;
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( double& i ) {
        if ( type() == ConfigValue::VALUE_FLOAT )  {
          i = ( double )_val;
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( float& i ) {
        if ( type() == ConfigValue::VALUE_FLOAT )  {
          i = ( float )_val;
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( std::string& i ) {
        std::ostringstream str;
        if ( type() == ConfigValue::VALUE_STRING )  {
          str << static_cast<std::string>(_val).c_str();
          i = str.str();
          return true;
        } else
          return false;
      }


      AGX_FORCE_INLINE bool getVal( Array& i ) {
        if ( type() == ConfigValue::VALUE_FLOAT_ARRAY || type() == ConfigValue::VALUE_STRING_ARRAY )  {
          i = ( Array )_val;
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( agx::RealVector& i ) {
        if ( type() == ConfigValue::VALUE_FLOAT_ARRAY )  {
          ( ( Array* )( &_val ) )->getFloatVector( i );
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( agx::Vector<std::string>& i ) {
        if ( type() == ConfigValue::VALUE_STRING_ARRAY )  {
          ( ( Array* )( &_val ) )->getStringVector( i );
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( StructMap*& i ) {
        if ( type() == ConfigValue::VALUE_STRUCT )  {
          i = ( StructMap* )_val;
          return true;
        } else
          return false;
      }

      AGX_FORCE_INLINE bool getVal( Expression& i ) {
        if ( type() == ConfigValue::VALUE_EXPRESSION )  {
          i = ( Expression )_val;
          return true;
        } else
          return false;
      }

    protected:

      ///
      AGX_FORCE_INLINE virtual ~Item() {}

      T _val;
  };

  class ExpressionItem : public Item<Expression>
  {
    public:
      ExpressionItem( const Expression& e ) : Item<Expression>( e, ConfigValue::VALUE_EXPRESSION ) {}

      // recursively recalculates all sub elements that are expressions
      virtual void update() {}

      ConfigValue* copy( void ) {
        ExpressionItem* ptr = new ExpressionItem ( _val );
        return ptr;
      }

      const char* description() const {
        return "Expression";
      }

    protected:
      virtual ~ExpressionItem() {}

  };

///
  class IntItem : public Item<int>
  {
    public:
      ///
      IntItem( int i ) : Item<int>( i, ConfigValue::VALUE_INT ) {}
      AGX_FORCE_INLINE ConfigValue* copy( void ) { /*std::cerr << _val << std::endl; */
        IntItem* ptr = new IntItem( _val );
        return ptr;
      };
      const char* description() const {
        return "integer";
      };

    protected:
      virtual ~IntItem() {}


  };

  ///
  class FloatItem : public Item<agx::Real>
  {
    public:
      ///
      AGX_FORCE_INLINE FloatItem( agx::Real i ) : Item<agx::Real>( i, ConfigValue::VALUE_FLOAT ) {}

      ConfigValue* copy( void ) { /*std::cerr << _val << std::endl; */
        FloatItem* ptr = new FloatItem( _val );
        return ptr;
      };

      AGX_FORCE_INLINE std::ostream& put ( std::ostream& os ) const {
        return agxCFG::streamReal( os, _val);
      }

      const char* description() const {
        return "float";
      }

    protected:
      AGX_FORCE_INLINE ~FloatItem() {}

  };

///
  class StringItem : public Item<std::string>
  {
    public:
      ///
      StringItem( std::string i = "" ) : Item<std::string>( i, ConfigValue::VALUE_STRING ) {}


      AGX_FORCE_INLINE ConfigValue* copy( void ) { /*std::cerr << _val.c_str() << std::endl; */
        StringItem* ptr = new StringItem( _val );
        return ptr;
      }

      AGX_FORCE_INLINE std::ostream& put ( std::ostream& os ) const {
        os << "\"" << _val.c_str() << "\"";
        return os;
      }

      const char* description() const {
        return "string";
      }


    protected:
      virtual ~StringItem() {}

  };



  class FloatVectorItem : public Item<Array>
  {
    public:
      FloatVectorItem( const Array& i, ConfigValue::ValueType type = ConfigValue::VALUE_FLOAT_ARRAY ) : Item<Array>( i, type ) {}
      FloatVectorItem( const FloatVector& vec ) : Item<Array>( vec, ConfigValue::VALUE_FLOAT_ARRAY ) {}
      FloatVectorItem( const agx::Vector<std::string>& vec ) : Item<Array>( vec, ConfigValue::VALUE_STRING_ARRAY ) {}

      ConfigValue* copy( void ) {
        FloatVectorItem* ptr = new FloatVectorItem( _val );
        return ptr;
      }

      const char* description() const {
        return "array";
      };

      virtual void update() {}


    protected:
      virtual ~FloatVectorItem() {}

  };


  typedef std::multimap<std::string, ConfigValueObserver>::iterator StructMapIterator;

  ///
  class StructMap : public ItemMap
  {
    public:

      StructMap() : m_parent( nullptr ), m_deleted( false ) {}

      ConfigValue* copy( void );

      ConfigValue* findConfigValue( const std::string& key ) {
        StructMapIterator smi;
        smi = this->find( key );
        if ( smi != this->end() )
          return ( smi->second );

        return nullptr;
      }

      /**
      Replace the key with name \p s with value.
      \param typeCheck - If true, then replace will only happen if:
      1. key does not exist or
      2. key is of same type as value
      If neither of the above is true, this method will return end()
      */
      StructMapIterator replace( const std::string& s, ConfigValue* value, bool typeCheck = false );
      StructMapIterator typedInsert(
        const std::string& s, ConfigValue* value, bool onlyIfNotExist = false, bool typeCheck = false);


      bool getVal( const std::string& key, FloatVector& result );
      bool getVal( const std::string& key, agx::Vector<std::string>& result );

      bool getVal( const std::string& key, Array& result );

      bool getVal( const std::string& key, std::string& result );

      bool getVal( const std::string& key, int& result );

      bool getVal( const std::string& key, double& result );
      bool getVal( const std::string& key, float& result );

      bool getVal( const std::string& key, StructMap*& result );
      bool getVal( const std::string& key, ConfigValue*& result );

      // recursively recalculates all sub elements that are expressions
      void update() {}

      StructMap* getParent() const {
        return m_parent;
      }
      void setParent( StructMap* sm ) {
        m_parent = sm;
      }

      /**
      Tries to find the item named name. name must be a float, integer or an expression item
      First the current scope is searched (within this StructItem), if it is not found, it will be searched for
      in the parents scope, etc, until the top scope is reached.
      \return true if variable name is found, otherwise false
      */
      bool getScopeVariable( const std::string& name, agx::Real& value );

      /**
      This method will take the content in StructItem snd and insert it into this StructMap.
      \param typeHandling - If typeHandling is true then this method will fail and report false if a
      key in snd of type A has the same name as a key in this StructMap which is of a different type.
      If typeHandling is false, then the key/value in snd is silently ignored.
      */
      bool merge( const StructMap* from, bool typeHandling = false );

      ///
      virtual ~StructMap();

      bool isDeleted() const {
        return m_deleted;
      }
    protected:


    private:
      StructMap* m_parent;
      bool m_deleted;


  };

  typedef agx::observer_ptr<StructMap> StructMapObserver;

  typedef Item<StructMapObserver> StructItem;

  ///
  class StructMapItem : public Item<StructMapObserver>
  {
    public:
      ///
      StructMapItem( StructMap* i ) : StructItem( i, ConfigValue::VALUE_STRUCT ) {}


      inline StructMap::iterator begin() {
        return _val->begin();
      }

      inline StructMap::iterator end() {
        return _val->end();
      }

      // recursively recalculates all sub elements that are expressions
      virtual void update() {}

      const char* description() const {
        return "struct";
      };

    protected:
      virtual ~StructMapItem ();

  };

  inline bool StructMap::getVal( const std::string& key, StructMap*& result )
  {

    bool flag = false;
    StructMapIterator smi;
    smi = find( key );
    if ( smi != end() ) {
      flag = ( ( StructItem* )smi->second.get() )->getVal( result );
    }

    return flag;
  }

  inline bool StructMap::getVal( const std::string& key, ConfigValue*& result )
  {

    bool flag = false;
    StructMapIterator smi;
    smi = find( key );
    if ( smi != end() ) {
      result = smi->second;
      return true;
    }

    return flag;
  }

/// Prints a FloatVector
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, const agxCFG::FloatVector& v );

/// Prints a StringVector
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, const agxCFG::StringVector& v );

/// Prints a Array
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, agxCFG::Array* );

/// Prints a StructMap
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, agxCFG::StructMap* );

/// Prints an Expression
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, agxCFG::Expression* );

/// Prints an Expression
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, agxCFG::Expression e );

/// Prints a Array
  AGXPHYSICS_EXPORT std::ostream&   operator <<( std::ostream& os, const Array& v );

}
DOXYGEN_END_INTERNAL_BLOCK()

#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif
