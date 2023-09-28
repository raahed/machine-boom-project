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

#ifndef AGXCFG_CONFIGSCRIPT_H
#define AGXCFG_CONFIGSCRIPT_H


#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif


#include <string>
#include <agx/Vector.h>
#include <agxCFG/ItemMap.h>
#include <agxCFG/Item.h>
#include <agxIO/FileState.h>
#include <agx/Referenced.h>
#include <agx/ref_ptr.h>
#include <agx/agx.h>
#include <agxCFG/Preprocessor.h>
#include <agx/Vec2.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/agx_vector_types.h>
#include <agx/ThreadSynchronization.h>

#include <agxIO/FileState.h>

#include <agx/agxPhysics_export.h>

/// Contains classes for reading the ConfigScript file format, used for configuration, .scene files etc.
namespace agxCFG
{

/// Class for parsing and accessing data in configuration scripts.

  /**
   * Instead of setting parameters for applications explicitly this class helps you
   * by letting you setting all parameters in a configuration script and then parsing the
   * script, and later letting the application accessing the data in the script in a typesafe way.
   * The file format of the script is:

   <key> <item>

   Where <key> can contain alphanumeric letters including _ (underscore).
   <key> is the "hook" for accessing the data <item> later on.
   There can be many <key> with the same name. All Get and Return methods have a index for accessing
   the nth <key>.

  <item> Is describing the data. Valid types are:
  integer, float, string, struct and vector

  integer is obvious, just make sure you think of the type for default parameters.
  float will be treated as agx::Real and can thus be either float or double.

  string is declared as "a string" and must start and end on the same line.
  vector is declared as [float or int ... ] can be as many as you want, but must start and
  end on the same line.

  struct is declared as { <key> <item> ... } and can span over as many lines as you want. Example:

  Sven {
    age 23
    Children {
      child {
        name "john"
        age 3
        sex "male"
      }
      child {
        name "Eva"
        age 2
        sex "female"
      }
    }
  }
  */
  AGX_DECLARE_POINTER_TYPES( ConfigScript );
  class AGXPHYSICS_EXPORT ConfigScript : public agx::Referenced
  {

    public:

      typedef ConfigValue::ValueType Type;

      /**
      * Arguments to ConfigStructPop.
      * ONE - Pop only one level
      * ALL - Pops all levels and empties the stack of pushed structs
      */
      enum StructSelection { ONE = 0, ALL = 1};

      /**
      * Constructor
      * Opens the ConfigScript file filename.
      * Checks that the first line in the file contains header
      * Parses the file for configuration data.
      * Throws and std::runtime_error exception in case of an error
      * \param filename - The file to open
      * \param header - The header of the file that will be opened. This header must match the one in the file
      * \param map_id - Associates an id with the opened configfile for later retrieval with ConfigSetActive
      * \return bool - true if opening and parsing is successful.
      */
      ConfigScript( const std::string& filename, const std::string& header = std::string( "" ) );

      /**
        Uses the stream in data to parse and create items for later access.
        Can be used instead of opening a file and parsing that.
        Will append the data read from the data string to the current map

        \param title - The title of the stream of data to be parsed. Can be used to get better information
        from error messages.
        \param data - A stream of data which is the script to be parsed.
        \param header - The first line in any script to be parsed, including the stream of data.
        \return true if parsing was successful, otherwise false
      */
      bool readString( const std::string& title, const std::string& data, const std::string& header = std::string( "" ) );

      /**
        Constructor
        Uses the parameter item_map to search for data.
        Doesn't parse any files, that is assumed to be done when creating the item_map.
        \param item_map - The Item map containing data that will be parsed
      */
      ConfigScript( agxCFG::StructMap* item_map );

      /**
      * Constructor Does almost nothing. Requires that  Open is called later on.
      */
      ConfigScript();


      /**
      * Open a new empty ConfigScript DB. Clears the previous one. Including closing any files.

      */
      void open( );

      /**
      * Opens the ConfigScript file filename.
      * Checks that the first line in the file contains header
      * Parses the file for configuration data
      * \param filename - The file to open
      * \param header - The header of the file that will be opened. This header must match the one in the file
      * \param map_id - Associates an id with the opened configfile for later retrieval with ConfigSetActive
      * \return bool - true if opening and parsing is successful.
      */
      bool open( const std::string& filename, const std::string& header = std::string( "" ) );

      /**
      If ConfigScript is opened in read-mode, this method returns true if one of the files related
      to the current db has been changed, (in size/modified date).
      \return true if a file related to an opened CFG database has been changed.
      */
      bool hasChanged() const;

      /**
      * Opens the ConfigScript file filename.
      * Checks that the first line in the file contains header
      * Parses the file for configuration data into the already existing itemmap (it if exists).

      If no file was parsed prior to this call, read() == open()

      * \param filename - The file to open
      * \param header - The header of the file that will be opened. This header must match the one in the file
      * \param map_id - Associates an id with the opened configfile for later retrieval with ConfigSetActive
      * \return bool - true if opening and parsing is successful.
      */
      bool read( const std::string& filename, const std::string& header = std::string( "" ) );


      /**
        Uses an existing ItemMap for parsing.


      * \param item_map - The map of keys and data (items) that will be used for this instance of ConfigScript
      * \return bool - always true
      */
      bool open( StructMap* item_map ) {
        close();
        _allocated_item_map = false;
        _item_map = item_map;
        return true;
      }

      /**
        This method takes the existing data and writes it to a named file as a structured
        configscript file.

        \param fileName - Name of the file the data will be written to.
        \param header - The header (first line) that will be written to the file.
        \return bool - true if writing to the file was successful.
      */
      bool write( const std::string& fileName, const std::string& header = std::string( "" ) );


      /**
        This method takes the existing data and writes it to an output stream.

        \param fileName - Name of the file the data will be written to.
        \param header - The header (first line) that will be written to the file.
        \return bool - true if writing to the file was successful.
      */
      bool write( std::ostream& outStream, const std::string& header = std::string( "" ) );

      /**
       * Explicitly closes the script db. If \p deallocateItemMap==true memory will be deallocated
       * \return bool - true if it can be closed (is open).
       */
      bool close( bool deallocateItemMap = true );

      /**
       * \return bool - true if script is opened and parsed with no problems.
      */
      bool isOpen();

      /**
      * Any error message set during a call to any configfunctions can be retrieved with ConfiggetLastError()
      * \return std::string - A message containing the last set error message.
      */
      std::string getLastError( void );


      bool validate( const std::string& schemaFile );
      bool validateString( const std::string& schemaString );
      /**
      * Pops the stack of opened structs
      * \param selection - If == ONE only one level is popped from the stack.
      * If == ALL all pushed structs are popped
      * \return bool - true if stack before the call to ConfigStructPop was not empty otherwise false.
      */
      bool pop( StructSelection selection = ONE );


      /**
      * Pushes a given (in key) struct to the opened stack. This will cause all successive data retrieves to
      * work from the opened struct and beneath.
      * i.e. PushOpen("aaa"); Will work on the struct aaa { ... }
      * \param key - The name of the struct to be opened
      * \param index - If the key exists several times (multiple key) index selects which one to open (1 = first)
      * \return true if the struct key was found.
      */
      bool push( const std::string& key, int index = 1 );


      /**
      * Returns true if the given key exists
      * \param key - The name of the key to search for
      * \param index - If the key exists several times (multiple key) index selects which one to open (1 = first)
      * \return true if the key was found.
      */
      bool exist( std::string key, Type type = ConfigValue::VALUE_ANY, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      * \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, int& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, float& data, int index = 1 );
      bool get( const std::string& key, double& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, std::string&, int index = 1 );
      bool get( const std::string& key, agx::String&, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, agx::RealVector& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, agx::Vec3& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, agx::Vec2& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, agx::Vec4& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      \return false when key is either missing or of wrong type
      */
      bool get(const std::string& key, agx::Vec4f& data, int index = 1);

      /** Overloaded function to retrieve data associated to key read from configfile
          \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, agx::Vector<std::string>& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return false when key is either missing or of wrong type
      */
      bool get( const std::string& key, Expression& data, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return the value if found, otherwise the specified default value will be returned.
      */
      std::string returns( const std::string& key, const std::string& def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return the value if found, otherwise the specified default value will be returned.
      */
      int returns( const std::string& key, int def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      returns the value if found, otherwise the specified default value will be returned.
      */
      double returns( const std::string& key, double def, int index = 1 );
      float returns( const std::string& key, float def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          returns the value if found, otherwise the specified default value will be returned.
      */
      agx::RealVector returns( const std::string& key, const agx::RealVector& def, int index = 1 );


      /** Overloaded function to retrieve data associated to key read from configfile
      returns the value if found, otherwise the specified default value will be returned.
      */
      agx::Vec2 returns( const std::string& key, const agx::Vec2& def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      returns the value if found, otherwise the specified default value will be returned.
      */
      agx::Vec3 returns( const std::string& key, const agx::Vec3& def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      returns the value if found, otherwise the specified default value will be returned.
      */
      agx::Vec4 returns( const std::string& key, const agx::Vec4& def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
      returns the value if found, otherwise the specified default value will be returned.
      */
      agx::Vec4f returns(const std::string& key, const agx::Vec4f& def, int index = 1);

      /** Overloaded function to retrieve data associated to key read from configfile
          \return the value if found, otherwise the specified default value will be returned.
      */
      agx::Vector<std::string> returns( const std::string& key, const agx::Vector<std::string>& def, int index = 1 );

      /** Overloaded function to retrieve data associated to key read from configfile
          \return the value if found, otherwise the specified default value will be returned.
      */
      Expression returns( const std::string& key, const Expression& def, int index = 1 );

      /**
        \return the type of a named key of a given index.
        If the key does not exist, it returns NO_TYPE
      */
      Type getType( const std::string& key, int index = 1 );

      /**
        Add a macro definition to the list of macros. This macro can then be used when parsing
        configscripts and tested for with #ifdef and also used in expressions.
        It has to be called before the Open method, otherwise it is useless, the parsing is already been done.
        It cant be used with the Constructor that calls Open ConfigScript(const std::string& filename)
        because that constructor calls Open.
      */
      void addMacro( const std::string& macro, const std::string& value );

      /**
      Change the value of a specified key.
      \param key - The key to look for
      \param data - new data value
      \return true if value was found, of correct type and changed. Otherwise false.
      */
      bool set( const std::string& key, const int& data, unsigned int nth = 1 );

      /**
      Change the value of a specified key.
      \param key - The key to look for
      \param data - new data value
      \return true if value was found, of correct type and changed. Otherwise false.
      */
      bool set( const std::string& key, const float& data, unsigned int nth = 1 );
      bool set( const std::string& key, const double& data, unsigned int nth = 1 );

      /**
      Change the value of a specified key.
      \param key - The key to look for
      \param data - new data value
      \return true if value was found, of correct type and changed. Otherwise false.
      */
      bool set( const std::string& key, const std::string& data, unsigned int nth = 1 );

      /**
      Change the value of a specified key.
      \param key - The key to look for
      \param data - new data value
      \return true if value was found, of correct type and changed. Otherwise false.
      */
      bool set( const std::string& key, const  agx::RealVector& data, unsigned int nth = 1 );

      /**
      Change the value of a specified key.
      \param key - The key to look for
      \param data - new data value
      \return true if value was found, of correct type and changed. Otherwise false.
      */
      bool set( const std::string& key, const  agx::Vector<std::string>& data, unsigned int nth = 1 );

      /**
      Change the value of a specified key.
      \param key - The key to look for
      \param data - new data value
      \return true if value was found, of correct type and changed. Otherwise false.
      */
      bool set( const std::string& key, const agx::Vec2& data,  unsigned int nth = 1 );
      bool set( const std::string& key, const agx::Vec3& data,  unsigned int nth = 1 );
      bool set( const std::string& key, const agx::Vec4& data,  unsigned int nth = 1 );
      bool set( const std::string& key, const agx::Vec4f& data, unsigned int nth = 1 );

      /**
        Add a key with its associated data (int) to the current scope
        \return the index of the newly added key
      */
      int add( const std::string& key, const int& data );

      /**
        Add a key with its associated data (int) to the current scope
        \return the index of the newly added key
      */
      int add( const std::string& key, const float& data );
      int add( const std::string& key, const double& data );

      /**
        Add a key with its associated data (int) to the current scope
        \return the index of the newly added key
      */
      int add( const std::string& key, const std::string& data);

      /**
        Add a key with its associated data (int) to the current scope
        \return the index of the newly added key
      */
      int add( const std::string& key, const agx::RealVector& data );
      int add( const std::string& key, const agx::Vec2& data );
      int add( const std::string& key, const agx::Vec3& data );
      int add( const std::string& key, const agx::Vec4& data );
      int add( const std::string& key, const agx::Vec4f& data );

      /**
        Add a key with its associated data (int) to the current scope
        \return the index of the newly added key
      */
      int add( const std::string& key, const agx::Vector<std::string>& data );

      /**
        Add a key with its associated data (int) to the current scope
        \return the index of the newly added key
      */
      int add( const std::string& key, const Expression& data );

      /**
        Add an empty struct named key to the map at the current position
        \return the index of the newly added key
      */
      int add( const std::string& key );


      /**
        Remove a named key of the specified type at index idx.
        \param key - Name of key to be removed
        \param type - The type of the key we want to remove
        \param nth - index of key to be removed (default first==1)
        \return true if removal was successful
      */
      bool remove( const std::string& key, Type type = ConfigValue::VALUE_ANY, unsigned int nth = 1 );

      /**
      Remove a named key at index idx no matter which type
      \param key - Name of key to be removed
      \param nth - index of key to be removed
      \return true if removal was successful
      */
      inline bool remove( const std::string& key, unsigned int nth ) {
        return remove( key, ConfigValue::VALUE_ANY, nth );
      }

      /**
        \return the top of the stack of pushed structures.
        This is the scope in which any Return and get Call would operate.
      */
      std::string currentScope();

      /**
        Return the number of occurrences of the named key
        \param key - The key we are looking for
        \param type - The type of the key we are looking for

      */
      unsigned int getNumberOfItems( const std::string& key, agxCFG::ConfigScript::Type type );

      typedef StructMap::const_iterator const_iterator;

      /// \return iterator to the beginning of the map of the current scope
      const_iterator begin() const {
        if ( !_struct_stack.empty() )
          return _struct_stack.top().get()->begin();
        else
          return _item_map->begin();
      }

      /// \return iterator to the end of the map of the current scope
      const_iterator end() const {
        if ( !_struct_stack.empty() )
          return _struct_stack.top().get()->end();
        else
          return _item_map->end();
      }
      const ItemMap* getItemMap() const {
        return _item_map;
      }
      ItemMap* getItemMap() {
        return _item_map;
      }


    protected:

      bool validate( ConfigScript* schema );

      /**
      * Destructor. Protected so we won't be able to create an object on the stack
      */
      virtual ~ConfigScript();



      //typedef std::pair<std::string, StructMapItem *> ScopeStackContent;
    public:

      class ScopeStackContent
      {
        public:
          ScopeStackContent() : data(0) {}
          StructMapItem* get() {
            return data;
          }
          std::string getPath() {
            return path;
          }

        private:
          friend class ConfigScript;
          std::string path;
          StructMapItem* data;

      };

      typedef std::deque<ScopeStackContent> ScopeStack;

      // Class for storing the current Push/Pop scope
      class Scope : private ScopeStack
      {
        public:
          friend class ConfigScript;
          Scope() : ScopeStack() {}
          Scope( const Scope& s ) : ScopeStack() {
            if ( this == &s )
              return;

            *this = s;
          }

          Scope& operator=( const Scope& s ) {
            if ( this == &s )
              return *this;

            this->clear();
            Scope::const_iterator it;
            for ( it = s.begin(); it != s.end(); ++it ) {
              this->push_back( *it );
            }
            return *this;
          }


          //std::ostream &operator <<(std::ostream& stream) {
          //  stream << (this->top().second);
          //}
          std::ostream& operator <<( std::ostream& os )const;

          /// Return the top of the stack
          ScopeStackContent top() {
            return std::deque<ScopeStackContent>::front();
          }

          /// Return the top of the stack
          ScopeStackContent top() const {
            return std::deque<ScopeStackContent>::front();
          }

          /// return true if the stack is empty
          bool empty() const {
            return std::deque<ScopeStackContent>::empty();
          }

        private:
          void push( ScopeStackContent& s ) {
            std::deque<ScopeStackContent>::push_front( s );
          }
          void pop() {
            std::deque<ScopeStackContent>::erase( std::deque<ScopeStackContent>::begin() );
          }

      }; // Scope class

    public:

      /// return the type in readable text
      std::string getTypeString( Type type ) const;
      //std::string getTypeString( ConfigValue::ValueType type ) const;


      /** Return the current Push/Pop scope
        By storing the scope externally it is possible to manipulate it
        and restore it later on.
        \return Scope
      */
      const Scope getScope() const {
        return _struct_stack;
      }

      /**
        Set scope to be the current Push()/Pop() scope.
      */
      void restoreScope( const Scope& scope ) {
        _struct_stack = scope;
      }

      /**
        From a point separated key string (a.b.c.d) this method generates a vector of strings where each
        item in the vector contains the keys i.e. [a, b, c, d].
        \return the size of the vector. 1 means that no . were found at all
      */
      static std::size_t createKeyVector( std::string str, agx::Vector<std::string>& vec );
      static void createString( const agx::Vector<std::string>& vec , std::string& str );

      enum Status { OK, INVALID_MAP_ID, INVALID_KEY, INVALID_TYPE, INVALID_MEMBER, MISSING_KEY, INTERNAL_ERROR, STACK_EMPTY, INVALID_LENGTH };
      Status getStatus() {
        return _status;
      }

    private:

      bool equal( ConfigScript* schema, const std::string& path, ConfigValue* item, int index );

      // No copying available
      ConfigScript( const ConfigScript& ) : Referenced() {}

      // No assignment operator available.
      ConfigScript& operator=( const ConfigScript& ) {
        return *this;
      }

      Status _status;

      bool validateItem( ConfigScript* schema, ConfigValue* item, std::string& path, ConfigValue** currentItem );



      //ConfigValue::ValueType toInternalType( Type type ) const;

      //Type fromInternalType( ConfigValue::ValueType type ) const;

      bool _get( const std::string& key, agx::RealVector& data, int index = 1 );
      template<typename T>
      bool _get( const std::string& key, T& data, int index = 1 );
      //bool _get( const std::string& key, double& data, int index = 1 );


      agxIO::FileStateVector m_fileStates;
      agx::ReentrantMutex m_mutex;

      Preprocessor::MacroMap _macro_map;


      Scope _struct_stack;

      /// Contains the last error message
      std::string _last_error;


      //ItemMap *_item_map;
      StructMap* _item_map;
      bool _allocated_item_map;



      ConfigScript::Status structAccess( const std::string& str, std::string& key, std::string& struct_key,
                                         ConfigValue*& result, unsigned int nth = 1 );

      inline bool check();

      template <class T>
      int getNumberofItems( const std::string& key, T* data ) {
        int idx = 0;

        std::pair<typename T::iterator, typename T::iterator> range;
        range = data->equal_range( key );

        for ( typename T::iterator it = range.first; it != range.second; ++it )
          idx++;

        return idx;
      }

      template <class T>
      int addItem( const std::string& keyName, T* value ) {
        int idx = 0;

        std::string key = keyName;

        agx::Vector<std::string> keys;
        createKeyVector( key, keys );

        std::string add_key = key;
        // Are there a bunch of keys, then it is a path, last is the actual key
        // rest is just structure access
        if ( keys.size() > 0 ) {
          add_key = keys.back();
          keys.pop_back();
        }

        std::string new_key;
        createString( keys, new_key );
        if ( new_key.length() > 1 )
          key = new_key;

        // Are we somewhere deep into the tree of the data?
        if ( _struct_stack.size() || keys.size() > 0 ) {
          // Use the current scope
          ConfigValue* item = nullptr;
          std::string s_key, parent_key;

          if ( _struct_stack.size() && !keys.size() ) {
            item = _struct_stack.top().data;
          } else {
            if ( structAccess( key, s_key, parent_key, item, 1 ) )
              return 0;
          }

          if ( item->type() != ConfigValue::VALUE_STRUCT ) {
            std::ostringstream str;
            str << "Trying to add key " << key << " to a non existing structure";
            str << " Current scope: \n" << currentScope();
            _last_error = str.str();
            return 0;
          }

          //idx = getNumberofItems<StructMap>( key, smptr );

          StructMap* smptr = 0;
          smptr = ( ( StructItem* )item )->getVal();
          smptr->insert( std::make_pair( add_key, value ) );
        } // No we are at the root and we are not accessing a struct
        else {
          idx = getNumberofItems<ItemMap>( key, _item_map );
          _item_map->insert( std::make_pair( key, value ) );
        }
        return idx + 1;
      }


  }; // class ConfigScript

  inline bool ConfigScript::check()
  {
    if ( !_item_map ) {
      _last_error = "configscript db is unitialized. The open method must be executed";
      return false;
    }
    return true;
  }

  /// The constructor does a Push() and the destructor automatically calls Pop()

  /**
    The purpose of this class is to automatically manage a pop of a ConfigScript object
    when the scope of a ScopedPush is ended. In this way an exception can be thrown anywhere
    in the scope, and the pop method will still be managed by this class, not leaving a
    dangling state of a ConfigScript object.
  */
  class AGXPHYSICS_EXPORT ScopedPush
  {

    private:
      template <typename T>
      struct IsADot {
        bool operator()( const T& c1 ) {
          return c1 == '.';
        }
      };

    public:
      /**
        Constructor.
        \param cfg - The ConfigScript object that will be managed
        \param id - The id of the struct that will be pushed
        \param n - The level of the struct (n:th struct)
        \param will_throw - If true this constructor will Throw a runtime_error if the struct is not found.
      */
      ScopedPush( ConfigScript* cfg, const std::string& id, unsigned int n = 1, bool will_throw = false );

      inline bool operator !() const {
        return !valid();
      }

      /// \return true if the constructor managed to successfully push the named struct
      inline bool valid() const {
        return m_pushed;
      }

      /// Destructor, will pop the pushed struct
      virtual ~ScopedPush() {
        if ( m_pushed ) {
          for ( size_t i = 0; i < m_numLevels; i++ )
            m_cfg->pop();
        }
      }

    private:
      agx::ref_ptr<ConfigScript> m_cfg;
      bool m_pushed;
      size_t m_numLevels;


  }; // class ScopedPush


  /// Class that stores the current state of the Scope at construction and restores it at destruction

  /**
    The purpose of this class is to automatically store the current scope of a ConfigScript object
    and to restore it at destruction of this class.
  */
  class AGXPHYSICS_EXPORT ScopeRestore
  {
    public:

      /**
        Constructor.
        \param cfg - The ConfigScript object that will be managed
        \param id - The id of the struct that will be pushed
        \param n - The level of the struct (n:th struct)
        \param will_throw - If true this constructor will Throw a runtime_error if the struct is not found.
      */
      ScopeRestore( ConfigScript* cfg ) : m_cfg( cfg ) {
        m_scope = m_cfg->getScope();
      }

      /// Destructor, will pop the pushed struct
      virtual ~ScopeRestore() {
        m_cfg->restoreScope( m_scope );
      }

    private:

      agx::ref_ptr<ConfigScript> m_cfg;
      ConfigScript::Scope m_scope;
  }; // Class ScopeRestorer


  AGXPHYSICS_EXPORT std::ostream&  operator <<( std::ostream& os, const agxCFG::ConfigScript::Scope& scope );

} //namespace agxCFG

#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif // AGXCFG_CONFIGSCRIPT_H
