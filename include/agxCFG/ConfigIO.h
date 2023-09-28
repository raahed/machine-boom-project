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

#ifndef AGXCFG_CONFIGIO_H
#define AGXCFG_CONFIGIO_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif

#include <agx/macros.h>

DOXYGEN_START_INTERNAL_BLOCK()

#include <agxCFG/utils.h>
#include <agx/agx.h>


#include <fstream>
#include <string>
#include <sstream>
#include <stack>

#include <agxCFG/ConfigValue.h>
#include <agxCFG/ExpressionSolver.h>
#include <agxCFG/ConfigScript.h>
#include <stdexcept>
#include <agxCFG/Preprocessor.h>
#include <agx/Referenced.h>
#include <agx/ref_ptr.h>
#include <agx/agxPhysics_export.h>

namespace agxCFG
{
  class StructMap;

/// A class for indicating parsing error
  class AGXPHYSICS_EXPORT InternalError : public std::exception
  {
    public:
      ///
      InternalError( const std::string& fname, int line_no, const std::string& m ) {
        std::ostringstream msg;
        msg << "InternalError: " << fname << ": " << line_no << ": " << m;
        std::runtime_error( msg.str() );
      }
  };


  class ConfigIO;

  class AGXPHYSICS_EXPORT ConfigExpressionSolver : public ExpressionSolver
  {
    public:
      ConfigExpressionSolver( ConfigIO* cio ) : ExpressionSolver(), m_cio( cio ) {}
      virtual ~ConfigExpressionSolver() {}

      virtual bool getVariableValue( const std::string& name, ExpressionSolver::ExpressionType& value );
      ConfigIO* getCIO() {
        return m_cio;
      }
    private:
      ConfigValue* recursiveSearchValue( StructMap* struct_map, ConfigValue::ValueType type, agx::Vector<std::string>& key_vector, bool up = true );
      ConfigIO* m_cio;
  };

/// Class for reading/writing configuration files
  class AGXPHYSICS_EXPORT ConfigIO
  {

    public:
      friend class ConfigExpressionSolver;

      /// Opens a config file
      ConfigIO( const std::string& filename, const std::string& header = std::string( "" ) );

      /// Constructor
      ConfigIO( void );

      ///
      ~ConfigIO( );

      /// Closes a configuration file
      void close( );

      /// Opens a config file for reading, checks so the first line contains header
      void openRead( const std::string& filename, const std::string& header = std::string( "" ) );

      /// Reads data from the string data instead of a file.
      void readString( const std::string& title, const std::string& data, const std::string& header = "" );

      /// Opens a config file for writing (makes it empty), Writes the header to the first line
      void openWrite( const std::string& filename, const std::string& header = std::string( "" ) );

      void openWrite( std::ostream& outStream, const std::string& header = std::string( "" )  );

      /// Parses the opened file for an item, throws exceptions on error
      ConfigValue* getItem( std::string& key, int start_line );

      /// Writes an item to the config file
      void putItem( const std::string& key, const ConfigValue* value );

      /// Writes a whole item map to the config file
      void putMap( const StructMap& item_map );

      /// Parses the whole file and returns a item map as the result
      void getMap( StructMap& item_map );


      void addTemplate( std::pair<std::string, ConfigValue*> item );
      StructMap* findTemplate( const std::string& key );
      void applyTemplate( ConfigValue* value, const std::string& template_key );

      /// Changes the default comment string "//"
      void setCommentString( std::string cstr ) {
        _comment_string = cstr;
      };

      /**
      Add a macro definition to the preprocessors map of macros. This macro can then be used when parsing
      configscripts and tested for with #ifdef and also used in expressions.
      It has to be called before the OpenRead method, otherwise it is useless, the parsing is already been done.
      It cant be used with the Constructor that calls Open ConfigIO(const std::string& filename, ...
      because that constructor calls OpenRead.
      */
      void addMacro( const std::string& macro, const std::string& value ) {
        _preprocessor.addMacro( macro, value );
      }





      const agxIO::FileStateVector getFileStateVector() {
        return _fileDependentStates;
      }

    private:

      /** Opens a config file for reading, checks so the first line contains header
          also handles the IOState stack and the work queue of IOStates
      */
      void _openRead( const std::string& filename, const std::string& header );

      ConfigValue* getReferencedValue( const std::string& key );


      ///
      StructMap* getItemMap() {
        return _item_map;
      }
      ///
      StructMap* getCurrentScope() {
        return _current_scope;
      }

      agxCFG::ConfigScript* getConfigScript() {
        if ( !_config_script.get() ) _config_script = new agxCFG::ConfigScript();
        _config_script->open( _item_map );
        return _config_script.get();
      }

      class Value
      {
        public:
          enum Type { SINGLE, STRUCT };

          Type type() {
            return type_;
          }

          int start() {
            return start_;
          }
          int end() {
            return end_;
          }

          std::string val() {
            agxAssert( type_ == SINGLE );
            return value_;
          }
          int pos() {
            agxAssert( type_ == STRUCT );
            return pos_;
          }

        protected:
          Value( Type t, int start, int end, std::string str = std::string( "" ), int pos = 0 ) : type_( t ), start_( start ),
            end_( end ), value_( str ), pos_( pos ) {}

        private:

          Type type_;
          int start_, end_;
          std::string value_;
          int pos_;
      };

      class SingleValue : public Value
      {
        public:
          SingleValue( std::string& str, int start, int end ) : Value( SINGLE, start, end, str ) {}
      };

      class StructValue : public Value
      {
        public:
          StructValue( int start, int end, int pos ) : Value( STRUCT, start, end, std::string( "" ), pos ) {}

      };


    public:
      class IOState : public agx::Referenced
      {
        public:

          /// Default constructor
          IOState();

          /**
          Set method.
          Opens filename for reading, throws an IOError exception on error
          \param filename - path to file to open for reading
          */
          void set( const std::string& filename );

          /**
          Constructor
          Uses in_data to initialize a strstream used for reading later on.
          \param title - Title of data, used in error messages
          \param in_data - String containing the data that will be parsed
          */
          IOState( const std::string& title, const std::string& in_data );

          /// Closes any opened files
          void finish() {
            if ( !m_use_sstream ) m_ifstream.close();
          }

          bool haveSStream() const {
            return m_use_sstream;
          }

          void setItemStart( size_t i ) {
            m_item_start = i;
          }
          size_t getItemStart( ) const {
            return m_item_start;
          }
          void setCurrentLine( size_t i ) {
            m_curr_line = i ;
          }


          /// Return the active stream.
          std::istream& getInStream() {
            if ( haveSStream() )
              return m_istrstream;
            else
              return m_ifstream;
          }


          /// Return the include path used when opening included files
          std::string getIncludePath() const {
            return m_include_path;
          }

          void setIncludePath( const std::string& path ) {
            m_include_path = path;
            m_finished = true;
          }

          bool ok() {
            return m_finished;
          }

          size_t incNumLines() {
            m_num_lines++;
            return m_num_lines;
          }
          size_t getNumLines() const {
            return m_num_lines;
          }

          void setFilename( const std::string& filename ) {
            m_filename = filename;
          }
          const std::string& getFilename() const {
            return m_filename;
          }
          const std::string& getTitle() const {
            return m_title;
          }

          void setDBStartline( size_t i ) {
            m_db_start_line = i;
          }


          ///
          std::ostream out;

        protected:

          virtual ~IOState();

        private:

          std::ifstream m_ifstream;
          ///
          std::string m_filename; /// Name of the file
          std::string m_title;

          bool m_finished;
          bool m_use_sstream;

          std::istringstream m_istrstream;

          std::string m_include_path;
          size_t m_num_lines;
          size_t m_db_start_line;
          size_t m_item_start; /// The last item started at line
          size_t m_curr_line; /// Current line

      }; // IOState

      typedef agx::ref_ptr<IOState> IOStateRef;

      struct CurrentKey {
        CurrentKey( const std::string& k, IOState* s, size_t start, size_t curr ) : key ( k ), state( s ), startLineNum( start ), currentLineNum( curr ), m_valid( true ) {}
        CurrentKey() : m_valid( false ) {}
        std::string key;
        IOStateRef state;
        size_t startLineNum, currentLineNum;

        bool m_valid;

        bool valid() const {
          return m_valid;
        }
      };


    private:


      /// Throws an exception if the header does not match
      void checkHeader( void );

      /// Do a preprocessing parsing of the key
      void preprocessKey( const std::string& key );

      /// Reads a line from the stream, removes any non-printable letters
      bool readLine( std::string& line  );

      /// Reads the whole file into the _file_content
      void readFile( void );

      /// Reads a line, removes any non-printable letters
      void putLine( const std::string& line );

      //void BuildLineVector( std::string& line, std::queue<std::string>& line_queue);

      /// Reads a line and removes all comments
      bool readLineNoComments( std::string& line );

      /// Continues to read the file until the delimiter is found
      std::string getLineUntilDelimiter( char delimiter );

      /// Parses a string for values, returns a ConfigValue
      ConfigValue* parseValue( Value* value, ConfigValue::ValueType type );

      /// Parses an array value
      ConfigValue* parseArray( Value* value );

      /// Parses a structure value
      ConfigValue* parseStruct( Value* val );

      ConfigValue* parseExpression( Value* value );


      ConfigValue* getValue( int start_line );


      /** Extracts the value from line
       If the type of the value seems to be a struct then it will
       continue to read and try to find balancing {}
       */
      ConfigValue* extractValue( std::string& line, int start_line );

      //ConfigIO::Value *getString( std::string& line, int start_line, char delimiter );

      Value* getValueBasedOnDelimiter( int start_line, char delimiter );
      ///
      Value* getStruct( int start_line );

      /** Checks so value is of the same type indicated by type
      * If not it trows a char *message
      */
      void validValue( Value* value, ConfigValue::ValueType& type );

      void preProcessLine( std::string& line );

      class ContentDB : public agx::Referenced
      {
        public:
          ContentDB() : last_accessed_( 0 ), m_eof( true ) {}


          class Entry : public agx::Referenced
          {
            public:
              Entry( IOState* ioState, size_t linenum, const std::string& line ) : m_ioState( ioState ), m_line( line ), m_line_num( linenum ) {}
              IOState* getIOState() {
                return m_ioState.get();
              }
              size_t getLineNum() const {
                return m_line_num;
              }
              std::string& getLine() {
                return m_line;
              }
            private:
              Entry() : Referenced() {}
              Entry( const Entry& ) : Referenced() {}
              virtual ~Entry() {}

              IOStateRef m_ioState;
              // int m_fileReference;
              std::string m_line;
              size_t m_line_num;
          };

          typedef agx::Vector<agx::ref_ptr<Entry> > EntryVector;
          EntryVector m_line_db;


          void push_back( IOState* ioState, size_t lineNum, const std::string& line ) {
            m_eof = false;
            m_line_db.push_back( new Entry( ioState, lineNum, line ) );
          }
          AGX_FORCE_INLINE bool next() {
            agxAssert( m_line_db.size() );

            if ( last_accessed_ < ( m_line_db.size() - 1 ) ) {
              last_accessed_++;
              m_eof = false;
            } else  {
              m_eof = true;
            }
            return !m_eof;
          }
          AGX_FORCE_INLINE bool end() {
            return m_eof;
          }

          bool valid( size_t i ) {
            if ( !m_eof && m_line_db.size() ) {
              m_eof = i > m_line_db.size();
              return !m_eof;
            } else { // empty DB
              m_eof = true;
              return false;
            }
          }
          AGX_FORCE_INLINE std::string& refLine() {
            agxAssert( m_line_db.size() );
            agxAssert( last_accessed_ < m_line_db.size() );
            return  m_line_db[last_accessed_]->getLine();
          }

          std::string val() {
            agxAssert( m_line_db.size() );
            return  m_line_db[last_accessed_]->getLine();
          }
          AGX_FORCE_INLINE bool set( unsigned pos ) {
            //agxAssert(content_.size());
            if ( !m_line_db.size() )
              return false;

            if ( pos < m_line_db.size() ) {
              last_accessed_ = pos;
              return true;
            } else
              return false;
          }

          void clear() {
            m_eof = true;
            last_accessed_ = 0;
            m_line_db.clear();
          }
          AGX_FORCE_INLINE std::size_t size() {
            return m_line_db.size();
          }

          AGX_FORCE_INLINE IOState* getIOState() {
            if ( m_line_db.size() ) {
              return m_line_db[last_accessed_]->getIOState();
            } else
              return nullptr;
          }




          typedef agx::Vector<CurrentKey> CurrentKeyStack;
          CurrentKeyStack m_currentKeyStack;


          CurrentKeyStack::reverse_iterator rbeginKeys()  {
            return m_currentKeyStack.rbegin();
          }
          CurrentKeyStack::reverse_iterator rendKey()  {
            return m_currentKeyStack.rend();
          }

          void pushCurrentKey( const std::string& key ) {
            m_currentKeyStack.push_back( CurrentKey( key, m_line_db[last_accessed_]->getIOState(),
                                         m_line_db[last_accessed_]->getIOState()->getItemStart(),
                                         currentLineNum() ) );
          }

          void popCurrentKey(  ) {
            if ( m_currentKeyStack.size() )
              m_currentKeyStack.pop_back( );
          }

          CurrentKey getCurrentKey() {
            CurrentKey key;

            if ( m_currentKeyStack.size() )
              key = m_currentKeyStack.back();

            return key;
          }


          size_t currentLineNum() {
            if ( m_line_db.size() ) return m_line_db[last_accessed_]->getLineNum();
            else return 0;
          }
          std::string currentFileName() {
            if ( m_line_db.size() ) return m_line_db[last_accessed_]->getIOState()->getFilename();
            else return "";
          }

          unsigned int lastAccessed() const {
            return last_accessed_;
          }

        protected:
          virtual ~ContentDB();

        private:

          unsigned int last_accessed_;

          bool m_eof;
      }; // ContentDB



    public:


      ContentDB* getContentDB() {
        return m_contentDB.get();
      }

      Preprocessor& getPreprocessor() {
        return _preprocessor;
      }

    private:

      agx::ref_ptr<ContentDB> m_contentDB;
      typedef agx::Vector<IOStateRef> IOStateVector_t;

      class IOStateStack
      {
        public:
          virtual ~IOStateStack();

          void pop();
          IOState* top() {
            if ( size() ) return *( m_vec.end() - 1 ); //[size()-1];
#ifdef AGX_DEBUG
            else agxThrow InternalError( __FILE__, __LINE__, "Getting top of empty stack, serious stuff" );
#else
            else agxThrow InternalError( "", __LINE__, "Getting top of empty stack, serious stuff" );
#endif
            //return nullptr; // Satisfies the compiler
          }

          void push( IOState* s ) {
            m_vec.push_back( s );
          }
          size_t size() {
            return m_vec.size();
          }
          std::string getIncludePath();
        private:
          IOStateVector_t m_vec;

      }; // IOStateStack


      IOStateStack _iostate_stack;


      agxIO::FileStateVector _fileDependentStates;


      AGX_FORCE_INLINE IOState* currIOState() {
        return _iostate_stack.top();
      }

      IOStateRef _out_state;
      std::ofstream m_outStream;

      void popIOState() {
        _iostate_stack.pop();
      }
      void pushIOState( IOState* s ) {
        _iostate_stack.push( s );
        _fileDependentStates.push_back( s->getFilename() );
      }

      IOState* topIOState() {
        return _iostate_stack.top();
      }
      size_t sizeIOStateStack() {
        return _iostate_stack.size();
      }


      std::string getIncludePath() {
        return _iostate_stack.getIncludePath();
      }


      /// The string used for comments
      std::string _comment_string;

      /// The maximum number of lines a configfile can contain (just for precaution)
      const unsigned int MAX_LINES;

      bool _have_header;

      StructMap* _item_map;
      StructMap* _current_scope;

      /// The maximum number of included files (just to avoid recursive includes)
      const unsigned int MAX_INCLUDE_DEPTH;

      std::string _header;


      agx::ref_ptr<agxCFG::ConfigScript> _config_script;
      agx::ref_ptr<agxCFG::ConfigScript> _templates;

      Preprocessor _preprocessor;

      bool m_recursiveStop;

#define CONFIGSCRIPT_FILE_PATH "CFG_FILE_PATH"

    private:
      ConfigIO& operator =( const ConfigIO& ) {
        return *this;
      }

  }; // ConfigIO

  std::ostream&  operator<<( std::ostream& os, const ConfigIO::IOState* s );


/// A class for indicating an io error (EOF)
  class AGXPHYSICS_EXPORT IOError : public std::exception
  {
    public:
      ///
      IOError( const ConfigIO::IOState* s, const std::string& m ) throw();

      const char* what() const throw() {
        return m_msg.c_str();
      }
      ~IOError() throw() {}

    private:
      std::string m_msg;

  };

/// A class for indicating parsing error
  class AGXPHYSICS_EXPORT ParseError : public std::exception
  {
    public:
      ///
      ParseError( const ConfigIO::CurrentKey& key, const std::string& m ) throw() {
        std::ostringstream msg;
        msg << "ParseError: " << m;
        if ( key.valid() ) {
          msg << " Key: " << key.key;
          msg << " (" << key.currentLineNum << ") " << key.startLineNum << " File: " << key.state->getFilename();
        }
        m_msg = msg.str();
        //Throw std::runtime_error( msg.str() );
      }

      ParseError( const std::string& filename, const std::string& m ) throw() {
        std::ostringstream msg;
        msg << "ParseError: " << m << " File: " << filename;
        m_msg = msg.str();
        //Throw std::runtime_error( msg.str() );
      }

      const char* what() const throw() {
        return m_msg.c_str();
      }

      ~ParseError() throw() {}
    private:
      std::string m_msg;
  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

DOXYGEN_END_INTERNAL_BLOCK()

#endif
