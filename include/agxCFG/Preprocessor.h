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

#ifndef AGXCFG_Preprocessor_h
#define AGXCFG_Preprocessor_h

#include <agx/macros.h>
DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) // Disable warnings about long names
#endif

#include <string>
#include <agx/HashTable.h>
#include <agx/Vector.h>
#include <stack>

#include <agx/agxPhysics_export.h>

namespace agxCFG
{

/// Class for handling preprocessor directives in a config script
  /*!
  */
  class AGXPHYSICS_EXPORT Preprocessor
  {
    public:
      /// Constructor
      Preprocessor();

      // Destructor
      ~Preprocessor() {}

      /// Processes a line
      void processLine( std::string& line );

      /// returns true if there is currently a balance between #ifdef and #endif
      bool checkMatch() const ;

      /*! Adds the macro with its associated value to the existing map of macros
          Replaces any existing occurrences of macro
      */
      void addMacro( const std::string& macro, const std::string& value );

      /*! \return the found include path during the last call to processLine.
          returns "" if none were found the last time processLine were called.
      */
      std::string getIncludePath() const  {
        return _include_path;
      }

      typedef agx::HashTable<std::string, std::string> MacroMap;
      typedef MacroMap::const_iterator MacroMapIterator;
      typedef MacroMap::value_type MacroMapType;


      MacroMap& getMacroMap() {
        return _macro_map;
      }

    private:

      /// Replaces any matching macros with its associated values
      void replacePhase( std::string& line );

      /*! Extract any macro definitions from line and put them into map of macros.
          If two macros with same name is added, it is always the last that is stored.
      */
      void extractMacroPhase( std::string& line );

      void undefMacroPhase( std::string& line );
      /*!
        Checks for existence of INCLUDE. If found it extracts the path and stores it in _include_path
        This can later be checked with getIncludePath. _include_path is "":ed during each call to processLine
      */
      void includePhase( std::string& line );

      /// Performs a logical ifdef else endif check to remove lines that are in a "false" area
      void logicIfPhase( std::string& line );

      /// Specifies the allowed preprocessor keys
      enum PreprocessorDirectives { UNDEF, DEFINE, INCLUDE, PRAGMA, IF, IFDEF, ELSE, ENDIF, ELSEIF, ALWAYS_LAST };

      /// returns true if Preprocessor directive dir exists in line
      bool preprocessorDirExist( PreprocessorDirectives dir, std::string& line ) const ;

      agx::Vector<std::string> _preprocessor_directives;
      unsigned int MAX_LOOPS;


      MacroMap _macro_map;

      std::string _include_path;

      typedef std::stack< std::pair< bool, unsigned short > > IfStateStack_t;
      typedef IfStateStack_t::value_type IfState;


      /// Stores the nested if:s according to its nested level and its state (true or not)
      class IfStateStack
      {
        public:
          /// Pushes a state to the stack
          void push( IfState s ) {
            m_stack.push( s );
          }
          /// Pops a state from the stack
          void pop( void ) {
            m_stack.pop();
          }

          /// Returns the last state
          IfState top( void ) const {
            if ( size() ) return m_stack.top();
            else return IfState( true, IfState::second_type( 0 ) );
          }

          /// Inverts the state of the last state
          void invert() {
            if ( size() ) {
              IfState s = m_stack.top();
              pop();
              push( IfState( !s.first, s.second ) );
            }
          }

          /// Returns the size of the stack
          size_t size() const {
            return m_stack.size();
          }
        private:
          IfStateStack_t m_stack;
      };

      IfStateStack _if_stack;

      int _line_no;
      int _if_start;

  };

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif

DOXYGEN_END_INTERNAL_BLOCK()

