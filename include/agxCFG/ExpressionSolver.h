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

#ifndef AGXCFG_EXPRESSIONSOLVER_H
#define AGXCFG_EXPRESSIONSOLVER_H

#include <agx/macros.h>
DOXYGEN_START_INTERNAL_BLOCK()

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4786) /// Disable warnings about long names

#endif

#include <agx/HashTable.h>
#include <string>
#include <agx/Vector.h>
#include <agx/agxPhysics_export.h>

namespace agxCFG
{

/// Solves mathematical expressions
  class AGXPHYSICS_EXPORT ExpressionSolver
  {

    public:

      /// The resulting type of the expression.
      typedef double ExpressionType; /* Type of numbers to work with */

/// Codes returned from the evaluator
      enum ReturnCode  {E_OK,      /// Successful evaluation
                        E_SYNTAX,  /// Syntax error
                        E_UNBALAN, /// Unbalanced parenthesis
                        E_DIVZERO, /// Attempted division by zero
                        E_UNKNOWN, /// Reference to unknown variable
                        E_MAXVARS, /// Maximum variables exceeded
                        E_BADFUNC, /// Unrecognized function
                        E_NUMARGS, /// Wrong number of arguments to function
                        E_NOARG,   /// Missing an argument to a function
                        E_EMPTY,    /// Empty expression
                        E_INVALID_VALUE /// A value is in an invalid form
                       };

      /// Constructor
      ExpressionSolver();

      // Destructor
      virtual ~ExpressionSolver();

      /*!
        Evaluates the expression given in e. The result is returned in result.
        If the expression was an assignment the *a is set to true (unless the pointer is null)
        \return E_OK (0) if the calculation was successful, otherwise an error code.
      */
      int evaluate( const std::string& e, ExpressionType& result, bool* a = 0 );

      /// Returns the error message if the evaluate method failed.
      std::string getLastError() {
        return m_last_error;
      }

      /*! Specifies if the class should allow assignments or not.
        if assignments is not allowed the setVariable will never be called.
      */
      void enableAssignments( bool flag ) {
        m_enable_assignments = flag;
      }

      /// Removes all variables from symboltable
      void clearAllVars();

    private:
      /// removes the variable named name from the symbol table
      void clearVar( const std::string& name );

      /*! Get the value of a named variable
      \param name - The variable to look up.
      \param value - The value will be put here
      \return true if variable is found otherwise false
      */
      virtual bool getVariableValue( const std::string& name, ExpressionType& value );

      /*! Get the value of a named constant
      \param name - The constant to look up.
      \param value - The value will be put here
      \return true if constant is found otherwise false
      */
      virtual bool getConstantValue( const std::string& name, ExpressionType& value );

      /// Get the value of a named variable/constant
      ExpressionType getValue( const std::string& name );

      /// Store the value in a variable named name
      void setValue( const std::string& name, const ExpressionType& value );

      /// Superlevel parser
      void parse();

      /// Parses 1st level assignments a=expr
      bool level1( ExpressionType& r );

      /// Handles addition and subtractions (a+b, a-b)
      void level2( ExpressionType& r );

      // Multiplication, division and modulo (a*b, a/b, a%b)
      void level3( ExpressionType& r );

      /// Handles power of (a^b)
      void level4( ExpressionType& r );

      /// Handles unary - and + signs: (-a, +a)
      void level5( ExpressionType& r );

      /// Handles literals, variables and function calls
      void level6( ExpressionType& r );

      void allocateMemory( std::size_t size );

#ifdef _WIN32
#define __CDECL         __cdecl
#else
#define __CDECL
#endif

      typedef ExpressionType ( __CDECL* FuncPtr1 )( ExpressionType );
      typedef ExpressionType ( __CDECL* FuncPtr2 )( ExpressionType, ExpressionType );
      typedef ExpressionType ( __CDECL* FuncPtr3 )( ExpressionType, ExpressionType, ExpressionType );
      typedef ExpressionType ( __CDECL* FuncPtr4 )( ExpressionType, ExpressionType, ExpressionType, ExpressionType );

      /// Helper class for storing pointers to predefined functions such as sin, cos, etc...
      class Function
      {
        public:
          Function( int a, FuncPtr1 fp ) : func1( fp ), func2( nullptr ), func3( nullptr ), func4( nullptr ), args( a )  {}
          Function( int a, FuncPtr2 fp ) : func1( nullptr ), func2( fp ), func3( nullptr ), func4( nullptr ), args( a )  {}
          Function( int a, FuncPtr3 fp ) : func1( nullptr ), func2( nullptr ), func3( fp ), func4( nullptr ), args( a ) {}
          Function( int a, FuncPtr4 fp ) : func1( nullptr ), func2( nullptr ), func3( nullptr ), func4( fp ), args( a ) {}
          Function( ) : func1( nullptr ), func2( nullptr ), func3( nullptr ), func4( nullptr ), args( 0 ) {}


          FuncPtr1 func1;                        /* Pointer to function */
          FuncPtr2 func2;                        /* Pointer to function */
          FuncPtr3 func3;                        /* Pointer to function */
          FuncPtr4 func4;                        /* Pointer to function */
          int   args;                          /* Number of arguments to expect */
      };

    private:
      std::string m_last_error;
      unsigned int m_error_pos;

      typedef agx::HashTable<std::string, ExpressionType> VariableMap;
      typedef VariableMap::value_type VariableType;
      typedef VariableMap::iterator VariableMapIterator;
      VariableMap m_constants;

      VariableMap m_variables;

      typedef agx::HashTable<std::string, Function> FunctionMap;
      typedef FunctionMap::iterator FunctionMapIterator;
      typedef FunctionMap::value_type FunctionType;
      FunctionMap m_functions;

      char* m_expression;
      char* m_exp_start;
      char* m_buffer;
      unsigned int m_buffer_size;
      /// Ouch, only static size in this class, the maximum size if a variable name
#define MAX_VARIABLE_LENGTH 80
      char m_token[MAX_VARIABLE_LENGTH + 1];

      /// Specifies the type of a character
      enum CharacterType {
        NOTYPE, /// No type specified yet
        VARIABLE, /// A variable
        DELIMITER,  /// A operator/delimiter
        NUMBER /// A number
      };
      CharacterType m_type;

      agx::Vector<std::string> m_err_messages;

      bool m_enable_assignments;
  };

}
#ifdef _MSC_VER
# pragma warning(pop)
#endif

DOXYGEN_END_INTERNAL_BLOCK()

#endif


