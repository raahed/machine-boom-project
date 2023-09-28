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

#ifndef AGXIO_ARGUMENT_PARSER_H
#define AGXIO_ARGUMENT_PARSER_H

#include <string>
#include <sstream>
#include <agx/HashVector.h>
#include <agx/Referenced.h>
#include <agx/String.h>
#include <iostream>
#include <agx/agxCore_export.h>

namespace agxIO
{

  /**
  A small argument parser class. Copies over the content of all c strings to std::strings.

  There is no differentiation between keys and values, which means that
  if the arguments are "a b", then "a" could be key and "b" value, and/or both
  could be keys (without values).
  */
  class AGXCORE_EXPORT ArgumentParser : public agx::Referenced
  {

    public:
      /**
      Help class for handling arguments.
      Contains the argument name and a bool indicating if the argument has been handled.
      */
      class AGXCORE_EXPORT Argument
      {
        public:
          /// Constructs an empty argument.
          Argument();

          /// Constructs and argument given its name.
          Argument (const std::string& argumentName);

          /// Gets the name of the argument.
          std::string getName() const;

          /// Has the argument been handled?
          bool getHandled() const;

          /// Set if the argument has been handled.
          void setHandled( bool handled);

          /// Is the argument an option (starting with "--")?
          bool isOption() const;


        private:
          agx::String m_argumentName;
          bool m_handled;
      };

      typedef agx::Vector<Argument> ArgumentVector;

    public:

      /// Constructs an argument parser from the program command line input.
      ArgumentParser();

#ifndef SWIG
      /**
      Constructs an argument parser from the program command line input.
      Each white-space separated string is treated as a separate argument,
      so that the number of arguments will correspond to argc.
      The content of argv is copied and stored internally.
      */
      ArgumentParser( int argc, char** argv );
#endif

      /**
      Constructs an argument parser from the program command line input given as an array of strings
      */
      ArgumentParser(const agx::StringVector& arguments);

      /**
      Finds the position of an argument. Returns -1 if not found.
      \note Will not set the argument to 'handled'.
      \param key Argument name
      \param startPosition Position in argument array where to start searching from.
      \param findAlsoHandled Should also arguments been found which have been treated as 'handled'?
      \retval position of argument if found, -1 if not found
      */
      int find( const std::string& key, size_t startPosition = 0, bool findAlsoHandled = true);

      /**
      Looks for an argument, and returns true if it finds it.
      \note Will set the argument to 'handled'.
      \param key Argument name
      \param startPosition Position in argument array where to start searching from.
      \param findAlsoHandled Should also arguments been found which have been treated as 'handled'?
      \retval Returns if the argument was found.
      */
      bool read( const std::string& key, size_t startPosition = 0, bool findAlsoHandled = true );

      /**
      Looks for an argument, and reads it into a variable if found.
      \note Will set the argument to 'handled'.
      \param key Key argument
      \param value Value argument, which has to come directly after the key argument
             in the argument list.
      \param startPosition Position in argument array where to start searching from.
      \param findAlsoHandled Should also arguments been found which have been treated as 'handled'?
      \retval   Were the key and value argument found, and was the conversion successful?
      */
      template <typename T>
      bool read1( const std::string& key, T& value, size_t startPosition = 0, bool findAlsoHandled = true );

      /**
      Reads two argument into variables.
      \note Will set the argument to 'handled'.
      \param key Key argument
      \param value1 Value argument, which has to come directly after the key argument
             in the argument list.
      \param value2 Value argument, which has to come directly after the key argument
             in the argument list.
      \param startPosition Position in argument array where to start searching from.
      \param findAlsoHandled Should also arguments been found which have been treated as 'handled'?
      \retval   Were the key and value argument found, and was the conversion successful?
      */
      template <typename T1, typename T2>
      bool read2( const std::string& key, T1& value1, T2& value2, size_t startPosition = 0, bool findAlsoHandled = true );

      /**
      Reads three argument into variables.
      \note Will set the argument to 'handled'.
      \param key Key argument
      \param value1 Value argument, which has to come directly after the key argument
             in the argument list.
      \param value2 Value argument, which has to come directly after the key argument
             in the argument list.
      \param value3 Value argument, which has to come directly after the key argument
             in the argument list.
      \param startPosition Position in argument array where to start searching from.
      \param findAlsoHandled Should also arguments been found which have been treated as 'handled'?
      \retval   Were the key and value argument found, and was the conversion successful?
      */
      template <typename T1, typename T2, typename T3>
      bool read3( const std::string& key, T1& value1, T2& value2, T3& value3, size_t startPosition = 0, bool findAlsoHandled = true );

      /**
      Adds an argument manually. Not recommended unless you know what you're doing.
      \param argumentName Name of the argument to be added.
      */
      void addArgument( const char* argumentName);

      /**
      Reads an argument into a variable.
      \note Will set the argument to 'handled'.
      \param key Key argument
      \param defaultValue Value to be returned if read is not successful
      \param startPosition Position in argument array where to start searching from.
      \param findAlsoHandled Should also arguments been found which have been treated as 'handled'?
      \retval   The value of the value argument, if the read was successful, otherwise
             the defaultValue.
      */
      template <typename T>
      T defaultRead(const std::string& key, const T& defaultValue, size_t startPosition = 0, bool findAlsoHandled = true);

      /**
      Writes all arguments which have not been handled to a stream.
      \param out An std::ostream that should be written to.
      \return true if all arguments were read. All is ok.
      */
      bool reportUnreadArguments( std::ostream& out = std::cout );

      /// Returns the number of arguments.
      size_t getNumArguments() const;

      /// Gets an argument at a position in the argument container.
      Argument* getArgument( size_t index );

      /**
      Gets the original array of c-strings.
      The content has been copied to an internal data structure.
      */
      char** getArgv();

    protected:
      virtual ~ArgumentParser();

    private:
      // Relies on std streams to cast values in strings to templated type.
      template<typename T>
      static bool assignStringContent (const std::string& key, T& type);

      static bool assignStringContent (const std::string& key, std::string& type);


      /// Hide copy and assignment
      ArgumentParser(const ArgumentParser&);
      ArgumentParser& operator=(const ArgumentParser&);

    private:
      ArgumentVector m_arguments;
      char** m_argv;

  };


// IMPLEMENTATIONS //
  template <typename T>
  bool ArgumentParser::read1(const std::string& key, T& value, size_t startPosition /*= 0*/,
                             bool findAlsoHandled /*= true*/)
  {
    int pos = this->find(key, startPosition, findAlsoHandled);
    if (pos >= 0 && pos + 1 < (int)m_arguments.size()) {
      if (!assignStringContent (m_arguments[pos + 1].getName(), value))
        return false;
      m_arguments[pos].setHandled( true ); // key handled
      m_arguments[pos + 1].setHandled( true ); // value handled
      return true;
    }
    return false;
  }


  template <typename T1, typename T2>
  bool ArgumentParser::read2( const std::string& key, T1& value1, T2& value2, size_t startPosition /*= 0*/,
                              bool findAlsoHandled /*= true*/)
  {
    int pos = find(key, startPosition, findAlsoHandled);
    if (pos >= 0 && size_t(pos + 2) < m_arguments.size()) {
      if (!assignStringContent (m_arguments[pos + 1].getName(), value1) ||
          !assignStringContent (m_arguments[pos + 2].getName(), value2))
        return false;
      m_arguments[pos].setHandled( true ); // key handled
      m_arguments[pos + 1].setHandled( true ); // value1 handled
      m_arguments[pos + 2].setHandled( true ); // value2 handled
      return true;
    }
    return false;
  }


  template <typename T1, typename T2, typename T3>
  bool ArgumentParser::read3( const std::string& key, T1& value1, T2& value2, T3& value3, size_t startPosition /*= 0*/,
                              bool findAlsoHandled /*= true*/)
  {
    int pos = this->find(key, startPosition, findAlsoHandled);
    if (pos >= 0 && pos + 3 < (int)m_arguments.size()) {
      if (!assignStringContent (m_arguments[pos + 1].getName(), value1) ||
          !assignStringContent (m_arguments[pos + 2].getName(), value2) ||
          !assignStringContent (m_arguments[pos + 3].getName(), value3))
        return false;
      m_arguments[pos].setHandled( true ); // key handled
      m_arguments[pos + 1].setHandled( true ); // value1 handled
      m_arguments[pos + 2].setHandled( true ); // value2 handled
      m_arguments[pos + 3].setHandled( true ); // value3 handled
      return true;
    }
    return false;
  }


  template <typename T>
  T ArgumentParser::defaultRead(const std::string& key, const T& defaultValue, size_t startPosition /*= 0*/,
                                bool findAlsoHandled /*= true*/)
  {
    T ret;
    if (this->read1( key, ret, startPosition, findAlsoHandled ))
      return ret;
    return defaultValue;
  }


  template <typename T>
  bool ArgumentParser::assignStringContent (const std::string& key, T& type)
  {
    std::stringstream ss( key );
    ss >> type;
    return (ss.good() || ss.rdstate() == std::ios_base::eofbit);
  }

  template <>
  bool AGXCORE_EXPORT ArgumentParser::assignStringContent<std::string>(const std::string& key, std::string& type);

  template <>
  bool AGXCORE_EXPORT ArgumentParser::assignStringContent<agx::String>(const std::string& key, agx::String& type);

  typedef agx::ref_ptr<ArgumentParser> ArgumentParserRef;
}

#endif

