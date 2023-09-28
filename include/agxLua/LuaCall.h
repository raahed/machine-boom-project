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

#ifndef AGXLUA_LUACALL_H
#define AGXLUA_LUACALL_H

#include <agx/config/AGX_USE_LUA.h>
#include <agx/config.h>

#if AGX_USE_LUA()
#include <sstream>
#include <cassert>
#include <agxLua/export.h>
#include <agxLua/luaUtils.h>
#include <agxStream/Serializable.h>
#include <agxUnit/UnitMacros.h>
#include <agxStream/ClassInformation.h>


namespace agxLua
{

  /**
    The class encapsulates method calls from C++ to lua.
    The usage of this class enables a call such as:

    return_value = this->method(args...);

    The constructor takes a pointer to an object, this is the "this" pointer which is of a known lua user type.
    The name of the method is given as a string parameter.

    By default the constructor will Throw an exception if the function does not exist.

    IMPORTANT: Make sure you make a call to LuaCall:call() if isValid() returns true
    after the constructor have been called. Otherwise the lua stack would most probably
    be corrupted.

    Example usage:
     MyClass::callLua(MyArgument& arg) {

        LuaCall lc(luaVM, this, "MyClass", "luaMethod", false);
        if (!lc.valid()) {
          cerr << "Invalid call to non existing method";
          return;
        }

        // The argument to the method
        lc.pushUsertype(&arg, "MyArgument");
        lc.call(0); //Make the call, return 0 arguments
      }

      The lua code for this setup would then be:

      mc= MyClass:new()
      function mc:luaMethod(arg)
        print("The call succeeded")
      end

      ma = MyArgument:new()
      mc:callLua(ma) // Called from lua
  */


  /// Class that encapsulates method calls from C++ to lua
  class AGXLUA_EXPORT LuaCall
  {
    public:

      /**
        Constructor.
        \param lua - A pointer to the VM of lua
        \param type - A string that is the type of the this pointer (must be known by lua as usertype)
        \param this_ptr - A void pointer to this->
        \param method - A string containing the name of the method to be called
      */
      LuaCall( lua_State *lua, const std::string& type, void *this_ptr, const std::string& method);

      LuaCall( lua_State *lua, const std::string& functionName );

      LuaCall( lua_State *lua, const agxStream::ClassInformation &classInfo, const void *this_ptr);

      // Pop the stack of this methodcall, but only if it was a valid call!
      ~LuaCall();

      /// Returns true if a call to the method can safely be done ( the method exists in lua)
      bool isValid() {
        return m_valid;
      }

      /// Push an integer as an argument
      void pushInteger ( int value ) {
        m_num_args++;
        lua_pushinteger( m_lua, value );
      }

      /// Push a value as an argument
      void pushValue ( int lo ) {
        m_num_args++;
        lua_pushvalue ( m_lua,  lo );
      }


      /// Push a boolean as an argument
      void pushBoolean ( bool value ) {
        m_num_args++;
        lua_pushboolean ( m_lua, value );
      }

      /// Push a number as an argument
      void pushNumber ( double value ) {
        m_num_args++;
        lua_pushnumber ( m_lua, value );
      }


      /// Push a string as an argument
      void pushString ( const std::string& value ) {
        m_num_args++;
        agxLua::pushstring ( m_lua, value.c_str() );
      }


      /// Push a pointer of userdata as an argument
      bool pushUsertype ( void* value, const char *type ) {
        m_num_args++;
        return agxLua::pushusertype( m_lua, value, type );
      }


      /// Push a pointer to a known usertype
//     void pushUsertype ( void* value, const std::string& type) {
//       m_num_args++;
//       tolua_pushusertype (m_lua, value, type.c_str());
//     }

      /**
        Perform the actual call.
        If the function call is valid (the method exists in lua) a call will be made
        with the pushed arguments.
        \param num_return_values - Specifies the number of arguments that will be returned by the lua method
        \return true - If call was successful, false if failure occurred. getLastError() contain error message.
      */
      bool call( int num_return_values );

      /// Return message string of last error message (from previous call to call() method). "" if no error occurred.
      const std::string& getLastError() const;

      /// Return the current stack trace
      std::string getStackTrace() const;


    private:

      lua_State *m_lua;
      unsigned int m_num_args;
      bool m_valid;
      bool m_isFunction;
      std::string m_errorMessage;
  };



} // namespace agxLua

#define AGXLUA_LUACALL_CLASS_DECLARE( LC ) agxLua::LuaCall LC( m_lua, agxStream::ClassInformation( AGX_FUNCTION ), this );


#define AGXLUA_REPORT_ERROR(errorMessage)                                 \
  if (g_luaScriptManager->getTreatInvalidAsErrors() || agxUnit::isUnittestingEnabled()) \
  {                                                                       \
    LOGGER_ERROR() << (errorMessage) << std::endl << LOGGER_END();        \
  }                                                                       \
  else                                                                    \
  {                                                                       \
    LOGGER_WARNING() << (errorMessage) << std::endl << LOGGER_END();      \
    g_luaScriptManager->setLastErrorMessage(errorMessage);                   \
  }


#define AGXLUA_LUACALL_CLASS_CALL( N )       \
  if (!lc.call(N))                           \
  {                                          \
    AGXLUA_REPORT_ERROR(lc.getLastError())   \
  }


inline std::ostream& operator <<( std::ostream& ostr, const agxStream::ClassInformation &c )
{
  ostr << c.getClassType() << "::" << c.method;
  return ostr;

}
#endif

#endif

