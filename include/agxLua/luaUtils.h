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

#ifndef AGXLUA_LUAUTILS_H
#define AGXLUA_LUAUTILS_H

#include <agx/config/AGX_USE_LUA.h>
#include <agx/config.h>
#if AGX_USE_LUA()

extern "C" {
  #include <lua.h>
  #include <lualib.h>
  #include <lauxlib.h>
  struct tolua_Error;
}

#include <agx/Referenced.h>
#include <agxLua/export.h>
#include <stdexcept>



/**
The agxLua namespace contain classes for coupling AGX to Lua.
*/
namespace agxLua
{

  class LuaError : public std::runtime_error
  {
  public:
    explicit LuaError( lua_State *L ) : std::runtime_error(""), m_msg("")
    {
      if (lua_isstring(L, -1)) {
        m_msg = lua_tostring(L, -1);
      }
    }

    const char *what() const throw() { return m_msg; }

    explicit LuaError( const char *msg ) : std::runtime_error(msg), m_msg(msg)
    {
    }

  private:
    const char *m_msg;
  };

  AGXLUA_EXPORT bool pushusertype (lua_State *L, void *p, const char *name);

  AGXLUA_EXPORT void ensurepointer (lua_State *L, const void *p);

  AGXLUA_EXPORT void getpointertable (lua_State *L);

  AGXLUA_EXPORT void pushstring (lua_State* L, const char* value);

  AGXLUA_EXPORT const char* tostring (lua_State* L, int narg, const char* def="");

  AGXLUA_EXPORT bool testuserdata (lua_State *L, int index, const char *name);

  AGXLUA_EXPORT int pcall (lua_State *L, int nargs, int nresults, int errfunc);

  AGXLUA_EXPORT void error( lua_State * L);

  AGXLUA_EXPORT int requestPlugin(lua_State *L);

  AGXLUA_EXPORT int doFile(lua_State *L);
  AGXLUA_EXPORT int doString(lua_State *L);


//   template<typename T>
//   T *touserdata (lua_State *L, int index, const char *name) {
//     void *ret = 0;
//     if (!agxLua::testuserdata(L, index, name)) return 0;
//     void **pp = static_cast<void**>(lua_touserdata(L, index));
//     ret = *pp;
//     lua_getfield(L, index, name);
//     ret = (void*)(lua_tointeger(L, -1) + (char*)ret);
//     lua_pop(L, 1);
//     return static_cast<T *>(ret);
//   }


  /**
  This method will return a variable of a specified stack index.
  It will be tested for type (using tolua++) and if it is the correct user type, it will
  be statically cast and returned,
  \param L - Lua state
  \param index - Index of the Lua stack where to find the variable to be queried
  \param type - C++ type of the variable that should be returned.
  \return a pointer to the specified variable if found and of correct type
  */
  AGXLUA_EXPORT void *_tousertype(lua_State *L, int index, const char *type);

  template<typename T>
  T *tousertype (lua_State *L, int index, const char *type) {
    return static_cast<T *>(_tousertype(L, index, type));
  }



  /**
  This method will return a variable of a global named variable
  It will be tested for type (using tolua++) and if it is the correct user type, it will
  be statically cast and returned,
  \param L - Lua state
  \param variableName - Name of a global variable that should be queried
  \param type - C++ type of the variable that should be returned.
  \return a pointer to the specified variable if found and of correct type
  */
  template<typename T>
  T *tousertype (lua_State *L, const char *variableName, const char *type) {
    lua_getglobal(L, variableName);
    if (lua_isnil(L,-1))
      return nullptr;

    return tousertype<T>(L, -1, type);
  }

  AGXLUA_EXPORT std::string getStackTrace( lua_State *L );
  AGXLUA_EXPORT void error(lua_State* L, const char* msg, tolua_Error* err);


  /**
  Function that will take a table of tables and convert it to a agx::RealPairVector
  The table has to follow the format {{1,2},{3,4},...} otherwise an error is generated

  Can be used in Lua as:

  vector = agx.tableToRealPairVector( {{1,2},{3,4},{5,6}}
  The type of vector will be "agx::RealPairVector"
  */
  AGXLUA_EXPORT int tableToRealPairVector(lua_State* L);


  /**
  Function that will take a table of numbers and convert it to a agx::RealVector
  The table has to follow the format {{1,2,3,4,...} otherwise an error is generated

  Can be used in Lua as:

  vector = agx.tableToRealVector( {{1,2,3,4,5,6}
  The type of vector will be "agx::RealVector"
  */
  AGXLUA_EXPORT int tableToRealVector(lua_State* L);

}
#endif

#endif
