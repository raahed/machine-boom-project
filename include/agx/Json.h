/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

/*
This source code has been taken and modified by Algoryx Simulation AB
from the source and under the license given below.
*/

#ifndef AGX_JSON_H
#define AGX_JSON_H

#include <agx/debug.h>

#include <agx/PushDisableWarnings.h>
#include <external/json/json.h>
#include <agx/PopDisableWarnings.h>

#define agxJsonVerify(x, element)                                                                                                                                    \
if (!(x))                                                                                                                                                            \
{                                                                                                                                                                    \
  std::stringstream __stream;                                                                                                                                          \
  __stream << agx::buildErrorString("[%s:%u] agxVerify failed: `%s'", "", __FILE__, __LINE__, #x) << std::endl;                                                            \
  __stream << "==============================================================================" << std::endl;                                                           \
  __stream << agxJson::StyledWriter().write(element);                                                                                                                  \
  __stream << "==============================================================================" << std::endl;                                                           \
  const std::string& __string = __stream.str();                                                                                                                          \
  std::cerr << __string << std::endl;                                                                                                                                  \
  agxThrow std::runtime_error(__string);                                                                                                                                  \
}

#define agxJsonVerify1(x, element, msg)                                                                                                                              \
if (!(x))                                                                                                                                                            \
{                                                                                                                                                                    \
  std::stringstream __stream;                                                                                                                                          \
  __stream << agx::buildErrorString("[%s:%u] agxVerify failed: `%s', ", "%s", __FILE__, __LINE__, #x, msg) << std::endl;                                 \
  __stream << "==============================================================================" << std::endl;                                                           \
  __stream << agxJson::StyledWriter().write(element);                                                                                                                  \
  __stream << "==============================================================================" << std::endl;                                                           \
  const std::string& __string = __stream.str();                                                                                                                          \
  std::cerr << __string << std::endl;                                                                                                                                  \
  agxThrow std::runtime_error(__string);                                                                                                                                  \
}

#define agxJsonVerifyN(x, element, format, ...)                                                                                                                      \
if (!(x))                                                                                                                                                            \
{                                                                                                                                                                    \
  std::stringstream __stream;                                                                                                                                          \
  __stream << agx::buildErrorString("[%s:%u] agxVerify failed: `%s', ", format, __FILE__, __LINE__, #x, ##__VA_ARGS__) << std::endl;                               \
  __stream << "==============================================================================" << std::endl;                                                           \
  __stream << agxJson::StyledWriter().write(element);                                                                                                                  \
  __stream << "==============================================================================" << std::endl;                                                           \
  const std::string& __string = __stream.str();                                                                                                                          \
  std::cerr << __string << std::endl;                                                                                                                                  \
  agxThrow std::runtime_error(__string);                                                                                                                                  \
}


namespace agx
{

}

#endif /* AGX_JSON_H */
