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

#ifndef AGX_XML_H
#define AGX_XML_H

#include <agx/String.h>
#include <agx/Name.h>
#include <agx/debug.h>
#include <external/tinyxml/tinyxml.h>




#define agxXmlVerify(x, element)                                                                                 \
if (!(x))                                                                                                        \
{                                                                                                                \
  if (!element)                                                                                                                \
    agxThrow std::runtime_error(std::string("Error in xml-check: Both x and element are nullptr."));                                                                              \
  std::stringstream __stream;                                                                                      \
  __stream << "[" << __FILE__ << ":" << __LINE__ << "] XML error at row " << (element)->Row() << std::endl;          \
  __stream << "File: " << (element)->GetDocument()->Value() << std::endl;                                          \
  __stream << "==============================================================================" << std::endl;       \
  (element)->Print(__stream, 0);                                                                                     \
  __stream << std::endl;                                                                                           \
  __stream << "==============================================================================" << std::endl;       \
  const std::string& __string = __stream.str();                                                                      \
  std::cerr << __string << std::endl;                                                                              \
  agxThrow std::runtime_error(__string);                                                                              \
}

#define agxXmlVerify1(x, element, msg)                                                                                                 \
if (!(x))                                                                                                                              \
{                                                                                                                                      \
  std::stringstream __stream;                                                                                                            \
  __stream << "[" << __FILE__ << ":" << __LINE__ << "] XML error: " << msg << ". Error at row " << (element)->Row() << std::endl;          \
  __stream << "File: " << (element)->GetDocument()->Value() << std::endl;                                                                \
  __stream << "==============================================================================" << std::endl;                             \
  (element)->Print(__stream, 0);                                                                                                           \
  __stream << std::endl;                                                                                                                 \
  __stream << "==============================================================================" << std::endl;                             \
  const std::string& __string = __stream.str();                                                                                            \
  std::cerr << __string << std::endl;                                                                                                    \
  agxThrow std::runtime_error(__string);                                                                                                    \
}

#define AGX_XML_VERIFY_BUFFER_SIZE 8192
#define agxXmlVerifyN(x, element, format, ...)                                                                                                                       \
if (!(x))                                                                                                                                                            \
{                                                                                                                                                                    \
  char __str[AGX_XML_VERIFY_BUFFER_SIZE];                                                                                                                              \
  agx_snprintf(__str, AGX_XML_VERIFY_BUFFER_SIZE, format, ##__VA_ARGS__);                                                                                              \
  std::stringstream __stream;                                                                                                                                          \
  __stream << "[" << __FILE__ << ":" << __LINE__ << "] XML error: " << __str << ". Error at row " << (element)->Row() << std::endl;                                        \
  __stream << "File: " << (element)->GetDocument()->Value() << std::endl;                                                                                              \
  __stream << "==============================================================================" << std::endl;                                                           \
  (element)->Print(__stream, 0);                                                                                                                                         \
  __stream << std::endl;                                                                                                                                               \
  __stream << "==============================================================================" << std::endl;                                                           \
  const std::string& __string = __stream.str();                                                                                                                          \
  std::cerr << __string << std::endl;                                                                                                                                  \
  agxThrow std::runtime_error(__string);                                                                                                                                  \
}


#define agxXmlAbort(element, format, ...)                                                                                                                            \
{                                                                                                                                                                    \
  char __str[AGX_XML_VERIFY_BUFFER_SIZE];                                                                                                                              \
  agx_snprintf(__str, AGX_XML_VERIFY_BUFFER_SIZE, format, ##__VA_ARGS__);                                                                                              \
  std::stringstream __stream;                                                                                                                                          \
  __stream << "[" << __FILE__ << ":" << __LINE__ << "] XML error: " << __str << ". Error at row " << (element)->Row() << std::endl;                                        \
  __stream << "File: " << (element)->GetDocument()->Value() << std::endl;                                                                                              \
  __stream << "==============================================================================" << std::endl;                                                           \
  (element)->Print(__stream, 0);                                                                                                                                         \
  __stream << std::endl;                                                                                                                                               \
  __stream << "==============================================================================" << std::endl;                                                           \
  const std::string& __string = __stream.str();                                                                                                                          \
  std::cerr << __string << std::endl;                                                                                                                                  \
  agxThrow std::runtime_error(__string);                                                                                                                                  \
}




namespace agx
{
  template <typename T>
  inline T parseAttribute(const TiXmlElement *element, const char *attributeName)
  {
    agxXmlVerifyN(element->Attribute(attributeName), element, "Could not locate attribute \'%s\'", attributeName);

    const char *attrString = element->Attribute(attributeName);
    std::stringstream stream(attrString);
    T val;
    stream >> std::boolalpha >> val;
    agxXmlVerifyN(stream, element, "Unable to parse attribute %s", attributeName);
    return val;
  }

  template <>
  inline String parseAttribute<String>(const TiXmlElement *element, const char *attributeName)
  {
    agxXmlVerifyN(element->Attribute(attributeName), element, "Could not locate attribute \'%s\'", attributeName);
    return element->Attribute(attributeName);
  }

  template <>
  inline Name parseAttribute<Name>(const TiXmlElement *element, const char *attributeName)
  {
    agxXmlVerifyN(element->Attribute(attributeName), element, "Could not locate attribute \'%s\'", attributeName);
    return element->Attribute(attributeName);
  }

}


#endif /* AGX_XML_H */
