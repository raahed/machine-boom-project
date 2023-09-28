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
#pragma once

// TODO Integrate with agxStream serialization...

namespace agxNet
{
  static inline size_t stringSize(const agx::String& string)
  {
    return string.size() + sizeof(agx::UInt32);
  }

  static inline agx::UInt8 *writeString(agx::UInt8 *buffer, const agx::String& string)
  {
    *(agx::UInt32 *)buffer = htonl((agx::UInt32)string.size());
    buffer += sizeof(agx::UInt32);
    strncpy((char *)buffer, string.c_str(), string.size());
    return buffer + string.size();
  }

  static inline agx::UInt8 *writeUInt32(agx::UInt8 *buffer, agx::UInt32 value)
  {
    *(agx::UInt32 *)buffer = htonl(value);
    return buffer + sizeof(agx::UInt32);
  }

#ifdef AGX_DEBUG
  static inline agx::UInt32 parseUInt32(agx::UInt8 *& buffer, agx::UInt8 *end)
#else
  static inline agx::UInt32 parseUInt32(agx::UInt8 *& buffer, agx::UInt8 * /*end*/)
#endif
  {
    agxAssert(end - buffer >= (ptrdiff_t)sizeof(agx::UInt32));
    agx::UInt32 result = ntohl(*(agx::UInt32 *)buffer);
    buffer += sizeof(agx::UInt32);
    return result;
  }

#ifdef AGX_DEBUG
  static inline agx::String parseString(agx::UInt8 *& buffer, agx::UInt8 *end)
#else
  static inline agx::String parseString(agx::UInt8 *& buffer, agx::UInt8 * /*end*/)
#endif
  {
    agxAssert(end - buffer >= (ptrdiff_t)sizeof(agx::UInt32));
    agx::UInt32 stringSize = ntohl(*(agx::UInt32 *)buffer);
    buffer += sizeof(agx::UInt32);
    agxAssert(agx::UInt32(end - buffer) >= stringSize);
    agx::String result((const char *)buffer, stringSize);
    buffer += stringSize;
    return result;
  }

}


#endif /* AGXNET_UTIL_H */
