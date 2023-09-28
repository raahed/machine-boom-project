/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/


#pragma once

#include <cstddef>
#include <agx/agxCore_export.h>

namespace agx
{
  class Container;

  class AGXCORE_EXPORT ThreadLocalAllocator
  {
  public:
    ThreadLocalAllocator();
    ThreadLocalAllocator(const ThreadLocalAllocator& other);
    ThreadLocalAllocator(ThreadLocalAllocator&& other);
    ~ThreadLocalAllocator();

    ThreadLocalAllocator& operator=(const ThreadLocalAllocator& other);
    ThreadLocalAllocator& operator=(ThreadLocalAllocator&& other);


    void setContainer(Container* container);
    void* allocateBytes(size_t numBytes);
    void deallocateBytes(void* buffer, size_t numBytes);

  private:
    void clear();

  private:
    // m_container is not a salient attribute. It represents an owned-by
    // relationship that should reflect the physical composition and
    // copying/moving containers should not "leak" this pointer into the new
    // container.
    Container* m_container;
  };
}
