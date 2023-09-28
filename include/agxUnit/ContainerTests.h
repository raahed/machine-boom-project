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

#ifndef AGXUNIT_CONTAINERTESTS_H
#define AGXUNIT_CONTAINERTESTS_H

#include <agx/agxPhysics_export.h>
#include <agx/String.h>
#include <fstream>
#include <agx/Allocator.h>

namespace agxUnit
{

  /// Never allocates anything, throws always an exception. Only for testing.
  template <typename T>
  class AlwaysFailingAllocator : agx::Allocator<T>
  {
    public:
      void *allocateBytes(size_t numBytes, size_t alignment = 16)
      {
        agxThrow std::runtime_error(agx::String::format("AlwaysFailingAllocator called with numBytes %lu\n and alignment %lu.", numBytes, alignment));
      }

      void deallocateBytes(void* /*buffer*/, size_t /*numBytes*/)
      {
      }

      void setContainer(class agx::Container *)
      {
      }
  };



  template<typename T>
  /**
  The purpose of this test is to test how the container recovers from bad alloc.
  1. It should through an exception.
  2. Otherwise, its state should be intact.
  Only call with containers that have AlwaysFailingAllocator!
  */
  void testBadAlloc(const agx::String& /*typeName*/)
  {

    AGXUNIT_BEGIN_TEST(typeName + ": badAlloc");
    for (size_t i = 0; i < 4; ++i) {
      // Create a simple T.
      T t;
      size_t oldCapacity = t.capacity();
      void* oldPtr = (void*)t.ptr(); // Save to cast away the constness here, we will only use the pointer for comparisons.

      bool hasBeenInException = false;

      try {
        // Here, we try to provoke an exception.
        if (i % 2 == 0)
          t.reserve(2);
        else
          insertElement(t);
      }
      catch (std::exception& e)
      {
        std::cout << "\nCaught (intentionally provoked) exception: " << e.what() << std::endl;
        hasBeenInException = true;
      }

      AGXUNIT_ASSERT_MESSAGE("Did not Throw exception.", hasBeenInException);
      AGXUNIT_ASSERT_MESSAGE("Size is not 0.", t.size() == 0);
      AGXUNIT_ASSERT_MESSAGE("Capacity has changed by bad alloc.", t.capacity() == oldCapacity);
      AGXUNIT_ASSERT_MESSAGE("Pointer has changed by bad alloc.", t.ptr() == oldPtr);

      if (i < 2)
        t.clear(); // Test deallocating once with clear, and once without.
    }

    AGXUNIT_END_TEST(typeName + ": badAlloc");
  }



  template<typename T>
  /**
  The purpose of this test is to test for overflow in allocation size.
  Either it should Throw an exception, or have the right size in allocation.
  Only call with containers that have AlwaysFailingAllocator!
  */
  void testAllocationOverflow(const agx::String& /*typeName*/)
  {
    // The purpose of this test is to test how the container recovers from bad alloc.
    // 1. It should through an exception.
    // 2. Otherwise, its state should be intact.

    AGXUNIT_BEGIN_TEST(typeName + ": testAllocationOverflow");
#if 0 // This test will fail in many containers. \todo: Fix containers!

    for (size_t i = 0; i < 2; ++i) {
      // Create a simple T.
      T t;
      insertElement(t);
      t.reserve(2);
      size_t oldCapacity = t.capacity();
      void* oldPtr = (void*)t.ptr(); // Save to cast away the constness here, we will only use the pointer for comparisons.

      bool hasBeenInException = false;
      const size_t tooBigToAlloc = std::numeric_limits<size_t>::max() - 2;


      try {
        // Here, we try to provoke an exception.
        t.reserve(tooBigToAlloc);

        oldCapacity = t.capacity();
        oldPtr = (void*)t.ptr(); // Save to cast away the constness here, we will only use the pointer for comparisons.
      }
      catch (std::exception& e)
      {
        std::cout << "\nCaught (intentionally provoked) exception: " << e.what() << std::endl;
        hasBeenInException = true;
      }
      if (hasBeenInException) {
        AGXUNIT_ASSERT_MESSAGE("Size is not 1.", t.size() == 1);
        AGXUNIT_ASSERT_MESSAGE("Capacity has changed by bad alloc.", t.capacity() == oldCapacity);
        AGXUNIT_ASSERT_MESSAGE("Pointer has changed by bad alloc.", t.ptr() == oldPtr);
      }
      else {
        const size_t minNewCapacity = oldCapacity + tooBigToAlloc;
        AGXUNIT_ASSERT_MESSAGE(
          agx::String::format("Did not reserve enough capacity, wanted: %lu; reserved: %lu.\n", minNewCapacity, t.capacity()),
          t.capacity() >= minNewCapacity);
      }

      AGXUNIT_ASSERT_MESSAGE("Did not Throw exception.", hasBeenInException);

      if (i == 1)
        t.clear(); // Test deallocating once with clear, and once without.
    }
#endif
    AGXUNIT_END_TEST(typeName + ": testAllocationOverflow");
  }

}

#endif

