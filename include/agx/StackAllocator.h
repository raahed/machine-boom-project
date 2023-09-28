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


#ifndef AGX_STACK_ALLOCATOR_H
#define AGX_STACK_ALLOCATOR_H

#ifndef SWIG_CPPWRAPPER_BUILD

#include <agx/agxPhysics_export.h>

#include <agx/agx.h>

namespace agx {


    /**
       The StackAllocator class is designed to be used with the STL vector
       class.  It allocates the required amount of space directly on the
       stack where an STL container is declared.  Because the STL template
       for the allocator must have a single argument, this template must be
       specialized to the desired value of the template argument N before
       being used in an STL container declaration.   The source code for
       the present class is adapted from the standard definition of the
       allocator class in STL, as in the book "The C++ Standard Library: A
       Tutorial and Reference" by Nicolai M. Josuttis, Addison Wesley,
       1999.  The main difference here is that all calls to  the method
       allocate() return a pointer to m_data.

       The explicit assumption made here is that the STL container requires
       only space for N objects and that any other overhead for begin, end,
       and capacity pointers are allocated in the stack of the STL class
       itself.   In particular, this allocator cannot work with std::list
       at this time.

       In addition, the rebind utility is not provided and neither is the
       copy constructor.

     */
 template <class T, int N  >
 class StackAllocator {
  private:
    T m_data[N];
   public:
     // type definitions
     typedef T        value_type;
     typedef T*       pointer;
     typedef const T* const_pointer;
     typedef T&       reference;
     typedef const T& const_reference;
     typedef std::size_t    size_type;
     typedef std::ptrdiff_t difference_type;

     // return address of values
     pointer address (reference value) const {
         return &value;
     }
     const_pointer address (const_reference value) const {
         return &value;
     }

     /* constructors and destructor
      * - nothing to do because the allocator has no state
      */
     StackAllocator() throw() {
     }

     ~StackAllocator() throw() {
     }

     // return maximum number of elements that can be allocated
     size_type max_size () const throw() {
         return N;
     }

     // allocate but don't initialize num elements of type T
     pointer allocate (size_type num, const void* = 0) {
         return m_data;
     }

     // initialize elements of allocated storage p with value value
     void construct (pointer p, const T& value) {
         // initialize memory with placement new
         new((void*)p)T(value);
     }

     // destroy elements of initialized storage p
     void destroy (pointer p) {
         // destroy objects by calling their destructor
         p->~T();
     }

     // deallocate storage p of deleted elements
     void deallocate (pointer p, size_type num) {
     }
 };
} // namespace agx

#endif // SWIG_CPPWRAPPER_BUILD
#endif
