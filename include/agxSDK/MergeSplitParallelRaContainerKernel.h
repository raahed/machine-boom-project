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

#pragma once

#include <agx/ParallelRaContainerKernel.h>

namespace agxSDK
{
  class MergeSplitHandler;

  template<typename T>
  class MergeSplitParallelRaContainerKernel : public agx::ParallelRaContainerKernel<T>
  {
    public:
      /**
      Construct given name of this kernel.
      \param name - name of this kernel
      */
      MergeSplitParallelRaContainerKernel( const agx::Name& name );

      /**
      Execute parallel iterate over given container.
      \param container - container to iterate in parallel
      \param handler - the merge split handler
      */
      void doExecute( const T* container,
                      const agxSDK::MergeSplitHandler* handler,
                      const agxSDK::MergeSplitPostSolveData* postSolveData = nullptr );

      /**
      Do not use this method.
      */
      using agx::ParallelRaContainerKernel<T>::doExecute;

      /**
      \return the merge split handler during execute - otherwise null
      */
      const agxSDK::MergeSplitHandler* getHandler() const;

      /**
      \return the post solve data if given to execute - otherwise null
      */
      const agxSDK::MergeSplitPostSolveData* getPostSolveData() const;

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~MergeSplitParallelRaContainerKernel();

    private:
      const MergeSplitHandler* m_handler;
      const MergeSplitPostSolveData* m_postSolveData;
  };

  template<typename T>
  MergeSplitParallelRaContainerKernel<T>::MergeSplitParallelRaContainerKernel( const agx::Name& name )
    : agx::ParallelRaContainerKernel<T>( name ),
      m_handler( nullptr ),
      m_postSolveData( nullptr )
  {
  }

  template<typename T>
  MergeSplitParallelRaContainerKernel<T>::~MergeSplitParallelRaContainerKernel()
  {
  }

  template<typename T>
  void MergeSplitParallelRaContainerKernel<T>::doExecute( const T* container,
                                                          const MergeSplitHandler* handler,
                                                          const MergeSplitPostSolveData* postSolveData /*= nullptr*/ )
  {
    m_handler = handler;
    m_postSolveData = postSolveData;
    agx::ParallelRaContainerKernel<T>::doExecute( container );
    m_postSolveData = nullptr;
    m_handler = nullptr;
  }

  template<typename T>
  const MergeSplitHandler* MergeSplitParallelRaContainerKernel<T>::getHandler() const
  {
    return m_handler;
  }

  template<typename T>
  const MergeSplitPostSolveData*  MergeSplitParallelRaContainerKernel<T>::getPostSolveData() const
  {
    return m_postSolveData;
  }
}
