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

#include <agx/Kernel.h>

namespace agx
{
  template<typename T, typename argument_type = typename T::const_reference>
  class ParallelRaContainerKernel : public agx::LambdaKernel
  {
    public:
      /**
      Execute the jobs (blocking mode) given container. The reference
      to this container (e.g., this->getContainer()) is only valid during
      onPreExecute, onElement and onPostExecute.
      \param container - container to iterate in parallel
      */
      virtual void doExecute( const T* container );

    protected:
      /**
      Construct given name.
      \param name - name of this kernel
      */
      ParallelRaContainerKernel( const agx::Name& name );

      /**
      \return the total number of elements
      */
      agx::UInt getNumElements() const;

      /**
      \return the container we're iterating - non-null when this kernel is executing
              and otherwise null
      */
      const T* getContainer() const;

    protected:
      /**
      Callback before the kernel (onElement callback) is executed.
      */
      virtual void onPreExecute();

      /**
      Callback with current element.
      \param index - index of current element
      \param element - current element, i.e., element == this->getContainer()[ index ]
      */
      virtual void onElement( agx::UInt index, argument_type element ) = 0;

      /**
      Callback after the kernel (onElement callback) is executed.
      */
      virtual void onPostExecute();

    private:
      ParallelRaContainerKernel( const ParallelRaContainerKernel<T>& ) = delete;
      ParallelRaContainerKernel<T>& operator=( const ParallelRaContainerKernel<T>& ) = delete;

      /**
      Hiding blocking/non-blocking method since it's only valid to call
      execute with the container to iterate.
      */
      using agx::LambdaKernel::execute;

      /**
      Callback from task manager with current range job.
      */
      void executeJob( const agx::RangeJob& job );

    private:
      const T* m_container;
  };

  template<typename T, typename argument_type>
  ParallelRaContainerKernel<T, argument_type>::ParallelRaContainerKernel( const agx::Name& name )
    : LambdaKernel( name,
                    std::bind( &ParallelRaContainerKernel<T, argument_type>::executeJob, this, std::placeholders::_1 ),
                    std::bind( &ParallelRaContainerKernel<T, argument_type>::getNumElements, this ) ),
      m_container( nullptr )
  {
  }

  template<typename T, typename argument_type>
  agx::UInt ParallelRaContainerKernel<T, argument_type>::getNumElements() const
  {
    return m_container != nullptr ? agx::UInt( m_container->size() ) : agx::UInt( 0 );
  }

  template<typename T, typename argument_type>
  const T* ParallelRaContainerKernel<T, argument_type>::getContainer() const
  {
    return m_container;
  }

  template<typename T, typename argument_type>
  void ParallelRaContainerKernel<T, argument_type>::onPreExecute()
  {
  }

  template<typename T, typename argument_type>
  void ParallelRaContainerKernel<T, argument_type>::doExecute( const T* container )
  {
    m_container = container;
    if ( m_container != nullptr ) {
      onPreExecute();
      agx::LambdaKernel::execute();
      onPostExecute();
    }
    m_container = nullptr;
  }

  template<typename T, typename argument_type>
  void ParallelRaContainerKernel<T, argument_type>::onPostExecute()
  {
  }

  template<typename T, typename argument_type>
  void ParallelRaContainerKernel<T, argument_type>::executeJob( const agx::RangeJob& job )
  {
    for ( agx::UInt i = job.range().begin(), end = job.range().end(); i != end; ++i )
      onElement( i, const_cast<T&>(*m_container)[ i ] );
  }
}
