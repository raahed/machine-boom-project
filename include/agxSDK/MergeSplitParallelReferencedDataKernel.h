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

#include <agxSDK/MergeSplitPostSolveData.h>

#include <agx/Kernel.h>

namespace agxSDK
{
  template<typename T>
  class MergeSplitParallelReferencedDataKernel : public agx::LambdaKernel
  {
    public:
      void execute( const T* data,
                    const agxSDK::MergeSplitHandler& handler,
                    const agxSDK::MergeSplitPostSolveData& postSolveData );

    protected:
      virtual void onPreExecute();
      virtual void onEntityPtr( agxData::EntityPtr ptr,
                                const agxSDK::MergeSplitHandler& handler,
                                const agxSDK::MergeSplitPostSolveData& postSolveData ) = 0;
      virtual void onPostExecute();

    protected:
      MergeSplitParallelReferencedDataKernel( const agx::Name& name );

      agx::UInt getNumElements() const;
      const T* getData() const;

      using agx::Task::execute;

    private:
      void executeJob( const agx::RangeJob& job );

    private:
      MergeSplitParallelReferencedDataKernel( const MergeSplitParallelReferencedDataKernel<T>& ) = delete;
      MergeSplitParallelReferencedDataKernel& operator=( const MergeSplitParallelReferencedDataKernel<T>& ) = delete;

    private:
      const T* m_data;
      const MergeSplitHandler* m_handler;
      const MergeSplitPostSolveData* m_postSolveData;
  };

  template<typename T>
  MergeSplitParallelReferencedDataKernel<T>::MergeSplitParallelReferencedDataKernel( const agx::Name& name )
    : LambdaKernel( name,
                    std::bind( &MergeSplitParallelReferencedDataKernel<T>::executeJob, this, std::placeholders::_1 ),
                    std::bind( &MergeSplitParallelReferencedDataKernel<T>::getNumElements, this ) ),
      m_data( nullptr ),
      m_handler( nullptr ),
      m_postSolveData( nullptr )
  {
  }

  template<typename T>
  agx::UInt MergeSplitParallelReferencedDataKernel<T>::getNumElements() const
  {
    return m_data != nullptr ? m_data->instance.size() : 0u;
  }

  template<typename T>
  const T* MergeSplitParallelReferencedDataKernel<T>::getData() const
  {
    return m_data;
  }

  template<typename T>
  void MergeSplitParallelReferencedDataKernel<T>::onPreExecute()
  {
  }

  template<typename T>
  void MergeSplitParallelReferencedDataKernel<T>::execute( const T* data,
                                                           const MergeSplitHandler& handler,
                                                           const MergeSplitPostSolveData& postSolveData )
  {
    m_data = data;
    if ( m_data != nullptr ) {
      m_handler = &handler;
      m_postSolveData = &postSolveData;

      this->onPreExecute();
      agx::LambdaKernel::execute();
      this->onPostExecute();
    }


    m_data = nullptr;
    m_handler = nullptr;
    m_postSolveData = nullptr;
  }

  template<typename T>
  void MergeSplitParallelReferencedDataKernel<T>::onPostExecute()
  {
  }

  template<typename T>
  void MergeSplitParallelReferencedDataKernel<T>::executeJob( const agx::RangeJob& job )
  {
    for ( agx::UInt i = job.range().begin(), end = job.range().end(); i != end; ++i )
      onEntityPtr( m_data->instance[ i ], *m_handler, *m_postSolveData );
  }
}
