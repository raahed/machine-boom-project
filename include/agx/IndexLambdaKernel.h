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

#include <stack>

namespace agx
{
  /**
  Utility entrance to the task system, simplifying parallelization of algorithms.
  Example:
  void collectConstraintForces( agxSDK::Simulation* simulation )
  {
    struct ConstraintForceData
    {
      agx::Bool valid;
      agx::Vec3 force[2];
      agx::Vec3 torque[2];
    };
    using ConstraintForceDataVector = agx::VectorPOD<ConstraintForceData>;

    const agx::ConstraintRefSetVector& constraints = simulation->getConstraints();
    ConstraintForceData dataVector;
    dataVector.resize( constraints.size() );

    auto kernel = simulation->getDynamicsSystem()->getIndexLambdaKernel();
    kernel->execute( "Fetch constraint forces",
                     constraints.size(), // Number of elements.
                     []( agx::UInt index,
                         const agx::ConstraintRefSetVector& constraints,
                         ConstraintForceData& dataVector )
                     {
                       auto& data = dataVector[ index ];
                       const auto constraint = constraints[ index ];
                       data.valid = constraint->getLastForce( 0, data.force[ 0 ], data.torque[ 0 ] );
                       data.valid = data.valid && constraint->getLastForce( 1, data.force[ 1 ], data.torque[ 1 ] );
                     },
                     std::cref( constraints ), // second function argument (first is always the current index)
                     std::ref( dataVector ) ); // third function argument
  }
  */
  class AGXPHYSICS_EXPORT IndexLambdaKernel : public agx::LambdaKernel
  {
    public:
      /**
      Default callback signature with current index. Additional
      arguments are binded after the index.
      */
      using IndexCallback = std::function<void( agx::UInt )>;

    public:
      /**
      Generates IndexCallback given function and arguments. Note that the
      \p callback must have UInt index as first arugment and \p args
      are added after.
      \param callback - callback function
      \param args - additional arguments
      \return callback function object including arguments
      */
      template<typename CallbackT, typename... Args>
      static IndexCallback generate( CallbackT callback, Args&&... args );

    public:
      /**
      Default Constructor.
      */
      IndexLambdaKernel();

      /**
      Execute \p callback given \p name, \p numElements and \p args (additional arguments to \p callback).
      \param name - name of the kernel
      \param numElements - total number of elements
      \param callback - callback to execute
      \param args - additional arguments to \p callback
      */
      template<typename CallbackT, typename... Args>
      void execute( const agx::Name& name, agx::UInt numElements, CallbackT callback, Args&&... args );

      /**
      Execute \p callback given \p name, \p numElements and \p args (additional arguments to \p callback).
      \param name - name of the kernel
      \param numElements - total number of elements
      \param callback - callback to execute
      \param args - additional arguments to \p callback
      \param jobSize - number of items per job
      */
      template<typename CallbackT, typename... Args>
      void executeEx( const agx::Name& name, agx::UInt numElements, agx::UInt jobSize, CallbackT callback, Args&&... args );

      /**
      Execute \p callback given \p name and \p numElements.
      \param callback - index callback to execute
      \param name - name of the kernel
      \param numElements - total number of elements
      \param jobSize - number of items per job (default settings if jobSize == agx::InvalidIndex)
      */
      void execute( IndexCallback callback,
                    const agx::Name& name,
                    agx::UInt numElements,
                    agx::UInt jobSize = agx::InvalidIndex );

    protected:
      /**
      Reference counted object - protected destructor.
      */
      virtual ~IndexLambdaKernel();

    private:
      struct ExecData
      {
        ExecData( IndexCallback callback, const agx::Name& name, agx::UInt numElements )
          : callback( callback ), name( name ), numElements( numElements )
        {
        }

        IndexCallback callback;
        agx::Name name;
        agx::UInt numElements;
      };
      using ExecStack = std::stack<ExecData>;

    private:
      /**
      \return the number of elements in current kernel
      */
      agx::UInt getNumElements() const;

      /**
      Callback from system with job in current kernel.
      \param job - job to execute
      */
      void executeJob( const agx::RangeJob& job );

      /**
      Executes callback given index range.
      \param begin - first element index
      \param end - last (not included) element index
      */
      void executeIndexRange( agx::UInt begin, agx::UInt end );

      /**
      Reset job size to default values.
      */
      void resetJobSize();

    private:
      ExecStack m_execStack;
      Mutex m_mutex;
  };
  using IndexLambdaKernelRef = agx::ref_ptr<IndexLambdaKernel>;

  template<typename CallbackT, typename... Args>
  IndexLambdaKernel::IndexCallback IndexLambdaKernel::generate( CallbackT callback, Args&&... args )
  {
    // Placeholder for UInt index as first argument.
    return std::bind( callback, std::placeholders::_1, std::forward<Args>( args )... );
  }

  template<typename CallbackT, typename... Args>
  void IndexLambdaKernel::execute( const agx::Name& name, agx::UInt numElements, CallbackT callback, Args&&... args )
  {
    this->executeEx( name, numElements, agx::InvalidIndex, std::forward<CallbackT>( callback ), std::forward<Args>( args )... );
  }

  template<typename CallbackT, typename... Args>
  void IndexLambdaKernel::executeEx( const agx::Name& name, agx::UInt numElements, agx::UInt jobSize, CallbackT callback, Args&&... args )
  {
    this->execute( generate( std::forward<CallbackT>( callback ), std::forward<Args>( args )... ), name, numElements, jobSize );
  }
}
