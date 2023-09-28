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

#ifndef AGX_KERNEL_H
#define AGX_KERNEL_H


#include <agx/Referenced.h>
#include <agx/String.h>
#include <agx/Task.h>
#include <agxData/Value.h>
#include <agx/Vector.h>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable : 4251 ) // class X needs to have dll-interface to be used by clients of class Y
#endif


namespace agx
{
  class TiXmlElement;

  AGX_DECLARE_POINTER_TYPES(Kernel);
  AGX_DECLARE_VECTOR_TYPES(Kernel);
  AGX_DECLARE_POINTER_TYPES(CpuKernel);
  AGX_DECLARE_VECTOR_TYPES(CpuKernel);

  /**
  A kernel is an abstract representation of data parallel tasks.
  It may be implemented in arbitrary language, using a common parameter interface.
  */
  class AGXCORE_EXPORT Kernel : public ParallelTask
  {
  public:
    static agx::Model *ClassModel();

    static Kernel *load(TiXmlElement *eKernel, Device *device);
    virtual void configure(TiXmlElement *eKernel) override;

  public:
    Kernel(const agx::Name& name, agx::Device* device, const agx::Job::Callback& dispatcher);

    /**
    \return The number of work items for the currently bound parameters.
    */
    agx::UInt getNumWorkItems() const;

    /**
    \return The compute time from the last execution.
    */
    virtual float getComputeTime();

    /**
    \return The number of jobs per thread.
    */
    agx::UInt getNumJobsPerThread() const;

    /**
    \return The minimum number of work elements per job.
    */
    agx::UInt getMinJobSize() const;

    /**
    \return The maximum number of work elements per job.
    */
    agx::UInt getMaxJobSize() const;

    /**
    \return The per-element cost estimate.
    */
    agx::UInt getCostPerElement() const;

    /**
    Set the number of jobs per thread.
    */
    void setNumJobsPerThread(agx::UInt numJobs);

    /**
    Set the minimum number of work elements per job.
    */
    void setMinJobSize(agx::UInt min);

    /**
    Set the maximum number of work elements per job.
    */
    void setMaxJobSize(agx::UInt max);

    /**
    Set the per-element cost estimate.
    */
    void setCostPerElement(agx::UInt cost);



    /**
    Explicitly set the splitting parameter.
    */
    // void setSplittingParameter(ArrayParameter *parameter);

    virtual void preDispatch() {}

    virtual void rebind() override;

  protected:
    virtual ~Kernel();
    virtual void parameterAddedCallback(Parameter* parameter) override;
    virtual void parameterRemovedCallback(Parameter* parameter) override;
    void locateNumWorkItemsValue();
    agxData::ValueRefT<UInt> m_numWorkItems;

  private:
    agxData::Val<UInt> m_numJobsPerThread;
    agxData::Val<UInt> m_minJobSize;
    agxData::Val<UInt> m_maxJobSize;
    agxData::Val<UInt> m_costPerElement;
    // ArrayParameterRef m_splittingParameter;
  };


  //---------------------------------------------------------------

  /**
  A kernel which executes on the CPU using native C/C++ code.
  */
  class AGXCORE_EXPORT CpuKernel : public Kernel
  {
  public:
    typedef RangeJob::RangeCallback RangeDispatch;

  public:
    static CpuKernel *load(TiXmlElement *eKernel, Device *device);

  public:
    CpuKernel(const agx::Name& name, RangeDispatch rangeDispatch);
    CpuKernel(const agx::Name& name, Job::Callback serialDispatch);

    UInt getNumJobs() const;

    const RangeDispatch& getRangeDispatch() const;

  protected:
    virtual ~CpuKernel();
    void generateJobs();
    virtual RangeJob *allocateRangeJobs(size_t numJobs) = 0;

  protected: // Should be private, is protected for custom job generators
    // Vector<RangeJob> m_rangeJobs;
    RangeDispatch m_rangeDispatch;
    UInt m_numJobs;
  };


  //---------------------------------------------------------------

  /**
  A kernel wrapper to be used with custom lambda function callbacks.
  */
  AGX_DECLARE_POINTER_TYPES(LambdaKernel);
  class AGXCORE_EXPORT LambdaKernel : public CpuKernel
  {
  public:
    typedef std::function<void (const RangeJob&)> LambdaCallback;
    typedef std::function<UInt ()> DataSetSizeCallback;

  public:
    /**
    Constructor
    \param name The name of the kernel
    \param lambdaCallback The parallel operator to be called asynchronously with the workblock ranges
    \param sizeCallback Callback that must return the size of the full dataset, which will be partitioned
                        by the kernel during execution
    */
    LambdaKernel(const agx::Name& name, LambdaCallback lambdaCallback, DataSetSizeCallback sizeCallback);

  private:
    void generateJobs();
    virtual RangeJob *allocateRangeJobs(size_t numJobs) override;
    void callOnJobRange(const RangeJob& job);

  protected:
    Vector<RangeJob> m_rangeJobs;

  private:
    DataSetSizeCallback m_sizeCallback;
    LambdaCallback m_lambda;
  };

  //---------------------------------------------------------------


  /* Implementation */
  // inline const Kernel::ImplementationTable& Kernel::getImplementations() const { return m_implementations; }
  // inline const Kernel::Implementation& Kernel::getImplementation(const String& name) const { agxAssert(m_implementations.contains(name)); return m_implementations.find(name)->second; }
  // inline const Kernel::Implementation& Kernel::getActiveImplementation() const { return m_activeImplementation; }
  AGX_FORCE_INLINE UInt Kernel::getNumJobsPerThread() const { return m_numJobsPerThread; }
  AGX_FORCE_INLINE UInt Kernel::getMinJobSize() const { return m_minJobSize; }
  AGX_FORCE_INLINE UInt Kernel::getMaxJobSize() const { return m_maxJobSize; }
  AGX_FORCE_INLINE UInt Kernel::getCostPerElement() const { return m_costPerElement; }

  AGX_FORCE_INLINE UInt Kernel::getNumWorkItems() const
  {
    if (!m_numWorkItems)
      const_cast<Kernel *>(this)->locateNumWorkItemsValue();

    return m_numWorkItems->get();
  }

  AGX_FORCE_INLINE const CpuKernel::RangeDispatch& CpuKernel::getRangeDispatch() const { return m_rangeDispatch; }

  AGX_FORCE_INLINE UInt CpuKernel::getNumJobs() const { return m_numJobs; }
}


#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif /* _AGX_KERNEL_H_ */
