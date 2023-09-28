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

#ifndef AGX_GLOBALRESULT_H
#define AGX_GLOBALRESULT_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4290 ) // C++ exception specification ignored except to indicate a function is not __declspec(nothrow)
#endif

#include <agx/agx.h>
#include <agx/agxCore_export.h>
#include <agxData/AttributeContainer.h>
#include <agx/Object.h>

#include <atomic>

namespace agxData
{
  class Buffer;
  class EntityStorage;
}

namespace agx
{
  class Task;
  class Parameter;
  class ArrayParameter;
  class EntityDataParameter;

  /**
  GlobalResult is used by kernels to generate result data from
  different threads. The initial result size estimation may be
  insufficient in which case the GlobalResult must be reallocated.
  Transactions are used to handle overflows in a thread safe manner.
  */
  class AGXCORE_EXPORT GlobalResult : public Object
  {
  public:
    AGX_DECLARE_POINTER_TYPES(BufferSet);

    enum Settings
    {
      // Mode
      TRUNCATE = 0x1, /// Clear the target data set before the kernel executes
      APPEND   = 0x2, /// Append to the existing data

      // Ordering
      ORDER    = 0x4, /// Build a permutation table for deterministic ordering of result data
      SORT     = 0x8  /// Build a permutation table and then use permutation to reorder the data
    };


    /**
    Transactions are used to reserve exclusive memory areas
    in which the threads can write their data.
    */
    class AGXCORE_EXPORT Transaction
    {
    public:
      /**
      \return The allocated range for this transaction.
      */
      const IndexRange& range() const;

      /**
      Commit the transaction after copying data to global buffer(s)
      \param sortIndex A sort index used to build a deterministic ordering of the result data
      \return false If the transaction failed, if so it must be restarted until
        commit returns true. NOTE: Do _not_ create a new transaction if the current
        one fails, just re-write the data using the existing transaction.
      */
      bool commit(Index sortIndex);

      /**
      Same as above but without deterministic sorting.
      */
      bool commit();


      /**
      \return The active buffer set for the transaction.
      */
      BufferSet *getBufferSet();

      /**
      \return The target global result.
      */
      GlobalResult *getTarget();

    private:
      friend class agx::GlobalResult;
      Transaction(GlobalResult *target, const IndexRange& range, BufferSet *bufferSet);

    private:
      GlobalResult *m_target;
      IndexRange m_range;
      BufferSet *m_bufferSet;
    };


    /**
    Allocate a transaction.
    \param numElements The number of elements to be written
    \return A transaction handle that must be committed after writing the result data
    */
    Transaction allocateResult(size_t numElements);

    /**
    Convenience wrapper to perform a complete transaction, this consists of:
    1. Init the transaction
    2. Copy the data to global result
    3. Commit transaction, go back to step 2 if commit fails

    \param numElements The number of elements to be written
    \param target The target buffer
    \param localResult The local result buffer to copy data from
    \param sortIndex A sort index used to build a deterministic ordering of the result data
    \return The offset where the data was committed
    */
    size_t commit(const agxData::AbstractArray& target, size_t numElements, const void *localResult, Index sortIndex);

    template <typename T>
    size_t commit(const agxData::AbstractArray& target, const T& vector, Index sortIndex);

    size_t commit(const agxData::AbstractArray& target, size_t numElements, const void *localResult);

    template <typename T>
    size_t commit(const agxData::AbstractArray& target, const T& vector);


    /**
    \return True if the GlobalResult did overflow during the last result session.
    */
    bool didOverflow() const;

    /**
    \return The number of overflows during the last result session.
    */
    agx::UInt getNumOverflows() const;

    agx::Mutex& getMutex();

  public:
    GlobalResult(UInt settings, const Path& bindPath);

    BufferSet *getActiveSet();

    void prepareResultSorting(size_t numJobs);

  protected:
    struct OverflowBuffer
    {
      OverflowBuffer();
      OverflowBuffer(agxData::Buffer* buffer_);

      agxData::Buffer* buffer;
      void* ptr;
      size_t size;
    };

  protected:
    virtual ~GlobalResult();

    virtual void handleOverflow(size_t overflowSize) = 0;

    template <typename T>
    size_t calculateInitialCapacity(const T *target);

    size_t calculateIncreasedCapacity(size_t currentCapacity, size_t overflowSize);

    void finalizeTransactions();
    void finalizeResultSorting(IndexRange range);
    virtual void sortResult(const agxData::IndexArray& permutation) = 0;

    void init(const IndexRange& range);
    BufferSet *createNewSet(size_t capacity);
    BufferSet *reportFailedTransaction(Transaction *transaction);
    void reportFailedTransaction(const IndexRange& transactionRange, BufferSet *startSet);
    void reportCompletedTransaction(Transaction *transaction);

  protected:
    UInt m_settings;
    BufferSetRef m_activeSet;
    Vector<BufferSetRef> m_bufferSets;

    std::atomic<size_t> m_capacity;
    size_t m_initialSize;
    Real m_smoothed;

    std::atomic<agx::Int32> m_head;
    agx::Mutex m_mutex;
    VectorPOD<Transaction> m_transactions;
    std::atomic<agx::Int32> m_numTransactions;
    std::atomic<agx::Int32> m_numCompletedTransactions;
    UInt m_numOverflows;

    VectorPOD<IndexRange> m_sortedRanges;
    agxData::Buffer *m_resultOrderBuffer;
  };

  //---------------------------------------------------------------

  /**
  The set of active buffers in a GlobalResult.
  */
  class AGXCORE_EXPORT GlobalResult::BufferSet : public Referenced
  {
  public:
    BufferSet(size_t index, size_t capacity);

    /// Initialize a buffer set
    void init(size_t index, size_t capacity);

    /// \return The index of the set
    size_t getIndex() const;

    /// \return The capacity of the set
    size_t getCapacity() const;

    /// Report an overflown buffer
    void addOverflow(const OverflowBuffer& buffer);

    /// Report a failed transaction
    void addFailedTransaction(const IndexRange& transactionRange);

    /// \return The list of overflows
    VectorPOD<OverflowBuffer>& getOverflows();

    /// \return The list of failed transaction
    IndexRangeVector& getFailedTransactions();

    /// Commit partial result to final buffers
    void commitRanges(const IndexRangeVector& ranges);

  protected:
    virtual ~BufferSet() {}

  private:
    VectorPOD<OverflowBuffer> m_overflows;
    IndexRangeVector m_failedTransactions;
    size_t m_index;
    size_t m_capacity;
  };

  //---------------------------------------------------------------

  AGX_DECLARE_POINTER_TYPES(GlobalResultStorage);
  class AGXCORE_EXPORT GlobalResultStorage : public GlobalResult
  {
  public:

    GlobalResultStorage(agxData::EntityStorage *targetStorage, UInt settings);

    void prepare(Task *task);
    void finalize();

    // agxData::EntityStorage *getStorage();

  protected:
    virtual ~GlobalResultStorage();

    virtual void handleOverflow(size_t overflowSize) override;
    virtual void sortResult(const agxData::IndexArray& permutation) override;

  private:
    void locateTargets(Task *task);

    friend class EntityDataParameter;
    void locateTargets(EntityDataParameter *parameter);

    template <typename ParameterVectorT>
    void locateTargets(const ParameterVectorT& parameters);

  private:
    agxData::EntityStorage *m_targetStorage;
    VectorPOD<ArrayParameter *> m_targets;
    VectorPOD<agxData::Buffer *> m_implicitBuffers;
    bool m_targetInstanceBuffer;
  };

  //---------------------------------------------------------------

  AGX_DECLARE_POINTER_TYPES(GlobalResultBuffer);
  class AGXCORE_EXPORT GlobalResultBuffer : public GlobalResult
  {
  public:
    GlobalResultBuffer(agxData::Buffer *target, UInt settings);

    void prepare(Task *task);
    void finalize();

    bool isActive() const;

  protected:
    virtual void handleOverflow(size_t overflowSize) override;
    virtual void sortResult(const agxData::IndexArray& permutation) override;

    virtual ~GlobalResultBuffer();

  private:
    agxData::Buffer *m_target;
    bool m_isActive;
  };




  /* Implementation */
  AGX_FORCE_INLINE GlobalResult::Transaction::Transaction(GlobalResult *target, const IndexRange& range, BufferSet *bufferSet)
      : m_target(target)
      , m_range(range)
      , m_bufferSet(bufferSet)
  {
    agxAssert(target);
    agxAssert(bufferSet);
  }


  AGX_FORCE_INLINE const IndexRange& GlobalResult::Transaction::range() const { return m_range; }
  AGX_FORCE_INLINE GlobalResult::BufferSet *GlobalResult::Transaction::getBufferSet() { return m_bufferSet; }
  AGX_FORCE_INLINE GlobalResult *GlobalResult::Transaction::getTarget() { return m_target; }


  AGX_FORCE_INLINE bool GlobalResult::Transaction::commit(Index sortIndex)
  {
    m_target->getMutex().lock();
    auto currentActiveSet = m_target->getActiveSet();
    m_target->getMutex().unlock();

    if (m_bufferSet == currentActiveSet)
    {
      #ifdef AGX_DEBUG
      m_target->reportCompletedTransaction(this);
      #endif

      agxAssert(sortIndex < m_target->m_sortedRanges.size());
      m_target->m_sortedRanges[sortIndex] = m_range;

      return true;
    }

    m_bufferSet = m_target->reportFailedTransaction(this);
    return false;
  }

  AGX_FORCE_INLINE bool GlobalResult::Transaction::commit()
  {
    agxAssertN(!(m_target->m_settings & (SORT|ORDER)) || m_target->m_sortedRanges.empty(), "%s: Must use extended commit method if SORT or ORDER is specified", m_target->getPath().c_str());

    m_target->getMutex().lock();
    BufferSet *currentActiveSet = m_target->getActiveSet();
    m_target->getMutex().unlock();

    if (m_bufferSet == currentActiveSet)
    {
      #ifdef AGX_DEBUG
      m_target->reportCompletedTransaction(this);
      #endif

      return true;
    }

    m_bufferSet = m_target->reportFailedTransaction(this);
    return false;
  }


  //---------------------------------------------------------------

  AGX_FORCE_INLINE GlobalResult::BufferSet *GlobalResult::getActiveSet() { return m_activeSet; }
  AGX_FORCE_INLINE bool GlobalResult::didOverflow() const { return m_numOverflows > 0; }
  AGX_FORCE_INLINE agx::UInt GlobalResult::getNumOverflows() const { return m_numOverflows; }

  //---------------------------------------------------------------

  AGX_FORCE_INLINE agx::Mutex& GlobalResult::getMutex()
  {
    return m_mutex;
  }

  template <typename T>
  AGX_FORCE_INLINE size_t GlobalResult::commit(const agxData::AbstractArray& target, const T& vector, Index sortIndex)
  {
    return this->commit(target, vector.size(), vector.ptr(), sortIndex);
  }

  template <typename T>
  AGX_FORCE_INLINE size_t GlobalResult::commit(const agxData::AbstractArray& target, const T& vector)
  {
    return this->commit(target, vector.size(), vector.ptr());
  }

  AGX_FORCE_INLINE bool GlobalResultBuffer::isActive() const { return m_isActive; }



}

#ifdef _MSC_VER
# pragma warning(pop)
#endif
#endif /* AGX_GLOBALRESULT_H */
