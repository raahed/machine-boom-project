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


// Inspired by Emil Ernerfeldt:
//   https://github.com/emilk/emilib/tree/master/emilib
//   LICENSE:
//     This software is dual-licensed to the public domain and under the following
//     license: you are granted a perpetual, irrevocable license to copy, modify,
//     publish, and distribute this file as you see fit.





#ifndef AGX_LINEAR_PROBING_HASH_TABLE_H
#define AGX_LINEAR_PROBING_HASH_TABLE_H

#include <cstdlib>
#include <iterator>
#include <utility>

#include <agx/ref_ptr.h>
#include <agx/hash.h>

#include <agx/Allocator.h>

namespace agx
{
/**
 * A cache-friendly hash table with open addressing, linear probing and power-of-two capacity.
 */
template <typename KeyT, typename ValueT, typename HashT = agx::HashFn<KeyT>, typename AllocatorT = ByteAllocator>
class LinearProbingHashTableImplementation
{
private:
  using MyType = LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>;
  using PairT = std::pair<KeyT, ValueT>;
  enum class State : uint8_t
  {
      INACTIVE, // Never been touched
      ACTIVE,   // Is inside a search-chain, but is empty
      FILLED    // Is set with key/value
  };


public:
  using key_type        = KeyT;
  using size_type       = size_t;
  using value_type      = ValueT;
  using bucket_type     = PairT;
  using reference       = PairT&;
  using const_reference = const PairT&;

  // The quadratic probing hash table supports various types of behavior when
  // clearing. The linear probing hash table does not.
  //
  // We repeat the clear policy type here so that users can compile, even though
  // the value is ignored for now.
  enum ClearPolicy
  {
    SHRINK_BUFFER,           //!< Buffer is deallocated and replaced by an newly allocated empty buffer.
    SHRINK_BUFFER_AVERAGED,  //!< Buffer is shrunk if a smoothing average (which is updated each clear call) goes below a threshold.
    MAINTAIN_BUFFER          //!< Buffer is maintained (normal stl behavior).
  };



  class iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = size_t;
    using distance_type     = size_t;
    using bucket_type       = std::pair<KeyT, ValueT>;
    using pointer           = bucket_type*;
    using reference         = bucket_type&;

    iterator() { }

    iterator(MyType* hash_map, size_t bucket) : _map(hash_map), _bucket(bucket)
    {
    }

    iterator& operator++()
    {
      this->goto_next_element();
      return *this;
    }

    iterator operator++(int)
    {
      size_t old_index = _bucket;
      this->goto_next_element();
      return iterator(_map, old_index);
    }

    /// Does not do any bounds checking, so only call this if there really is a
    /// previous element.
    iterator operator--()
    {
      this->goto_previous_element();
      return *this;
    }

    /// Does not do any bounds checking, so only call this if there really is a
    /// previous element.
    iterator operator--(int)
    {
      size_t old_index = _bucket;
      this->goto_previous_element();
      return iterator(_map, old_index);
    }

    reference operator*() const
    {
      return _map->_pairs[_bucket];
    }

    pointer operator->() const
    {
      return _map->_pairs + _bucket;
    }

    bool operator==(const iterator& rhs) const
    {
      agxAssert(_map == rhs._map);
      return this->_bucket == rhs._bucket;
    }

    bool operator!=(const iterator& rhs) const
    {
      agxAssert(_map == rhs._map);
      return this->_bucket != rhs._bucket;
    }

  private:
    void goto_next_element()
    {
      agxAssert(_bucket < _map->_num_buckets);
      do {
        _bucket++;
      } while (_bucket < _map->_num_buckets && _map->_states[_bucket] != State::FILLED);
    }

    // This is tremendously dangerous. There is currently no way of detecting
    // that we've reached rend since there is no rend.
    void goto_previous_element()
    {
      agxAssert(_bucket > 0);

      do {
        _bucket--;
      } while(_bucket > 0 && _map->_states[_bucket] != State::FILLED);

      agxAssert(_map->_states[_bucket] == State::FILLED);
    }

  public:
    MyType* _map;
    size_t  _bucket;
  };

  class const_iterator
  {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = size_t;
    using distance_type     = size_t;
    using bucket_type       = std::pair<KeyT, ValueT>;
    using pointer           = bucket_type*;
    using reference         = bucket_type&;

    const_iterator() { }

    const_iterator(iterator proto) : _map(proto._map), _bucket(proto._bucket)
    {
    }

    const_iterator(const MyType* hash_map, size_t bucket) : _map(hash_map), _bucket(bucket)
    {
    }

    const_iterator& operator++()
    {
      this->goto_next_element();
      return *this;
    }

    const_iterator operator++(int)
    {
      size_t old_index = _bucket;
      this->goto_next_element();
      return const_iterator(_map, old_index);
    }

    /// Does not do any bounds checking, so only call this if there really is a
    /// previous element.
    const_iterator operator--()
    {
      this->goto_previous_element();
      return *this;
    }

    /// Does not do any bounds checking, so only call this if there really is a
    /// previous element.
    const_iterator operator--(int)
    {
      size_t old_index = _bucket;
      this->goto_previous_element();
      return iterator(_map, old_index);
    }

    reference operator*() const
    {
      return _map->_pairs[_bucket];
    }

    pointer operator->() const
    {
      return _map->_pairs + _bucket;
    }

    bool operator==(const const_iterator& rhs) const
    {
      agxAssert(_map == rhs._map);
      return this->_bucket == rhs._bucket;
    }

    bool operator!=(const const_iterator& rhs) const
    {
      agxAssert(_map == rhs._map);
      return this->_bucket != rhs._bucket;
    }

  private:
    void goto_next_element()
    {
      agxAssert(_bucket < _map->_num_buckets);
      do {
        _bucket++;
      } while (_bucket < _map->_num_buckets && _map->_states[_bucket] != State::FILLED);
    }

    // This is tremendously dangerous. There is currently no way of detecting
    // that we've reached rend since there is no rend.
    void goto_previous_element()
    {
      agxAssert(_bucket > 0);

      do {
        _bucket--;
      } while(_bucket > 0 && _map->_states[_bucket] != State::FILLED);

      agxAssert(_map->_states[_bucket] == State::FILLED);
    }

  //private:
  //  friend class MyType;
  public:
    const MyType* _map;
    size_t        _bucket;
  };

  class insert_iterator : public iterator
  {
    public:
      insert_iterator() : iterator(), m_overwrite(false)
      {
      }

      insert_iterator(MyType* hash_map, size_t bucket, bool overwrite) :
          iterator(hash_map, bucket), m_overwrite(overwrite)
      {
      }

      /**
      \return True if the insert operation did overwrite an existing entry.
      */
      bool didOverwrite() const
      {
        return m_overwrite;
      }

    private:
      bool m_overwrite;
  };

  // ------------------------------------------------------------------------

  LinearProbingHashTableImplementation() = default;

  LinearProbingHashTableImplementation(const LinearProbingHashTableImplementation& other)
  {
    reserve(other.size());
    insert(cbegin(other), cend(other));
  }

  LinearProbingHashTableImplementation(LinearProbingHashTableImplementation&& other)
  {
    *this = std::move(other);
  }

  LinearProbingHashTableImplementation& operator=(const LinearProbingHashTableImplementation& other)
  {
    clear();
    reserve(other.size());
    insert(cbegin(other), cend(other));
    return *this;
  }

  LinearProbingHashTableImplementation& operator=(LinearProbingHashTableImplementation&& other)
  {
    this->swap(other);
    return *this;
  }

  ~LinearProbingHashTableImplementation()
  {
    for (size_t bucket=0; bucket<_num_buckets; ++bucket) {
      if (_states[bucket] == State::FILLED) {
        _pairs[bucket].~PairT();
      }
    }

    _allocator.deallocateBytes(_states, _num_buckets * sizeof(State));
    _allocator.deallocateBytes(_pairs,  _num_buckets * sizeof(PairT));
  }

  void swap(LinearProbingHashTableImplementation& other)
  {
    std::swap(_hasher,           other._hasher);
    std::swap(_allocator,        other._allocator);
    std::swap(_states,           other._states);
    std::swap(_pairs,            other._pairs);
    std::swap(_num_buckets,      other._num_buckets);
    std::swap(_num_filled,       other._num_filled);
    std::swap(_max_probe_length, other._max_probe_length);
    std::swap(_mask,             other._mask);
  }

  // -------------------------------------------------------------

  /**
  Iterator to first element in hash table.
  */
  iterator begin()
  {
    size_t bucket = 0;
    while (bucket<_num_buckets && _states[bucket] != State::FILLED) {
      ++bucket;
    }
    return iterator(this, bucket);
  }

  /**
  Iterator to first element in hash table.
  */
  const_iterator begin() const
  {
    size_t bucket = 0;
    while (bucket<_num_buckets && _states[bucket] != State::FILLED) {
      ++bucket;
    }
    return const_iterator(this, bucket);
  }


  /**
  Iterator marking end of hash table.
  */
  iterator end()
  { return iterator(this, _num_buckets); }


  /**
  Iterator marking end of hash table.
  */
  const_iterator end() const
  { return const_iterator(this, _num_buckets); }

  /**
   * \return The number of key/value pairs in the hash table.
   */
  size_t size() const
  {
    return _num_filled;
  }

  /**
   * \return The total number of buckets in the hash table.
   */
  size_t capacity() const
  {
    return _num_buckets;
  }

  /**
   * \return True if the hash table contains no key/value pairs. False otherwise.
   */
  bool empty() const
  {
    return _num_filled==0;
  }

  // ------------------------------------------------------------

  /**
   * Find a key/value pair in the hash table given a key.
   * \param key The key to look for.
   * \return Iterator to the key/value pair, or the end iterator if no such key exists.
   */
  iterator find(const KeyT& key)
  {
    auto bucket = this->find_filled_bucket(key);
    if (bucket == (size_t)-1) {
      return this->end();
    }
    return iterator(this, bucket);
  }

  /**
   * Find a key/value pair in the hash table given a key.
   * \param key The key to look for.
   * \return Iterator to the key/value pair, or the end iterator if no such key exists.
   */
  const_iterator find(const KeyT& key) const
  {
    auto bucket = this->find_filled_bucket(key);
    if (bucket == (size_t)-1)
    {
      return this->end();
    }
    return const_iterator(this, bucket);
  }

  /**
   * Check if the hash table contains a key/value pair for the given key.
   * \param k Key to look for.
   * \return True if the key exists in the hash table. False otherwise.
   */
  bool contains(const KeyT& k) const
  {
    return find_filled_bucket(k) != (size_t)-1;
  }

  /**
   * Count the number of key/value pairs matching the given key. Will always be
   * either 0 or 1.
   * \param k The key to count.
   * \return 1 if the key exists in the hash table, 0 otherwise.
   */
  size_t count(const KeyT& k) const
  {
    return find_filled_bucket(k) != (size_t)-1 ? 1 : 0;
  }

  /**
   * Find the value associated with the given key. Returns nullptr if no such value exists.
   * \param k The key to look for.
   * \return Pointer to the corresponding value, or nullptr.
   */
  ValueT* try_get(const KeyT& k)
  {
    auto bucket = find_filled_bucket(k);
    if (bucket != (size_t)-1) {
      return &_pairs[bucket].second;
    } else {
      return nullptr;
    }
  }

  /**
   * Find the value associated with the given key. Returns nullptr if no such value exists.
   * \param k The key to look for.
   * \return Pointer to the corresponding value, or nullptr.
   */
  const ValueT* try_get(const KeyT& k) const
  {
    auto bucket = find_filled_bucket(k);
    if (bucket != (size_t)-1) {
      return &_pairs[bucket].second;
    } else {
      return nullptr;
    }
  }

  // Convenience function.
  const ValueT get_or_return_default(const KeyT& k) const
  {
    const ValueT* ret = try_get(k);
    if (ret) {
      return *ret;
    } else {
      return ValueT();
    }
  }

  // -----------------------------------------------------

  /**
   * Insert a key/value pair into the hash table. The current value is overwritten
   * if a key/value pair with an equal key already exists.
   * \param key The key to insert.
   * \param value The value to associate with the given key.
   * \return An input iterator to the inserted key/value pair.
   */
  insert_iterator insert(const KeyT& key, const ValueT value)
  {
    check_expand_need();

    auto bucket = find_or_allocate(key);
    bool didOverwrite;
    if (_states[bucket] == State::FILLED) {
      (_pairs + bucket)->second = value;
      didOverwrite = true;
    } else {
      _states[bucket] = State::FILLED;
      new(_pairs + bucket) PairT(key, value);
      _num_filled++;
      didOverwrite = false;
    }

    return insert_iterator(this, bucket, didOverwrite);
  }

  insert_iterator insert(const std::pair<KeyT, ValueT>& p)
  {
    return insert(p.first, p.second);
  }

  /**
   * Insert all elements in the given range.
   *
   * \param begin Beginning of the range.
   * \param end End of the range.
   */
  void insert(const_iterator begin, const_iterator end)
  {
    for (; begin != end; ++begin) {
      insert(begin->first, begin->second);
    }
  }

  /**
   * Insert a key/value pair into the hash table assuming that the given key
   * does not already exist in the hash table. It is undefined behavior to call
   * this method with a key that does already exist in the hash table.
   *
   * \param key The key to insert.
   * \param value The value to associate with the key.
   */
  void insert_unique(KeyT&& key, ValueT&& value)
  {
    agxAssert(!contains(key));
    check_expand_need();
    auto bucket = find_empty_bucket(key);
    _states[bucket] = State::FILLED;
    new(_pairs + bucket) PairT(std::move(key), std::move(value));
    _num_filled++;
  }

  void insert_unique(std::pair<KeyT, ValueT>&& p)
  {
    insert_unique(std::move(p.first), std::move(p.second));
  }

  // Return the old value or ValueT() if it didn't exist.
  ValueT set_get(const KeyT& key, const ValueT& new_value)
  {
    check_expand_need();

    auto bucket = find_or_allocate(key);

    // Check if inserting a new value rather than overwriting an old entry
    if (_states[bucket] == State::FILLED) {
      ValueT old_value = _pairs[bucket].second;
      _pairs[bucket] = new_value.second;
      return old_value;
    } else {
      _states[bucket] = State::FILLED;
      new(_pairs + bucket) PairT(key, new_value);
      _num_filled++;
      return ValueT();
    }
  }

  /**
   * Return the value associated with the given key. A new value is default
   * created, inserted and then returned if the given key does not yet exist
   * in the hash table.
   * \param key The key to find.
   * \return The value associated with the given key. A default constructed one if no prior value existed.
   */
  ValueT& operator[](const KeyT& key)
  {
    check_expand_need();

    auto bucket = find_or_allocate(key);

    /* Check if inserting a new value rather than overwriting an old entry */
    if (_states[bucket] != State::FILLED) {
      _states[bucket] = State::FILLED;
      new(_pairs + bucket) PairT(key, ValueT());
      _num_filled++;
    }

    return _pairs[bucket].second;
  }

  // -------------------------------------------------------

  /**
   * Erase an element from the hash table.
   * \return True if an element was removed, false otherwise.
   */
  bool erase(const KeyT& key)
  {
    auto bucket = find_filled_bucket(key);
    if (bucket != (size_t)-1) {
      this->erase_bucket(bucket);
      return true;
    } else {
      return false;
    }
  }

  /**
   * Erase an element from the hash table.
   * \return An iterator to the next element, or the end iterator if there is no such element.
   */
  iterator erase(iterator it)
  {
    agxAssert(it._map == this);
    agxAssert(it._bucket < _num_buckets);
    this->erase_bucket(it._bucket);
    return ++it;
  }

  /**
   *  Remove all elements, keeping full capacity.
   */
  void clear(int /*IGNORED_AGX_PARAMETER*/ = 0)
  {
    for (size_t bucket=0; bucket<_num_buckets; ++bucket) {
      if (_states[bucket] == State::FILLED) {
        _states[bucket] = State::INACTIVE;
        _pairs[bucket].~PairT();
      }
    }
    _num_filled = 0;
    _max_probe_length = -1;
  }

  /**
   * Make room for this many elements in the hash table.
   */
  void reserve(size_t num_elems)
  {
    size_t required_buckets = num_elems + num_elems/2 + 1;
    if (required_buckets <= _num_buckets) {
      return;
    }
    size_t num_buckets = 4;
    while (num_buckets < required_buckets) { num_buckets *= 2; }

    auto new_states = (State*)_allocator.allocateBytes(num_buckets * sizeof(State));
    auto new_pairs  = (PairT*)_allocator.allocateBytes(num_buckets * sizeof(PairT));

    if (!new_states || !new_pairs) {
      _allocator.deallocateBytes(new_states, num_buckets * sizeof(State));
      _allocator.deallocateBytes(new_pairs, num_buckets * sizeof(PairT));
      throw std::bad_alloc();
    }

    //auto old_num_filled  = _num_filled;
    auto old_num_buckets = _num_buckets;
    auto old_states      = _states;
    auto old_pairs       = _pairs;

    _num_filled  = 0;
    _num_buckets = num_buckets;
    _mask        = _num_buckets - 1;
    _states      = new_states;
    _pairs       = new_pairs;

    std::fill_n(_states, num_buckets, State::INACTIVE);

    _max_probe_length = -1;

    for (size_t src_bucket=0; src_bucket<old_num_buckets; src_bucket++) {
      if (old_states[src_bucket] == State::FILLED) {
        auto& src_pair = old_pairs[src_bucket];

        auto dst_bucket = find_empty_bucket(src_pair.first);
        agxAssert(dst_bucket != (size_t)-1);
        agxAssert(_states[dst_bucket] != State::FILLED);
        _states[dst_bucket] = State::FILLED;
        new(_pairs + dst_bucket) PairT(std::move(src_pair));
        _num_filled += 1;

        src_pair.~PairT();
      }
    }

    //agxAssert(old_num_filles == _num_filled);

    _allocator.deallocateBytes(old_states, old_num_buckets * sizeof(State));
    _allocator.deallocateBytes(old_pairs,  old_num_buckets * sizeof(PairT));
  }


  /**
   * \return Raw memory access to the key/value buckets.
   */
  const void *ptr() const
  {
    return _pairs;
  }

protected:
  // Find the bucket with this key, or return nullptr
  template<typename T2>
  size_t find_filled_bucket(const T2& key) const
  {
    if (empty()) { return (size_t)-1; } // Optimization

    auto hash_value = _hasher(key);
    for (int offset=0; offset<=_max_probe_length; ++offset) {
      auto bucket = (hash_value + offset) & _mask;
      if (_states[bucket] == State::FILLED && agx::hashKeyEqual(_pairs[bucket].first, key)) {
        return bucket;
      }
      if (_states[bucket] == State::INACTIVE) {
        return (size_t)-1; // End of the chain!
      }
    }
    return (size_t)-1;
  }

    void erase_bucket(size_t bucket) {
      _states[bucket] = State::ACTIVE;
      _pairs[bucket].~PairT();
      _num_filled -= 1;
    }


private:
  // Can we fit another element?
  void check_expand_need()
  {
    reserve(_num_filled + 1);
  }
  // Find the bucket with this key, or return a good empty bucket to place the key in.
  // In the latter case, the bucket is expected to be filled.
  size_t find_or_allocate(const KeyT& key)
  {
    auto hash_value = _hasher(key);
    size_t hole = (size_t)-1;
    int offset=0;
    for (; offset<=_max_probe_length; ++offset) {
      auto bucket = (hash_value + offset) & _mask;

      if (_states[bucket] == State::FILLED) {
        if (agx::hashKeyEqual(_pairs[bucket].first, key)) {
          return bucket;
        }
      } else if (_states[bucket] == State::INACTIVE) {
        return bucket;
      } else {
        // ACTIVE: keep searching
        if (hole == (size_t)-1) {
          hole = bucket;
        }
      }
    }

    // No key found - but maybe a hole for it

    agxAssert(offset == _max_probe_length+1);

    if (hole != (size_t)-1) {
      return hole;
    }

    // No hole found within _max_probe_length
    for (; ; ++offset) {
      auto bucket = (hash_value + offset) & _mask;

      if (_states[bucket] != State::FILLED) {
        _max_probe_length = offset;
        return bucket;
      }
    }
  }

  // key is not in this map. Find a place to put it.
  size_t find_empty_bucket(const KeyT& key)
  {
    auto hash_value = _hasher(key);
    for (int offset=0; ; ++offset) {
      auto bucket = (hash_value + offset) & _mask;
      if (_states[bucket] != State::FILLED) {
        if (offset > _max_probe_length) {
          _max_probe_length = offset;
        }
        return bucket;
      }
    }
  }

private:
  HashT   _hasher;
  AllocatorT _allocator     = AllocatorT();
  State*  _states           = nullptr;
  PairT*  _pairs            = nullptr;
  size_t  _num_buckets      =  0;
  size_t  _num_filled       =  0;
  int     _max_probe_length = -1; // Our longest bucket-brigade is this long. ONLY when we have zero elements is this ever negative (-1).
  size_t  _mask             = 0;  // _num_buckets minus one
};


  template <typename KeyT, typename ValueT, typename HashT, typename AllocatorT>
  typename LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>::iterator begin(LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>& table)
  {
    return table.begin();
  }

  template <typename KeyT, typename ValueT, typename HashT, typename AllocatorT>
  typename LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>::iterator end(LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>& table)
  {
    return table.end();
  }


  template <typename KeyT, typename ValueT, typename HashT, typename AllocatorT>
  typename LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>::const_iterator cbegin(const LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>& table)
  {
    return table.begin();
  }

  template <typename KeyT, typename ValueT, typename HashT, typename AllocatorT>
  typename LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>::const_iterator cend(const LinearProbingHashTableImplementation<KeyT, ValueT, HashT, AllocatorT>& table)
  {
    return table.end();
  }


  /**
  Inheritance with partial specialization due to bug with ref_ptr containers.
  Partial specialization only works on classes, not methods which would have
  been a cleaner solution.

  See full comment in HashSet.h
  */
  template <typename KeyT, typename DataT, typename HashT = agx::HashFn<KeyT>, typename AllocatorT = ByteAllocator>
  class LinearProbingHashTable : public LinearProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT>
  {
    public:
      typedef LinearProbingHashTableImplementation<KeyT, DataT, HashT, AllocatorT> Implementation;

      LinearProbingHashTable() : Implementation()
      {
      }
  };



  template <typename KeyT, typename DataT, typename HashT, typename AllocatorT>
  class LinearProbingHashTable< agx::ref_ptr<KeyT>, DataT, HashT, AllocatorT> : public LinearProbingHashTableImplementation<agx::ref_ptr<KeyT>, DataT, HashT, AllocatorT>
  {
    public:
      typedef LinearProbingHashTableImplementation<agx::ref_ptr<KeyT>, DataT, HashT, AllocatorT> Implementation;
      typedef typename Implementation::iterator iterator;
      typedef typename Implementation::const_iterator const_iterator;

      LinearProbingHashTable() : Implementation()
      {}

      using Implementation::contains;
      bool contains(const KeyT *key) const
      {
        return this->find_filled_bucket(key) != (size_t)-1;
      }

      using Implementation::find;
      iterator find(const KeyT *key)
      {
        auto bucket = this->find_filled_bucket(key);
        if (bucket == (size_t)-1) {
          return this->end();
        }
        return iterator(this, bucket);
      }

      const_iterator find(const KeyT *key) const
      {
        auto bucket = this->find_filled_bucket(key);
        if (bucket == (size_t)-1) {
          return this->end();
        }
        return const_iterator(this, bucket);
      }

      using Implementation::erase;
      bool erase(const KeyT *key)
      {
        auto bucket = this->find_filled_bucket(key);
        if (bucket != (size_t)-1) {
          this->erase_bucket(bucket);
          return true;
        } else {
          return false;
        }
      }

  };
}

#endif
