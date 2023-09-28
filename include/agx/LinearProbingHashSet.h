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




#pragma once

#include <cstdlib>
#include <iterator>
#include <utility>

#include <agx/ref_ptr.h>
#include <agx/hash.h>

namespace agx
{
  /**
  * A cache-friendly hash set with open addressing, linear probing and power-of-two capacity
  */
  template <typename KeyT, typename HashT = agx::HashFn<KeyT>>
  class LinearProbingHashSetImplementation
  {
  private:
    using MyType = LinearProbingHashSetImplementation<KeyT, HashT>;
    enum class State : uint8_t
    {
        INACTIVE, // Never been touched
        ACTIVE,   // Is inside a search-chain, but is empty
        FILLED    // Is set with key/value
    };

  public:
    using size_type       = size_t;
    using value_type      = KeyT;
    using reference       = KeyT&;
    using const_reference = const KeyT&;

    class iterator
    {
    public:
      using iterator_category = std::forward_iterator_tag;
      using difference_type   = size_t;
      using distance_type     = size_t;
      using value_type        = KeyT;
      using pointer           = value_type*;
      using reference         = value_type&;

      iterator()
      {
      }

      iterator(MyType* hash_set, size_t bucket) : _set(hash_set), _bucket(bucket)
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
        return iterator(_set, old_index);
      }

      reference operator*() const
      {
        return _set->_keys[_bucket];
      }

      pointer operator->() const
      {
        return _set->_keys + _bucket;
      }

      bool operator==(const iterator& rhs)
      {
        agxAssert(_set == rhs._set);
        return this->_bucket == rhs._bucket;
      }

      bool operator!=(const iterator& rhs)
      {
        agxAssert(_set == rhs._set);
        return this->_bucket != rhs._bucket;
      }

    private:
      void goto_next_element()
      {
        agxAssert(_bucket < _set->_num_buckets);
        do {
          _bucket++;
        } while (_bucket < _set->_num_buckets && _set->_states[_bucket] != State::FILLED);
      }

    //private:
    //  friend class MyType;
    public:
      MyType* _set;
      size_t  _bucket;
    };

    class const_iterator
    {
    public:
      using iterator_category = std::forward_iterator_tag;
      using difference_type   = size_t;
      using distance_type     = size_t;
      using value_type        = const KeyT;
      using pointer           = value_type*;
      using reference         = value_type&;

      const_iterator() { }

      const_iterator(iterator proto) : _set(proto._set), _bucket(proto._bucket)
      {
      }

      const_iterator(const MyType* hash_set, size_t bucket) : _set(hash_set), _bucket(bucket)
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
        return const_iterator(_set, old_index);
      }

      reference operator*() const
      {
        return _set->_keys[_bucket];
      }

      pointer operator->() const
      {
        return _set->_keys + _bucket;
      }

      bool operator==(const const_iterator& rhs)
      {
        agxAssert(_set == rhs._set);
        return this->_bucket == rhs._bucket;
      }

      bool operator!=(const const_iterator& rhs)
      {
        agxAssert(_set == rhs._set);
        return this->_bucket != rhs._bucket;
      }

    private:
      void goto_next_element()
      {
        agxAssert(_bucket < _set->_num_buckets);
        do {
          _bucket++;
        } while (_bucket < _set->_num_buckets && _set->_states[_bucket] != State::FILLED);
      }

    //private:
    //  friend class MyType;
    public:
      const MyType* _set;
      size_t        _bucket;
    };


    class insert_iterator : public iterator
    {
      public:
        insert_iterator() : iterator(), m_overwrite(false)
        {
        }

        insert_iterator(MyType* hash_set, size_t bucket, bool overwrite)
            : iterator(hash_set, bucket)
            , m_overwrite(overwrite)
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

    LinearProbingHashSetImplementation() = default;

    LinearProbingHashSetImplementation(const LinearProbingHashSetImplementation& other)
    {
      reserve(other.size());
      insert(cbegin(other), cend(other));
    }

      LinearProbingHashSetImplementation(LinearProbingHashSetImplementation&& other)
    {
      *this = std::move(other);
    }

    LinearProbingHashSetImplementation& operator=(const LinearProbingHashSetImplementation& other)
    {
      clear();
      reserve(other.size());
      insert(cbegin(other), cend(other));
      return *this;
    }

    LinearProbingHashSetImplementation& operator=(LinearProbingHashSetImplementation&& other)
    {
      this->swap(other);
      return *this;
    }

    ~LinearProbingHashSetImplementation()
    {
      for (size_t bucket=0; bucket<_num_buckets; ++bucket) {
        if (_states[bucket] == State::FILLED) {
          _keys[bucket].~KeyT();
        }
      }
      free(_states);
      free(_keys);
    }

    void swap(LinearProbingHashSetImplementation& other)
    {
      std::swap(_hasher,           other._hasher);
      std::swap(_states,           other._states);
      std::swap(_keys,             other._keys);
      std::swap(_num_buckets,      other._num_buckets);
      std::swap(_num_filled,       other._num_filled);
      std::swap(_max_probe_length, other._max_probe_length);
      std::swap(_mask,             other._mask);
    }

    // -------------------------------------------------------------

    iterator begin()
    {
      size_t bucket = 0;
      while (bucket<_num_buckets && _states[bucket] != State::FILLED) {
        ++bucket;
      }
      return iterator(this, bucket);
    }

    const_iterator begin() const
    {
      size_t bucket = 0;
      while (bucket<_num_buckets && _states[bucket] != State::FILLED) {
        ++bucket;
      }
      return const_iterator(this, bucket);
    }

    iterator end()
    { return iterator(this, _num_buckets); }

    const_iterator end() const
    { return const_iterator(this, _num_buckets); }

    size_t size() const
    {
      return _num_filled;
    }

    bool empty() const
    {
      return _num_filled==0;
    }

    // ------------------------------------------------------------

    iterator find(const KeyT& key)
    {
      auto bucket = this->find_filled_bucket(key);
      if (bucket == (size_t)-1) {
        return this->end();
      }
      return iterator(this, bucket);
    }

    const_iterator find(const KeyT& key) const
    {
      auto bucket = this->find_filled_bucket(key);
      if (bucket == (size_t)-1) {
        return this->end();
      }
      return const_iterator(this, bucket);
    }

    bool contains(const KeyT& k) const
    {
      return find_filled_bucket(k) != (size_t)-1;
    }

    size_t count(const KeyT& k) const
    {
      return find_filled_bucket(k) != (size_t)-1 ? 1 : 0;
    }

    // -----------------------------------------------------

    // Insert an element, unless it already exists.
    // Returns a pair consisting of an iterator to the inserted element
    // (or to the element that prevented the insertion)
    // and a bool denoting whether the insertion took place.
    insert_iterator insert(const KeyT& key)
    {
      check_expand_need();

      auto bucket = find_or_allocate(key);

      if (_states[bucket] == State::FILLED) {
        _keys[bucket] = key;
        return insert_iterator(this, bucket, true);
      } else {
        _states[bucket] = State::FILLED;
        new(_keys + bucket) KeyT(key);
        _num_filled++;
        return insert_iterator(this, bucket, false);
      }
    }

    // Insert an element, unless it already exists.
    // Returns a pair consisting of an iterator to the inserted element
    // (or to the element that prevented the insertion)
    // and a bool denoting whether the insertion took place.
    insert_iterator insert(KeyT&& key)
    {
      check_expand_need();

      auto bucket = find_or_allocate(key);

      if (_states[bucket] == State::FILLED) {
        _keys[bucket] = std::move(key);
        return insert_iterator(this, bucket, true);
      } else {
        _states[bucket] = State::FILLED;
        new(_keys + bucket) KeyT(std::move(key));
        _num_filled++;
        return insert_iterator(this, bucket, false);
      }
    }

    template<class... Args>
    insert_iterator emplace(Args&&... args)
    {
      return insert(KeyT(std::forward<Args>(args)...));
    }

    void insert(const_iterator begin, const_iterator end)
    {
      for (; begin != end; ++begin) {
        insert(*begin);
      }
    }

    // Same as above, but contains(key) MUST be false
    void insert_unique(KeyT&& key)
    {
      agxAssert(!contains(key));
      check_expand_need();
      auto bucket = find_empty_bucket(key);
      _states[bucket] = State::FILLED;
      new(_keys + bucket) KeyT(std::move(key));
      _num_filled++;
    }

    // -------------------------------------------------------

    /* Erase an element from the hash set.
       return false if element was not found */
    bool erase(const KeyT& key)
    {
      auto bucket = find_filled_bucket(key);
      if (bucket != (size_t)-1) {
        erase_bucket(bucket);
        return true;
      } else {
        return false;
      }
    }

    /* Erase an element using an iterator.
       Returns an iterator to the next element (or end()). */
    iterator erase(iterator it)
    {
      agxAssert(it._set == this);
      agxAssert(it._bucket < _num_buckets);
      erase_bucket(it._bucket);
      return ++it;
    }

    // Remove all elements, keeping full capacity.
    void clear()
    {
      for (size_t bucket=0; bucket<_num_buckets; ++bucket) {
        if (_states[bucket] == State::FILLED) {
          _states[bucket] = State::INACTIVE;
          _keys[bucket].~KeyT();
        }
      }
      _num_filled = 0;
      _max_probe_length = -1;
    }

    // Make room for this many elements
    void reserve(size_t num_elems)
    {
      size_t required_buckets = num_elems + num_elems/2 + 1;
      if (required_buckets <= _num_buckets) {
        return;
      }
      size_t num_buckets = 4;
      while (num_buckets < required_buckets) { num_buckets *= 2; }

      auto new_states = (State*)malloc(num_buckets * sizeof(State));
      auto new_keys  = (KeyT*)malloc(num_buckets * sizeof(KeyT));

      if (!new_states || !new_keys) {
        free(new_states);
        free(new_keys);
        throw std::bad_alloc();
      }

      // auto old_num_filled  = _num_filled;
      auto old_num_buckets = _num_buckets;
      auto old_states      = _states;
      auto old_keys        = _keys;

      _num_filled  = 0;
      _num_buckets = num_buckets;
      _mask        = _num_buckets - 1;
      _states      = new_states;
      _keys        = new_keys;

      std::fill_n(_states, num_buckets, State::INACTIVE);

      _max_probe_length = -1;

      for (size_t src_bucket=0; src_bucket<old_num_buckets; src_bucket++) {
        if (old_states[src_bucket] == State::FILLED) {
          auto& src = old_keys[src_bucket];

          auto dst_bucket = find_empty_bucket(src);
          agxAssert(dst_bucket != (size_t)-1);
          agxAssert(_states[dst_bucket] != State::FILLED);
          _states[dst_bucket] = State::FILLED;
          new(_keys + dst_bucket) KeyT(std::move(src));
          _num_filled += 1;

          src.~KeyT();
        }
      }

      // DCHECK_EQ_F(old_num_filled, _num_filled);

      free(old_states);
      free(old_keys);
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
        if (_states[bucket] == State::FILLED && agx::hashKeyEqual(_keys[bucket], key)) {
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
      _keys[bucket].~KeyT();
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
          if (agx::hashKeyEqual(_keys[bucket], key)) {
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
    State*  _states           = nullptr;
    KeyT*   _keys             = nullptr;
    size_t  _num_buckets      =  0;
    size_t  _num_filled       =  0;
    int     _max_probe_length = -1; // Our longest bucket-brigade is this long. ONLY when we have zero elements is this ever negative (-1).
    size_t  _mask             = 0;  // _num_buckets minus one
  };

  template <typename KeyT, typename HashT>
  typename LinearProbingHashSetImplementation<KeyT, HashT>::iterator begin(LinearProbingHashSetImplementation<KeyT, HashT>& set)
  {
    return set.begin();
  }

  template <typename KeyT, typename HashT>
  typename LinearProbingHashSetImplementation<KeyT, HashT>::iterator end(LinearProbingHashSetImplementation<KeyT, HashT>& set)
  {
    return set.end();
  }


  template <typename KeyT, typename HashT>
  typename LinearProbingHashSetImplementation<KeyT, HashT>::const_iterator cbegin(const LinearProbingHashSetImplementation<KeyT, HashT>& set)
  {
    return set.begin();
  }

  template <typename KeyT, typename HashT>
  typename LinearProbingHashSetImplementation<KeyT, HashT>::const_iterator cend(const LinearProbingHashSetImplementation<KeyT, HashT>& set)
  {
    return set.end();
  }


  /**
  Inheritance with partial specialization due to bug with ref_ptr containers.
  Partial specialization only works on classes, not methods which would have
  been a cleaner solution.

  See full comment in HashSet.h
  */
  template <typename KeyT, typename HashT = agx::HashFn<KeyT>>
  class LinearProbingHashSet : public LinearProbingHashSetImplementation<KeyT, HashT>
  {
    public:
      typedef LinearProbingHashSetImplementation<KeyT, HashT> Implementation;

      LinearProbingHashSet() : Implementation()
      {
      }
  };



  template <typename KeyT, typename HashT>
  class LinearProbingHashSet< agx::ref_ptr<KeyT>, HashT> : public LinearProbingHashSetImplementation<agx::ref_ptr<KeyT>, HashT>
  {
    public:
      typedef LinearProbingHashSetImplementation<agx::ref_ptr<KeyT>, HashT> Implementation;
      typedef typename Implementation::iterator iterator;
      typedef typename Implementation::const_iterator const_iterator;

      LinearProbingHashSet() : Implementation()
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
