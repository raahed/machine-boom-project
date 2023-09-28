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


#include <agx/Vector.h>
#include <agx/HashTable.h>

namespace agx
{

  /**
  This class is a combined container which has the find complexity of a HashTable,
  deterministic iteration of a Vector (using begin(), end()).
  The cost is storage requirement. Data is stored in a vector<pair<KeyT, DataT> >, which contains the Hash key and the data.
  A hash table in parallel store the index to the vector and the corresponding key.

  Beware of accessing the iterator of the vector, and change the Key value (pair.first). This will effectively invalidate the
  data structure. Unfortunately, there is no easy way protecting this unless only a const_iterator is used for accessing the vector, which
  would reduce the practical functionality of the container.
  */
  template <typename DataT, typename HashT = HashFn<DataT> >
  class SetVector
  {
  public:

    typedef DataT value_type;
    typedef const DataT& const_reference;
    typedef size_t size_type;
    typedef SetVector<DataT, HashT> ContainerType;
    typedef Vector<DataT> VectorType;
    typedef HashTable<DataT, size_t, HashT> HashType;

  private:
    VectorType m_vector;
    HashType m_hash;

  public:
#ifndef SWIG
    enum ClearPolicy
    {
      SHRINK_BUFFER=VectorType::SHRINK_BUFFER,           // Buffer is deallocated and replaced by an empty buffer
      SHRINK_BUFFER_AVERAGED=VectorType::SHRINK_BUFFER_AVERAGED,  // Buffer is shrunk if a smoothing average (which is updated each clear call) goes below a threshold
      MAINTAIN_BUFFER=VectorType::MAINTAIN_BUFFER          // Buffer is maintained
    };
#endif

    typedef typename VectorType::iterator iterator;
    typedef typename VectorType::const_iterator const_iterator;
    typedef typename VectorType::reverse_iterator reverse_iterator;
    typedef typename VectorType::const_reverse_iterator const_reverse_iterator;


    /** Constructor */
    SetVector()
    {}

    /** Copy constructor */
    SetVector(const ContainerType& other) : m_vector( other.m_vector ), m_hash( other.m_hash )
    {
      this->clear(SHRINK_BUFFER);
      *this = other;
    }

    /** Assignment operator */
    ContainerType& operator= (const ContainerType& other)  // NOLINT clang-tidy-7 doesn't realize that ContainerType is a SetVector.
    {
      m_vector = other.m_vector;
      m_hash = other.m_hash;
      return *this;
    }


    /** Destructor */
    ~SetVector()
    {
      // don't change buffer, will be dealloc'ed soon anyway
      clear(MAINTAIN_BUFFER);
    }

    /**
    \return the elements
    */
    const VectorType& vector() const
    {
      return m_vector;
    }

    /**
    \return the hash table containing vector indices for existing elements
    */
    const HashType& table() const
    {
      return m_hash;
    }

    /**
    Return the number of inserted elements.
    */
    inline size_t size() const
    {
      agxAssert( checkSize() );

      return m_vector.size();
    }

    /**
    Return true if the container is empty
    */
    inline bool empty() const
    {
      return m_vector.empty();
    }

    /**
    Insert a data element in to the end of the container.
    The element will end up at the end of the vector (analogue to a vector push_back!).
    What also happens is an insert into the hash table.

    If the data already exists in the Container, it will analogue to the HashTable
    replace the already existing data element in the vector.

    Identical to the insert() method

    Complexity: O(HashTable::find()+HashTable::insert()+Vector::push_back())

    \return iterator to the inserted pair (the last element if key did not exist before).
    */
    template< typename T2 >
    inline iterator push_back(const T2& data)
    {
      agxAssert( checkSize() );

      typename HashType::iterator it = m_hash.find(data);
      if (it != m_hash.end())
      {
        // Replace the already existing data value with the new
        m_vector[it->second] = data;
        return &m_vector[it->second];
      }

      // Key did not exist, insert at the end of the vector
      m_vector.push_back( data );

      // Hash on data, index is data value
      m_hash.insert( data, m_vector.size()-1 );

      return end()-1; // Return iterator to the newly inserted element.
    }

    /**
    Insert a data element in to the end of the container.
    The element will end up at the end of the vector (analogue to a vector push_back!).
    What also happens is an insert into the hash table.

    Identical to the push_back() method

    Complexity: O(HashTable::find()+HashTable::insert()+Vector::push_back())


    \return iterator to the inserted pair (at the end of the vector)
    */
    template< typename T2 >
    inline iterator insert(const T2& data)
    {
      agxAssert( checkSize() );

      return push_back(data);
    }

    /**
    Replace data in set vector.

    Similar to erase + insert but preserve iteration order.

    \note If \p oldData doesn't exist, the operation is aborted and end() is returned.
    \param oldData - old key (fails if not present in this hash vector)
    \param newData - new key for data
    \return iterator to the inserted data
    */
    template< typename OldDataT, typename NewDataT >
    iterator replace( const OldDataT& oldData, const NewDataT& newData )
    {
      agxAssert( checkSize() );

      auto oldIt = m_hash.find( oldData );
      if ( oldIt == m_hash.end() )
        return end();

      size_t index = oldIt->second;
      m_vector[ index ] = newData;
      m_hash.erase( oldIt );
      m_hash.insert( newData, index );

      return begin() + index;
    }

    /**
    Hash search for the data in the container.

    Complexity: O(HashTable::contains())


    \return true if key exists in the container.
    */
    template< typename T2 >
    inline bool contains(const T2& key) const
    {
      agxAssert( checkSize() );

      return m_hash.contains( key );
    }

    /**
    Perform a hash search for a given key.

    Complexity: O(HashTable::find())


    \return iterator to the found element, end() if none is found.
    */
    template< typename T2 >
    iterator find(const T2& key)
    {
      agxAssert( checkSize() );

      typename HashType::iterator it = m_hash.find( key );
      if (it == m_hash.end())
        return end();

      return &m_vector[it->second];
    }

    /**
    Perform a hash search for a given key.

    Complexity: O(HashTable::find())

    \return const_iterator to the found element, end() if none is found.
    */
    template< typename T2 >
    const_iterator find(const T2& key) const
    {
      agxAssert( checkSize() );

      typename HashType::const_iterator it = m_hash.find( key );
      if (it == m_hash.end())
        return end();

      // Return index to the element in the vector
      return &m_vector[it->second];
    }

#ifndef SWIG
    /**
    \return iterator to first element in the vector container
    */
    inline iterator begin()
    {
      agxAssert( checkSize() );

      return m_vector.begin();
    }

    /**
    \return const_iterator to first element in the vector container
    */
    inline const_iterator begin() const
    {
      agxAssert( checkSize() );

      return m_vector.begin();
    }

    /**
    \return reverse iterator to first element in the vector container
    */
    inline reverse_iterator rbegin()
    {
      agxAssert( checkSize() );

      return m_vector.rbegin();
    }

    /**
    \return reverse const_iterator to first element in the vector container
    */
    inline const_reverse_iterator rbegin() const
    {
      agxAssert( checkSize() );

      return m_vector.rbegin();
    }

    /**
    \return iterator marking end of the vector container
    */
    inline iterator end()
    {
      agxAssert( checkSize() );

      return m_vector.end();
    }

    /**
    \return const_iterator marking end of the vector container
    */
    inline const_iterator end() const
    {
      agxAssert( checkSize() );
      return m_vector.end();
    }

    /**
    \return reverse iterator marking end of the vector container
    */
    inline reverse_iterator rend()
    {
      agxAssert( checkSize() );

      return m_vector.rend(); }

    /**
    \return reverse const_iterator marking end of the vector container
    */
    inline const_reverse_iterator rend() const
    {
      agxAssert( checkSize() );
      return m_vector.rend();
    }
#endif
    /**
    \return a reference to the first element in the vector container
    */
    inline DataT& front()
    {
      agxAssert( checkSize() );
      return m_vector[0];
    }

   /**
    \return a const reference to the first element in the vector container
    */
    inline const DataT& front() const
    {
      agxAssert( checkSize() );
      return m_vector[0];
    }

    /**
    \return a reference to the last element in the vector container
    */
    inline DataT& back()
    {
      agxAssert( checkSize() );
      return m_vector[m_vector.size()-1];
    }

    /**
    \return a const reference to the last element in the vector container
    */
    inline const DataT& back() const
    {
      agxAssert( checkSize() );
      return m_vector[m_vector.size()-1];
    }


    /**
    Erase an element from the container.
    The elements in the vector CAN change order due to the use
    of Vector::fastErase, where the erased and the last element swap places.

    Complexity: O(HashTable::find() + HashTable::erase() + Vector::fastErase() )

    \return false if element was not found
    */
    template< typename T2 >
    bool erase(const T2& key)
    {
      agxAssert( checkSize() );

      typename HashType::iterator it = m_hash.find( key );
      if (it == m_hash.end())
        return false;

      erase( it );
      return true;
    }

    /**
    Erase an element using an iterator.

    This method assumes that it is a valid/existing iterator in the container.
    If it is not, there is a chance that the iterator is point at something we do not want to
    touch, hence could cause crash. it is checked to see if it is == end(), then the method will return false.
    For other invalid iterators, behavior is undefined.

    Complexity: O(HashTable::find() + HashTable::erase() + Vector::fastErase() )

    \param it - Iterator to the element to be removed.
    \return true if erasing the element referred by it was successful. False if it==end()
    */
    bool erase(const iterator& it)
    {

      agxAssert( checkSize() );

      if (it == end())
        return false;
      agxAssert( it != end() );

      DataT& key = *it;
      typename HashType::iterator hashIt = m_hash.find( key );

      agxAssert ( hashIt != m_hash.end() );

      erase( hashIt );

      return true;
    }

    /**
    Remove all elements.
    \param policy - specifies the erase policy.
    */
#ifndef SWIG
    void clear(ClearPolicy policy = SHRINK_BUFFER_AVERAGED)
#else
    void clear()
#endif
    {

      agxAssert( checkSize() );

      m_hash.clear( (typename HashType::ClearPolicy)policy );
      m_vector.clear( (typename VectorType::ClearPolicy)policy );
    }

    /**
    Access an element in the vector using an index
    \return reference to the Data in the i:th element in the vector
    */
    inline DataT& operator[](size_t i )
    {
      agxAssert( checkSize() );

      agxAssert(i < m_vector.size());
      return m_vector[i];
    }

    /**
    Access an element in the vector using an index
    \return const reference to the Data in the i:th element in the vector
    */
    inline const DataT& operator[](size_t i ) const
    {
      agxAssert( checkSize() );

      agxAssert(i < m_vector.size());
      return m_vector[i];
    }


    /**
    Access an element in the vector using an index
    \return reference to the Data in the i:th element in the vector
    */

    inline const DataT& at(size_t index) const
    {
      return m_vector.at(index);
    }


    /**
    Access an element in the vector using an index
    \return reference to the Data in the i:th element in the vector
    */

    inline DataT& at(size_t index)
    {
      return m_vector.at(index);
    }


    /**
    Reserve capacity in the hash table.
    \param size - reserve size elements
    */
    void reserve(size_t size)
    {
      agxAssert( checkSize() );

      m_vector.reserve( size );
      m_hash.reserve( size );
    }



    /**
    Purge the hash set. Good to use when removing multiple items, since iterators
    are invalidated after any modification. The purger should be a functor class
    which takes the key as parameter and return true if the entry should
    be removed from the hash set.

    Complexity N - size of container, n - number of purged elements: O(n*HashTable::erase())

    Example of a purger implementation:

    template < typename T1, typename T2 >
    class MyPurger
    {
    public:
      MyPurger() {}
      inline bool operator()(const T1& a, const T2& b)
      {
        return a > b;
      }
    };

    \param purger - An instance of a class with an bool () operator which when it returns true, the element will be removed
    from the container
    \return number of purged elements
    */
    template<typename T>
    size_t purge(T purger)
    {
      if (empty()) // Nothing to do
        return 0;

#if 0 // Slower implementation
      typename VectorType savedElements; // Elements to keep
      savedElements.reserve( m_vector.size());
      typename VectorType::const_iterator it = m_vector.begin();
      for(; it != m_vector.end(); ++it)
      {
        // Should we keep it?
        if (!purger(*it))
          savedElements.push_back(*it); // store all elements that should remain
      }
      m_vector.clear(agx::Container::SHRINK_BUFFER_AVERAGED);
      m_hash.clear(agx::Container::SHRINK_BUFFER_AVERAGED);

      // Make sure we can fit all the saved elements
      m_vector.reserve( savedElements.size() );
      m_hash.reserve( savedElements.size() );

      for(typename VectorType::const_iterator it = savedElements.begin(); it != savedElements.end(); ++it)
      {
        m_vector.push_back( *it );
        // Hash on data, index is data value
        m_hash.insert( *it, m_vector.size()-1 );
      }
#else // Faster implementation
      typename agx::Vector<typename VectorType::value_type *> removeElements; // Safe to use pointers?
      removeElements.reserve( m_vector.size());
      for(typename VectorType::iterator it = m_vector.begin(); it != m_vector.end(); ++it)
      {
        // Should we keep it?
        if (!purger(*it))
          removeElements.push_back(&(*it)); // store all elements that should remain
      }

      for(typename agx::Vector<typename VectorType::value_type *>::const_iterator it = removeElements.begin(); it != removeElements.end(); ++it)
      {
        this->erase( **it );
      }

      return removeElements.size();
#endif
    }

  private:

    // Return true if size of vector and hash is equal
    AGX_FORCE_INLINE bool checkSize() const {
      return m_vector.size() == m_hash.size();
    }

    void erase( typename HashType::iterator it )
    {
      size_t idx = it->second;

      m_hash.erase( it ); // remove from hash

      // Remove from vector too:
      if (m_vector.size() > 1 && idx != (m_vector.size()-1))
      {
        // Get the last element
        typename VectorType::value_type lastElement = m_vector.back();

        // Put it where we have removed an element
        m_vector[idx] = lastElement;

        // remove the last now
        m_vector.pop_back();

        // Rehash the moved one with the new index
        m_hash.insert( lastElement, idx );
      }
      else
        // Only one element, OR its the last we are removing: just remove it
        m_vector.pop_back();

      agxAssert( checkSize() );
    }


  };

}

