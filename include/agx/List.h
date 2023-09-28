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

#ifndef AGX_LIST_H
#define AGX_LIST_H

#define AGX_USE_AGX_LIST_IMPL 1

#if AGX_USE_AGX_LIST_IMPL

#include <agx/Math.h>
#include <iterator>

#ifdef __linux__
#include <cstddef>
#endif

namespace agx
{
#define AGX_LIST_ITERATOR_TYPEDEFS                           \
  typedef std::bidirectional_iterator_tag iterator_category; \
  typedef T&                              reference;         \
  typedef const T&                        const_reference;   \
  typedef T*                              pointer;           \
  typedef const T*                        const_pointer;     \
  typedef T                               value_type;        \
  typedef ptrdiff_t                       difference_type

  template < typename T >
  class LNode
  {
    public:
      LNode() : m_element( T() ), m_next( nullptr ), m_prev( nullptr ) {}
      LNode( T element ) : m_element( element ), m_next( nullptr ), m_prev( nullptr ) {}
      T      m_element;
      LNode* m_next;
      LNode* m_prev;
  };

  class ItOp
  {
    public:
      typedef agx::UInt value_type;

    public:
      struct PPOp
      {
        template< typename T >
        T& operator()( T& val ) const
        {
          return ++val;
        }
      };

      struct MMOp
      {
        template< typename T >
        T& operator()( T& val ) const
        {
          return --val;
        }
      };

    public:
      explicit ItOp( value_type num )
        : m_num( num ) {}

      template< typename ItT, typename Op >
      ItT operator()( const ItT& org, Op op ) const
      {
        ItT ret = org;
        for ( value_type i = value_type( 0 ); i < m_num; ++i )
          op( ret );
        return ret;
      }

    private:
      value_type m_num;
  };

  template< typename ItT >
  ItT operator + ( const ItT& it, const agx::ItOp& inc )
  {
    return inc( it, agx::ItOp::PPOp() );
  }

  template< typename ItT >
  ItT operator - ( const ItT& it, const agx::ItOp& inc )
  {
    return inc( it, agx::ItOp::MMOp() );
  }

  template < typename T >
  class List
  {
    public:
      typedef List< T >     my_type;
      typedef T             value_type;
      typedef T&            reference;
      typedef const T&      const_reference;
      typedef T*            pointer;
      typedef const T*      const_pointer;
      typedef size_t        size_type;
      typedef LNode<T>      node_type;
      typedef LNode<T>*     node_ptr_type;
      typedef LNode<T>&     node_ref_type;

    public:
      class const_iterator
      {
        public:
          AGX_LIST_ITERATOR_TYPEDEFS;

        public:
          friend class List;

          inline const_iterator() : m_ptr( nullptr ) {}
          inline explicit const_iterator( node_ptr_type ptr ) : m_ptr( ptr ) {}
          inline const_iterator( const const_iterator& rhs )
          {
            *this = rhs;
          }
          inline const_iterator& operator=( const const_iterator& rhs )
          {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline const_iterator& operator++()
          {
            m_ptr = m_ptr->m_next;
            return *this;
          }
          inline const_iterator operator++(int)
          {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
          }
          inline const_iterator operator+( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::PPOp() );
          }
          inline const_iterator& operator--()
          {
            m_ptr = m_ptr->m_prev;
            return *this;
          }
          inline const_iterator operator--(int)
          {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
          }
          inline const_iterator operator-( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::MMOp() );
          }
          inline const_reference operator*() const
          {
            return m_ptr->m_element;
          }
          inline const_pointer operator->() const
          {
            return &(m_ptr->m_element);
          }
          inline bool operator==(const const_iterator& i) const
          {
            return m_ptr == i.m_ptr;
          }
          inline bool operator!=(const const_iterator& i) const
          {
            return m_ptr != i.m_ptr;
          }
        private:
          node_ptr_type m_ptr;
      };

      class iterator
      {
        public:
          AGX_LIST_ITERATOR_TYPEDEFS;

        public:
          friend class List;

          inline iterator() : m_ptr( nullptr ) {}
          inline explicit iterator( node_ptr_type ptr ) : m_ptr( ptr ) {}
          inline iterator( const iterator& rhs )
          {
            *this = rhs;
          }
          inline iterator& operator=( const iterator& rhs )
          {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline iterator& operator++()
          {
            m_ptr = m_ptr->m_next;
            return *this;
          }
          inline iterator operator++(int)
          {
            iterator tmp = *this;
            ++(*this);
            return tmp;
          }
          inline iterator operator+( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::PPOp() );
          }
          inline iterator& operator--()
          {
            m_ptr = m_ptr->m_prev;
            return *this;
          }
          inline iterator operator--(int)
          {
            iterator tmp = *this;
            --(*this);
            return tmp;
          }
          inline iterator operator-( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::MMOp() );
          }
          inline reference operator*() const
          {
            return m_ptr->m_element;
          }
          inline pointer operator->() const
          {
            return &(m_ptr->m_element);
          }
          inline bool operator==(const iterator& i) const
          {
            return m_ptr == i.m_ptr;
          }
          inline bool operator==(const const_iterator& i) const
          {
            return m_ptr == i.m_ptr;
          }
          inline bool operator!=(const iterator& i) const
          {
            return m_ptr != i.m_ptr;
          }
          inline bool operator!=(const const_iterator& i) const
          {
            return m_ptr != i.m_ptr;
          }
          inline operator const_iterator() const
          {
            return const_iterator(m_ptr);
          }
        private:
          node_ptr_type m_ptr;
      };

      class const_reverse_iterator
      {
        public:
          AGX_LIST_ITERATOR_TYPEDEFS;

        public:
          friend class List;

          inline const_reverse_iterator() : m_ptr( nullptr ) {}
          inline explicit const_reverse_iterator( node_ptr_type ptr ) : m_ptr( ptr ) {}
          inline const_reverse_iterator( const const_reverse_iterator& rhs )
          {
            *this = rhs;
          }
          inline const_reverse_iterator( const iterator& rhs )
          {
            *this = rhs;
          }
          inline const_reverse_iterator( const const_iterator& rhs )
          {
            *this = rhs;
          }
          inline const_reverse_iterator& operator=( const const_reverse_iterator& rhs )
          {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline const_reverse_iterator& operator=( const iterator& rhs )
          {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline const_reverse_iterator& operator=( const const_iterator& rhs )
          {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline const_reverse_iterator& operator++()
          {
            m_ptr = m_ptr->m_prev;
            return *this;
          }
          inline const_reverse_iterator operator++(int)
          {
            const_reverse_iterator tmp = *this;
            ++(*this);
            return tmp;
          }
          inline const_reverse_iterator operator+( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::PPOp() );
          }
          inline const_reverse_iterator& operator--()
          {
            m_ptr = m_ptr->m_next;
            return *this;
          }
          inline const_reverse_iterator operator--(int)
          {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
          }
          inline const_reverse_iterator operator-( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::MMOp() );
          }
          inline const_reference operator*() const
          {
            return m_ptr->m_prev->m_element;
          }
          inline const_pointer operator->() const
          {
            return &(m_ptr->m_prev->m_element);
          }
          inline bool operator==(const const_reverse_iterator& i) const
          {
            return m_ptr == i.m_ptr;
          }
          inline bool operator!=(const const_reverse_iterator& i) const
          {
            return m_ptr != i.m_ptr;
          }
        private:
          node_ptr_type m_ptr;
      };

      class reverse_iterator
      {
        public:
          AGX_LIST_ITERATOR_TYPEDEFS;

        public:
          friend class List;

          inline reverse_iterator() : m_ptr( nullptr ) {}
          inline explicit reverse_iterator( node_ptr_type ptr ) : m_ptr( ptr ) {}
          inline reverse_iterator( const reverse_iterator& rhs )
          {
            *this = rhs;
          }
          inline reverse_iterator( const iterator& rhs )
          {
            *this = rhs;
          }
          inline reverse_iterator& operator=( const iterator& rhs ) {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline reverse_iterator& operator=( const const_reverse_iterator& rhs ) {
            m_ptr = rhs.m_ptr;
            return *this;
          }
          inline reverse_iterator& operator++()
          {
            m_ptr = m_ptr->m_prev;
            return *this;
          }
          inline reverse_iterator operator++(int)
          {
            reverse_iterator tmp = *this;
            ++(*this);
            return tmp;
          }
          inline reverse_iterator operator+( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::PPOp() );
          }
          inline reverse_iterator& operator--()
          {
            m_ptr = m_ptr->m_next;
            return *this;
          }
          inline reverse_iterator operator--(int)
          {
            reverse_iterator tmp = *this;
            --(*this);
            return tmp;
          }
          inline reverse_iterator operator-( const agx::ItOp& inc ) const
          {
            return inc( *this, ItOp::MMOp() );
          }
          inline reference operator*() const
          {
            return m_ptr->m_prev->m_element;
          }
          inline pointer operator->() const
          {
            return &(m_ptr->m_prev->m_element);
          }
          inline bool operator==(const reverse_iterator& i) const
          {
            return m_ptr == i.m_ptr;
          }
          inline bool operator!=(const reverse_iterator& i) const
          {
            return m_ptr != i.m_ptr;
          }
          inline operator const_reverse_iterator()
          {
            return const_reverse_iterator(m_ptr);
          }
        private:
          node_ptr_type m_ptr;
      };

      class EqualPred1
      {
        public:
          EqualPred1( const_reference obj ) : m_obj( obj ) {}
          bool operator()( const_reference obj ) const { return m_obj == obj; }
        private:
          value_type m_obj;
        // Not defined.
        private:
          EqualPred1();
      };

      class EqualPred2
      {
        public:
          bool operator()( const_reference left, const_reference right ) const { return left == right; }
      };

      class LessPred1
      {
        public:
          LessPred1( const_reference obj ) : m_obj( obj ) {}
          bool operator()( const_reference obj ) const { return m_obj == obj; }
        private:
          value_type m_obj;
        // Not defined.
        private:
          LessPred1();
      };

      class LessPred2
      {
        public:
          bool operator()( const_reference left, const_reference right ) const { return left < right; }
      };

    public:
      /**
      Default constructor, size zero and begin() == end().
      */
      List()
        : m_size( 0 ), m_end( allocate() ), m_front( m_end )
      {
      }

      /**
      Copy constructor.
      */
      List( const List& other )
        : m_size( 0 ), m_end( allocate() ), m_front( m_end )
      {
        *this = other;
      }

      /**
      Destructor.
      */
      ~List()
      {
        clear();
        if ( m_front == m_end && m_front != nullptr )
          deallocate( m_front );
      }

      /**
      Assignment operator. Clears current state and copies nodes
      from \p other to this.
      \param other - list to copy from
      \return self
      */
      List& operator=( const my_type& other )
      {
        if ( this == &other )
          return *this;

        clear();
        insert( end(), other.begin(), other.end() );
        return *this;
      }

      /**
      Comparison of lists of same size.
      \param other - other list
      \return true if all elements in this list are identical to the values in the other list
      */
      bool operator==( const my_type& other ) const
      {
        if ( size() != other.size() )
          return false;

        for ( const_iterator myIt = begin(), otherIt = other.begin(), endIt = end(); myIt != endIt; ++myIt, ++otherIt )
          if ( *myIt != *otherIt )
            return false;

        return true;
      }

      /**
      Find iterator to FIRST element with value \p _val.
      \param _val - value to search for
      \return iterator to element with value \p _val if found - otherwise end iterator
      */
      const_iterator find( const_reference _val ) const
      {
        iterator i(m_front);
        iterator iend(m_end);
        for ( ; i != iend; ++i ) {
          if ( *i == _val ) return i;
        }
        return end();
      }

      /**
      Find iterator to FIRST element with value \p _val.
      \param _val - value to search for
      \return iterator to element with value \p _val if found - otherwise end iterator
      */
      iterator find( const_reference _val )
      {
        iterator i(m_front);
        iterator iend(m_end);
        for ( ; i != iend; ++i ) {
          if ( *i == _val ) return i;
        }
        return end();
      }

      /**
      Check if a value \p _val is part of this list (at least one time).
      \param _val - value to check for
      \return true if value \p _val shows up at least one time in this list
      */
      bool contains( const_reference _val )
      {
        return find( _val ) != end();
      }

      /**
      \return the begin iterator of this list (end if empty)
      */
      inline       iterator begin()       { return       iterator( m_front ); }

      /**
      \return the begin iterator of this list (end if empty)
      */
      inline const_iterator begin() const { return const_iterator( m_front ); }

      /**
      \return the end iterator of this list
      */
      inline       iterator end()       { return       iterator( m_end ); }

      /**
      \return the end iterator of this list
      */
      inline const_iterator end() const { return const_iterator( m_end ); }

      /**
      \return the reverse begin iterator of this list
      */
      inline       reverse_iterator rbegin()       { return       reverse_iterator( m_end ); }

      /**
      \return the reverse begin iterator of this list
      */
      inline const_reverse_iterator rbegin() const { return const_reverse_iterator( m_end ); }

      /**
      \return the reverse end iterator of this list
      */
      inline       reverse_iterator rend()       { return       reverse_iterator( m_front ); }

      /**
      \return the reverse end iterator of this list
      */
      inline const_reverse_iterator rend() const { return const_reverse_iterator( m_front ); }

      /**
      \return the size of this list
      */
      inline size_type size() const { return m_size; }

      /**
      \return true if this list is empty - otherwise false
      */
      inline bool empty() const { return m_size == 0; }

      /**
      \note It's not valid to call this method if this list is empty.
      \return the front element of this list
      */
      inline       reference front()       { agxAssert( !empty() ); return m_front->m_element; }

      /**
      \note It's not valid to call this method if this list is empty.
      \return the front element of this list
      */
      inline const_reference front() const { agxAssert( !empty() ); return m_front->m_element; }

      /**
      \note It's not valid to call this method if this list is empty.
      \return the last element of this list
      */
      inline       reference back()       { agxAssert( !empty() ); return m_end->m_prev->m_element; }

      /**
      \note It's not valid to call this method if this list is empty.
      \return the last element of this list
      */
      inline const_reference back() const { agxAssert( !empty() ); return m_end->m_prev->m_element; }

      /**
      Insert an element \p c_ref before \p _where.
      \param _where - iterator to place where to insert element
      \param c_ref - element to insert
      \return iterator to the inserted element
      */
      inline iterator insert( const_iterator _where, const_reference c_ref )
      {
        node_ptr_type node = allocate();
        node->m_element = c_ref;
        node->m_prev = _where.m_ptr->m_prev;
        node->m_next = _where.m_ptr;
        _where.m_ptr->m_prev = node;
        ++m_size;
        if ( node->m_prev == nullptr ) {
          m_front = node;
        }
        else {
          node->m_prev->m_next = node;
        }
        return iterator(node);
      }

      /**
      Insert N elements before \p _where.
      \param _where - where to insert the elements
      \param _first - any compatible iterator to the first element to insert
      \param _last - any compatible iterator, AFTER \p _first, to define the end (i.e., \p _last element will not be inserted)
      */
      template< typename InputIterator >
      void insert( const_iterator _where, InputIterator _first, InputIterator _last )
      {
        for ( ; _first != _last; ++_first ) {
          insert( _where, *_first );
        }
      }

      /**
      Add element to the back of this list.
      \param c_ref - element to add
      */
      inline void push_back( const_reference c_ref )
      {
        insert( end(), c_ref );
      }

      /**
      Add element to the front of this list.
      \param c_ref - element to add
      */
      inline void push_front( const_reference c_ref )
      {
        insert( begin(), c_ref );
      }

      /**
      Remove first element in this list.
      \note It's not valid to call this method if this list is empty.
      */
      inline void pop_front()
      {
        agxAssert( !empty() );
        erase( begin() );
      }

      /**
      Remove last element in this list.
      \note It's not valid to call this method if this list is empty.
      */
      inline void pop_back()
      {
        agxAssert( !empty() );
        erase( --end() );
      }

      /**
      Clear this list of all elements.
      */
      void clear()
      {
        while ( !empty() ) {
          pop_back();
        }
      }

      /**
      Erase element at \p _where.
      \param _where - iterator to element to erase
      \return iterator to the element before the erased element
      */
      inline iterator erase( const_iterator _where )
      {
        if ( _where.m_ptr == m_front ) {
          m_front         = m_front->m_next;
          m_front->m_prev = nullptr;

          deallocate( _where.m_ptr );

          agxAssert( m_size > 0 );
          --m_size;
          agxAssert( m_size > 0 || (m_size == 0 && m_front == m_end) );

          return iterator(m_front);
        }
        else {
          _where.m_ptr->m_prev->m_next = _where.m_ptr->m_next;
          _where.m_ptr->m_next->m_prev = _where.m_ptr->m_prev;

          node_ptr_type ret = _where.m_ptr->m_prev;

          deallocate( _where.m_ptr );

          agxAssert( m_size > 0 );
          --m_size;

          return iterator(ret);
        }
      }

      /**
      Erase (all) elements matching \p pred.
      \param pred - operator to trigger remove
      */
      template< typename Pred >
      void remove_if( Pred pred )
      {
        node_ptr_type node = m_front;

        while ( node != m_end ) {
          if ( pred( node->m_element ) )
            node = erase( const_iterator( node ) ).m_ptr;
          else
            node = node->m_next;
        }
      }

      /**
      Erase (all) elements matching \p valArg.
      \param valArg - value for erase
      */
      inline void remove( const_reference valArg )
      {
        remove_if( EqualPred1( valArg ) );
      }

      /**
      Merge in elements from \p other, both this and \p other are ordered by \p pred.
      Note that \p other will be empty after the merge.
      \param other - the other list to merge into this
      \param pred - operator for merge
      */
      template< typename Pred >
      void merge( my_type& other, Pred pred )
      {
        if ( this != &other ) {
          iterator myIt  = begin();
          iterator myEnd = end();

          const_iterator otherIt  = other.begin();
          const_iterator otherEnd = other.end();

          while ( myIt != myEnd && otherIt != otherEnd ) {
            if ( pred( *otherIt, *myIt ) ) {
              const_iterator otherNext = otherIt;
              splice( myIt, other, otherIt, ++otherNext, 1 );
              otherIt = otherNext;
            }
            else ++myIt;
          }

          if ( otherIt != otherEnd )
            splice( myEnd, other, otherIt, otherEnd, other.m_size );

          agxVerify( other.empty() );
        }
      }

      /**
      Merge in elements from \p other, both this and \p other are ordered by operator<.
      Note that \p other will be empty after the merge.
      \param other - the other list to merge into this
      */
      void merge( my_type& other )
      {
        merge( other, LessPred2() );
      }

      /**
      Erases each element satisfying \p pred with previous.
      \param pred - operator to trigger erase
      */
      template< typename Pred >
      void unique( Pred pred )
      {
        if ( empty() )
          return;

        const const_iterator endIt = end();
        iterator curr              = begin();
        iterator prev              = curr++;
        while ( curr != endIt ) {
          // Erase? Yes - let prev be what it was and increment curr.
          // Note that erase wont return end since we always have prev.
          if ( pred( *prev, *curr ) )
            curr = ++erase( curr );
          // Erase? No - continue.
          else
            prev = curr++;
        }
      }

      /**
      Erases each element satisfying operator== with previous.
      */
      void unique()
      {
        unique( EqualPred2() );
      }

      /**
      Reverse sequence.
      */
      void reverse()
      {
        if ( m_size < 2 )
          return;

        node_ptr_type node = m_front;
        for ( ; ; ) {
          const node_ptr_type next = node->m_next;
          node->m_next = node->m_prev;
          node->m_prev = next;
          if ( node == m_end ) {
            node_ptr_type tmp = m_front;
            m_front = m_end->m_next;
            m_front->m_prev = nullptr;
            m_end->m_prev = tmp;
            tmp->m_next = m_end;
            break;
          }
          node = next;
        }
      }

      /**
      Swap this with \p other.
      */
      void swap( my_type& other )
      {
        std::swap( m_front, other.m_front );
        std::swap( m_end, other.m_end );
        std::swap( m_size, other.m_size );
      }

      /**
      Sorts the content in this list given \p pred.
      */
      template< typename Pred >
      void sort( Pred pred )
      {
        if ( m_size < 2 )
          return;

        const size_type MAXBINS = 25;
        my_type tmpList;
        my_type binList[ MAXBINS + 1 ];

        size_type maxBin = 0;
        while ( !empty() ) {
          tmpList.splice( tmpList.begin(), *this, begin(), ++begin(), 1 );

          size_type bin = 0;
          for ( ; bin < maxBin && !binList[ bin ].empty(); ++bin ) {
            binList[ bin ].merge( tmpList, pred );
            binList[ bin ].swap( tmpList );
          }

          if ( bin == MAXBINS )
            binList[ bin - 1 ].merge( tmpList, pred );
          else {
            binList[ bin ].swap( tmpList );
            maxBin += size_type( bin == maxBin );
          }
        }

        for ( size_type bin = 1; bin < maxBin; ++bin )
          binList[ bin ].merge( binList[ bin - 1 ], pred );

        splice( end(), binList[ maxBin - 1 ], binList[ maxBin - 1 ].begin(), binList[ maxBin - 1 ].end(), binList[ maxBin - 1 ].size() );
      }

      /**
      Splice \p other [first, last) at \p _where. Size of \p other after this operation is other.size() - count.
      \param _where - where at this
      \param other - other list
      \param first - iterator to first element in \p other
      \param last - iterator to last element i \p other
      \param count - number of elements [first, last)
      */
      void splice( const_iterator _where, my_type& other, const_iterator first, const_iterator last, size_type count )
      {
        if ( count == 0 )
          return;

        if ( this != &other ) {
          m_size += count;
          other.m_size -= count;
        }

        // If first is other.begin(), other gets a new m_front.
        if ( first.m_ptr->m_prev == nullptr )
          other.m_front = last.m_ptr;
        else
          first.m_ptr->m_prev->m_next = last.m_ptr;

        last.m_ptr->m_prev->m_next = _where.m_ptr;

        if ( _where.m_ptr->m_prev == nullptr )
          m_front = first.m_ptr;
        else
          _where.m_ptr->m_prev->m_next = first.m_ptr;

        node_ptr_type node   = _where.m_ptr->m_prev;
        _where.m_ptr->m_prev = last.m_ptr->m_prev;
        last.m_ptr->m_prev   = first.m_ptr->m_prev;
        first.m_ptr->m_prev  = node;
      }

      /**
      Splice \p other [other.begin(), other.end()) at \p _where.
      \param _where - where at this
      \param other - other list
      */
      void splice( const_iterator _where, my_type& other )
      {
        splice( _where, other, other.begin(), other.end(), other.size() );
      }

      /**
      Splice \p other [first, last) at \p _where.
      \note If the number of elements between \p first and \p last is known, use splice given count instead.
      \param _where - where at this
      \param other - other list
      \param first - iterator to first element in \p other
      \param last - iterator to last element i \p other
      */
      void splice( const_iterator _where, my_type& other, const_iterator first, const_iterator last )
      {
        size_type count = 0;
        if ( first == other.begin() && last == other.end() )
          count = other.size();
        else for ( const_iterator i = first; i != last; ++i, ++count )
          ;
        splice( _where, other, first, last, count );
      }

      /**
      Sorts the content in this list given operator<.
      */
      void sort()
      {
        sort( LessPred2() );
      }

    private:
      node_ptr_type allocate()
      {
        return new node_type();
      }

      void deallocate( node_ptr_type node )
      {
        delete node;
      }

    private:
      size_type         m_size;
      node_ptr_type     m_end;
      node_ptr_type     m_front;
  };
}


#else

#include <list>

namespace agx
{
  template< typename T >
  class List : public std::list< T > {};
}

#endif

#endif
