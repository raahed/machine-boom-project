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


#include <agx/StrongInteraction.h>
#include <agx/agx_valarray_types.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( AddedMassInteraction );

  /**
  Interface to write added mass matrices for one or two rigid bodies.
  For two dynamic rigid bodies an off diagonal block may be written
  as well:
  [ M_0                 ]
  [  .  .               ]
  [  .    .             ]
  [  .      .           ]
  [ A_i0      M_i       ]
  [               .     ]
  [                 .   ]
  [                   . ]
  */
  class AGXPHYSICS_EXPORT AddedMassInteraction : public agx::StrongInteraction
  {
    public:
      /**
      6x6 matrix class with blockwise interface to get or set 3x3 matrices:
      [ M(0, 0)    M(0, 1) ]
      [ M(1, 0)    M(1, 1) ]
      */
      class AGXPHYSICS_EXPORT Matrix6x6
      {
        public:
          /**
          Default constructor, sets all elements to 0.
          */
          Matrix6x6();

          /**
          Copy constructor.
          */
          Matrix6x6( const Matrix6x6& other );

          /**
          Assignment operator.
          \return this
          */
          Matrix6x6& operator=( const Matrix6x6& other );

          /**
          \param i - row to access
          \note Debug bounds check for row only.
          \sa at( size_t i, size_t j )
          \return row i of this 6x6 matrix (i < 6)
          */
          const agx::Real* operator[]( size_t i ) const;

          /**
          \param i - row to access
          \note Debug bounds check for row only.
          \sa at( size_t i, size_t j )
          \return row i of this 6x6 matrix (i < 6)
          */
          agx::Real* operator[]( size_t i );

          /**
          \param i - row to access
          \param j - column to access
          \return element at row i and column j
          */
          const agx::Real& at( size_t i, size_t j ) const;

          /**
          \param i - row to access
          \param j - column to access
          \param value - The new value for i,j
          \return element at row i and column j
          */
          void set( size_t i, size_t j, agx::Real value );


          /**
          \param i - row to access
          \param j - column to access
          \return element at row i and column j
          */
          agx::Real& at( size_t i, size_t j );

          /**
          \return the transpose of this matrix
          */
          Matrix6x6 transposed() const;

          /**
          \param i - block row index
          \param j - block column index
          \note Changing the values of the returned 3x3 block wont affect this matrix.
          \return the values of block row i and block row j (i < 2 and j < 2)
          */
          agx::Matrix3x3 block( size_t i, size_t j ) const;

          /**
          \param i - block row index
          \param j - block column index
          \note Changing the values of the returned 3x3 block wont affect this matrix.
          \return the values of block row i and block row j (i < 2 and j < 2)
          */
          void block( size_t i, size_t j, const agx::Matrix3x3& matrix );

          /**
          Returns a transformed version of this matrix:
          [ left M(0, 0) right   left M(0, 1) right ]
          [ left M(1, 0) right   left M(1, 1) right ]
          \param left - 3x3 matrix to multiply each block from the left
          \param right - 3x3 matrix to multiply each block from the right
          \return transformed version of this matrix
          */
          Matrix6x6 transform( const agx::Matrix3x3& left, const agx::Matrix3x3& right ) const;

          /**
          Performs result += M * [v1]
                                 [v2].
          */
          void multiply( const agx::Vec3 v1, const agx::Vec3 v2, agx::Real* result ) const;

          /**
          Resets all values to zero.
          */
          void reset();

        public:
          /**
          Internal. Store this object.
          */
          void store( agxStream::OutputArchive& out ) const;

          /**
          Internal. Restore this object.
          */
          void restore( agxStream::InputArchive& in );

        private:
          agx::Real m_data[ 6 ][ 6 ];
      };

      /**
      Class containing a rigid body and the 6x6 matrix to be written
      in the diagonal of the system matrix. This object also keeps
      track if a rigid body has been deleted.
      */
      class AGXPHYSICS_EXPORT RigidBodyStorage
      {
        public:
          /**
          Construct given rigid body.
          \param rb - rigid body for this storage
          \note If rigid body is null, getValid will always return true.
                If the rigid body is valid (not null) and later is deleted,
                getValid returns false.
          */
          RigidBodyStorage( agx::RigidBody* rb );

          /**
          \return rigid body of this storage, can be null
          */
          agx::RigidBody* getRigidBody() const;

          /**
          \return entity pointer to the rigid body of this storage
          */
          agx::Physics::RigidBodyPtr getRigidBodyEntity() const;

          /**
          \return true if the rigid body is valid or if this storage was constructed
                  with a null rigid body pointer
          */
          agx::Bool getValid() const;

          /**
          \return the 6x6 matrix block associated to the rigid body in this storage
          */
          const agx::AddedMassInteraction::Matrix6x6& getBlock() const;

          /**
          \return the 6x6 matrix block associated to the rigid body in this storage
          */
          agx::AddedMassInteraction::Matrix6x6& getBlock();

        public:
          /**
          Internal. Store this object.
          */
          void store( agxStream::OutputArchive& out ) const;

          /**
          Internal. Restore this object.
          */
          void restore( agxStream::InputArchive& in );

        private:
          agx::RigidBodyObserver m_rb;
          agx::Bool m_hadValidRb;
          agx::AddedMassInteraction::Matrix6x6 m_block;
      };

    public:
      /**
      Construct given either one or two rigid bodies.
      \param rb1 - first rigid body
      \param rb2 - second rigid body
      */
      AddedMassInteraction( agx::RigidBody* rb1, agx::RigidBody* rb2 = nullptr );

      /**
      Storage of 'rb1' from construction containing the rigid body and a 6x6 matrix block.
      \return storage of rigid body 1
      */
      const agx::AddedMassInteraction::RigidBodyStorage* getRigidBody1Storage() const;

      /**
      Storage of 'rb1' from construction containing the rigid body and a 6x6 matrix block.
      \return storage of rigid body 1
      */
      agx::AddedMassInteraction::RigidBodyStorage* getRigidBody1Storage();

      /**
      Storage of 'rb2' from construction containing the rigid body and a 6x6 matrix block.
      \return storage of rigid body 2
      */
      const agx::AddedMassInteraction::RigidBodyStorage* getRigidBody2Storage() const;

      /**
      Storage of 'rb2' from construction containing the rigid body and a 6x6 matrix block.
      \return storage of rigid body 2
      */
      agx::AddedMassInteraction::RigidBodyStorage* getRigidBody2Storage();

      /**
      Off diagonal block (6x6 matrix). Defines how acceleration of the first body results in
      a force acting on the second body.
      \return the off diagonal block storage (6x6 matrix)
      */
      const agx::AddedMassInteraction::Matrix6x6& getOffDiagonalBlock() const;

      /**
      Off diagonal block (6x6 matrix). Defines how acceleration of the first body results in
      a force acting on the second body.
      \return the off diagonal block storage (6x6 matrix)
      */
      agx::AddedMassInteraction::Matrix6x6& getOffDiagonalBlock();

      /**
      Transform the mass matrix according to a translate of center of mass.
      */
      void translateWorldDiagonalBlock( const agx::Vec3 cmOffset,
                                        agx::AddedMassInteraction::Matrix6x6& matrix ) const;

      void translateWorldDiagonalBlock( const agx::Vec3 cmOffsetRb1,
                                        const agx::Vec3 cmOffsetRb2,
                                        agx::AddedMassInteraction::Matrix6x6& matrix ) const;

    public:
      /**
      \return first rigid body of this interaction
      */
      virtual agx::RigidBody* getRigidBody1() const override;

      /**
      \return second rigid body of this interaction
      */
      virtual agx::RigidBody* getRigidBody2() const override;

      AGXSTREAM_DECLARE_SERIALIZABLE( agx::AddedMassInteraction );

    protected:

      /**
      Default constructor used by serialization.
      */
      AddedMassInteraction();

      /**
      Reference counted object, protected destructor.
      */
      virtual ~AddedMassInteraction();

      /**
      Checks if this interaction is valid for the solver.
      */
      virtual void prepare() override;

      /**
      Call from the direct solver with pointers where to write diagonal block(s) and the off diagonal block.
      \note \p rb1DiagonalBlock and \p rb2DiagonalBlock can/will come in a different order than what
            m_rb1Storage and m_rb2Storage are.
      \param rb1DiagonalBlock - block for what the solver thinks is rb1
      \param rb2DiagonalBlock - block for what the solver thinks is rb2 (data is zero if the solver doesn't need the data to be written)
      \param offDiagonalBlock - off diagonal block. Location of this block in the system matrix is defined by blockRowIndex fields in
                                rb1DiagonalBlock and rb2DiagonalBlock.
      */
      virtual void writeMatrixData( agx::StrongInteraction::MatrixData rb1DiagonalBlock,
                                    agx::StrongInteraction::MatrixData rb2DiagonalBlock,
                                    agx::StrongInteraction::MatrixData offDiagonalBlock ) override;

      /**
      Right hand side update.
      */
      virtual void writeRhs( agx::Physics::RigidBodyPtr rb1,
                             agx::Real* rb1Rhs,
                             agx::Physics::RigidBodyPtr rb2,
                             agx::Real* rb2Rhs,
                             agx::Bool isRestingSolve ) override;

    private:
      struct Transformed
      {
        AddedMassInteraction::Matrix6x6 rb1DiagonalBlock;
        AddedMassInteraction::Matrix6x6 rb2DiagonalBlock;
        AddedMassInteraction::Matrix6x6 offDiagonalBlock;

        void reset()
        {
          rb1DiagonalBlock.reset();
          rb2DiagonalBlock.reset();
          offDiagonalBlock.reset();
        }
      };

    private:
      agx::AddedMassInteraction::RigidBodyStorage m_rb1Storage;
      agx::AddedMassInteraction::RigidBodyStorage m_rb2Storage;
      agx::AddedMassInteraction::Matrix6x6        m_offDiagonalBlock;
      Transformed                                 m_transformed;
  };

  AGXPHYSICS_EXPORT agx::Bool equivalent( const agx::AddedMassInteraction::Matrix6x6& lhs, const agx::AddedMassInteraction::Matrix6x6& rhs, agx::Real eps = agx::RealEpsilon );
  AGXPHYSICS_EXPORT agx::Bool equalsZero( const agx::AddedMassInteraction::Matrix6x6& matrix, agx::Real eps = agx::RealEpsilon );

  AGX_FORCE_INLINE const Real* AddedMassInteraction::Matrix6x6::operator[]( size_t i ) const
  {
    agxAssert( i < 6 );
    return m_data[ i ];
  }

  AGX_FORCE_INLINE Real* AddedMassInteraction::Matrix6x6::operator[]( size_t i )
  {
    agxAssert( i < 6 );
    return m_data[ i ];
  }

  AGX_FORCE_INLINE const Real& AddedMassInteraction::Matrix6x6::at(size_t i, size_t j ) const
  {
    agxAssert( j < 6 );
    return (*this)[ i ][ j ];
  }

  AGX_FORCE_INLINE void AddedMassInteraction::Matrix6x6::set( size_t i, size_t j, agx::Real value )
  {
    agxAssert( i < 6 );
    agxAssert( j < 6 );
    (*this)[ i ][ j ] = value;
  }


  AGX_FORCE_INLINE Real& AddedMassInteraction::Matrix6x6::at( size_t i, size_t j )
  {
    agxAssert( j < 6 );
    return (*this)[ i ][ j ];
  }

  AGX_FORCE_INLINE const AddedMassInteraction::Matrix6x6& AddedMassInteraction::RigidBodyStorage::getBlock() const
  {
    return m_block;
  }

  AGX_FORCE_INLINE AddedMassInteraction::Matrix6x6& AddedMassInteraction::RigidBodyStorage::getBlock()
  {
    return m_block;
  }

  AGX_FORCE_INLINE const AddedMassInteraction::Matrix6x6& AddedMassInteraction::getOffDiagonalBlock() const
  {
    return m_offDiagonalBlock;
  }

  AGX_FORCE_INLINE AddedMassInteraction::Matrix6x6& AddedMassInteraction::getOffDiagonalBlock()
  {
    return m_offDiagonalBlock;
  }
}
