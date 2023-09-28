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

#ifndef AGXMODEL_DEFORMABLE1DBULKPROPERTIES_H
#define AGXMODEL_DEFORMABLE1DBULKPROPERTIES_H

#include <agxModel/export.h>

#include <agxStream/Serializable.h>

namespace agx
{
  class Constraint;
}

namespace agxModel
{
  class Deformable1D;
  class Deformable1DNode;

  AGX_DECLARE_POINTER_TYPES( Deformable1DBulkProperties );

  /**
  Class containing and updating material properties of a Deformable1D object.
  */
  class AGXMODEL_EXPORT Deformable1DBulkProperties : public agx::Referenced, public agxStream::Serializable
  {
    public:
      /**
      Directions in each constraint. The constraint axis
      is assumed to be along the z-axis.
      */
      enum Direction
      {
        SHEAR_X = 0,   /**< Along x axis. */
        SHEAR_Y,       /**< Along y axis. */
        TENSILE,       /**< Along z axis. */

        LATERAL,       /**< About x axis. */
        VERTICAL,      /**< About y axis. */
        TORSIONAL,     /**< About z axis. */
        ALL_DIRECTIONS
      };

    public:
      /**
      Default constructor.
      */
      Deformable1DBulkProperties();

      /**
      Assign Young's Modulus to a given direction. If \p direction is ALL_DIRECTIONS \p youngsModulus will be
      assigned to all directions.
      \param youngsModulus - new value of Young's Modulus
      \param direction - direction
      */
      void setYoungsModulus( agx::Real youngsModulus, agxModel::Deformable1DBulkProperties::Direction direction );

      /**
      \param direction - direction
      \return current value of Young's Modulus for a given direction
      */
      agx::Real getYoungsModulus( agxModel::Deformable1DBulkProperties::Direction direction ) const;

      /**
      Assign Poisson's Ratio to a given direction. If \p direction is ALL_DIRECTIONS \p poissonsRatio will be
      assigned to all directions.
      \param poissonsRatio - new value of Poisson's Ratio
      \param direction - direction
      */
      void setPoissonsRatio( agx::Real poissonsRatio, agxModel::Deformable1DBulkProperties::Direction direction );

      /**
      \param direction - direction
      \return current value of Poisson's Ratio for a given direction
      */
      agx::Real getPoissonsRatio( agxModel::Deformable1DBulkProperties::Direction direction ) const;

      /**
      Assign yield point to a given direction. If \p direction is ALL_DIRECTIONS \p yieldPoint will be
      assigned to all directions.
      \param yieldPoint - new value of the yield point
      \param direction - direction
      */
      void setYieldPoint( agx::Real yieldPoint, agxModel::Deformable1DBulkProperties::Direction direction );

      /**
      \param direction - direction
      \return current value of the yield point for a given direction
      */
      agx::Real getYieldPoint( agxModel::Deformable1DBulkProperties::Direction direction ) const;

      /**
      Assign break point to a given direction. If \p direction is ALL_DIRECTIONS \p breakPoint will be
      assigned to all directions.
      \param breakPoint - new value of the break point
      \param direction - direction
      */
      void setBreakPoint( agx::Real breakPoint, agxModel::Deformable1DBulkProperties::Direction direction );

      /**
      \param direction - direction
      \return current value of the break point for a given direction
      */
      agx::Real getBreakPoint( agxModel::Deformable1DBulkProperties::Direction direction ) const;

      /**
      Assign SPOOK damping along given direction to the constraints linking the
      segments of the Deformable1D. If \p direction is \p ALL_DIRECTIONS then
      \p spookDamping will be assigned to all directions.

      \param damping - The new value to use for SPOOK damping.
      \param direction - direction
      */
      void setDamping( agx::Real damping, agxModel::Deformable1DBulkProperties::Direction direction );

      /**
      \return The SPOOK damping in the given \p direction.
      */
      agx::Real getDamping( agxModel::Deformable1DBulkProperties::Direction direction ) const;


      /**
      Check if the bulk parameters are flagged as dirty and have to be read by Deformable1D constraints.
      \return true if dirty and the bulk properties have to be updated
      */
      agx::Bool isDirty() const;

      /**
      Clear the dirty flag. This bulk material will not be used to update any
      Deformable1D constraint parameters until made dirty again.
      */
      void clearDirtyFlag();

      /**
      Updates the bulk properties of the whole Deformable1D object, ignoring dirty flag.
      After calling this method this object is no longer dirty.
      \param d1d - the Deformable1D object
      */
      virtual void updateBulkProperties( const agxModel::Deformable1D* d1d ) const;

      /**
      Updates the bulk properties of a given node. The constraint between \p node and the
      previous node will be updated.
      \param node - node to update (constraint between previous node and \p node)
      \param d1d the Deformable1D object
      */
      virtual void updateBulkProperties( agxModel::Deformable1DNode* node, const agxModel::Deformable1D* d1d ) const;

      /**
      Updates the bulk properties given node before -> constraint -> node after where
      "node after" is \p curr (current node), the target node for the update, i.e., the
      one assumed to be carrying the constraint.
      \param prev - node before \p curr
      \param constraint - constraint to assign properties to
      \param curr - current node
      \param d1d - the Deformable1D object
      */
      virtual void updateBulkProperties( const agxModel::Deformable1DNode* prev,
                                         agx::Constraint* constraint,
                                         const agxModel::Deformable1DNode* curr,
                                         const agxModel::Deformable1D* d1d ) const;

      AGXSTREAM_DECLARE_SERIALIZABLE(agxModel::Deformable1DBulkProperties);

      /**
      Store structural independent data to stream.
      */
      virtual void storeLightData( agxStream::StorageStream& str ) const override;

      /**
      Restore structural independent data from stream.
      */
      virtual void restoreLightData( agxStream::StorageStream& str ) override;

    protected:
      /**
      Reference counted object, protected destructor.
      */
      virtual ~Deformable1DBulkProperties();

      /**
      Assigns default values to all parameters.
      */
      void setDefaultValues();

    protected:
      agx::Real         m_youngsModulus[ ALL_DIRECTIONS ];
      agx::Real         m_poissonsRatio[ ALL_DIRECTIONS ];
      agx::Real         m_yieldPoint[ ALL_DIRECTIONS ];
      agx::Real         m_breakPoint[ ALL_DIRECTIONS ];
      agx::Real         m_spookDamping[ ALL_DIRECTIONS ];
      mutable agx::Bool m_dirty;
  };
}

#endif // AGXMODEL_DEFORMABLE1DBULKPROPERTIES_H
