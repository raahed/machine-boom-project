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

#ifndef AGXPOWERLINE_TRANSLATIONAL_UNIT_H
#define AGXPOWERLINE_TRANSLATIONAL_UNIT_H

#include <agxModel/export.h>
#include <agxPowerLine/TranslationalDimension.h>
#include <agxPowerLine/Unit.h>

namespace agxPowerLine
{
  DOXYGEN_START_INTERNAL_BLOCK()
  class AGXMODEL_EXPORT TranslationalUnit : public agxPowerLine::Unit
  {
  public:

    /**
    Create a translational unit.
    */
    TranslationalUnit();


    /**
    \returns the last delivered force.
    */
    virtual agx::Real getDeliveredForce() const {return m_translationalDimension->getOutputLoad();}

    /**
    \returns pointer to the translational dimension (the only dimension of a translational unit).
    */
    agxPowerLine::TranslationalDimension* getTranslationalDimension() const;

    /**
    Set velocity damping for the local direction.
    */
    void setVelocityDamping(agx::Real damping);


    /**
    Set the mass for the one dimensional body.
    */
    void setMass( agx::Real mass );

    /**
    returns a vector in world coordinates that defines the linear axis of the transational unit.
    */
    virtual agx::Vec3 getWorldDirection() const;

    /**
    Functions that have to be public since they are called by static functions. Do NOT use them.
    */
    virtual bool preUpdate(agx::Real timeStep) override;
    /**
    Stores internal data into stream.
    */
    virtual bool store(agxStream::StorageStream& str) const override;

    /**
    Restores internal data from stream.
    */
    virtual bool restore(agxStream::StorageStream& str) override;

    AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::TranslationalUnit);

  protected:
    virtual ~TranslationalUnit();

  protected:
    agxPowerLine::TranslationalDimensionRef m_translationalDimension;
  };

  DOXYGEN_END_INTERNAL_BLOCK()


  typedef agx::ref_ptr<TranslationalUnit> TranslationalUnitRef;
}

#endif // AGXMODEL_TRANSLATIONAL_UNIT_H
