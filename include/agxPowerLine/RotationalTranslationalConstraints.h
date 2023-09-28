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

#ifndef AGXPOWERLINE_ROTATIONAL_TRANSLATIONAL_CONSTRAINTS_H
#define AGXPOWERLINE_ROTATIONAL_TRANSLATIONAL_CONSTRAINTS_H

#include <agxPowerLine/PowerLineConstraints.h>
#include <agxPowerLine/RotationalDimension.h>
#include <agxPowerLine/TranslationalDimension.h>
#include <agxPowerLine/PowerLineController.h>

namespace agxPowerLine
{
  class RotationalTranslationalConnector;

  DOXYGEN_START_INTERNAL_BLOCK()

  //Constrains a rotational dimension to a translational dimension
  class ElementaryRotationalTranslationalConstraint : public agxPowerLine::ElementaryPhysicalDimensionMultiBodyConstraint
  {
    public:
      ElementaryRotationalTranslationalConstraint();
      using ElementaryPhysicalDimensionConstraint::getJacobian;
      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::ElementaryRotationalTranslationalConstraint);
  };

  typedef agx::ref_ptr<ElementaryRotationalTranslationalConstraint> ElementaryRotationalTranslationalConstraintRef;



  class RotationalTranslationalConstraintImplementation : public PhysicalDimensionMultiBodyConstraintImplementation
  {
    public:
      RotationalTranslationalConstraintImplementation();
      AGXSTREAM_DECLARE_SERIALIZABLE(agxPowerLine::RotationalTranslationalConstraintImplementation);
  };


  DOXYGEN_END_INTERNAL_BLOCK()
}

#endif // AGXMODEL_ROTATIONAL_TRANSLATIONAL_CONSTRAINTS_H
