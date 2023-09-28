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

#ifndef AGX_CUSTOMGRAVITYFIELD_H
#define AGX_CUSTOMGRAVITYFIELD_H

#include <agx/GravityField.h>

namespace agx
{

  /**
  The custom gravity field allow a user to derive from this class and implement their own gravity function based on the
  center of mass of a particle/rigid body.

  Compared to UniformGravityField and PointGravityField this implementation will give some overhead due to the virtual call to calculateGravity for each object.
  So for a very large set of objects, such as hundreds of thousands of particles this might slow down the simulation somewhat.

  Also the calculateGravity method will be called from multiple threads if AGX is using more than one thread so make sure it is thread safe. It should as it is a const method.
  */
  class AGXPHYSICS_EXPORT CustomGravityField : public agx::GravityField
  {
    public:
      /// Constructor
      CustomGravityField();

      /**
      Given \p position, a gravity acceleration will be calculated and returned.
      Make sure this method is thread safe as it will be called from several threads at the same time.

      \param position - The position at which the acceleration will be given.
      \return The calculated acceleration at position. Default implementation returns agx::Vec3()
      */
      virtual agx::Vec3 calculateGravity(const agx::Vec3& position) const override;

    public:
      /// Internal use
      virtual Task* createRigidBodyTask() override;

      /// Internal use
      virtual Task* createParticleSystemTask() override;

      AGXSTREAM_DECLARE_ABSTRACT_SERIALIZABLE( agx::CustomGravityField );

    private:

    protected:
      /// Destructor
      virtual ~CustomGravityField();

      virtual void store(agxStream::OutputArchive& out) const override;
      virtual void restore( agxStream::InputArchive& in ) override;
  };
  AGX_DECLARE_POINTER_TYPES(CustomGravityField);

} //namespace agx


#endif
