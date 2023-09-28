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

#include <agx/ElementaryContactPoint.h>
#include <agx/NlmcpCallback.h>

namespace agx
{
  AGX_DECLARE_POINTER_TYPES( ElementaryMultiTangentsContactPoint );
  AGX_DECLARE_POINTER_TYPES( ElementaryMultiTangentsContactPointFactory );

  /**
  Elementary contact point supporting N >= 0 tangents.
  */
  class AGXPHYSICS_EXPORT ElementaryMultiTangentsContactPoint : public ElementaryContactPoint
  {
    public:
      /**
      Construct given number of friction directions for each contact point.
      \param numTangents - number of tangents >= 0
      */
      ElementaryMultiTangentsContactPoint( UInt numTangents );

      /**
      Set number of friction directions (tangents) to be used for each contact point.
      \param numTangents - number of tangents
      */
      void setNumTangents( UInt numTangents );

      DOXYGEN_START_INTERNAL_BLOCK()

    public:
      virtual void configure( agxCollide::ContactPoint& contactPoint, const InputData& inputData ) override;

      virtual UInt getJacobian( Jacobian6DOFElement* G, UInt numBlocks, UInt row, GWriteState::Enum writeState ) override;

      virtual ConstraintNlmcpCallback* getNlCallback() const override;

      virtual void onPostSolve( agxCollide::ContactPoint& contactPoint, const Real* solution, Real h ) const override;

      virtual void renderDebug() const override;

    protected:
      virtual ~ElementaryMultiTangentsContactPoint();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      ConstraintNlmcpCallbackRef m_nlCallback;
  };

  /**
  Factory for the ElementaryMultiTangentsContactPoint contact model.
  */
  class AGXPHYSICS_EXPORT ElementaryMultiTangentsContactPointFactory : public ElementaryContactPointFactory
  {
    public:
      /**
      Construct given number of friction directions (tangents) to be used
      for each contact point.
      \param numTangents - number of tangents >= 0
      */
      ElementaryMultiTangentsContactPointFactory( UInt numTangents );

      /**
      Set number of friction directions (tangents) to be used for each contact point.
      \param numTangents - number of tangents
      */
      void setNumTangents( UInt numTangents );

      /**
      \return the number of tangents used for each contact point
      */
      UInt getNumTangents() const;

      DOXYGEN_START_INTERNAL_BLOCK()

    public:
      virtual ElementaryContactPoint* createInstance() override;

    protected:
      virtual ~ElementaryMultiTangentsContactPointFactory();

      DOXYGEN_END_INTERNAL_BLOCK()

    private:
      UInt m_numTangents;
  };
}
