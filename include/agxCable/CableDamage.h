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

#include <agxCable/export.h>

#include <agxCable/Cable.h>
#include <agxCable/Direction.h>
#include <agxCable/CableDamageState.h>
#include <agxCable/CableDamageTypes.h>
#include <agxCable/SegmentDamage.h>

#include <agxSDK/LinkedStructureComponent.h>


namespace agxCable
{
  AGX_DECLARE_POINTER_TYPES(CableDamage);


  /**
  The CableDamage class is used to estimate damage caused to a cable during a
  simulation. Damage is estimated from a number of damage sources and weighted
  together to form a total.

  The damage sources are categorized in two ways: dimension and cause. The
  dimension is one of the regular cable dimensions: bend, twist, or stretch. The
  cause is the state of the cable that is producing the damage and is either
  tension, deformation or deformation rate. There is also contacts, which are
  treated separately.

  Given three dimensions; bend, twist, and stretch, and three sources; tension,
  deformation and deformation rate, we get a total of nine damage types, plus
  two contact types normal force and friction force. They are enumerated in \p
  agxCable::DamageTypes.

  The damage model incorporates the notion of "safe deformations". These are
  deformation thresholds below which no damage is being produced due to
  deformation or deformation rate. Tension and contact damage still apply.

  Damage is estimated per cable segment.
  */
  class AGXCABLE_EXPORT CableDamage : public agxSDK::LinkedStructureComponent
  {
    public:
      /**
      Create a new CableDamage instance. The CableDamage is activated when
      added to a Cable.
      */
      CableDamage();

      /**
      Set the weight, or scaling, to use for the given damage source when
      summing it with all the other damage sources to form a total damage
      estimate.

      \param type - The damage type to set the weight for.
      \param weight - The new weight.
      */
      void setWeight(agxCable::DamageTypes::DamageType type, agx::Real weight);

      /**
      Get the weight, or scaling, used for the given damage source when summing
      it with all the other damage sources to form a total damage estimate.

      \param type - The damage type to get the weight for.
      \return The weight for the given damage type.
       */
      agx::Real getWeight(agxCable::DamageTypes::DamageType type) const;

      /**
      Set the weight to use for damage caused by stretch rate.

      Stretch rate is a measure of how fast the cable is being stretched,
      measured in lengths per unit length per unit time.

      \param weight - The new weight for stretch rate damage estimation.
      */
      void setStretchRateWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by bend rate.

      Bend rate is a measure of how fast the cable is being bent, measured as
      the change in curvature per unit time.

      \param weight - The new weight for bend rate damage estimation.
      */
      void setBendRateWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by twist rate.

      Twist rate is a  measure of how fast the cable is being twisted, measured
      in radians per unit length per unit time.

      \param weight - The new weight for twist rate damage estimation.
      */
      void setTwistRateWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by stretch tension.

      \param weight - The new weight for stretch tension damage estimation.
      */
      void setStretchTensionWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by bend tension.

      \param weight - The new weight for bend tension damage estimation.
      */
      void setBendTensionWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by twist tension.

      \param weight - The new weight for twist tension damage estimation.
      */
      void setTwistTensionWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by bend deformation.

      \param weight - The new weight for bend deformation damage estimation.
      */
      void setBendDeformationWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by twist deformation.

      \param weight  - The new weight for twist deformation damage estimation.
      */
      void setTwistDeformationWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by stretch deformation.

      \param weight - The new weight for stretch deformation damage estimation.
      */
      void setStretchDeformationWeight(agx::Real weight);



      /**
      Set the weight to use for damage caused by normal forces from contacts.

      \param weight - The new weight for normal force damage estimation.
      */
      void setNormalForceWeight(agx::Real weight);

      /**
      Set the weight to use for damage caused by friction forces from contacts.

      \param weight - The new weight for friction force damage estimation.
      */
      void setFrictionForceWeight(agx::Real weight);


      /**
      Set the deformation threshold for the given direction that must be
      surpassed before the deformation begin to contribute to the damage
      estimation.

      The unit of the threshold depends on the direction. See
      setStretchThreshold, setBendThreshold, and setTwistThreshold.

      \param direction - The direction to set deformation threshold for.
      \param threshold - The new deformation threshold.

      \see setStretchThreshold
      \see setBendThreshold
      \see setTwistThreshold
      */
      void setThreshold(agxCable::Direction direction, agx::Real threshold);


      /**
      Set the stretch deformation threshold that must be surpassed before the
      stretch deformation begin to contribute to the damage estimation.

      The threshold is given in length per unit length.

      \param threshold - The minimum stretch deformation for stretch damage to be applied.
      */
      void setStretchThreshold(agx::Real threshold);

      /**
      Set the bend deformation threshold that must be surpassed before the bend
      deformation begin to contribute to the damage estimation.

      The threshold is measured in radians per unit length.

      \param threshold - The minimum bend deformation for bend damage to be applied.
      */
      void setBendThreshold(agx::Real threshold);


      /**
      Set the twist deformation threshold that must be surpassed before the
      twist deformation begin to contribute to the damage estimation.

      The twist threshold is measured in radians per unit length.

      \param threshold - The minimum twist deformation for twist damage to be applied.
      */
      void setTwistThreshold(agx::Real threshold);



      /**
      \see setBendDeformationWeight
      \return The bend deformation weight.
      */
      agx::Real getBendDeformationWeight() const;

      /**
      \see setTwistDeformationWeight
      \return The twist deformation weight.
      */
      agx::Real getTwistDeformationWeight() const;

      /**
      \see setStretchDeformationWeight
      \return The stretch deformation weight.
      */
      agx::Real getStretchDeformationWeight() const;

      /**
      \see setStretchRateWeight
      \return The stretch rate weight.
      */
      agx::Real getStretchRateWeight();

      /**
      \see setBendRateWeight
      \return The bend rate weight.
      */
      agx::Real getBendRateWeight();

      /**
      \see setTwistRateWeight
      \return The twist rate weight.
      */
      agx::Real getTwistRateWeight();


      /**
      \see setThreshold
      \param direction - Direction to get deformation threshold for.
      \return The deformation threshold for the given direction.
      */
      agx::Real getThreshold(agxCable::Direction direction) const;

      /**
      \see setStretchrateThreshold
      \return The stretch deformation threshold.
      */
      agx::Real getStretchThreshold();

      /**
      \see setBendRateThreshold
      \return The bend deformation threshold
      */
      agx::Real getBendThreshold();

      /**
      \see setTwistRateThreshold
      \return The twistdeformation threshold
      */
      agx::Real getTwistThreshold();

      /**
      \see setStretchTensionWeight
      \return The stretch tension weight.
      */
      agx::Real getStretchTensionWeight();

      /**
      \see setBendTensionWeight
      \return The bend tension weight.
      */
      agx::Real getBendTensionWeight();

      /**
      \see setTwistTensionWeight
      \return The twist tension weight.
      */
      agx::Real getTwistTensionWeight();


      /**
      \see setNormalForceWeight
      \return The normal force weight.
      */
      agx::Real getNormalForceWeight();

      /**
      \see setFrictionForceWeight
      \return  The friction force weight.
      */
      agx::Real getFrictionForceWeight();


      /**
      Provides access to the per-segment damage estimate contributions that the
      last time step produced.

      \return A collection of SegmentDamage instances holding the last time step's damage contributions.
      */
#ifndef SWIG
      const agxCable::SegmentDamageVector& getCurrentDamages() const;
#endif
      agxCable::SegmentDamagePtrVector getCurrentDamagePtrs() const;

      /**
      Provides access to the per-segment damage estimates that has been
      accumulated over the simulation so far.

      \return A collection of SegmentDamage instances holding the sum of all time step's damage contributions.
      */
#ifndef SWIG
      const agxCable::SegmentDamageVector& getAccumulatedDamages() const;
#endif
      agxCable::SegmentDamagePtrVector getAccumulatedDamagePtrs() const;

      /**
      May return zero until the cable for which damage is estimated has been
      fully initialized.

      \return The number of segments for which damage is estimated.
      */
      size_t getNumDamages() const;

      /**
      Provides access to the last time step's damage contribution for a
      particular cable segment. \p index must be less than the value returned by
      \p getNumDamages.

      \param index - The index of the SegmentDamage instance to access.
      \return The SegmentDamage at the given index.
      */
      const agxCable::SegmentDamage& getCurrentDamageAt(size_t index) const;

      /**
      Provides access to the accumulated damage estimation for a particular
      cable segment. \p index must be less than the value returned by \p
      getNumDamages.

      \param index - The index of the SegmentDamage instance to access.
      \return The SegmentDamage at the given index.
      */
      const agxCable::SegmentDamage& getAccumulatedDamageAt(size_t index) const;

      /**
      Get the cable that this CableDamage is estimating damage for. Will return
      nullptr if this CableDamage hasn't been added to a cable yet.

      \return The cable for which this CableDamage is estimating damage.
      */
      agxCable::Cable* getCable();

      /**
      Get the cable that this CableDamage is estimating damage for. Will return
      nullptr if this CableDamage hasn't been added to a cable yet.

      \return The cable for which this CableDamage is estimating damage.
      */
      const agxCable::Cable* getCable() const;

      /**
      Find the CableDamage instances that estimate damage for the given cable.

      \param cable - The cable for which a CableDamage instance should be found.
      \return The CableDamage instance that is part of the given cable, or nullptr if there is no such CableDamage.
       */
      static agxCable::CableDamage* getCableDamage(agxCable::Cable* cable);

      AGXSTREAM_DECLARE_SERIALIZABLE(agxCable::CableDamage);

    protected:
      virtual ~CableDamage();

      virtual void onAddNotification(agxSDK::LinkedStructure* linkedStructure) override;
      virtual void onAddNotification(agxSDK::Simulation* simulation) override;
      virtual void onPreStep() override;
      virtual void onPostStep() override;

    private:
      CableDamage(const CableDamage&) = delete;
      void operator=(const CableDamage&) = delete;

      void recomputeCurrentDamages();

    private:
      agxCable::CableObserver m_cable;
      agxCable::CableDamageStateRef m_cableState;

      agx::Real m_weights[agxCable::DamageTypes::NUM_CABLE_DAMAGE_TYPES];
      agx::Real m_thresholds[agxCable::NUM_DIRECTIONS];

      agxCable::SegmentDamageVector m_currentDamages;
      agxCable::SegmentDamageVector m_accumulatedDamages;
  };
}
