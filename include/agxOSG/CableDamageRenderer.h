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

#include <agxOSG/export.h>

#include <agxCable/CableDamage.h>

#include <agxSDK/StepEventListener.h>

#include <agx/Vector.h>
#include <agxOSG/Node.h>
#include <agxOSG/GeometryNode.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/observer_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.



namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(CableDamageRenderer);


  /**
  Class that creates OSG rendering nodes for each segment of a cable and colors
  them with a color gradient based on the current total damage estimate for the
  segment.
  */
  class AGXOSG_EXPORT CableDamageRenderer : public agxSDK::StepEventListener
  {
    public:
      /**
      Create a damage renderer for the given CableDamage instance. The color
      gradient used goes from green for zero damage and red for the given \p
      colorRangeMax.

      Created rendering nodes are added as children of the given \p root.

      \param damage - The CableDamage object from which segment damage is read.
      \param colorRangeMax - The damage for which the color gradient is clamped.
      \param root - The OSG group to which the new rendering nodes are added.
      */
      CableDamageRenderer(agxCable::CableDamage* damage, agx::Real colorRangeMax, osg::Group* root);

      void renderCurrent();
      void renderAccumulated();

      bool isRenderingCurrent() const;
      bool isRenderingAccumulated() const;

      /**
      Enable or disable rendering by adding or removing the created OSG nodes
      from the node that was passed to the CableDamageRenderer constructor.

      \param enable - True to enable damage rendering. False to disable.
      */
      void setEnableRendering(bool enable);


      /**
      \return True of rendering is enabled. False otherwise.
      */
      bool getEnableRendering() const;

      /**
      \return The segment damage at which the color gradient reaches its
      maximum. All segment damages at or above this level get the same color.
      */
      agx::Real getColorRangeMax() const;

      /**
      Set the damage at which the color gradient should reach its maxmimu. All
      segment damages at or above the given maximum will get the same color.

      \param colorRangeMax - The upper damage limit of the color gradient.
      */
      void setColorRangeMax(agx::Real colorRangeMax);

      /**
      Gives access to the OSG nodes created for the cable segments. \p index
      must be less than the number of segments in the cable for which segment
      damages are being rendered.

      \param index - The index of the OSG node to access.
      \return The OSG node at the given index.
      */
      agxOSG::GeometryNode* getOsgNode(size_t index);

      /**
      Update the rendering colors based on the current segment damages.
      \param timeStamp - Not used.
      */
      virtual void last(const agx::Real& timeStamp) override;

    protected:
      virtual ~CableDamageRenderer() {}

    private:
      CableDamageRenderer& operator=(CableDamageRenderer&) = delete;

    private:
      agxCable::CableDamageRef m_damage;
      osg::observer_ptr<osg::Group> m_parent;
      agx::Vector<osg::ref_ptr<GeometryNode>> m_nodes;
      agx::Real m_colorRangeMax; /// \todo Better name needed. It's damage max and not color max.
      bool m_renderAccumulatedInsteadOfCurrent;
  };

}
