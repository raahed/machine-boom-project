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

#include <agx/config/AGX_USE_AGXTERRAIN.h>
#if AGX_USE_AGXTERRAIN()


#include <agxOSG/export.h>

#include <agxTerrain/Terrain.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Drawable>
#include <osg/Version>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace agxOSG
{
  /**
  Internal drawable class for drawing the Voxel grid structure.
  */
  class AGXOSG_EXPORT SoilParticleSystemDrawable : public osg::Drawable
  {
    public:
      SoilParticleSystemDrawable(agxTerrain::Terrain* terrain);

      void setEnable(bool enable);
      bool getEnable() const;

      /**
      \return true when the render task has been added to the simulation
      */
      bool isInitialized() const;

      virtual osg::Object* cloneType() const { return new SoilParticleSystemDrawable(m_terrain); }
      virtual osg::Object* clone(const osg::CopyOp&) const { return new SoilParticleSystemDrawable(*this); }
      virtual bool isSameKindAs(const osg::Object* obj) const { return dynamic_cast<const SoilParticleSystemDrawable *>(obj) != nullptr; }
      virtual const char* libraryName() const { return "agxOSG"; }
      virtual const char* className() const { return "SoilParticleSystemDrawable"; }

      /** Compute the bounding box around Drawables's geometry.*/
#if OSG_VERSION_GREATER_OR_EQUAL(3,4,0)
      virtual osg::BoundingSphere computeBound() const;
      virtual osg::BoundingBox computeBoundingBox() const;
#else
      virtual osg::BoundingBox computeBound() const;
#endif

      /// The actual draw method for the merged bodies
      virtual void drawImplementation(osg::RenderInfo& renderInfo) const;

      agx::ParticleSystem* getParticleSystem();

    public:
      void onAddNotification( agxSDK::Simulation* simulation );
      void onRemoveNotification( agxSDK::Simulation* simulation );

    protected:
      virtual ~SoilParticleSystemDrawable();

      //////////////////////////////////////////////////////////////////////////
      // Variables
      //////////////////////////////////////////////////////////////////////////
    protected:
      agxTerrain::TerrainRef            m_terrain;
      osg::Group*                       m_group;
      agx::TaskGroupRef                 m_renderTask;
      bool                              m_enable;
  };
}
#endif
