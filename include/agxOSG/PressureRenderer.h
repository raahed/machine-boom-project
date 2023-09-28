/*
Copyright 2007-2023. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, is copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or having been
advised so by Algoryx Simulation AB for a time limited evaluation, or having purchased a
valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#ifndef AGXOSG_PRESSURE_RENDERER_H
#define AGXOSG_PRESSURE_RENDERER_H

#include <agxCollide/Trimesh.h>
#include <agxSDK/StepEventListener.h>
#include <agxSDK/ContactEventListener.h>

#include <agxOSG/export.h>
#include <agxOSG/PressureAtlasManager.h>
#include <agxOSG/Node.h>
#include <agxOSG/SceneDecorator.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/observer_ptr>
#include <osgSim/ColorRange>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agxRender/Color.h>
#include <agxOSG/PressureFromContacts.h>

namespace osg
{
  class Group;
}



namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(PressureRenderer);
  /**
  An utility class for handling of pressure rendering and legend.
  */
  class AGXOSG_EXPORT PressureRenderer : public agx::Referenced
  {
    public:
      /**
      Constructor using osg color range directly.
      \param root The scene root.
      \param decorator The scene decorator, used here for the hud-legend.
      \param simulation The simulation
      \param colorRange The color range, containing colors, and min and max pressure.
      \param texelsPerMeter The amount of texels per length unit (meter)
      \param rangeFactor Relationship between a contact force and area of the pressure circle. In m/N.
      \param forceType The type of the force to add. Tangential (friction) only, normal only, or both.
      */
      PressureRenderer(
        osg::Group* root,
        agxOSG::SceneDecorator* decorator,
        agxSDK::Simulation* simulation,
        osgSim::ColorRange* colorRange = new osgSim::ColorRange(0.0f, 1.0e5f),
        const agx::Real texelsPerMeter = agx::Real(50),
        const agx::Real rangeFactor = agx::Real(2.0e-5),
        agxOSG::PressureFromContacts::ForceType forceType = agxOSG::PressureFromContacts::TOTAL_FORCE
        );

      /**
      Constructor using agx data.
      \param root The scene root.
      \param decorator The scene decorator, used here for the hud-legend.
      \param simulation The simulation
      \param minPressure The min pressure for texturing
      \param maxPressure The max pressure for texturing
      \param colors A vector of colors for coloring. See osgSim::ColorRange for details.
      \param texelsPerMeter The amount of texels per length unit (meter)
      \param rangeFactor Relationship between a contact force and area of the pressure circle. In m/N.
      \param forceType The type of the force to add. Tangential (friction) only, normal only, or both.
      */
      PressureRenderer(
        osg::Group* root,
        agxOSG::SceneDecorator* decorator,
        agxSDK::Simulation* simulation,
        const agx::Real minPressure,
        const agx::Real maxPressure,
        const agxRender::ColorVector& colors,
        const agx::Real texelsPerMeter = agx::Real(50),
        const agx::Real rangeFactor = agx::Real(2.0e-5),
        agxOSG::PressureFromContacts::ForceType forceType = agxOSG::PressureFromContacts::TOTAL_FORCE
        );

      /**
      Adds a trimesh to the pressure rendering.
      The trimesh's geometry has to be part of the simulation,
      and have an osg-node which is in the root's tree.
      \param trimesh The trimesh.
      \param keepGeometryNode Should existing geometry nodes be kept?
        Interesting when blending render states.
      \returns True if successful, false otherwise
      */
      bool addTrimesh(agxCollide::Trimesh* trimesh, bool keepGeometryNode = false);

      /**
      Adds all shapes in a geometry to the pressure rendering.
      \note: right now, only trimeshes are supported.
      The geometry has to be part of the simulation,
      and have an osg-node which is in the root's tree.
      \param geometry The geometry.
      \param keepGeometryNode Should existing geometry nodes be kept?
        Interesting when blending render states.
      \returns True if successful, false otherwise
      */
      bool addGeometry(agxCollide::Geometry* geometry, bool keepGeometryNode = false);


      /**
       * Start recording of maximum and average pressure for each pressure
       * atlas currently owned by this PressureRenderer.
       *
       * Should not be called while a recording is already in progress.
       *
       * @return True is a recording was started. False if a recording was already taking place.
       */
      bool startMaxAvgRecording();

      /**
       * Stop the current recording. An ID for the recording is returned. The
       * recorded data can be rendered on the geometries using
       * 'renderMaxPressure(ID)' or 'renderAvgPressure(ID)'.
       *
       * Should only be called while recording.
       *
       * @return A recording ID for the new recording, or agx::InvalidIndex if not currently recording.
       */
      size_t stopMaxAvgRecording();

      /**
       * @return True if a currently recording. False otherwise.
       */
      bool isRecording();

      /**
       * @return The number of finished recordings.
       */
      size_t getNumRecordings();

      /**
       *
       * Replace the texture data for measured geometries with a recorded
       * maximum pressure. Each measured trimesh must be part of a geometry
       * rendered using the OSG mesh created by the owning PressureAtlas, and
       * the geometry must have an associated texture.
       *
       * Will change the osg::Image object held by the osg::Texture2D object
       * that is part of the OSG render tree.
       *
       * @param recording The recording ID from which the maximum pressure should be fetched.
       * @return True if the given recording was selected. False if the given recording ID is invalid.
       */
      bool renderMaxPressure(size_t recording);

      /**
       *
       * Replace the texture data for measured geometries with a recorded
       * average pressure. Each measured trimesh must be part of a geometry
       * rendered using the OSG mesh created by the owning PressureAtlas, and
       * the geometry must have an associated texture.
       *
       * Will change the osg::Image object held by the osg::Texture2D object
       * that is part of the OSG render tree.
       *
       * @param recording The recording ID from which the average pressure should be fetched.
       * @return True if the given recording was selected. False if the given recording ID is invalid.
       */
      bool renderAvgPressure(size_t recording);

      /**
       * Disable rendering of recorded maximum or average data. Each texture
       * will be returned the osg::Image object it held when the recording
       * started.
       *
       * @return True if original texture data was restored, fase if currently not rendering maximum or average pressure.
       */
      bool restoreRendering();

      /// Enables/disables legend in HUD
      void setEnableLegend(bool flag);

      /// Returns the legend center position in relative screen space (0 to 1).
      agx::Vec2 getLegendPosition() const;

      /// Sets the legend center position in relative screen space (0 to 1).
      void setLegendPosition(const agx::Vec2& pos);

      /// Gets the color range.
      osgSim::ColorRange* getColorRange();

      /// Gets the pressure atlas manager.
      agxOSG::PressureAtlasManager* getPressureAtlasManager();

      /**
      Add a contact filter. A contact will only contribute pressure if it is
      accepted by all filters.
      Filters will only affect meshes and geometries which have been added after the filter
      has been added.
      \param filter
      */
      void addFilter(agxSDK::ExecuteFilter* filter);


    protected:
      virtual ~PressureRenderer();

      // Helper function used by render[Max|Avg]Pressure.
      bool renderPressure(size_t recording, bool useMax);

      // Trigger a color conversion for the recording currently being recorded.
      // Only valid between calls to startMaxAvgRecording() and
      // stopMaxAvgRecording(). Called from stopMaxAvgRecording() right before
      // copying color information to the recordings list.
      void updateTextureData();

      // Remove the maximum- and average gatherers from their respective PressureAtlases.
      void unregisterRecorders();

    protected:
      agx::Vector<agxCollide::TrimeshRef> m_meshes;
      osg::observer_ptr<osg::Group> m_root;
      osg::observer_ptr<agxOSG::SceneDecorator> m_decorator;
      agxOSG::PressureAtlasManagerRef m_atlasManager;
      osg::ref_ptr<osgSim::ColorRange> m_colorRange; // Holds min pressure, max pressure, and colors.
      osg::ref_ptr<agxOSG::Transform> m_legendTransform; // Holding the transform to the HUD-legend.
      agx::Vec2 m_legendPosition;
      agx::Real m_rangeFactor;
      agx::Vector<agxSDK::ExecuteFilterRef> m_filters;


      /*
       * Data used during recording. On per measured mesh.
       */
      struct MaxAvgData {
        MaxAvgData(const agxCollide::TrimeshConstObserver mesh, GatherMaxPressureRef maxGatherer, GatherAvgPressureRef avgGatherer, osg::Group* root);
        osg::Texture2D* getTexture(osg::Group* root);

        const agxCollide::TrimeshConstObserver mesh;
        GatherMaxPressureRef maxGatherer;
        GatherAvgPressureRef avgGatherer;
        osg::ref_ptr<osg::Image> originalTextureImage; // Borrowed from the osg::Texture2D originaly used to render the mesh.
      private:
        // Hiding assignment
        MaxAvgData& operator=(const MaxAvgData&) {return *this;}

      };

      // Non-empty while recording. One element per measured mesh.
      agx::Vector<MaxAvgData> m_maxAvgData;


      /*
       * Data held for each mesh for each finished recording.
       */
      struct MaxAvgRecording {
        MaxAvgRecording(const MaxAvgData& data);
        osg::Texture2D* getTexture(osg::Group* root);

        const agxCollide::TrimeshConstObserver mesh;
        osg::ref_ptr<osg::Image> maxPressure; // Copy of the color map generated by a MaxAvgData::maxGatherer.
        osg::ref_ptr<osg::Image> avgPressure; // Copy of the color map generated by a MaxAvgData::avgGatherer.
        osg::ref_ptr<osg::Image> originalTextureImage; // Given from MaxAvgData upon stopMaxAvgRecording().
      private:
        // Hiding assignment
        MaxAvgRecording& operator=(const MaxAvgRecording&) {return *this;}
      };

      // Vector holding all recorded data for a particular recording.
      typedef agx::Vector<MaxAvgRecording> MaxAvgRecordingVector;

      // Vector holding all recordings. Recording IDs are indices into this vector.
      agx::Vector<MaxAvgRecordingVector> m_maxAvgRecordings;

      // The record ID that is currently being used for rendering.
      size_t m_currentRecording;

      // The type of force to register. Normal, tangential or both.
      const agxOSG::PressureFromContacts::ForceType m_forceType;
    private:
      // Hiding assignment
      PressureRenderer& operator=(const PressureRenderer&) {return *this;}
  };
}


#endif
