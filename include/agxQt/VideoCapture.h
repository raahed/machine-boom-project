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

#ifndef AGXQT_QTSIMULATIONRECORDER_H
#define AGXQT_QTSIMULATIONRECORDER_H

#include <agxQt/export.h>
#include <agx/Vec3.h>
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osgViewer/ViewerEventHandlers>
#include <osgViewer/Renderer>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/Referenced.h>
#include <agxSDK/Simulation.h>
#include <agxOSG/ImageCapture.h>
#include <agxOSG/RenderToTexture.h>
#include <agxOSG/VideoFFMPEGPipeCapture.h>

namespace agxQt
{
  class QtAgxWrapper;

  /**
  * Helper class that records videos of simulations by sending images to a separate FFMPEG process.
  */
  AGX_DECLARE_POINTER_TYPES(VideoCapture);
  class AGXQT_EXPORT VideoCapture : public agx::Referenced
  {
  public:
    /// Default constructor
    VideoCapture(QtAgxWrapper * wrapper);

    /**
    Set the desired video frame rate of the capture
    */
    void setVideoFPS(agx::Real fps);

    /**
    Get the desired video frame rate of the capture
    */
    agx::Real getVideoFPS() const;

    /**
    Set to true/false if the video capture should synchronize with simulation time stepping.
    */
    void setEnableSyncWithSimulation(bool enable);

    /**
    Set the simulation that the video capture should synchronize with
    */
    void setSimulation(agxSDK::Simulation * simulation);

    /**
    Set the complete filename of the video file that should be generated, including postfix
    */
    void setCompleteFileName(const std::string& filename);

    /**
    Get the filename, without the postfix, of the video file that should be generated
    */
    agx::String getFileNameWithoutPostFix() const;

    /**
    Get the filename, including the postfix, of the video file that should be generated
    */
    agx::String getCompleteFileName() const;

    /**
    Set the resolution of the video that is to be generated
    */
    void setVideoResolution(const agx::Vec2u& resolution);

    /**
    Set the real time factor of the video, which is the ratio between the video FPS and image capture FPS
    example: videoFPS(60Hz) / imageFPS(30Hz)
    */
    void setRealTimeFactor(agx::Real factor);

    /**
    Get the real time factor of the video, which is the ratio between the video FPS and image capture FPS
    example: videoFPS(60Hz) / imageFPS(30Hz)
    */
    agx::Real getRealTimeFactor() const;

    /**
    Set the quality of the video which is a normalized value between 0.0-1.0. This is typically converted into a CRF value
    if h264 is used in video generation.
    */
    void setQuality(agx::Real quality);

    /**
    Get the quality of the video which is a normalized value between 0.0-1.0. This is typically converted into a CRF value
    if h264 is used in video generation.
    */
    agx::Real getQuality();

    /**
    Starts or resumes the video capture
    */
    void startCapture();

    /**
    Pauses the video capture, if active
    */
    void stopCapture();

    /**
    Returns true if there is an active video capture process, otherwise false.
    */
    bool hasActiveVideo();

    /**
    Returns the number of generated video frames.
    */
    agx::UInt getNumGeneratedImages();

    /**
    Finalizes the current video process by completing the video file and closing down the video capture process.
    Will call stopCapture() internally.
    */
    void finalizeVideo();

  protected:
    /// Protected destructor
    virtual ~VideoCapture();

  private:
    void initResolutionForRenderToTexture(const agx::Vec2u& resolution);

    //////////////////////////////////////////////////////////////////////////
    // Variables
    //////////////////////////////////////////////////////////////////////////
  protected:
    QtAgxWrapper*                                m_qtagxWrapper;
    agx::ref_ptr<agxOSG::VideoFFMPEGPipeCapture> m_videoCapture;
    agx::UInt                                    m_imageNumber;
    agxOSG::RenderToTextureRef                   m_renderToTexture;
    agx::Real m_realTimeFactor;
    agx::Real m_quality;
  };
}


#endif
