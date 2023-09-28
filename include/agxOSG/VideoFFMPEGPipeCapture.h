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


#ifndef AGXOSG_VIDEOFFMPEGPIPECAPTURE_H
#define AGXOSG_VIDEOFFMPEGPIPECAPTURE_H

#include <agxOSG/ImageCapture.h>
#include <agxOSG/ExternalProcess.h>

typedef void *HANDLE;

namespace agxOSG
{
  /**
  Calls that enabled video creation by sending raw captured image frames to an external FFMPEG process by named pipes.
  The class is responsible for starting and handling the external FFMPEG, the pipe connections and capturing images.
  The writeImage   method is overridden to write the raw image data via a named pipe to the started FFMPEG process.
  When the destructor is called, the named pipe will be closed, thus triggering the closure of FFMPEG so that the video is finalized.
  */
  AGX_DECLARE_POINTER_TYPES(VideoFFMPEGPipeCapture);
  class AGXOSG_EXPORT VideoFFMPEGPipeCapture : public ImageCaptureBase
  {
  public:
    class DrawCallback;

    /// Enum describing the different types of codecs that are available
    enum class OutputCodec : agx::UInt32
    {
      H264 = 0,
      H264RGB = 1,
      XVID = 2
    };

    // Codec hashFunc
    struct CodecHash
    {
      template <typename T>
      agx::UInt32 operator()(T t) const
      {
        return static_cast<agx::UInt32>(t);
      }
    };

    /// Map for storing codec string conversion for FFMPEG
    typedef agx::HashTable<OutputCodec, agx::String, CodecHash> CodecStringConversionMap;
    static CodecStringConversionMap OutputCodecStrings;

  public:
    /// Constructor
    /**
    \param width The width of the finalized video
    \param height The height of the finalized video
    \param fps The video FPS
    \param sim attached simulation, used for timestep synchronization
    */
    VideoFFMPEGPipeCapture(agx::UInt width, agx::UInt height, agx::Real fps, agxSDK::Simulation *sim = nullptr);

    /// Initializes the FFMPEG process and the named pipes
    void init();

    /// Sets the filename for the video, excluding the postfix.
    void setFilename(const agx::String& filename);

    /// Sets the video postfix
    void setVideoPostFix(const agx::String& postFix);

    /// Gets the video postfix
    agx::String getVideoPostFix() const;

    /// Gets the video filename, excluding the postfix.
    agx::String getFilename() const;

    /// Set the FPS of the video
    void setVideoFPS(agx::Real videoFPS);

    /// Get the current FPS of the video
    agx::Real getVideoFPS() const;

    /// Set the CRF of the video
    void setVideoCRF(agx::UInt crf);

    /// get the CRF of the video
    agx::UInt getVideoCRF() const;

    /// Set the image capture FPS. Synchronize this with video FPS to control the real time factor. (videoFPS / imageFPS == 1.0 for realtime)
    void setImageCaptureFPS(agx::Real imageFPS);

    /// Get the image capture FPS. Synchronize this with video FPS to control the real time factor. (videoFPS / imageFPS == 1.0 for realtime)
    agx::Real getImageCaptureFPS();

    /// Set true/false if lossless encoding should be enabled. This will produce videos not supported by older media players
    void setEnableLossless(bool enable);

    /// Get true/false if lossless encoding should be enabled. This will produce videos not supported by older media players
    bool getEnableLossless() const;

    /// Set the video codec for the output. Available codecs are given in OutPutCodecs
    void setOutputVideoCodec(OutputCodec codec);

    /// Get the video codec for the output. Available codecs are given in OutPutCodecs
    OutputCodec getOutputCodec() const;

    /// Returns true if we have active video that is being created
    bool hasActiveVideo() const;

    /**
    Stops the ffmpeg process and finalizes the video file, allowing a new video capture to be started using startCapture().
    \return false if the ffmpeg process returned a non-zero value, true otherwise
    */
    bool stopProcess();

  protected:
    virtual ~VideoFFMPEGPipeCapture();

    void startExternalFFMPEGProcess();

    void closeExternalFFMPEGProcess();

    agx::String generatePipeName() const;

    agx::String getPipeName() const;

    agx::String getFFMPEGBinaryPath() const;

    agx::String getCompleteFileName() const;

  private:

    virtual void writeImage(osg::Image *image, agx::Index index);

  private:
    agx::String            m_filename;
    agx::String            m_videoPostFix;
    agx::Real              m_videoFPS;
    agx::String            m_pipeName;
    OutputCodec            m_outputCodec;
    agx::UInt              m_crf;
    agx::UInt              m_numWrittenFramesToPipe;
    bool m_firstFrame;
    bool m_enableLossless;

#ifdef _MSC_VER
    HANDLE hPipe;
#else
    FILE* m_fifo;
#endif
    agxOSG::ExternalProcessRef m_imageServerProcess;
  };
}

#endif /* AGXOSG_VideoFFMPEGPipeCapture_H */
