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

#ifndef AGXOSG_IMAGECAPTURE_H
#define AGXOSG_IMAGECAPTURE_H

#include <queue>
#include <atomic>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Camera>
#include <osg/observer_ptr>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

#include <agx/ThreadSynchronization.h>

#include <agxSDK/Simulation.h>
#include <agx/Timer.h>
#include <agx/Thread.h>
#include <agx/Referenced.h>
#include <agxOSG/export.h>



namespace agxOSG {

  AGX_DECLARE_POINTER_TYPES(ImageCaptureBase);
  class AGXOSG_EXPORT ImageCaptureBase : public agx::Referenced
  {
  public:
    class DrawCallback;

  public:
    ImageCaptureBase(agxSDK::Simulation *sim = nullptr);

    /**
    Set the desired frame rate (rate of capture).
    */
    void setFps(agx::Real fps);

    /**
    Return the specified desired frame rate for capture
    */
    agx::Real getFps();

    /** Set a limit of the maximum number of images that should be captured
    Set -1 for no limit
    */
    void setMaxImages(int i);

    /** Enable/disable capturing */
    virtual void startCapture( int startImageNumber=0 );

    /// Return false if stopping an already stopped capture process.
    virtual bool stopCapture();

    /** Return whether the capturing is enabled or not. */
    bool getCaptureStarted() const;

    agxSDK::Simulation *getSimulation();
    void setSimulation(agxSDK::Simulation *simulation);

    void setImageResolution(agx::UInt width, agx::UInt height);
    agx::Vec2u getImageResolution() const;


    int getImageNum() const;

    DrawCallback *getDrawCallback();

    void setEnableSyncWithSimulation( bool enableSync );
    bool getEnableSyncWithSimulation(  ) const;

    void resetTime();

    virtual void apply( osg::RenderInfo& renderInfo );

    // Called in separate consumer thread
    virtual void writeImage(osg::Image *image, agx::Index index) = 0;

  protected:
    virtual ~ImageCaptureBase();
    void setCaptureStarted(bool f);
    void shutdown();

  private:
    class ImageVector;
    AGX_DECLARE_POINTER_TYPES(WriteImageThread);

    typedef std::queue<std::pair<osg::ref_ptr<osg::Image>, agx::Index> > ImageVectorT;

  private:
    agxSDK::SimulationObserver m_simulation;

    agx::UInt m_width;
    agx::UInt m_height;

    mutable agx::ReentrantMutex m_mutex;
    bool m_captureEnabled;
    int m_maxImages;
    int m_imageNumber;
    agx::Timer m_timer;
    agx::Real m_lastTime;
    bool m_first;
    float m_fps;
    bool m_sync;
    osg::ref_ptr<DrawCallback> m_drawCallback;

    WriteImageThreadRef m_writeImageThread;
  };

  /**
  A class for capturing the simulation rendering to images on the disk.
  */
  AGX_DECLARE_POINTER_TYPES(ImageCapture);
  class AGXOSG_EXPORT ImageCapture : public ImageCaptureBase
  {
  public:

    ImageCapture(agxSDK::Simulation *sim = nullptr);


    std::string getPostfix() const
    {
      return m_postfix;
    }

    /**
    Sets the postfix.
    Should be of form "bmp", "png" and so on.
    Default is "bmp".
    */
    void setPostfix(const std::string& postfix)
    {
      m_postfix = postfix;
    }

    std::string getPrefix() const
    {
      return m_prefix;
    }

    void setPrefix(const std::string& prefix)
    {
      m_prefix = prefix;
    }

    std::string getDirectoryPath() const
    {
      return m_directory;
    }

    void setDirectoryPath(const std::string& path)
    {
      m_directory = path;
    }

    virtual void writeImage(osg::Image *image, agx::Index index);

  protected:

    virtual ~ImageCapture();

  private:
    std::string m_prefix;
    std::string m_postfix;
    std::string m_directory;
  };


  class ImageCaptureBase::ImageVector : public ImageCaptureBase::ImageVectorT
  {
  public:
    ImageVector() {}

    void push(ImageVectorT::value_type image_set)
    {
      agx::ScopeLock<agx::ReentrantMutex> lock(m_mutex);
      ImageVectorT::push(image_set);
      m_block.release();
    }

    size_t size() const
    {
      agx::ScopeLock<agx::ReentrantMutex> lock(m_mutex);
      return ImageVectorT::size();
    }

    void clear()
    {
      agx::ScopeLock<agx::ReentrantMutex> lock(m_mutex);
      while(ImageVectorT::size())
        ImageVectorT::pop();
    }

    void lock()
    {
      m_mutex.lock();
    }

    void unlock()
    {
      m_mutex.unlock();
    }

    void block()
    {
      m_block.block();
      m_block.reset();
    }

    void release()
    {
      m_block.release();
    }

  private:
    agx::Block m_block;
    mutable agx::ReentrantMutex m_mutex;
  };



  class ImageCaptureBase::WriteImageThread : public agx::BasicThread, public agx::Referenced
  {
  public:
    WriteImageThread(ImageCaptureBase *cap);
    virtual ~WriteImageThread();

    void push(ImageVectorT::value_type image_set);

    virtual void run();
    void stop();

  private:
    ImageVectorT::value_type topAndPop();

  private:
    ImageCaptureBase *m_imageCapture;
    ImageVector m_queue;
    std::atomic<bool> m_stop;
    agx::Block m_joinBlock;
  };


  class AGXOSG_EXPORT ImageCaptureBase::DrawCallback : public osg::Camera::DrawCallback
  {
  public:
    DrawCallback(ImageCaptureBase* capture);

    virtual void operator () (osg::RenderInfo& renderInfo) const;
    using osg::Camera::DrawCallback::operator();
    ImageCaptureBase* getImageCapture();
    void setImageCapture( ImageCaptureBase* ic );

  private:
    ImageCaptureBase *m_imageCapture;
  };

}

#endif
