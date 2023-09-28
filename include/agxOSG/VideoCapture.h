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


#ifndef AGXOSG_VIDEOCAPTURE_H
#define AGXOSG_VIDEOCAPTURE_H

#include <agx/config/AGX_USE_FFMPEG.h>
#include <agxOSG/ImageCapture.h>
#include <agxData/BinaryData.h>

#if AGX_USE_FFMPEG()

extern "C"
{
  #include <agx/PushDisableWarnings.h>
  #include <libavutil/opt.h>
  #include <libavcodec/avcodec.h>
  #include <libavutil/channel_layout.h>
  #include <libavutil/common.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/mathematics.h>
  #include <libavutil/samplefmt.h>
  #include <libswscale/swscale.h>
  #include <libavformat/avformat.h>
  #include <agx/PopDisableWarnings.h>
}

namespace agxOSG
{
  AGX_DECLARE_POINTER_TYPES(VideoCapture);
  class AGXOSG_EXPORT VideoCapture : public ImageCaptureBase
  {
  public:
    class DrawCallback;

  public:

    VideoCapture(agx::UInt width, agx::UInt height, agx::UInt fps, agxSDK::Simulation *sim = nullptr);

    void init(agx::UInt width, agx::UInt height, agx::UInt fps, AVCodecID codec);

    typedef agx::Callback2<agx::UInt8 *, size_t> DataCallback;

    void setDataCallback(DataCallback callback);

    agxData::BinaryData *writeHeader();

    void setFilename(const agx::String& filename);

    agx::String getFilename() const;

    void setMovieFPS(agx::UInt fps);

    void finalizeMovie();

  protected:
    virtual ~VideoCapture();

  private:

    void finalize();

    bool encode(AVFrame *frame);

    virtual void writeImage(osg::Image *image, agx::Index index);
    static int writePacket(void *opaque, uint8_t *buf, int buf_size);

  private:
    AVFormatContext *      m_formatContext;
    AVOutputFormat *       m_format;
    AVCodecContext *       m_codecContext;
    AVCodec *              m_codec;
    AVStream *             m_stream;
    AVFrame *              m_frame;
    SwsContext *           m_swsContext;
    DataCallback           m_dataCallback;
    unsigned char *        m_fragmentBuffer;
    FILE *                 m_file;
    bool                   m_firstFrame;
    bool                   m_writingHeader;
    agxData::BinaryDataRef m_header;
    agx::String            m_filename;
    agx::UInt              m_videoFPS;
  };

}


#endif
#endif /* AGXOSG_VIDEOCAPTURE_H */
