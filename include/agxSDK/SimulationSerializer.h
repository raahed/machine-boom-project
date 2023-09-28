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
#ifndef AGXSDK_SIMULATIONSERIALIZER_H
#define AGXSDK_SIMULATIONSERIALIZER_H


#include <agxSDK/StepEventListener.h>
namespace agxSDK
{

  /// class for serializing a simulation into a series for files on disk \<filename\>_0001.agx etc.
  class AGXPHYSICS_EXPORT SimulationSerializer : public agxSDK::StepEventListener
  {
    public:

      /**
      Default constructor, sets interval to 30hz and filename to "agxFile_00001.agx", writing in POST event.
      */
      SimulationSerializer( );

      /**
      Specify in what frequency the serializing of a file should occur.
      \param interval in seconds
      */
      void setInterval(agx::Real interval);

      /**
      Specify the base filename used for dumping to disk:  etc.
      */
      void setFilename( const agx::String& filename );

      /// \return the filename
      agx::String getFilename() const;

      /// \return the interval in seconds
      agx::Real getInterval( ) const;

      // Specify in when the dumping should occur: PRE, POST etc.
      void setMode( ActivationMask m );

      // Returns the current frame.
      int getCurrentFrame();

      /// Reset the frame number so next starts at 0
      void resetFrame();

      /**
      Assembles current filename in form \<filename_prefix\>_0001.\<filename_postfix\>,
      with 0001 being an example for the current frame.
      */
      agx::String getCurrentFilename() const;

    protected:
      void post( const agx::TimeStamp& t );
      void pre( const agx::TimeStamp& t );
      void preCollide( const agx::TimeStamp& t );

      virtual void write( const agx::TimeStamp& t );

      virtual ~SimulationSerializer();

      bool m_isPov;
      agx::Real m_interval;
      int m_currentFrame;
      agx::String m_filename;
      agx::Real m_lastTime;
      ActivationMask m_mode;
  };

  typedef agx::ref_ptr<SimulationSerializer> SimulationSerializerRef;

}


#endif
