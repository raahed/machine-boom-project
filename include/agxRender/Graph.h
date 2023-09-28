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

#ifndef AGXRENDER_GRAPH_H
#define AGXRENDER_GRAPH_H


#include <agx/Vector.h>
#include <agx/Vec3.h>
#include <agx/Vec4.h>
#include <agx/List.h>

namespace agxRender
{


  AGX_DECLARE_POINTER_TYPES( Graph );

  /// Class that implements a simple graph for plotting scalar values
  class AGXPHYSICS_EXPORT Graph : public agx::Referenced
  {
    public:

      typedef agx::Vector<agx::Vec2> DataVector;

    AGX_DECLARE_POINTER_TYPES( GraphRenderer );

    /**
    Abstract base class that acts as an interface and can render the data from a Graph.
    This class has to be implemented for each rendering engine.
    */
    class AGXPHYSICS_EXPORT GraphRenderer : public agx::Referenced
      {
        public:
          GraphRenderer() {}

          /**
          Called when text should be drawn.
          \param pos - position in x,y in the range [0..1]
      \param str - text to draw
          */
          virtual void drawText( const agx::Vec2& pos, const agx::String& str ) = 0;

          /**
          Set the current color for items to be drawn next.
      \param color - Color to draw
          */
          virtual void setColor( const agx::Vec4& color ) = 0;

          /**
          Draw a line from \p p1 to \p p2
          */
          virtual void drawLine( const agx::Vec2& p1, const agx::Vec2& p2 ) const = 0;

          /**
          If the rendering of this graph requires some pre-state to be set (orthographic projection etc)
          this can be done in this method. It will be called before any draw* methods
          */
          virtual void preDraw() = 0;

          /**
          If the rendering of this graph requires some cleanup to be done it can be done in this method.
          It will be called AFTER any draw* calls.
          */
          virtual void postDraw() = 0;

          /**
          Destruction of any data for this graph. After this, no more calls to draw will be done.
          */
          virtual void clear() = 0;

          /**
          Implement functionality for enabling/disabling the rendering of this graph.
          Might involve removing stuff from your specific scenegraph or similar.
          */
          virtual void setEnable( bool flag ) = 0;

          /**
          Setup structures/data for another data channel. Will later be indexed from the drawData method.
          */
          virtual void addChannel() = 0;

          /// \return number of added channels
          virtual size_t getNumChannels() const = 0;

          /**
          Tear down data for the last added channel.
          */
          virtual void removeChannel() = 0;

          /**
          Update the line geometry for a specified channel given the data (points)
          */
          virtual void drawData( size_t channelIndex, const Graph::DataVector& data ) = 0;


        protected:

          friend class Graph;
          agx::observer_ptr<Graph> m_graph;
          virtual ~GraphRenderer() {}
      };


    /// Internal class for handling data for a channel.
    class AGXPHYSICS_EXPORT Channel : public agx::Referenced
      {
        public:

          Channel( const agx::String& title , const agx::Vec4& color,
                   float minVal, float maxVal,
                   float scaleX = 1.0f,  float scaleY = 1.0f );

          void setResolution( size_t r ) {
            m_cacheSize = r;
          }
          void update( float data );

          void setEnableMA( bool flag ) {
            m_maEnabled = flag;
          }
          bool getEnableMA( ) const {
            return m_maEnabled;
          }

          const agx::Vec4& getColor() const {
            return m_color;
          }
        protected:

          float mapToLimits( float val ) const;

          agx::String m_title;

          void drawTitle( const agx::Vec2& pos );
          void drawData() ;
          void drawMovingAverage() ;

          friend class Graph;

          virtual ~Channel() {}

        private:

          agx::observer_ptr<Graph> m_parent;

          bool m_maEnabled;
          float m_scaleX;
          float m_scaleY;

          agx::Vec4 m_color;

          float m_limits[2];

          size_t m_cacheSize;

          agx::List<float> m_data;
          agx::List<float> m_movingAverage;
      };

      Graph( float width = float( 0.9 ), float height = float( 0.2 ), const agx::Vec2& offset = agx::Vec2( agx::Real( 0.01 ), agx::Real( 0.01 ) ) );
      void setSize( float width, float height ) {
        m_width = width;
        m_height = height;
      }
      void setOffset( const agx::Vec2& offset ) {
        m_offset = offset;
      }

      /**
      Will check so that the number of channels for the associated GraphRenderer has equal # channels as this graph
      */
      bool validate() const;

      void addChannel( Channel *channel );
      size_t numChannels() const {
        return m_channels.size();
      }
      void clearChannels();

      void setEnable( bool flag );
      bool getEnable(  ) const {
        return m_enable;
      }

      Channel *getChannel( size_t idx ) {
        if ( idx >= m_channels.size() ) return nullptr;
        return m_channels[idx];
      }

      void setRenderer( GraphRenderer *renderer ) {
        m_renderer = renderer;
        renderer->m_graph = this;
      }
      GraphRenderer *getRenderer(  ) {
        return m_renderer;
      }
      const GraphRenderer *getRenderer(  ) const {
        return m_renderer;
      }

      const agx::Vec2& getOffset() const {
        return m_offset;
      }

      float getWidth() {
        return m_width;
      }
      float getHeight() {
        return m_height;
      }
      void draw();

      void clearPoints() {
        m_dataVector.clear();
      }
      void reserve( size_t size ) {
        m_dataVector.reserve( size );
      }
      void addPoint( const agx::Vec2& point );
      void addText( const agx::Vec2& point, const agx::String& str );

      void preDraw();
      void postDraw();

    protected:

      void drawData( size_t channelIndex ) const;

      virtual ~Graph();

    private:

      typedef agx::Vector< agx::ref_ptr<Channel> > ChannelVector;
      ChannelVector m_channels;

      agx::ref_ptr<GraphRenderer> m_renderer;


      float m_width, m_height;
      agx::Vec2 m_offset;
      DataVector m_dataVector;

      bool m_enable;
  };

}

#endif
