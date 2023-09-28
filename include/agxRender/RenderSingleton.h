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

#ifndef AGXRENDER_RENDERERSINGLETON_H
#define AGXRENDER_RENDERERSINGLETON_H

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning( disable : 4251 )
//  warning C4251:  'X' : class 'Y' needs to have dll-interface to be used by clients of class 'Z'
#endif

#include <agx/agxPhysics_export.h>
#include <agx/Singleton.h>
#include <agx/Vec4.h>
#include <agx/AffineMatrix4x4.h>

#include <agxRender/Color.h>

// Forward declarations
namespace agx
{
  class Constraint;
  class ConstraintImplementation;
}

namespace agxSDK
{
  class Simulation;
}

namespace agxRender
{
  class RenderManager;

  AGXPHYSICS_EXPORT void debugRenderFrame( const agx::AffineMatrix4x4& transform, float scale, const agx::Vec4f& colorIdentifier, agxRender::RenderManager* mgr = nullptr );
  AGXPHYSICS_EXPORT void debugRenderConstraintFrames( const agx::Constraint* constraint, float scale, agxRender::RenderManager* mgr = nullptr, agx::Bool overrideInvalid = false );
  AGXPHYSICS_EXPORT void debugRenderConstraintFrames( const agx::ConstraintImplementation* constraint, float scale, agxRender::RenderManager* mgr = nullptr, agx::Bool overrideInvalid = false );

  /// Utility class for creating debug rendering objects
  class AGXPHYSICS_EXPORT RenderSingleton : public agx::Singleton
  {
    public:

      /// \return a pointer to the RenderSingleton
      static RenderSingleton* instance();

      /// \return true if this singleton is initialized (instance would then return != 0)
      static bool isInitialized();

      /// Clear the list of temporary RenderProxies
      void clearTemporaries();

      /// Clear statistics
      void clearStatics();

      /// Clear everything
      void clearAll();

      /// Called by Simulation to update the debug rendering
      void render( agxRender::RenderManager* mgr );

      /// Defines a point to be used in DebugRendering.
      class AGXPHYSICS_EXPORT Point
      {
        public:
          Point() {}
          Point( const agx::Vec3& position, agx::Real size = agx::Real(0.4), const agx::Vec4f& color = agx::Vec4f( 1, 1, 1, 1 ) )
            : m_position( position ), m_size( size ), m_color( color ) {}

          agx::Vec3& position() {
            return m_position;
          }
          agx::Real& size() {
            return m_size;
          }
          agx::Vec4f& color() {
            return m_color;
          }

        private:
          agx::Vec3 m_position;
          agx::Real m_size;
          agx::Vec4f m_color;
      };

      /// Defines a Line to be used in DebugRendering.
      class AGXPHYSICS_EXPORT Line
      {
        public:
          Line() {}
          Line( const agx::Vec3& start, const agx::Vec3& end, agx::Real size = 0.02f, const agx::Vec4f& color = agx::Vec4f( 1, 1, 1, 1 ) )
            : m_start( start ), m_end( end ), m_size( size ), m_color( color ) {}

          agx::Vec3& start() {
            return m_start;
          }
          agx::Vec3& end() {
            return m_end;
          }
          agx::Real& size() {
            return m_size;
          }
          agx::Vec4f& color() {
            return m_color;
          }

        private:
          agx::Vec3 m_start;
          agx::Vec3 m_end;
          agx::Real m_size;
          agx::Vec4f m_color;
      };

      /// Defines Text to be used in DebugRendering.
      class AGXPHYSICS_EXPORT Text
      {
        public:
          Text() {}
          Text( const agx::String& text, const agx::Vec2& position, const agx::Vec4f& color = agx::Vec4f( 1, 1, 1, 1 ) )
            : m_text( text ), m_position( position ), m_color( color ) {}

          agx::String& text() {
            return m_text;
          }
          agx::Vec2& position() {
            return m_position;
          }
          agx::Vec4f& color() {
            return m_color;
          }

        private:
          agx::String m_text;
          agx::Vec2 m_position;
          agx::Vec4f m_color;
      };


      /// Defines a Triangle to be used in DebugRendering.
      class AGXPHYSICS_EXPORT Triangle
      {
        public:
          Triangle() {}
          Triangle( const agx::Vec3& v0, const agx::Vec3& v1, const agx::Vec3& v2, agx::Real size = 0.02f, const agx::Vec4f& color = agx::Vec4f( 1, 1, 1, 1 ) )
            : m_v0( v0 ), m_v1( v1 ), m_v2( v2 ), m_size(size), m_color( color ) {}

          agx::Vec4f& color() {
            return m_color;
          }

          agx::Vec3& v0() {
            return m_v0;
          }
          agx::Vec3& v1() {
            return m_v1;
          }
          agx::Vec3& v2() {
            return m_v2;
          }
          agx::Real& size() {
            return m_size;
          }

        private:
          agx::Vec3 m_v0;
          agx::Vec3 m_v1;
          agx::Vec3 m_v2;
          agx::Real m_size;
          agx::Vec4f m_color;
      };

      class AGXPHYSICS_EXPORT Frame
      {
        public:
          Frame() {}
          Frame( const agx::AffineMatrix4x4& frame, agx::Real scale = agx::Real( 0.5 ), const agx::Vec4f& color = agx::Vec4f( 1, 1, 1, 1 ) ) : m_frame( frame ), m_scale( scale ), m_color( color ) {}

          const agx::AffineMatrix4x4& getFrame() const { return m_frame; }
          agx::Real getScale() const { return m_scale; }
          agx::Vec4f getColor() const { return m_color; }

        private:
          agx::AffineMatrix4x4 m_frame;
          agx::Real m_scale;
          agx::Vec4f m_color;
      };

      /**
      Add temporary point (sphere).
      */
      void add( const Point& point );
      /**
      Add temporary point (sphere).
      */
      void add( const agx::Vec3& position, agx::Real size, const agx::Vec4f& color ) {
        add( Point( position, size, color ) );
      }
      /**
      Add static point (sphere).
      */
      void addStatic( const Point& point );
      /**
      Add static point (sphere).
      */
      void addStatic( const agx::Vec3& position, agx::Real size, const agx::Vec4f& color )  {
        addStatic( Point( position, size, color ) );
      }

      /**
      Add temporary line (cylinder)
      */
      void add( const Line& line );
      /**
      Add temporary line (cylinder)
      */
      void add( const agx::Vec3& start, const agx::Vec3& end, agx::Real size, const agx::Vec4f& color ) {
        add( Line( start, end, size, color ) );
      }
      /**
      Add static line (cylinder)
      */
      void addStatic( const Line& line );
      /**
      Add static line (cylinder)
      */
      void addStatic( const agx::Vec3& start, const agx::Vec3& end, agx::Real size, const agx::Vec4f& color ) {
        addStatic( Line( start, end, size, color ) );
      }

      /**
      Add temporary text
      */
      void add( const Text& text );
      /**
      Add temporary text
      */
      void add( const agx::String& text, const agx::Vec2& position, const agx::Vec4f& color ) {
        add( Text( text, position, color ) );
      }
      /**
      Add static text
      */
      void addStatic( const Text& text );
      /**
      Add static text
      */
      void addStatic( const agx::String& text, const agx::Vec2& position, const agx::Vec4f& color ) {
        addStatic( Text( text, position, color ) );
      }

      /**
      Add temporary triangle
      */
      void add(const Triangle& triangle);
      /**
      Add temporary triangle
      */
      void add(const agx::Vec3& v0, const agx::Vec3& v1, const agx::Vec3& v2, const agx::Real size, const agx::Vec4f& color) {
        add(Triangle(v0, v1, v2, size, color));
      }
      /**
      Add static triangle
      */
      void addStatic( const Triangle& triangle);
      /**
      Add static triangle
      */
      void addStatic(const agx::Vec3& v0, const agx::Vec3& v1, const agx::Vec3& v2, const agx::Real size, const agx::Vec4f& color) {
        addStatic(Triangle(v0, v1, v2, size, color));
      }

      /**
      Add temporary frame.
      */
      void add( const agx::AffineMatrix4x4& frame, agx::Real scale, const agx::Vec4f& color );

      /**
      Add static frame.
      */
      void addStatic( const agx::AffineMatrix4x4& frame, agx::Real scale, const agx::Vec4f& color );

      SINGLETON_CLASSNAME_METHOD();

    protected:
      RenderSingleton();
      virtual void shutdown() override;

    private:
      static RenderSingleton* s_instance;
      agx::Vector< Point > m_staticPoints;
      agx::Vector< Point > m_points;
      agx::Vector< Line > m_staticLines;
      agx::Vector< Line > m_lines;
      agx::Vector< Text > m_staticText;
      agx::Vector< Text > m_text;
      agx::Vector< Triangle > m_staticTriangles;
      agx::Vector< Triangle > m_triangles;
      agx::Vector< Frame > m_staticFrames;
      agx::Vector< Frame > m_frames;
  };
}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
