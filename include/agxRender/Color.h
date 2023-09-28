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

#ifndef AGXRENDER_COLOR_H
#define AGXRENDER_COLOR_H

#include <agx/Vector.h>
#include <agx/Vec4.h>
#include <agx/agxPhysics_export.h>

namespace agxRender
{

  /**
  Utility color class with "common colors".
  Ref: http://kb.iu.edu/data/aetf.html

  Stored as float [0..1]
  */
  class  Color : public agx::Vec4f
  {
    public:

      /// Default constructor. Set color to 0,0,0,1
      Color()
        : agx::Vec4f( 0.f, 0.f, 0.f, 1.f ) {}

      /// Constructor
      explicit Color( float r, float g, float b, float a = 1.f )
        : agx::Vec4f( r, g, b, a ) {}

      /// Constructor
      explicit Color(const agx::Vec3f& color, float a = 1.f)
        : agx::Vec4f(color[0], color[1], color[2], a) {}

      /**
      Constructor which set all elements of color to \p c
      \param c - Color (r,g,b,a) will be set to c
      */
      explicit Color(float c)
        : agx::Vec4f(c) {}


      AGX_FORCE_INLINE float  r() const { return x(); }
      AGX_FORCE_INLINE float& r()       { return x(); }
      AGX_FORCE_INLINE float  g() const { return y(); }
      AGX_FORCE_INLINE float& g()       { return y(); }
      AGX_FORCE_INLINE float  b() const { return z(); }
      AGX_FORCE_INLINE float& b()       { return z(); }
      AGX_FORCE_INLINE float  a() const { return w(); }
      AGX_FORCE_INLINE float& a()       { return w(); }

      inline operator agx::Vec3f() const
      {
        return agx::Vec3f( r(), g(), b() );
      }

      inline operator agx::Vec3d() const
      {
        return agx::Vec3d( (double)r(), (double)g(), (double)b() );
      }

      inline operator agx::Vec4d() const
      {
        return agx::Vec4d( (double)r(), (double)g(), (double)b(), (double)a() );
      }

      inline agx::Vec4 asVec4() const
      {
        return agx::Vec4((agx::Real)r(), (agx::Real)g(), (agx::Real)b(), (agx::Real)a() );
      }

      /// Set the color as HSV (Hue, Saturation, Value)
      AGXPHYSICS_EXPORT void fromHSV( float h, float s, float v );
      AGXPHYSICS_EXPORT void fromHSV( const Color& hsv );

      /// \return the color as HSV
      AGXPHYSICS_EXPORT Color asHSV() const;

      /// Convert a rgb color to HSV
      AGXPHYSICS_EXPORT static Color RGBToHSV( const Color& rgb );
      /// Convert a HSV color to RGB
      AGXPHYSICS_EXPORT static Color HSVToRGB( const Color& rgb );


    public:
      /// \return a color as defined by the list of static methods below. Index goes from 0 to the number of colors. Invalid index returns first color.
      AGXPHYSICS_EXPORT static agxRender::Color getColor( size_t );

      /// \return number of colors in predefined color table
      AGXPHYSICS_EXPORT static size_t getNumColors( );

      // If you add or remove a color, be sure to reflect that in the InitColorTable method!
      inline static agxRender::Color                Wheat() { return agxRender::Color( 0.960784f, 0.870588f, 0.701961f, 1.f ); }
      inline static agxRender::Color            DarkKhaki() { return agxRender::Color( 0.741176f, 0.717647f, 0.419608f, 1.f ); }
      inline static agxRender::Color        DarkGoldenrod() { return agxRender::Color( 0.721569f, 0.525490f, 0.043137f, 1.f ); }
      inline static agxRender::Color          YellowGreen() { return agxRender::Color( 0.603922f, 0.803922f, 0.196078f, 1.f ); }
      inline static agxRender::Color            PaleGreen() { return agxRender::Color( 0.596078f, 0.984314f, 0.596078f, 1.f ); }
      inline static agxRender::Color          SpringGreen() { return agxRender::Color( 0.000000f, 1.000000f, 0.498039f, 1.f ); }
      inline static agxRender::Color            AliceBlue() { return agxRender::Color( 0.941176f, 0.972549f, 1.000000f, 1.f ); }
      inline static agxRender::Color            OrangeRed() { return agxRender::Color( 1.000000f, 0.270588f, 0.000000f, 1.f ); }
      inline static agxRender::Color          LightSalmon() { return agxRender::Color( 1.000000f, 0.627451f, 0.478431f, 1.f ); }
      inline static agxRender::Color            SlateBlue() { return agxRender::Color( 0.415686f, 0.352941f, 0.803922f, 1.f ); }
      inline static agxRender::Color           Aquamarine() { return agxRender::Color( 0.498039f, 1.000000f, 0.831373f, 1.f ); }
      inline static agxRender::Color            OliveDrab() { return agxRender::Color( 0.419608f, 0.556863f, 0.137255f, 1.f ); }
      inline static agxRender::Color           PapayaWhip() { return agxRender::Color( 1.000000f, 0.937255f, 0.835294f, 1.f ); }
      inline static agxRender::Color         LemonChiffon() { return agxRender::Color( 1.000000f, 0.980392f, 0.803922f, 1.f ); }
      inline static agxRender::Color              OldLace() { return agxRender::Color( 0.992157f, 0.960784f, 0.901961f, 1.f ); }
      inline static agxRender::Color          FloralWhite() { return agxRender::Color( 1.000000f, 0.980392f, 0.941176f, 1.f ); }
      inline static agxRender::Color               Maroon() { return agxRender::Color( 0.501961f, 0.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color                Linen() { return agxRender::Color( 0.980392f, 0.941176f, 0.901961f, 1.f ); }
      inline static agxRender::Color       LightSlateBlue() { return agxRender::Color( 0.517647f, 0.439216f, 1.000000f, 1.f ); }
      inline static agxRender::Color        DarkSlateGray() { return agxRender::Color( 0.184314f, 0.309804f, 0.309804f, 1.f ); }
      inline static agxRender::Color            CadetBlue() { return agxRender::Color( 0.372549f, 0.619608f, 0.627451f, 1.f ); }
      inline static agxRender::Color          NavajoWhite() { return agxRender::Color( 1.000000f, 0.870588f, 0.678431f, 1.f ); }
      inline static agxRender::Color               Violet() { return agxRender::Color( 0.933333f, 0.509804f, 0.933333f, 1.f ); }
      inline static agxRender::Color        PaleVioletRed() { return agxRender::Color( 0.858824f, 0.439216f, 0.576471f, 1.f ); }
      inline static agxRender::Color       LightGoldenrod() { return agxRender::Color( 0.933333f, 0.866667f, 0.509804f, 1.f ); }
      inline static agxRender::Color             Cornsilk() { return agxRender::Color( 1.000000f, 0.972549f, 0.862745f, 1.f ); }
      inline static agxRender::Color               Silver() { return agxRender::Color( 0.752941f, 0.752941f, 0.752941f, 1.f ); }
      inline static agxRender::Color             DarkCyan() { return agxRender::Color( 0.000000f, 0.545098f, 0.545098f, 1.f ); }
      inline static agxRender::Color           DarkOrchid() { return agxRender::Color( 0.600000f, 0.196078f, 0.800000f, 1.f ); }
      inline static agxRender::Color         AntiqueWhite() { return agxRender::Color( 0.980392f, 0.921569f, 0.843137f, 1.f ); }
      inline static agxRender::Color    MediumSpringGreen() { return agxRender::Color( 0.000000f, 0.980392f, 0.603922f, 1.f ); }
      inline static agxRender::Color            IndianRed() { return agxRender::Color( 0.803922f, 0.360784f, 0.360784f, 1.f ); }
      inline static agxRender::Color             DarkBlue() { return agxRender::Color( 0.000000f, 0.000000f, 0.545098f, 1.f ); }
      inline static agxRender::Color               Sienna() { return agxRender::Color( 0.627451f, 0.321569f, 0.176471f, 1.f ); }
      inline static agxRender::Color          ForestGreen() { return agxRender::Color( 0.133333f, 0.545098f, 0.133333f, 1.f ); }
      inline static agxRender::Color       LightSlateGray() { return agxRender::Color( 0.466667f, 0.533333f, 0.600000f, 1.f ); }
      inline static agxRender::Color       BlanchedAlmond() { return agxRender::Color( 1.000000f, 0.921569f, 0.803922f, 1.f ); }
      inline static agxRender::Color             Moccasin() { return agxRender::Color( 1.000000f, 0.894118f, 0.709804f, 1.f ); }
      inline static agxRender::Color            LimeGreen() { return agxRender::Color( 0.196078f, 0.803922f, 0.196078f, 1.f ); }
      inline static agxRender::Color              HotPink() { return agxRender::Color( 1.000000f, 0.411765f, 0.705882f, 1.f ); }
      inline static agxRender::Color            PeachPuff() { return agxRender::Color( 1.000000f, 0.854902f, 0.725490f, 1.f ); }
      inline static agxRender::Color              DimGray() { return agxRender::Color( 0.411765f, 0.411765f, 0.411765f, 1.f ); }
      inline static agxRender::Color               Purple() { return agxRender::Color( 0.501961f, 0.000000f, 0.501961f, 1.f ); }
      inline static agxRender::Color      MediumSlateBlue() { return agxRender::Color( 0.482353f, 0.407843f, 0.933333f, 1.f ); }
      inline static agxRender::Color                Brown() { return agxRender::Color( 0.647059f, 0.164706f, 0.164706f, 1.f ); }
      inline static agxRender::Color           MediumBlue() { return agxRender::Color( 0.000000f, 0.000000f, 0.803922f, 1.f ); }
      inline static agxRender::Color                  Red() { return agxRender::Color( 1.000000f, 0.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color                 Teal() { return agxRender::Color( 0.000000f, 0.501961f, 0.501961f, 1.f ); }
      inline static agxRender::Color            Burlywood() { return agxRender::Color( 0.870588f, 0.721569f, 0.529412f, 1.f ); }
      inline static agxRender::Color            MintCream() { return agxRender::Color( 0.960784f, 1.000000f, 0.980392f, 1.f ); }
      inline static agxRender::Color                Khaki() { return agxRender::Color( 0.941176f, 0.901961f, 0.549020f, 1.f ); }
      inline static agxRender::Color           DodgerBlue() { return agxRender::Color( 0.117647f, 0.564706f, 1.000000f, 1.f ); }
      inline static agxRender::Color           DarkOrange() { return agxRender::Color( 1.000000f, 0.549020f, 0.000000f, 1.f ); }
      inline static agxRender::Color                White() { return agxRender::Color( 1.000000f, 1.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color          DeepSkyBlue() { return agxRender::Color( 0.000000f, 0.749020f, 1.000000f, 1.f ); }
      inline static agxRender::Color               Orchid() { return agxRender::Color( 0.854902f, 0.439216f, 0.839216f, 1.f ); }
      inline static agxRender::Color                Ivory() { return agxRender::Color( 1.000000f, 1.000000f, 0.941176f, 1.f ); }
      inline static agxRender::Color          DarkMagenta() { return agxRender::Color( 0.545098f, 0.000000f, 0.545098f, 1.f ); }
      inline static agxRender::Color            LightCyan() { return agxRender::Color( 0.878431f, 1.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color        LightSeaGreen() { return agxRender::Color( 0.125490f, 0.698039f, 0.666667f, 1.f ); }
      inline static agxRender::Color         MediumPurple() { return agxRender::Color( 0.576471f, 0.439216f, 0.858824f, 1.f ); }
      inline static agxRender::Color       LightSteelBlue() { return agxRender::Color( 0.690196f, 0.768627f, 0.870588f, 1.f ); }
      inline static agxRender::Color                Green() { return agxRender::Color( 0.000000f, 0.501961f, 0.000000f, 1.f ); }
      inline static agxRender::Color         DarkSeaGreen() { return agxRender::Color( 0.560784f, 0.737255f, 0.560784f, 1.f ); }
      inline static agxRender::Color         MediumOrchid() { return agxRender::Color( 0.729412f, 0.333333f, 0.827451f, 1.f ); }
      inline static agxRender::Color               Bisque() { return agxRender::Color( 1.000000f, 0.894118f, 0.768627f, 1.f ); }
      inline static agxRender::Color                 Cyan() { return agxRender::Color( 0.000000f, 1.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color              Thistle() { return agxRender::Color( 0.847059f, 0.749020f, 0.847059f, 1.f ); }
      inline static agxRender::Color            LightPink() { return agxRender::Color( 1.000000f, 0.713726f, 0.756863f, 1.f ); }
      inline static agxRender::Color       CornflowerBlue() { return agxRender::Color( 0.392157f, 0.584314f, 0.929412f, 1.f ); }
      inline static agxRender::Color         LightSkyBlue() { return agxRender::Color( 0.529412f, 0.807843f, 0.980392f, 1.f ); }
      inline static agxRender::Color                Beige() { return agxRender::Color( 0.960784f, 0.960784f, 0.862745f, 1.f ); }
      inline static agxRender::Color                 Aqua() { return agxRender::Color( 0.000000f, 1.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color             DarkGray() { return agxRender::Color( 0.662745f, 0.662745f, 0.662745f, 1.f ); }
      inline static agxRender::Color           PowderBlue() { return agxRender::Color( 0.690196f, 0.878431f, 0.901961f, 1.f ); }
      inline static agxRender::Color          GreenYellow() { return agxRender::Color( 0.678431f, 1.000000f, 0.184314f, 1.f ); }
      inline static agxRender::Color                 Lime() { return agxRender::Color( 0.000000f, 1.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color      MediumVioletRed() { return agxRender::Color( 0.780392f, 0.082353f, 0.521569f, 1.f ); }
      inline static agxRender::Color               Salmon() { return agxRender::Color( 0.980392f, 0.501961f, 0.447059f, 1.f ); }
      inline static agxRender::Color            Firebrick() { return agxRender::Color( 0.698039f, 0.133333f, 0.133333f, 1.f ); }
      inline static agxRender::Color            RosyBrown() { return agxRender::Color( 0.737255f, 0.560784f, 0.560784f, 1.f ); }
      inline static agxRender::Color             Lavender() { return agxRender::Color( 0.901961f, 0.901961f, 0.980392f, 1.f ); }
      inline static agxRender::Color                 Navy() { return agxRender::Color( 0.000000f, 0.000000f, 0.501961f, 1.f ); }
      inline static agxRender::Color        DarkSlateBlue() { return agxRender::Color( 0.282353f, 0.239216f, 0.545098f, 1.f ); }
      inline static agxRender::Color     MediumAquamarine() { return agxRender::Color( 0.400000f, 0.803922f, 0.666667f, 1.f ); }
      inline static agxRender::Color           DarkViolet() { return agxRender::Color( 0.580392f, 0.000000f, 0.827451f, 1.f ); }
      inline static agxRender::Color               Yellow() { return agxRender::Color( 1.000000f, 1.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color           Chartreuse() { return agxRender::Color( 0.498039f, 1.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color                 Gold() { return agxRender::Color( 1.000000f, 0.843137f, 0.000000f, 1.f ); }
      inline static agxRender::Color           WhiteSmoke() { return agxRender::Color( 0.960784f, 0.960784f, 0.960784f, 1.f ); }
      inline static agxRender::Color              SkyBlue() { return agxRender::Color( 0.529412f, 0.807843f, 0.921569f, 1.f ); }
      inline static agxRender::Color            VioletRed() { return agxRender::Color( 0.815686f, 0.125490f, 0.564706f, 1.f ); }
      inline static agxRender::Color        LavenderBlush() { return agxRender::Color( 1.000000f, 0.941176f, 0.960784f, 1.f ); }
      inline static agxRender::Color            Turquoise() { return agxRender::Color( 0.250980f, 0.878431f, 0.815686f, 1.f ); }
      inline static agxRender::Color              Magenta() { return agxRender::Color( 1.000000f, 0.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color                 Blue() { return agxRender::Color( 0.000000f, 0.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color               Tomato() { return agxRender::Color( 1.000000f, 0.388235f, 0.278431f, 1.f ); }
      inline static agxRender::Color                  Tan() { return agxRender::Color( 0.823529f, 0.705882f, 0.549020f, 1.f ); }
      inline static agxRender::Color             SeaGreen() { return agxRender::Color( 0.180392f, 0.545098f, 0.341176f, 1.f ); }
      inline static agxRender::Color           SandyBrown() { return agxRender::Color( 0.956863f, 0.643137f, 0.376471f, 1.f ); }
      inline static agxRender::Color                 Pink() { return agxRender::Color( 1.000000f, 0.752941f, 0.796078f, 1.f ); }
      inline static agxRender::Color            Goldenrod() { return agxRender::Color( 0.854902f, 0.647059f, 0.125490f, 1.f ); }
      inline static agxRender::Color      MediumTurquoise() { return agxRender::Color( 0.282353f, 0.819608f, 0.800000f, 1.f ); }
      inline static agxRender::Color           DarkSalmon() { return agxRender::Color( 0.913725f, 0.588235f, 0.478431f, 1.f ); }
      inline static agxRender::Color            Gainsboro() { return agxRender::Color( 0.862745f, 0.862745f, 0.862745f, 1.f ); }
      inline static agxRender::Color                 Peru() { return agxRender::Color( 0.803922f, 0.521569f, 0.247059f, 1.f ); }
      inline static agxRender::Color            SlateGray() { return agxRender::Color( 0.439216f, 0.501961f, 0.564706f, 1.f ); }
      inline static agxRender::Color                Olive() { return agxRender::Color( 0.501961f, 0.501961f, 0.000000f, 1.f ); }
      inline static agxRender::Color            LightBlue() { return agxRender::Color( 0.678431f, 0.847059f, 0.901961f, 1.f ); }
      inline static agxRender::Color             Seashell() { return agxRender::Color( 1.000000f, 0.960784f, 0.933333f, 1.f ); }
      inline static agxRender::Color            SteelBlue() { return agxRender::Color( 0.274510f, 0.509804f, 0.705882f, 1.f ); }
      inline static agxRender::Color             DeepPink() { return agxRender::Color( 1.000000f, 0.078431f, 0.576471f, 1.f ); }
      inline static agxRender::Color          SaddleBrown() { return agxRender::Color( 0.545098f, 0.270588f, 0.074510f, 1.f ); }
      inline static agxRender::Color            RoyalBlue() { return agxRender::Color( 0.254902f, 0.411765f, 0.882353f, 1.f ); }
      inline static agxRender::Color        PaleTurquoise() { return agxRender::Color( 0.686275f, 0.933333f, 0.933333f, 1.f ); }
      inline static agxRender::Color       DarkOliveGreen() { return agxRender::Color( 0.333333f, 0.419608f, 0.184314f, 1.f ); }
      inline static agxRender::Color           LightGreen() { return agxRender::Color( 0.564706f, 0.933333f, 0.564706f, 1.f ); }
      inline static agxRender::Color                 Gray() { return agxRender::Color( 0.501961f, 0.501961f, 0.501961f, 1.f ); }
      inline static agxRender::Color              DarkRed() { return agxRender::Color( 0.545098f, 0.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color        DarkTurquoise() { return agxRender::Color( 0.000000f, 0.807843f, 0.819608f, 1.f ); }
      inline static agxRender::Color                 Plum() { return agxRender::Color( 0.866667f, 0.627451f, 0.866667f, 1.f ); }
      inline static agxRender::Color       MediumSeaGreen() { return agxRender::Color( 0.235294f, 0.701961f, 0.443137f, 1.f ); }
      inline static agxRender::Color           GhostWhite() { return agxRender::Color( 1.000000f, 0.980392f, 0.980392f, 1.f ); }
      inline static agxRender::Color            DarkGreen() { return agxRender::Color( 0.000000f, 0.392157f, 0.000000f, 1.f ); }
      inline static agxRender::Color                 Snow() { return agxRender::Color( 1.000000f, 0.980392f, 0.980392f, 1.f ); }
      inline static agxRender::Color            LawnGreen() { return agxRender::Color( 0.486275f, 0.988235f, 0.000000f, 1.f ); }
      inline static agxRender::Color          LightYellow() { return agxRender::Color( 1.000000f, 1.000000f, 0.878431f, 1.f ); }
      inline static agxRender::Color                Azure() { return agxRender::Color( 0.941176f, 1.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color         MidnightBlue() { return agxRender::Color( 0.098039f, 0.098039f, 0.439216f, 1.f ); }
      inline static agxRender::Color                Black() { return agxRender::Color( 0.000000f, 0.000000f, 0.000000f, 1.f ); }
      inline static agxRender::Color            MistyRose() { return agxRender::Color( 1.000000f, 0.894118f, 0.882353f, 1.f ); }
      inline static agxRender::Color            LightGray() { return agxRender::Color( 0.827451f, 0.827451f, 0.827451f, 1.f ); }
      inline static agxRender::Color        PaleGoldenrod() { return agxRender::Color( 0.933333f, 0.909804f, 0.666667f, 1.f ); }
      inline static agxRender::Color           LightCoral() { return agxRender::Color( 0.941176f, 0.501961f, 0.501961f, 1.f ); }
      inline static agxRender::Color               Orange() { return agxRender::Color( 1.000000f, 0.647059f, 0.000000f, 1.f ); }
      inline static agxRender::Color             Honeydew() { return agxRender::Color( 0.941176f, 1.000000f, 0.941176f, 1.f ); }
      inline static agxRender::Color           BlueViolet() { return agxRender::Color( 0.541176f, 0.168627f, 0.886275f, 1.f ); }
      inline static agxRender::Color LightGoldenrodYellow() { return agxRender::Color( 0.980392f, 0.980392f, 0.823529f, 1.f ); }
      inline static agxRender::Color            Chocolate() { return agxRender::Color( 0.823529f, 0.411765f, 0.117647f, 1.f ); }
      inline static agxRender::Color              Fuschia() { return agxRender::Color( 1.000000f, 0.000000f, 1.000000f, 1.f ); }
      inline static agxRender::Color                Coral() { return agxRender::Color( 1.000000f, 0.498039f, 0.313726f, 1.f ); }
  };

  typedef agx::Vector<Color> ColorVector;
}

#endif
