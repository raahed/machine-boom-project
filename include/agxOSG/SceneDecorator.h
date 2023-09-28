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

#include <agxOSG/export.h>


#include <agx/Vec3.h>
#include <agx/Vec4.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Referenced>
#include <osg/Group>
#include <osg/Texture2D>
#include <osg/MatrixTransform>
#include <osg/Projection>
#include <osg/StateSet>
#include <osg/PositionAttitudeTransform>
#include <osg/Switch>
#include <osg/Material>
#include <osgShadow/ShadowedScene>
#include <osgShadow/ShadowTechnique>
#include <osg/LightSource>
#include <osg/Geometry>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.
#include <agxOSG/LightSource.h>


namespace agxRender
{
  class Color;
}

namespace osgText
{
  class Text;
}

/// Namespace holding utility classes and functions for connecting AGX to the OpenSceneGraph rendering API.
namespace agxOSG
{
  /// Decorates a scene with a specific AGX Rendering Style.

  /**
  SceneDecorator is a class that inherits from osg::Group and will decorate the sub-graph with a specific
  rendering style and a logo.
  */
  class AGXOSG_EXPORT SceneDecorator : public osg::Group
  {
  public:

    /**
    Constructor.
    \param windowSizeX - The size of the window in x coordinates.
    \param windowSizeY - The size of the window in y coordinates.
    \param enabledLights - Specifies (using the \p Lights enum) which light sources should be created and used
    */
    SceneDecorator( int windowSizeX, int windowSizeY, int enabledLights = LIGHT0 | LIGHT1 );

    /// Enable/disable shadows for the sub-graph
    void setEnableShadows( bool flag );

    /// \return true if shadowing is enabled for sub-graph
    bool getEnableShadows() const;

    /// Enable/disable AGX rendering style for sub-graph
    void setEnable( bool flag );

    /// Return true if AGX rendering style is enabled
    bool getEnable() const;

    /// Return true if AGX rendering style is enabled
    bool isEnabled() const;

    /// Specifies the location of the logo
    enum LogoLocation {
      UPPER_LEFT = 0x1,
      LOWER_LEFT = 0x2,
      UPPER_RIGHT = 0x3,
      LOWER_RIGHT = 0x4,
      FREE               /**< Any position can be specified with the setLogoPosition() method */
    };

    /**
    Set the position of the logo from a list of predefined positions.
    \param l - An enum specifying the position. FREE to make it possible to set any position with setLogoPosition()
    */
    void setLogoLocation( LogoLocation l );

    /// \return the specified logo position
    LogoLocation getLogoLocation() const;

    /**
    Set the position of logo into [\p x, \p y] if setLogoLocation(FREE) has been previously called
    Valid range is [0,1] where 0,0 is lower left corner
    */
    void setLogoPosition(float x, float y);

    /// Enable/disable the rendering of the logo
    void setEnableLogo( bool flag );

    /// \return true if rendering of logo is enabled
    bool getEnableLogo() const;

    /// Specify a path to the image with the logo and the pixel density (i.e. 2.0 for high DPI and 1.0 for 1:1)
    bool setLogoFile(const std::string& filename, float density = 1.0);

    /// Print a row of text
    void setText( int row, const agx::String& text );
    void setText( int row, const agx::String& text, const agx::Vec4f& color );

    /**
    Enable/disable the use of a shader based default render state
    */
    void setEnableShaderState( bool flag, bool lock=false );

    /// \return true if shader based default render state is being activated
    bool getEnableShaderState( ) const;

    /// Reset all text to empty strings
    void clearText();
    /**
    Specif the path to the base texture on all decorated objects
    \return true if loading and creation of texture was successful.
    */
    bool setBaseTexture(const std::string& imagePath);

    /**
    \return the current logo dimension
    */
    const osg::Vec2& getLogoDimension();

    /**
    \return the maximum logo dimension
    */
    osg::Vec2 getMaximumLogoDimension();

    /**
    Set the maximum logo dimension.
    Values between 0.0f and 1.0f.
    */
    void setMaximumLogoDimension(float x, float y);

    /**
    \return the default logo position
    */
    static osg::Vec2 getDefaultLogoPosition();

    /**
    Return a pointer to the hud group. Any node added here
    will be rendered in the same coordinate system as the logo.
    Good for example console text.
    Calls to resetHUD will invalidate earlier returned HUDs.
    */
    osg::MatrixTransform* getHud() { return m_hudRoot.get(); }

    // Overriding add/remove child methods
    virtual bool insertChild( unsigned int index, osg::Node *child ) { return m_currentRoot->insertChild(index, child); }
    virtual bool addChild(osg::Node *child) { return m_currentRoot->addChild(child); }
    inline bool removeChild( osg::Node *child ) { return m_currentRoot->removeChild(child); }
    inline bool removeChild( unsigned int pos, unsigned int numChildrenToRemove=1 ) { return m_currentRoot->removeChild(pos, numChildrenToRemove); }
    virtual bool replaceChild( osg::Node *origChild, osg::Node* newChild ) { return m_currentRoot->replaceChild(origChild, newChild); }
    virtual bool setChild( unsigned  int i, osg::Node* node ) { return m_currentRoot->setChild(i, node); }
    virtual bool removeChildren(unsigned int pos,unsigned int numChildrenToRemove) { return m_currentRoot->removeChildren(pos, numChildrenToRemove); }

    /**
    Calculate the positions of the light sources from the bound of
    either a specified node, or if bound_node == 0, the sub-graph of this Group.
    */
    void calculateLightPositions(osg::Node* bound_node = nullptr);

    void calculateLightPositions( const osg::BoundingBox& boundingBox );

    void calculateLightPositions( const osg::BoundingSphere& boundingSphere );

    /// Returns the font name.
    std::string getFontName() const;

    /// Sets the font name.
    void setFontName(const std::string& fontName);

    /// Updates information about the window size in x and y (for logo scaling).
    void updateWindowSize(int sizeX, int sizeY);

    /// Specify the screen pixel scale (i.e. 2.0 for high DPI and 1.0 for 1:1)
    void setScreenPixelScale(float scale);

    /// Returns the font Size.
    float getFontSize() const;

    /// Sets the font Size.
    void setFontSize(const float& fontSize);

    /// Return a pointer to the light source used for calculating shadows
    osg::LightSource* getShadowLightSource();

    enum Lights {
      LIGHT0 = 0x1,
      LIGHT1 = 0x2,
      LIGHT2 = 0x4,
      ALL = LIGHT0 | LIGHT1 | LIGHT2
    };


    static size_t lightToIndex(Lights l);

    void setEnableCalculateLightPositions( Lights l, bool f );

    bool getEnableCalculateLightPositions(Lights l) const;

    /// Specify which light source is used for calculating shadows.
    bool setShadowLightSource( Lights l);

    osg::StateSet* getDecoratorStateSet() { return m_decoratorStateSet; }

    osg::LightSource* getOSGLight(Lights l);

    /// Get a wrapper for a LightSource with only AGX types
    agxOSG::LightSource getLightSource(Lights l);

    enum ShadowMethod {
      SOFT_SHADOWMAP,
      SHADOWMAP,
      SHADOWVOLUME, /// Deprecated
      PARALLELLSPLIT_SHADOWMAP,
      LIGHTSPACEPERSPECTIVE_SHADOWMAP,
      SHADOWTEXTURE,
      NO_SHADOWS
    };

    /// Specify which method is used for generating shadows,
    void setShadowMethod(ShadowMethod m);

    /// \returns the current shadow method
    ShadowMethod getShadowMethod() const { return m_shadowMethod; }

    /// Set a uniform background color using \p color
    void setBackgroundColor(const agx::Vec4f& color);

     /**
    Set a gradient two color background.
    \p upper - Top color of the gradient color
    \p lower - Lower color of the gradient color
    */
    void setBackgroundColor(const agx::Vec4f& color1, const agx::Vec4f& color2);

    /**
    Set a gradient four color background (each corner)
    \p colorUL - Upper left color
    \p colorUR - Upper right color
    \p colorLL - Lower left color
    \p colorLR - Lower right color
    */
    void setBackgroundColor(const agx::Vec4f& colorUL, const agx::Vec4f& colorUR, const agx::Vec4f& colorLL, const agx::Vec4f& colorLR);


    /// Set a uniform background color using \p color
    void setBackgroundColor(agx::Vec3 color);

    /**
    Set a gradient two color background.
    \p upper - Top color of the gradient color
    \p lower - Lower color of the gradient color
    */
    void setBackgroundColor(agx::Vec3 upper, agx::Vec3 lower);

    /**
    Set a gradient four color background (each corner)
    \p colorUL - Upper left color
    \p colorUR - Upper right color
    \p colorLL - Lower left color
    \p colorLR - Lower right color
    */
    void setBackgroundColor(agx::Vec3 colorUL, agx::Vec3 colorUR, agx::Vec3 colorLL, agx::Vec3 colorLR);

    /**
    Reset and enable the specified lights
    \param lightMask - Bit mask from the Light enum
    */
    void setEnableLights(int lightMask = ALL);

    /**
    Enable/disable the mouse cursor
    \param enable - If true the cursor will be visible
    */
    void setEnableRenderedCursor( bool enable );
    bool getEnableRenderedCursor() const;
    void updateCursorPosition( float x, float y );

    osg::Texture2D* getLogo();

    /// Might want to call updateLogo afterwards.
    void setLogo(osg::Texture2D* logo, float density = 1.0);

    // Will reset the rendering HUD.
    void resetHUD();

    /**
    \return the scene casting and receiving shadows
    */
    osg::Group* getShadowedScene() const;

    /**
    \return the scene not casting nor receiving shadows
    */
    osg::Group* getNonShadowedScene() const;

    static const osg::Material* getDefaultDecoratorMaterial();

  protected:
    /**
    Updates the logo scale and position, depending on logo size, and window width and height.
    */
    void updateLogo();
    void createHUD();

    void createLogo();
    void createBackground();

    void init(Lights l);

    osg::ref_ptr<osg::Group> m_enabledRoot;
    osg::ref_ptr<osg::Group> m_enabledNonShadowRoot;
    osg::ref_ptr<osg::Group> m_currentRoot;
    osg::ref_ptr<osg::Group> m_disabledRoot;
    osg::ref_ptr<osg::Group> m_disabledNonShadowRoot;
    osg::ref_ptr<osg::Switch> m_backgroundRoot;
    osg::observer_ptr<osg::Geometry> m_backgroundQuad;
    LogoLocation m_logoLocation;
    osg::ref_ptr<osg::Texture2D> m_logoTexture;
    osg::ref_ptr<osg::PositionAttitudeTransform> m_logoTransform;
    osg::ref_ptr<osg::Switch> m_logoSwitch;
    osg::ref_ptr<osg::MatrixTransform> m_hudRoot;
    osg::ref_ptr<osg::StateSet> m_logoStateSet;
    osg::Vec2 m_logoPosition;
    bool m_enabledShadows;
    bool m_enabled;
    bool m_logoEnabled;
    ShadowMethod m_shadowMethod;
    std::string m_baseTexturePath;
    agx::Vector<osgText::Text *> m_text;
    std::string m_fontName;
    float m_fontSize;
    int m_windowSizeX;
    int m_windowSizeY;
    float m_maximumLogoDimensionX;
    float m_maximumLogoDimensionY;
    osg::Vec2 m_logoSize;
    float m_logoPixelDensity;
    float m_screenPixelScale;

    osg::ref_ptr<osgShadow::ShadowedScene> m_shadowedScene;
    osg::ref_ptr<osgShadow::ShadowTechnique> m_shadowTechnique;

    osg::ref_ptr<osg::StateSet> m_decoratorStateSet;

    agx::Vector<osg::ref_ptr<osg::LightSource> > m_lightSources;
    osg::ref_ptr<osg::LightSource> m_ShadowlightSource;
    osg::ref_ptr<osg::Vec3Array> m_backgroundColors;

    void createDecoratorState();

    virtual ~SceneDecorator();

    bool m_calculateLightPosition[3];

    osg::ref_ptr<osg::Node> m_cursorTransform;
    osg::ref_ptr<osg::Camera> m_hudCamera;
    osg::ref_ptr<osg::Projection> m_hudProjection;

    bool m_useShaderState;
    bool m_shaderSupportAvailable;

    static osg::ref_ptr<osg::Material> s_decoratorMaterial;
  };
}
