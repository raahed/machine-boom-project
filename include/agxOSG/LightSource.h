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

#ifndef AGXOSG_LIGHTSOURCE_H
#define AGXOSG_LIGHTSOURCE_H

#include <agx/config.h>
#include <agx/Vec4.h>
#include <agxOSG/export.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Light>
#include <osg/LightSource>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.


namespace agxOSG
{
  class AGXOSG_EXPORT LightSource
  {
  public:

    LightSource( osg::LightSource *lightSource )
    {
      if (lightSource)
        m_light = lightSource->getLight();
    }

    /** Get which OpenGL light this osg::Light operates on. */
    int getLightNum() const { if (m_light) return m_light->getLightNum(); else return -1; }

    /** Set the ambient component of the light. */
    inline void setAmbient( const agx::Vec4f& ambient ) { if (m_light) m_light->setAmbient( AGX_VEC4_TO_OSG(ambient)); }

    /** Get the ambient component of the light. */
    inline agx::Vec4f getAmbient() const { if (m_light) return OSG_VEC4F_TO_AGX(m_light->getAmbient()); else return agx::Vec4f(); }

    /** Set the diffuse component of the light. */
    inline void setDiffuse( const agx::Vec4f& diffuse ) { if (m_light) m_light->setDiffuse( AGX_VEC4_TO_OSG(diffuse)); }

    /** Get the diffuse component of the light. */
    inline agx::Vec4f getDiffuse() const { if (m_light) return OSG_VEC4F_TO_AGX(m_light->getDiffuse()); else return agx::Vec4f(); }

    /** Set the specular component of the light. */
    inline void setSpecular( const agx::Vec4f& specular ) { if (m_light) m_light->setSpecular( AGX_VEC4_TO_OSG(specular)); }

    /** Get the specular component of the light. */
    inline agx::Vec4f getSpecular() const { if (m_light) return OSG_VEC4F_TO_AGX(m_light->getSpecular()); else return agx::Vec4f(); }

    /** Set the position of the light. */
    inline void setPosition( const agx::Vec4& position ) { if (m_light) m_light->setPosition( AGX_VEC4_TO_OSG(position)); }

    /** Get the position of the light. */
    inline agx::Vec4 getPosition() const { if (m_light) return OSG_VEC4_TO_AGX(m_light->getPosition()); else return agx::Vec4(); }

    /** Set the direction of the light. */
    inline void setDirection( const agx::Vec3& direction ) { if (m_light) m_light->setDirection(AGX_VEC3_TO_OSG(direction)); }

    /** Get the direction of the light. */
    inline agx::Vec3 getDirection() const { if (m_light) return OSG_VEC3_TO_AGX(m_light->getDirection()); else return agx::Vec3(); }

    /** Set the constant attenuation of the light. */
    inline void setConstantAttenuation( float constant_attenuation ) { if (m_light) m_light->setConstantAttenuation(constant_attenuation);}

    /** Get the constant attenuation of the light. */
    inline float getConstantAttenuation() const { if (m_light) return m_light->getConstantAttenuation(); else return 0; }

    /** Set the linear attenuation of the light. */
    inline void setLinearAttenuation ( float linear_attenuation ) { if (m_light) m_light->setLinearAttenuation( linear_attenuation ); }

    /** Get the linear attenuation of the light. */
    inline float getLinearAttenuation () const { if (m_light) return m_light->getLinearAttenuation(); else return 0; }

    /** Set the quadratic attenuation of the light. */
    inline void setQuadraticAttenuation ( float quadratic_attenuation )  { if (m_light) m_light->setQuadraticAttenuation(quadratic_attenuation); }

    /** Get the quadratic attenuation of the light. */
    inline float getQuadraticAttenuation()  const { if (m_light) return m_light->getQuadraticAttenuation(); else return 0;}

    /** Set the spot exponent of the light. */
    inline void setSpotExponent( float spot_exponent ) { if (m_light) m_light->setSpotExponent(spot_exponent); }

    /** Get the spot exponent of the light. */
    inline float getSpotExponent() const { if (m_light) return m_light->getSpotExponent(); else return 0; }

    /** Set the spot cutoff of the light. */
    inline void setSpotCutoff( float spot_cutoff ) { if (m_light) m_light->setSpotCutoff(spot_cutoff); }

    /** Get the spot cutoff of the light. */
    inline float getSpotCutoff() const { if (m_light) return m_light->getSpotCutoff(); return 0; }

    ~LightSource() {}

  protected:
    osg::ref_ptr<osg::Light> m_light;

  };
}

#endif
