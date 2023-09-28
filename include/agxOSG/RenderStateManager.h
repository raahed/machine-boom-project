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
#ifndef AGXOSG_RenderStateManager_H
#define AGXOSG_RenderStateManager_H


#include <agxOSG/export.h>
#include <agx/Singleton.h>
#include <agxCFG/ConfigScript.h>
#include <agx/HashTable.h>
#include <agx/Vec4.h>
#include <agx/Logger.h>

#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!
#include <osg/Program>
#include <osg/Shader>
#include <osg/Uniform>
#include <osg/StateAttribute>
#include <osg/Texture2D>
#include <osg/TexMat>
#include <osg/TexGen>
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

namespace osg
{
  class Group;
}

namespace agxOSG {
  class Node;
  class Group;
  class GeometryNode;
  class SimulationObject;



  class AGXOSG_EXPORT RenderStateManager : public agx::Singleton
  {
  public:

    enum Values
    {
      /** means that associated GLMode and Override is disabled.*/
      OFF          = osg::StateAttribute::OFF,
      /** means that associated GLMode is enabled and Override is disabled.*/
      ON           = osg::StateAttribute::ON,
      /** Overriding of GLMode's or StateAttributes is enabled, so that state below it is overridden.*/
      OVERRIDE     = osg::StateAttribute::OVERRIDE,
      /** Protecting of GLMode's or StateAttributes is enabled, so that state from above cannot override this and below state.*/
      PROTECTED    = osg::StateAttribute::PROTECTED,
      /** means that GLMode or StateAttribute should be inherited from above.*/
      INHERIT      = osg::StateAttribute::INHERIT
    };

    /**
    Will call clear and then init to reinitialize back to default state.
    */
    void reset();

    class AGXOSG_EXPORT ShaderState : public agx::Referenced
    {
    public:
      ShaderState(const std::string& name, const std::string& program_name) : m_name(name), m_program_name(program_name) {}

      std::string getName() const { return m_name; }

      std::string getProgramID() const { return m_program_name; }

      void addUniform(osg::Uniform* uniform);
      osg::Uniform* findUniform(const std::string& name);

      /**
      Add a ShaderState to a specified stateset with given StateAttributes
      \param stateset - A pointer to the osg::StateSet where the ShaderState will be added.
      \param value - The StateAttribute used for the ShaderState, Default is ON.
      */
      void apply(osg::StateSet* stateset, RenderStateManager::Values value=RenderStateManager::ON);
      void apply(osg::Node* node, RenderStateManager::Values value=RenderStateManager::ON);

    private:
      virtual ~ShaderState();
      std::string m_name, m_program_name;

      typedef agx::HashTable<std::string, osg::ref_ptr<osg::Uniform> > UniformHashTable;
      UniformHashTable m_uniforms;
    };

    class AGXOSG_EXPORT RenderState : public agx::Referenced
    {
    public:

      enum Color {
        AMBIENT,
        DIFFUSE,
        SPECULAR,
        ALPHA,
        SHININESS,
        EMISSIVE
      };

      RenderState( const std::string& name ) : m_name( name ) {}
      void setName( const std::string &name ) { m_name = name; }
      std::string getName() const { return m_name; }

      void addTexture(int unit, osg::Texture* texture, osg::TexMat* tm=nullptr, osg::TexGen* texgen=nullptr )
      {
        m_textureHash[unit]= Texture(texture, tm, texgen);
      }

      void apply( osg::Group* node, RenderStateManager::Values value=RenderStateManager::ON );
      void apply( osg::Node* node, RenderStateManager::Values value=RenderStateManager::ON );
      void apply( agxOSG::GeometryNode* node, RenderStateManager::Values value=RenderStateManager::ON );
      void apply( osg::StateSet* stateSet, RenderStateManager::Values value=RenderStateManager::ON );
      void apply( agxOSG::SimulationObject& obj, RenderStateManager::Values value=RenderStateManager::ON );

      void addUniform(osg::Uniform* uniform);
      osg::Uniform *findUniform(const std::string& name);


      /// DIFFUSE, AMBIENT, SPECULAR, EMISSIVE, SHININESS, ALPHA
      void setColor( Color color, const agx::Vec4f& col  ) { m_colorHash[color] = osg::Vec4f(float(col[0]), float(col[1]), float(col[2]), float(col[3]) ); }
      agx::Vec4f getColor( Color color ) { return OSG_VEC4F_TO_AGX(m_colorHash[color]); }

      void setShaderState( ShaderState *shaderState ) { m_shaderState = shaderState; }
    protected:
      virtual ~RenderState() {}

      std::string m_name;
      struct Texture {
        Texture(osg::Texture *t, osg::TexMat *tm, osg::TexGen *tg ) : texture(t), texMat(tm), texGen(tg) {}
        Texture( ) {}
        osg::ref_ptr<osg::Texture> texture;
        osg::ref_ptr<osg::TexMat> texMat;
        osg::ref_ptr<osg::TexGen> texGen;
      };
      typedef agx::HashTable<int, Texture  > TextureHash;
      TextureHash m_textureHash;
      agx::ref_ptr<ShaderState> m_shaderState;
      typedef agx::HashTable<agx::UInt, osg::Vec4f > ColorHash;
      ColorHash m_colorHash;

      typedef agx::HashTable<std::string, osg::ref_ptr<osg::Uniform> > UniformHashTable;
      UniformHashTable m_uniforms;

    };

    typedef agx::ref_ptr<RenderState> RenderStateRef;


    /// Return the singleton object
    static RenderStateManager *instance( void );

    SINGLETON_CLASSNAME_METHOD();

    osg::Shader *findShader(const std::string& name, osg::Shader::Type type) ;
    osg::Program *findProgram(const std::string& name) ;

    RenderState *findRenderState(const std::string& name);
    void addRenderState( RenderState *renderState );

    void addTexture(const std::string& name, osg::Texture *texture);
    osg::Texture *findTexture(const std::string& name);

    osg::Program *createProgram(const std::string& name);

    osg::Shader *createShader(const std::string& name,
      const std::string& shader,
      osg::Shader::Type type, bool isAFile );

    //bool loadShaderSource(const std::string& shader_id);

    void addShaderState(ShaderState* shaderState);
    ShaderState *findShaderState(const std::string& name);

    void parse( agxCFG::ConfigScript* cfg);

    /**
    Remove all previous configurations, Shader/RenderStates. Begin fresh.
    */
    void clear();

    template<typename T>
    static void parseUniform(T* obj, agxCFG::ConfigScript *cfg)
    {
      std::string id;
      int i_val;
      float f_val;
      agx::RealVector v_val;

      if (!cfg->get("name", id))
        LOGGER_ERROR() << "Required key " << cfg->currentScope() << "name is missing" << LOGGER_END();

      if (cfg->exist("value",  agxCFG::ConfigValue::VALUE_FLOAT) ||
        cfg->exist("value",  agxCFG::ConfigValue::VALUE_EXPRESSION)) {
          f_val = cfg->returns("value", 0.0f);
          obj->addUniform(new osg::Uniform(id.c_str(), f_val));
      }
      else if (cfg->exist("value",  agxCFG::ConfigValue::VALUE_INT)) {
        i_val = cfg->returns("value", 0);
        obj->addUniform(new osg::Uniform(id.c_str(), i_val));
      }
      else if (cfg->exist("value",  agxCFG::ConfigValue::VALUE_FLOAT_ARRAY)) {
        cfg->get("value", v_val);
        if (!v_val.size())
          LOGGER_ERROR() << "Invalid vector size for data in Uniform: ("+id+")" << LOGGER_END();
        if (v_val.size()==2) {
          osg::Vec2 v2((float)v_val[0], (float)v_val[1]);
          obj->addUniform(new osg::Uniform(id.c_str(), v2));
        }
        else if (v_val.size()==3) {
          osg::Vec3 v3((float)v_val[0], (float)v_val[1], (float)v_val[2]);
          obj->addUniform(new osg::Uniform(id.c_str(),  v3));
        }
        else if (v_val.size()==4) {
          osg::Vec4f v4((float)v_val[0], (float)v_val[1], (float)v_val[2], (float)v_val[3]);
          obj->addUniform(new osg::Uniform(id.c_str(),  v4));
        }
        else if (v_val.size()==16) {
          osg::Matrixf m((float)v_val[0], (float)v_val[1], (float)v_val[2], (float)v_val[3],
            (float)v_val[4], (float)v_val[5], (float)v_val[6], (float)v_val[7],
            (float)v_val[8], (float)v_val[9], (float)v_val[10], (float)v_val[11],
            (float)v_val[12], (float)v_val[13], (float)v_val[14], (float)v_val[15]
          );
          obj->addUniform(new osg::Uniform(id.c_str(), m));
        }
        else
          LOGGER_ERROR() << "Invalid vector size for data in Uniform: (" << id << ")" << LOGGER_END();
      }
      else
        LOGGER_ERROR() << "Uniform.value is either missing or is of an invalid type in " << cfg->currentScope() << "(" << id << ")" << LOGGER_END();
    }


  private:

    void shutdown() override;

    void init();

    /// Destructor
    virtual ~RenderStateManager();

    /// Constructor
    RenderStateManager( void );

    typedef agx::HashTable<std::string, osg::ref_ptr<osg::Shader> > ShaderMap;
    ShaderMap m_vertexShaders;
    ShaderMap m_fragmentShaders;


    typedef agx::HashTable<std::string, osg::ref_ptr<osg::Program> > ProgramMap;
    ProgramMap m_programs;

    typedef agx::HashTable<std::string, agx::ref_ptr<ShaderState> > ShaderStateMap;
    ShaderStateMap m_shader_states;

    typedef agx::HashTable<std::string, agx::ref_ptr<RenderState> > RenderStateMap;
    RenderStateMap m_render_states;

    typedef agx::HashTable<std::string, osg::ref_ptr<osg::Texture> > TextureHashTable;
    TextureHashTable m_textures;

  private:
    static RenderStateManager *s_instance;
    bool m_initialized;

  };


} // namespace agxOSG


#endif
