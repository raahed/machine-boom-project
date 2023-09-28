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

#ifndef AGXGL_OPENGL_H
#define AGXGL_OPENGL_H


#include <agx/config/AGX_USE_OPENGL.h>
#include <agx/config/AGX_USE_EGL.h>
#include <agx/config/AGX_USE_OSG.h>
#include <agx/agx.h>
#include <agx/debug.h>
#include <agx/agxCore_export.h>
#include <agx/Device.h>
#include <agx/Task.h>
#include <agx/Kernel.h>
#include <agx/Vec4.h>


#if AGX_USE_OPENGL()
#include <agx/PushDisableWarnings.h> // Disabling warnings. Include agx/PopDisableWarnings.h below!

#if defined( __APPLE__ )
  #define GL_SILENCE_DEPRECATION 1

  #ifdef AGX_APPLE_IOS
    #include <OpenGLES/ES1/gl.h>
    #include <OpenGLES/ES1/glext.h>
    #include <OpenGLES/ES2/gl.h>
    #include <OpenGLES/ES2/glext.h>
    #include <CoreGraphics/CGContext.h>
  #else
    #include <OpenGL/OpenGL.h>
    // #include <OpenGL/gl3.h>
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
  #endif

#elif defined( _WIN32 )
  #define GL_GLEXT_PROTOTYPES
  #define AGX_CUSTOM_GLAD_API_CALL_EXPORT
  #include <agx/Windows.h>
  #include <agxGL/gl.h>
  #include <GL/glu.h>
  //#include <GL/glext.h>
  //#include <GL/wglext.h>


#else
  #define GL_GLEXT_PROTOTYPES

  #if AGX_USE_EGL()
    #include <EGL/egl.h>
  #endif

  #include <agxGL/gl.h>
  #include <GL/glx.h>
  #undef Status
  #undef Bool
  #undef Convex
  #include <GL/glu.h>
#endif
#include <agx/PopDisableWarnings.h> // End of disabled warnings.

// #if AGX_USE_OSG()
// #include <osg/GraphicsContext>
// #endif


  #define GL_REAL GL_DOUBLE
  #define glVertex3r glVertex3d
  #define glVertex3rv glVertex3dv
  #define glColor3r glColor3d
  #define glColor3rv glColor3dv
  #define glColor4r glColor4d
  #define glColor4rv glColor4dv
  #define glNormal3r glNormal3d
  #define glNormal3rv glNormal3dv
  #define glMultMatrixr glMultMatrixd
  #define glTranslater glTranslated
  #define glRotater glRotated


#define glCheckError() glCheckError_impl(__FILE__,__LINE__)
#define glCheckErrorTitle(title_) glCheckError_impl(__FILE__,__LINE__, (title_))

#ifdef AGX_APPLE_IOS
#define gluErrorString(x) agx::String::format("%d", x).c_str()
#endif

extern "C"
{
  void AGXCORE_EXPORT glCheckError_impl(const char *file, int line, const char* title=nullptr);
}

namespace agx
{
  class TiXmlElement;
}


namespace agxGL {

  AGX_DECLARE_POINTER_TYPES(Device);
  AGX_DECLARE_POINTER_TYPES(Buffer);
  AGX_DECLARE_POINTER_TYPES(GeometryShader);
  AGX_DECLARE_POINTER_TYPES(VertexShader);
  AGX_DECLARE_POINTER_TYPES(FragmentShader);
  AGX_DECLARE_POINTER_TYPES(State);
  AGX_DECLARE_POINTER_TYPES(Kernel);

  #ifdef __APPLE__
    #ifdef AGX_APPLE_IOS
    typedef CGContextRef ContextHandle;
    #else
    typedef CGLContextObj ContextHandle;
    #endif
  #elif defined( UNIX ) || defined( __linux__ )
    typedef GLXContext  ContextHandle;
  #elif defined _WIN32
    typedef HGLRC ContextHandle;
  #else
  #error "OpenGL context handling on other platforms"
  #endif

  void AGXCORE_EXPORT init();
  ContextHandle AGXCORE_EXPORT getCurrentContext();
  void AGXCORE_EXPORT setCurrentContext(ContextHandle context);

  extern AGXCORE_EXPORT agx::Event initEvent;



  class AGXCORE_EXPORT SyncTag : public agx::SyncTag
  {
  public:
    virtual Status wait() override { return SYNC_TAG_COMPLETED; }
    virtual Status getStatus() override { return SYNC_TAG_COMPLETED; }
  };

  class AGXCORE_EXPORT Device : public agx::Device
  {
  public:
    static Device *instance();


    using agx::Device::createBuffer;
    virtual agxData::Buffer* createBuffer(agxData::Format* format) override;

    virtual agx::SyncTag* sendData(
      const void* hostBuffer, agxData::Buffer* deviceBuffer,
      size_t offset, size_t size, bool blocking) override;

    virtual agx::SyncTag* receiveData(
      void* hostBuffer, agxData::Buffer* deviceBuffer,
      size_t offset, size_t size, bool blocking) override;

    // #if AGX_USE_OSG()
    // /**
    // Allow device to keep reference to context to prevent agxGL::Buffer from being deallocated
    // _after_ the OpenGL context is removed.
    // */
    // void setOsgContext(osg::GraphicsContext *context);
    // #endif

    static void clearGLDevice();
  protected:
    virtual ~Device();

  private:
    Device(const agx::String name = "OpenGlDevice");

    // #if AGX_USE_OSG()
    // osg::ref_ptr<osg::GraphicsContext> m_osgContext;
    // #endif
  };



  class AGXCORE_EXPORT Buffer : public agxData::Buffer
  {
  public:
    Buffer(agxData::Format *format, Device *device, GLenum usage = GL_STREAM_DRAW);
    GLuint getImplementation();


    // using agxData::Buffer::read;
    // virtual void read(const void* /*buffer*/, size_t /*numElements*/, const agxData::Format* /*format*/) override { agxAbort1("TODO"); }

  protected:
    virtual ~Buffer();
    virtual void reallocate(size_t size) override;
    static void initializeElements(agxData::Buffer *buffer, const agx::IndexRange& range);

  private:
    friend class Device;
    void activate();
    GLuint allocate(size_t memsize);
    bool deallocate(GLuint handle);

  private:
    GLuint m_bufferHandle;
    GLenum m_usageFlags;
  };


  class AGXCORE_EXPORT Task : public agx::ParallelTask
  {
  public:
    Task(const agx::Name& name, agx::Device *device = Device::instance());

  private:
    void pre(agx::Task *);
    void post(agx::Task *);

  private:
    agx::Task::ExecutionEvent::CallbackType m_preCallback;
    agx::Task::ExecutionEvent::CallbackType m_postCallback;
  };


  class AGXCORE_EXPORT Shader
  {
  public:
    Shader();
    ~Shader();

    GLuint getImplementation();
  protected:
    agx::String readSource();
    void build_impl(GLenum shaderType, const agx::String& buildFlags);

  protected:
    GLuint m_shaderHandle;
    agx::String m_sourceFilePath;
  };

  class AGXCORE_EXPORT GeometryShader : public Shader
  {
  public:
    void build(const agx::String& shaderSourcePath, const agx::String& buildFlags);
  private:
  };


  class AGXCORE_EXPORT VertexShader : public Shader
  {
  public:
    void build(const agx::String& shaderSourcePath, const agx::String& buildFlags);
  private:
  };

  class AGXCORE_EXPORT FragmentShader : public Shader
  {
  public:
    void build(const agx::String& shaderSourcePath, const agx::String& buildFlags);
  private:
  };


  class AGXCORE_EXPORT State : public agx::Object
  {
  public:
    State(const agx::String& name);
    static State* load(agx::TiXmlElement *eState, agx::Device* );
    virtual void apply() = 0;
    virtual void addedToKernel(Kernel* kernel);
  };

  class AGXCORE_EXPORT CapabilityState : public State
  {
  public:
    CapabilityState(GLenum capability, agx::String& capabilityName, agx::String& bindPath, agx::Bool enabled=true);
    virtual void apply();
    static CapabilityState* load(agx::TiXmlElement *eState);
    virtual void addedToKernel(Kernel* kernel);
  private:
    GLenum m_capability;
    agxData::Val<agx::Bool> m_enabled;
    //agx::Bool m_enabled;
  };

  class AGXCORE_EXPORT PointSizeState : public State
  {
  public:
    PointSizeState(GLfloat size);
    virtual void apply();
  private:
    GLfloat m_pointSize;
  };

  class AGXCORE_EXPORT LineWidthState : public State
  {
  public:
    LineWidthState(GLfloat size);
    virtual void apply();
  private:
    GLfloat m_lineWidth;
  };

  class AGXCORE_EXPORT DepthMaskState : public State
  {
  public:
    DepthMaskState(GLboolean enabled);
    virtual void apply();
  private:
    GLboolean m_enabled;
  };


  class AGXCORE_EXPORT TexEnvState : public State
  {
  public:
    TexEnvState(GLenum target, GLenum pname, GLint param);
    virtual void apply();
  private:
    GLenum m_target;
    GLenum m_pname;
    GLint m_param;
  };

  class AGXCORE_EXPORT PolygonModeState : public State
  {
  public:
    PolygonModeState(GLenum face, GLenum mode);
    virtual void apply();
  private:
    GLenum m_face;
    GLenum m_mode;
  };


  class AGXCORE_EXPORT Kernel : public agx::Kernel
  {
  public:
    static Kernel *load(agx::TiXmlElement *eKernel, agx::Device *device);
    virtual void configure(agx::TiXmlElement *eKernel) override;

  public:
    Kernel(const agx::Name& name);
    Kernel(const agx::Name& name, agx::Device *device, const agx::String& shaderSourcePath, const agx::String& buildFlags);

    void build(agx::Device *device);
    void dispatch();
    void voidDispatch();
    virtual void addState(State* state);
    GLuint getShaderProgram();

  protected:
    virtual ~Kernel();
    void parsePrimitive(agx::TiXmlElement* eKernel);
    void parseState(agx::TiXmlElement* eKernel);
    void locateUniforms();
    void locateAttributes();
    void updateUniforms();
    void updateAttributes();
    void addUniform(agx::TiXmlElement *eUniform);
    void addAttribute(agx::TiXmlElement *eAttribute);
    void addSharedAttribute(agx::TiXmlElement *eAttribute);
    bool generateVao();
    bool deallocateVao();

  protected:
    agx::String m_sourcePath;
    GeometryShader m_geometryShader;
    VertexShader m_vertexShader;
    FragmentShader m_fragmentShader;
    agx::String m_buildFlags;
    GLuint m_program;

    GLsizei m_numVerticesPerPrimitive;
    GLsizei m_numElements;
    GLenum m_primitive;
    agx::Vector<StateRef> m_states;

    GLuint m_vao;

    struct ShaderVariable
    {
      ShaderVariable();
      ShaderVariable(
        const agx::String& name, agx::Parameter* parameter_, GLint numComponents_ = 1,
        GLsizei stride_ = 1, agx::ScalarParameter* divisor_ = nullptr);

      GLuint id;
      GLint numComponents;
      GLsizei stride;
      agx::String variableName;
      agx::ParameterRef parameter;
      agx::ScalarParameterRef divisor;
    };

    agx::Vector<ShaderVariable> m_uniforms;
    agx::Vector<ShaderVariable> m_attributes;
    agx::ArrayParameterRef m_indexArgument;
    agx::ArrayParameterRef m_multiDrawCountArgument;
    agx::ArrayParameterRef m_multiDrawFirstArgument;
    bool m_multiDraw;
    agx::ScalarParameterRef m_numInstances;
  };

}

// AGX_USE_OPENGL
#endif

#endif /* _AGXGL_OPENGL_H_ */
