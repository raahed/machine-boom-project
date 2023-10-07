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

/*

Demonstrates various ways of reading/write resources from disk and memory.

*/

#include <unistd.h>

#include <agx/Logger.h>
#include <agx/Hinge.h>
#include <agx/Prismatic.h>
#include <agx/CylindricalJoint.h>

#include <agxCable/Cable.h>

#include <agxOSG/ExampleApplication.h>
#include <agxCollide/ConvexFactory.h>
#include <agxIO/ImageReaderPNG.h>
#include <agxIO/ReaderWriter.h>
#include <agxOSG/ReaderWriter.h>

#include <agxUtil/HeightFieldGenerator.h>

#include <agxCollide/Convex.h>

#include "tutorialUtils.h"
#include "collector.hpp"
#include "joints.hpp"
#include "boom.hpp"
#include "utils.hpp"


// Do this at every step
class MyStepEventListener : public agxSDK::StepEventListener
{
public:
  MyStepEventListener(Boom *boom)
      : m_boom(boom)
  {
    // Mask to tell which step events we're interested in. Available step events are:
    // Pre-collide: Before collision detection
    // Pre: After collision detection (contact list available) but before the solve stage
    // Post: After solve (transforms and velocities have been updated)
    setMask(PRE_COLLIDE | POST_STEP);

    // if ( m_rb.isValid() )
    //   m_rb->setPosition( m_targetPosition );
    period = {4, 3, 2, 2, 2, 5, 3, 5, 4};
    amplitude = {0.5, 1.0, 0.2, 0.2, 0.2, 0.5, 0.4, 0.3, 0.2};
  }
  virtual void preCollide(const agx::TimeStamp &t)
  {
    // TODO: refactor velocity assignment
    std::vector<double> velocities;
    for (std::size_t i = 0; i < period.size(); i++)
    {
      velocities.push_back(amplitude[i] * sin(2 * M_PI * t / period[i]));
    }

    m_boom->setJointVelocity(velocities);
  }

  virtual void post(const agx::TimeStamp &t)
  {
    std::vector<std::vector<double>> joint_angles = m_boom->getJointPositions();

    /* write csv */
    Collector::instance().append(joint_angles);

    // TODO: print cable points
    std::vector<double> nodes;
    m_boom->getCableNodesSerialized(nodes);
    for (auto itr = nodes.begin(); itr != nodes.end(); itr++)
    {
      std::cout << *itr << ",";
    }
    std::cout << "\n";
  }

private:
  //    agx::observer_ptr< agx::RigidBody > m_rb;
  Boom *m_boom;
  agx::Vec3 m_targetPosition;
  std::vector<double> period, amplitude;
};

/**
Read in the hoses scene
*/
osg::Group *loadHosesScene(agxSDK::Simulation *simulation, agxOSG::ExampleApplication *application)
{

  // application->setEnableDebugRenderer( true );

  agx::String filename = "BoomAndCables.agx"; // the file
  osg::Group *root = new osg::Group;
  agxSDK::AssemblyRef assembly = new agxSDK::Assembly();

  // load scene
  agxOSG::readFile(filename, simulation, root, assembly);

  simulation->add(assembly);
  std::vector<agx::Name> jointNames{agx::Name("Hinge8"), agx::Name("Hinge12"), agx::Name("Hinge15"), agx::Name("Hinge24"), agx::Name("Hinge29"), agx::Name("Cylindrical6"), agx::Name("Prismatic27")};
  std::vector<agx::Name> cableNames{agx::Name("AGXUnity.Cable"), agx::Name("AGXUnity.Cable (1)")};

  Boom::Builder* boomBuilder = new Boom::Builder(assembly);
  boomBuilder->addJoint<agx::Hinge>(jointNames[0])->addJoint<agx::Hinge>(jointNames[1])->addJoint<agx::Hinge>(jointNames[2]);
  boomBuilder->addJoint<agx::Hinge>(jointNames[3])->addJoint<agx::Hinge>(jointNames[4]);
  boomBuilder->addJoint<agx::CylindricalJoint>(jointNames[5])->addJoint<agx::Prismatic>(jointNames[6]);
  boomBuilder->addCable(cableNames[0])->addCable(cableNames[1]);
  Boom* myBoom = boomBuilder->build();

  simulation->setUniformGravity(agx::Vec3(0, 0, -9.81));

  Collector::instance().setup("data", filename, convertNameVector(jointNames));

  std::cerr << "Assembly loaded: " << assembly->getName().str() << std::endl;
  for (auto itr = assembly->getAssemblies().begin(); itr != assembly->getAssemblies().end(); itr++)
  {
    std::cerr << "sub-assembly " << (*itr)->getName().str() << std::endl;
  }

  simulation->add(new MyStepEventListener(myBoom));

  return root;
}

int main(int argc, char **argv)
{
  agx::AutoInit agxInit;
  std::cerr << "\t" << agxGetLibraryName() << " " << agxGetVersion() << " " << agxGetCompanyName() << "(C)\n "
            << "\tData Collection\n"
            << "\t--------------------------------\n\n"
            << std::endl;

  try
  {

    agxOSG::ExampleApplicationRef application = new agxOSG::ExampleApplication;

    application->addScene(loadHosesScene, '1');

    if (application->init(argc, argv))
    {
      return application->run();
    }
  }
  catch (std::exception &e)
  {
    std::cerr << "\nCaught exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
