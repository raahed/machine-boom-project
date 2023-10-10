#include <unistd.h>
#include <string>

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

#include "main.h"
#include "collector.hpp"
#include "joints.hpp"
#include "boom.hpp"
#include "utils.hpp"

using namespace std;

// Do this at every step
class MyStepEventListener : public agxSDK::StepEventListener {
public:
    MyStepEventListener(Boom *boom)
            : m_boom(boom) {
        // Mask to tell which step events we're interested in. Available step events are:
        // Pre-collide: Before collision detection
        // Pre: After collision detection (contact list available) but before the solve stage
        // Post: After solve (transforms and velocities have been updated)
        setMask(PRE_COLLIDE | POST_STEP);

        period = {4, 3, 2, 2, 2, 5, 3, 5, 4};
        amplitude = {0.5, 1.0, 0.2, 0.2, 0.2, 0.5, 0.4, 0.3, 0.2};
    }

    virtual void preCollide(const agx::TimeStamp &t) {
        // TODO: refactor velocity assignment
        vector<double> velocities;
        for (size_t i = 0; i < period.size(); i++) {
            velocities.push_back(amplitude[i] * sin(2 * M_PI * t / period[i]));
        }

        // FIXME:  m_boom->setJointVelocity(velocities);
    }

    virtual void post(const agx::TimeStamp &t) {

        /* get data, bring lowest_cable also in a vector */
        vector <vector<double>> joint_angles = m_boom->getJointPositions();
        vector<double> lowest_cable = m_boom->getLowestCableNode();

        vector <vector<double>> data_row;
        data_row.insert(data_row.end(), joint_angles.begin(), joint_angles.end());
        data_row.push_back(lowest_cable);

        /* write csv */
        Collector::instance().append(data_row);
        cout << "CSV file writen" << endl;
    }

private:
    Boom *m_boom;
    agx::Vec3 m_targetPosition;
    vector<double> period, amplitude;
};

/**
Read in the hoses scene
*/
osg::Group *loadHosesScene(agxSDK::Simulation *simulation, agxOSG::ExampleApplication *application) {

    application->setEnableDebugRenderer(true);

    /* basic setup */
    agx::String filename = "BoomAndCables.agx";
    osg::Group *root = new osg::Group;
    agxSDK::AssemblyRef assembly = new agxSDK::Assembly();

    /* load scene and add assembly */
    agxOSG::readFile(filename, simulation, root, assembly);
    simulation->add(assembly);
    Boom::Builder *boomBuilder = new Boom::Builder(assembly);

    /* grep cable names and add them to the boom */
    vector <string> cables = getListEnv("SIMULATION_CABLE");
    for (string cable: cables) {
        agx::Name name(cable);
        boomBuilder->addCable(name);
    }

    /* grep joint names, parse them and add to the boom */
    vector <string> joints = getListEnv("SIMULATION_JOINT");
    for (string joint: joints) {
        auto type = generateAGXTypeFromName(joint);
        agx::Name name(joint);
        boomBuilder->addJoint<decltype(type)>(name);
    }

    Boom *myBoom = boomBuilder->build();

    simulation->setUniformGravity(agx::Vec3(0, 0, -9.81));

    /* setup collector instance */
    vector <string> configuration;
    configuration.insert(configuration.end(), joints.begin(), joints.end());
    configuration.insert(configuration.end(), cables.begin(), cables.end());

    Collector::instance().setup("/mnt/data", "test", configuration);

    /* add movement events */
    simulation->add(new MyStepEventListener(myBoom));

    return root;
}

int main(int argc, char **argv) {

    agx::AutoInit agxInit;

    cout << "\t" << agxGetLibraryName() << " " << agxGetVersion() << " " << agxGetCompanyName() << "(C)\n "
         << "\tData Collection\n"
         << "\t--------------------------------\n\n"
         << endl;

    try {

        agxOSG::ExampleApplicationRef application = new agxOSG::ExampleApplication;

        application->addScene(loadHosesScene, '1');

        if (application->init(argc, argv)) {
            return application->run();
        }
    }
    catch (exception &e) {
        cerr << "\nCaught exception: " << e.what() << endl;
        return 1;
    }

    return 0;
}
