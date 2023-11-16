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

#include "global.hpp"
#include "collector.hpp"
#include "utils.hpp"
#include <unistd.h>


class Joint {
public:
    std::string name;
    JType type;
    double position;
    double velocity;
    double effort;
    agx::ConstraintRef handle;
};

class Boom : public agxSDK::Assembly {
public:

    void setJointVelocity(std::vector<double> &ref) {
        if (!initialized) return;
        for (unsigned i = 0; i < ref.size(); i++) {
            if (joints[i].type == JType::ROTARY) {
                auto hinge = joints[i].handle->as<agx::Hinge>();
                if (hinge != NULL) {
                    hinge->getMotor1D()->setEnable(true);
                    hinge->getMotor1D()->setLocked(false);
                    hinge->getMotor1D()->setSpeed(ref[i]);
                }
            }
            if (joints[i].type == JType::PRISMATIC) {
                auto prismatic = joints[i].handle->as<agx::Prismatic>();
                if (prismatic != NULL) {
                    prismatic->getMotor1D()->setEnable(true);
                    prismatic->getMotor1D()->setLocked(false);
                    prismatic->getMotor1D()->setSpeed(ref[i]);
                }
            }
            if (joints[i].type == JType::CYLINDRICAL) {
                auto cylindrical = joints[i].handle->as<agx::CylindricalJoint>();

                //FIXME: this sets the same speed to both cylidrical axes
                if (cylindrical != NULL) {
                    cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setEnable(true);
                    cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setLocked(false);
                    cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setSpeed(ref[i]);
                    cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setEnable(true);
                    cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setLocked(false);
                    cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setSpeed(ref[i]);
                }
            }

        }
    }

    std::vector<double> getLowestCablePosition() {
        std::vector<double> result;
        result.reserve(3);
        double lowest_z = std::numeric_limits<double>::infinity();
        agx::Vec3 lowest_pos;
        for (auto cable : cables) {
            for(agxCable::CableIterator itr = cable->begin(); itr != cable->end(); ++itr) {
                auto pos = itr->getCenterPosition();

                /* determ the lowest cable point */
                if (pos.z() < lowest_pos.z()) {
                    lowest_pos = pos;
                }
            }
        }

        result.push_back(lowest_pos.x());
        result.push_back(lowest_pos.y());
        result.push_back(lowest_pos.z());
        return result;
    }

    void getJointPosition(std::vector<std::vector<double>> &ref) {
        if (!initialized) return;
        ref.clear();
        for (auto itr = joints.begin(); itr != joints.end(); itr++) {
            if (itr->type == JType::ROTARY) {
                auto hinge = itr->handle->as<agx::Hinge>();
                if (hinge != NULL) {
                    std::vector<double> tm;
                    tm.push_back(hinge->getAngle());
                    ref.push_back(tm);
                }
            }
            if (itr->type == JType::PRISMATIC) {
                auto prismatic = itr->handle->as<agx::Prismatic>();
                if (prismatic != NULL) {
                    std::vector<double> tm;
                    tm.push_back(prismatic->getAngle());
                    ref.push_back(tm);
                }
            }
            if (itr->type == JType::CYLINDRICAL) {
                auto cylindrical = itr->handle->as<agx::CylindricalJoint>();
                if (cylindrical != NULL) {
                    std::vector<double> tm;
                    tm.push_back(cylindrical->getAngle(agx::Constraint2DOF::FIRST));
                    tm.push_back(cylindrical->getAngle(agx::Constraint2DOF::SECOND));
                    ref.push_back(tm);
                }
            }
        }
    }

    bool initialize(std::vector <std::string> &jointNames, std::vector <std::string> &cableNames) {

        /* setup joints */
        joints.clear();
        for (auto itr = jointNames.begin(); itr != jointNames.end(); itr++) {
            Joint q;
            q.name = *itr;
            q.type = guessJTypeFromString(q.name);
            agx::Name name(q.name);
            q.handle = this->getConstraint(name);
            q.handle->setEnable(true);
            if (q.type == JType::ROTARY) {
                auto hinge = q.handle->as<agx::Hinge>();
                if (hinge != NULL) {
                    std::cerr << q.name << " is a hinge. Enabled? " << hinge->isEnabled() << " Valid? "
                              << hinge->getValid() << " Locking it.\n";
                    hinge->getMotor1D()->setEnable(true);
                    hinge->getMotor1D()->setLocked(true);
                }
            } else if (q.type == JType::PRISMATIC) {
                auto prismatic = q.handle->as<agx::Prismatic>();
                if (prismatic != NULL) {
                    std::cerr << q.name << " is prismatic. Enabled? " << prismatic->isEnabled() << " Valid? "
                              << prismatic->getValid() << " Locking it.\n";
                    prismatic->getMotor1D()->setEnable(true);
                    prismatic->getMotor1D()->setLocked(true);
                }
            } else if (q.type == JType::CYLINDRICAL) {
                auto cylindrical = q.handle->as<agx::CylindricalJoint>();
                if (cylindrical != NULL) {
                    std::cerr << q.name << " is cylindrical. Enabled? " << cylindrical->isEnabled() << " Valid? "
                              << cylindrical->getValid() << " Locking it.\n";
                    cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setEnable(true);
                    cylindrical->getMotor1D(agx::Constraint2DOF::FIRST)->setLocked(true);
                    cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setEnable(true);
                    cylindrical->getMotor1D(agx::Constraint2DOF::SECOND)->setLocked(true);
                }
            }
            joints.push_back(q);
        }

        /* setup cables */
        for (auto itr = cableNames.begin(); itr != cableNames.end(); itr++) {
            agxSDK::Assembly *tmp = NULL;
            tmp = this->getAssembly(agx::Name(*itr));
            if (tmp != NULL) this->cables.push_back(dynamic_cast<agxCable::Cable *>(tmp));
        }

        initialized = true;
        return initialized;
    }

private:
    bool initialized = false;
    std::vector <Joint> joints;
    std::vector<agxCable::Cable *> cables;
};

class MyStepEventListener : public agxSDK::StepEventListener {
public:
    MyStepEventListener(Boom *boom)
            : m_boom(boom) {
        // Mask to tell which step events we're interested in. Available step events are:
        // Pre-collide: Before collision detection
        // Pre: After collision detection (contact list available) but before the solve stage
        // Post: After solve (transforms and velocities have been updated)
        setMask(PRE_COLLIDE | POST_STEP);

        //if ( m_rb.isValid() )
        //  m_rb->setPosition( m_targetPosition );
        period = {4, 3, 2, 2, 2, 5, 5};
        amplitude = {0.5, 1.0, 0.2, 0.2, 0.2, 0.5, 0.2};
    }

    virtual void preCollide(const agx::TimeStamp &t) {
        vector<double> velocities;
        for (unsigned i = 0; i < period.size(); i++) {
            velocities.push_back(amplitude[i] * sin(2 * M_PI * t / period[i]));
        }

        m_boom->setJointVelocity(velocities);

    }

    virtual void post(const agx::TimeStamp &t) {
        vector<vector<double>> joint_angles;
        m_boom->getJointPosition(joint_angles);

        joint_angles.push_back(m_boom->getLowestCablePosition());

        if(Collector::instance().endOfCollection())
        {
            std::cout << "Collection ended!" << endl;
            exit(0);
        }

        /* write csv */
        Collector::instance().append(joint_angles);
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
osg::Group *loadHosesScene(agxSDK::Simulation *simulation, agxOSG::ExampleApplication *application) {

    agx::String filename = "BoomAndCables.agx";
    osg::Group *root = new osg::Group;

    Boom *myBoom = new Boom();
    agxSDK::AssemblyRef assembly = myBoom;

    /* load */
    agxOSG::readFile(filename, simulation, root, assembly);

    simulation->add(assembly);

    /* load joint string from env */
    std::vector <std::string> joint_names = getListEnv("SIMULATION_JOINT");

    /* load cable string from env */
    std::vector <std::string> cable_names = getListEnv("SIMULATION_CABLE");

    /* init boom */
    myBoom->initialize(joint_names, cable_names);

    /* setup collector instance */
    std::vector <std::string> config = joint_names;
    config.push_back("Lowest-Cable-Position");
    Collector::instance().setup("/mnt/data", filename, config);

    /* set data size 'partlyMax times sizeCounterMax' */
    Collector::instance().setPartlyMax(atoi(getenv("SIMULATION_EXPORT_NUMBER_OF_FILES")));
    Collector::instance().setSizeCounterMax(atoi(getenv("SIMULATION_EXPORT_FILE_SIZE")));

    simulation->setUniformGravity(agx::Vec3(0, 0, -9.81));

    std::cerr << "Assembly loaded: " << assembly->getName().str() << std::endl;
    for (auto itr = assembly->getAssemblies().begin(); itr != assembly->getAssemblies().end(); itr++) {
        std::cerr << "\tsub-assembly " << (*itr)->getName().str() << std::endl;
    }

    simulation->add(new MyStepEventListener(myBoom));

    return root;

}

int main(int argc, char **argv) {
    agx::AutoInit agxInit;
    std::cerr <<
              "\tData Collection\n" <<
              "\t--------------------------------\n\n" << std::endl;

    try {

        agxOSG::ExampleApplicationRef application = new agxOSG::ExampleApplication;

        application->addScene(loadHosesScene, '1');

        if (application->init(argc, argv)) {
            return application->run();
        }
    }
    catch (std::exception &e) {
        std::cerr << "\nCaught exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
