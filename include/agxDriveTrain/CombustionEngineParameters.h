/*
Copyright 2007-2022. Algoryx Simulation AB.

All AGX source code, intellectual property, documentation, sample code,
tutorials, scene files and technical white papers, are copyrighted, proprietary
and confidential material of Algoryx Simulation AB. You may not download, read,
store, distribute, publish, copy or otherwise disseminate, use or expose this
material unless having a written signed agreement with Algoryx Simulation AB, or
having been advised so by Algoryx Simulation AB for a time limited evaluation,
or having purchased a valid commercial license from Algoryx Simulation AB.

Algoryx Simulation AB disclaims all responsibilities for loss or damage caused
from using this software, unless otherwise stated in written agreements with
Algoryx Simulation AB.
*/

#pragma once
#include <agxModel/export.h>
#include <agxDriveTrain/Shaft.h>


namespace agxDriveTrain
{
  /**
  Parameters that affect the engine performance.
  */
  struct AGXMODEL_EXPORT CombustionEngineParameters
  {
    /**
    The engine parameters are generally complicated so sensible defaults are provided.
    They are as follows:
    \param displacementVolume     - The total displacement volume of the engine, which is the sum of the volumes of the cylinders.
    \param maxTorque              - The maximum rated torque which is the highest brake torque that an engine is allowed to deliver
                                    over short/continous periods of operations.
    \param maxTorqueRPM           - The rated torque speed (i.e. the crankshaft rotational speed) in which the maximum rated torque is
                                    delivered.
    \param maxPower               - The maximum rated power which is the highest brake power that an engine is allowed to deliver
                                    over short/continous periods of operations.
    \param maxPowerRPM            - The rated power speed (i.e. the crankshaft rotational speed) in which the maximum rated power is
                                    delivered.
    \param idleRPM                - The crankshaft speed of the engine that is idling.
    \param crankShaftInertia      - The moment of inertia of the crankshaft.
    */
    agx::Real displacementVolume;
    agx::Real maxTorque;
    agx::Real maxTorqueRPM;
    agx::Real maxPower;
    agx::Real maxPowerRPM;
    agx::Real idleRPM;
    agx::Real crankShaftInertia;


    /**
    Saab 9000 passenger car B234i engine parameters:
        Source: Figure 4.4, Eriksson2014, Modelling and control of engines.
        This engine has been installed on the cars such as Saab 900 II, Saab 9000 I and Saab 9-3 I.
        Check the description of Saab H engine here: https://en.wikipedia.org/wiki/Saab_H_engine
        https://www.zemekoni.org/files/files/9000%202.1%20-%20Basic%20Engine%20B234%20M1990-.pdf
        - displacementVolume: 2.3 Liter
        - maxTorque: 212 N.m
        - maxTorqueRPM: 4000 RPM
        - maxPower: 110 kW
        - maxPowerRPM: 5500 RPM
        - idleRPM: 1000 RPM
        - crankShaftInertia: 0.2 kg.m^2
    */
    static CombustionEngineParameters Saab9000();

    /**
    Volvo passenger car engine D4204T14 parameters:
        Source: https://www.media.volvocars.com/global/en-gb/media/documentfile/224147/technical-specifications-new-volvo-v60
        This engine is installed on Volvo V60.
        - displacementVolume: 1.969 Liter
        - maxTorque: 400 N.m
        - maxTorqueRPM: 2125 RPM
        - maxPower: 140 kW
        - maxPowerRPM: 4250 RPM
        - idleRPM: 1000 RPM
        - crankShaftInertia: 0.4 kg.m^2
    */
    static CombustionEngineParameters VolvoV60();


    /**
    General motor truck engine parameters:
        Source : Figure 1-11, Pulkabek2004, Engineering fundamentals of the internal combustion engine.
        - displacementVolume: 7.4 Liter
        - maxTorque: 556 N.m
        - maxTorqueRPM: 3200 RPM
        - maxPower: 216 kW
        - maxPowerRPM: 4200 RPM
        - idleRPM: 1000 RPM
        - crankShaftInertia: 2.0 kg.m^2
    */
    static CombustionEngineParameters GMTruck();


    /**
    Volvo D15E540 engine parameters:
        Source - https://stpi.it.volvo.com/STPIFiles/Volvo/FactSheet/D16E540,%20EU4SCR_Eng_01_1163211.pdf.
        - displacementVolume: 16.1 Liter
        - maxTorque: 2600 N.m
        - maxTorqueRPM: 1225 RPM
        - maxPower: 397 kW
        - maxPowerRPM: 1625 RPM
        - idleRPM: 600 RPM
        - crankShaftInertia: 4.0 kg.m^2
    */
    static CombustionEngineParameters VolvoExcavator();

    /**
    Volvo TD 73 KDE engine parameters:
        Source - https://www.volvoce.com/-/media/volvoce/global/global-site/product-archive/documents/spec-sheets/wheel-loaders/v-l120d-213-2321-0004.pdf?v=_nxHPw.
        This engine is installed on Volvo Wheelloader L120D.
        - displacementVolume: 6.7 Liter
        - maxTorque: 920 N.m
        - maxTorqueRPM: 1100 RPM
        - maxPower: 148 kW
        - maxPowerRPM: 2100 RPM
        - idleRPM: 600 RPM
        - crankShaftInertia: 2.0 kg.m^2
    */
    static CombustionEngineParameters VolvoWheelLoader();


    /**
    John Deere Forwarder PowerTech Plus 6068 engine parameters:
        Source - https://www.deere.com/en/forwarders/1210g-forwarder/#tag-compare.
        https://www.deere.com/assets/pdfs/common/industries/engines-and-drivetrain/specsheets/6068hf485_aux.pdf
        This engine is installed on John Deere Forwarder 1210G.
        - displacementVolume: 6.8 Liter
        - maxTorque: 1025 N.m
        - maxTorqueRPM: 1400 RPM
        - maxPower: 187 kW
        - maxPowerRPM: 2200 RPM
        - idleRPM: 800 RPM
        - crankShaftInertia: 2.5 kg.m^2
    */
    static CombustionEngineParameters JohnDeereForwarder();

    /**
    Scania DC09 110 / 310 engine parameters:
        Source - https://www.scania.com/content/dam/scanianoe/market/au/products-and-services/buses-and-coaches/spec-sheets/SCA03642AxelCityBusSpecSheet_SAU2016-7-KCity_4x2_WEB.pdf
        This engine is installed on city bus 4x2 K series.
        - displacementVolume: 9.0 Liter
        - maxTorque: 1550 N.m
        - maxTorqueRPM: 1225 RPM
        - maxPower: 228 kW
        - maxPowerRPM: 1900 RPM
        - idleRPM: 900 RPM
        - crankShaftInertia: 3.5 kg.m^2
    */
    static CombustionEngineParameters ScaniaBus();


    /**
    Carterpillar haul truck C175-V 20 engine parameters:
    Source - https://en.wikipedia.org/wiki/Caterpillar_C175
    https://en.wikipedia.org/wiki/Caterpillar_797
    https://www.cat.com/en_US/products/new/equipment/off-highway-trucks/mining-trucks/18093014.html
    This engine is installed on Carterpillar 797F Heavy haul truck.
    - displacementVolume: 106.0 Liter
    - maxTorque: 9260 N.m
    - maxTorqueRPM: 1500 RPM
    - maxPower: 4000 kW
    - maxPowerRPM: 1800 RPM
    - idleRPM: 500 RPM
    - crankShaftInertia: 25 kg.m^2
    */
    static CombustionEngineParameters CarterpillarHaulTruck();


    /**
    Carterpillar bull dozer Cat C32 engine parameters:
    Source - https://www.teknoxgroup.com/fileadmin/user_upload/C32acert_ind18_1200.pdf
    https://www.eltrakbulgaria.com/uploads/Specalogs/2020/D11.pdf
    This engine is installed on Carterpillar D11 bull dozer.
    - displacementVolume: 32.1 Liter
    - maxTorque: 5861.0 N.m
    - maxTorqueRPM: 1200 RPM
    - maxPower: 895 kW
    - maxPowerRPM: 1800 RPM
    - idleRPM: 700 RPM
    - crankShaftInertia: 10 kg.m^2
    */
    static CombustionEngineParameters CarterpillarBullDozer();


    /**
    Komatsu wheel loader SAA6D125E-7 engine parameters:
    Source - https://www.komatsu.eu/Assets/GetBrochureByProductName.aspx?id=WA475-10&langID=en
    This engine is installed on Komatsu WA475-10 wheelloader.
    - displacementVolume: 11.04 Liter
    - maxTorque: 1560.0 N.m
    - maxTorqueRPM: 1330 RPM
    - maxPower: 217 kW
    - maxPowerRPM: 1600 RPM
    - idleRPM: 800 RPM
    - crankShaftInertia: 3.0 kg.m^2
    */
    static CombustionEngineParameters KomatsuWheelLoader();
  };

}
