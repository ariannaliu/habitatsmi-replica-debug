// Copyright (c) Meta Platforms, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "Corrade/Utility/Assert.h"
#include "Magnum/DebugTools/Screenshot.h"
#include "Magnum/GL/Context.h"
#include "Magnum/Magnum.h"
#include "Magnum/Trade/AbstractImageConverter.h"
#include "configure.h"

#include "esp/gfx/replay/Recorder.h"
#include "esp/gfx/replay/ReplayManager.h"
#include "esp/metadata/managers/ObjectAttributesManager.h"
#include "esp/physics/objectManagers/RigidObjectManager.h"
#include "esp/sensor/CameraSensor.h"
#include "esp/sensor/Sensor.h"
#include "esp/sim/AbstractReplayRenderer.h"
#include "esp/sim/BatchReplayRenderer.h"
#include "esp/sim/ClassicReplayRenderer.h"
#include "esp/sim/Simulator.h"

#include <Corrade/TestSuite/Compare/Numeric.h>
#include <Corrade/TestSuite/Tester.h>
#include <Magnum/DebugTools/CompareImage.h>
#include <Magnum/ImageView.h>

namespace Cr = Corrade;
namespace Mn = Magnum;

using esp::assets::ResourceManager;
using esp::metadata::MetadataMediator;
using esp::scene::SceneManager;
using esp::sim::ReplayRendererConfiguration;
using esp::sim::Simulator;
using esp::sim::SimulatorConfiguration;

namespace {

const std::string screenshotDir =
    Cr::Utility::Path::join(TEST_ASSETS, "screenshots/");

struct BatchReplayRendererTest : Cr::TestSuite::Tester {
  explicit BatchReplayRendererTest();

  void testIntegration();
  void testUnproject();

  const Magnum::Float maxThreshold = 255.f;
  const Magnum::Float meanThreshold = 0.75f;

  esp::logging::LoggingContext loggingContext;

};  // struct BatchReplayRendererTest

Mn::MutableImageView2D getRGBView(int width,
                                  int height,
                                  std::vector<char>& buffer) {
  Mn::Vector2i size(width, height);
  constexpr int pixelSize = 4;

  buffer.resize(std::size_t(width * height * pixelSize));

  auto view = Mn::MutableImageView2D(Mn::PixelFormat::RGB8Unorm, size, buffer);

  return view;
}

std::vector<esp::sensor::SensorSpec::ptr> getDefaultSensorSpecs(
    const std::string& sensorName = "my_rgb") {
  auto pinholeCameraSpec = esp::sensor::CameraSensorSpec::create();
  pinholeCameraSpec->sensorSubType = esp::sensor::SensorSubType::Pinhole;
  pinholeCameraSpec->sensorType = esp::sensor::SensorType::Color;
  pinholeCameraSpec->position = {0.0f, 0.f, 0.0f};
  pinholeCameraSpec->resolution = {512, 384};
  pinholeCameraSpec->uuid = sensorName;
  std::vector<esp::sensor::SensorSpec::ptr> sensorSpecifications = {
      pinholeCameraSpec};
  return sensorSpecifications;
}

const struct {
  const char* name;
  Cr::Containers::Pointer<esp::sim::AbstractReplayRenderer> (*create)(
      const ReplayRendererConfiguration& configuration);
} TestIntegrationData[]{
    {"classic renderer",
     [](const ReplayRendererConfiguration& configuration) {
       return Cr::Containers::Pointer<esp::sim::AbstractReplayRenderer>{
           new esp::sim::ClassicReplayRenderer{configuration}};
     }},
    {"batch renderer", [](const ReplayRendererConfiguration& configuration) {
       return Cr::Containers::Pointer<esp::sim::AbstractReplayRenderer>{
           new esp::sim::BatchReplayRenderer{configuration}};
     }}};

BatchReplayRendererTest::BatchReplayRendererTest() {
  addInstancedTests({&BatchReplayRendererTest::testIntegration},
                    Cr::Containers::arraySize(TestIntegrationData));

  // temp only enable testUnproject for classic
  addInstancedTests({&BatchReplayRendererTest::testUnproject}, 1);
  // addInstancedTests({&BatchReplayRendererTest::testUnproject},
  //                    Cr::Containers::arraySize(TestIntegrationData));
}  // ctor

// test recording and playback through the simulator interface
void BatchReplayRendererTest::testUnproject() {
  auto&& data = TestIntegrationData[testCaseInstanceId()];
  setTestCaseDescription(data.name);

  std::vector<esp::sensor::SensorSpec::ptr> sensorSpecifications;
  auto pinholeCameraSpec = esp::sensor::CameraSensorSpec::create();
  pinholeCameraSpec->sensorSubType = esp::sensor::SensorSubType::Pinhole;
  pinholeCameraSpec->sensorType = esp::sensor::SensorType::Color;
  pinholeCameraSpec->position = {1.0f, 2.f, 3.0f};
  pinholeCameraSpec->resolution = {512, 384};
  pinholeCameraSpec->uuid = "my_rgb";
  sensorSpecifications = {pinholeCameraSpec};

  ReplayRendererConfiguration batchRendererConfig;
  batchRendererConfig.sensorSpecifications = std::move(sensorSpecifications);
  batchRendererConfig.numEnvironments = 1;
  {
    Cr::Containers::Pointer<esp::sim::AbstractReplayRenderer> renderer =
        data.create(batchRendererConfig);

    const int h = pinholeCameraSpec->resolution.x();
    const int w = pinholeCameraSpec->resolution.y();
    constexpr int envIndex = 0;
    auto ray = renderer->unproject(envIndex, {0, 0});
    CORRADE_COMPARE(Mn::Vector3(ray.origin),
                    Mn::Vector3(pinholeCameraSpec->position));
    CORRADE_COMPARE(Mn::Vector3(ray.direction),
                    Mn::Vector3(-0.51544, 0.68457, -0.51544));

    // Ug, these tests reveal an off-by-one bug in our implementation in
    // RenderCamera::unproject. Depending on your convention, we would expect
    // some of these various corners to have exactly mirrored results, but they
    // don't.
    ray = renderer->unproject(envIndex, {w, 0});
    CORRADE_COMPARE(Mn::Vector3(ray.direction),
                    Mn::Vector3(0.51544, 0.68457, -0.51544));
    ray = renderer->unproject(envIndex, {w - 1, 0});
    CORRADE_COMPARE(Mn::Vector3(ray.direction),
                    Mn::Vector3(0.513467, 0.68551, -0.51615));

    ray = renderer->unproject(envIndex, {0, h});
    CORRADE_COMPARE(Mn::Vector3(ray.direction),
                    Mn::Vector3(-0.51355, -0.68740, -0.51355));
    ray = renderer->unproject(envIndex, {0, h - 1});
    CORRADE_COMPARE(Mn::Vector3(ray.direction),
                    Mn::Vector3(-0.51449, -0.68599, -0.51450));
  }
}

// test recording and playback through the simulator interface
void BatchReplayRendererTest::testIntegration() {
  auto&& data = TestIntegrationData[testCaseInstanceId()];
  setTestCaseDescription(data.name);

  const std::string vangogh = Cr::Utility::Path::join(
      SCENE_DATASETS, "habitat-test-scenes/van-gogh-room.glb");
  constexpr int numEnvs = 4;
  const std::string sensorName = "my_rgb";
  const std::string userPrefix = "sensor_";
  const std::string screenshotPrefix = "ReplayBatchRendererTest_env";
  const std::string screenshotExtension = ".png";

  std::vector<std::string> serKeyframes;
  for (int envIndex = 0; envIndex < numEnvs; envIndex++) {
    SimulatorConfiguration simConfig{};
    simConfig.activeSceneName = vangogh;
    simConfig.enableGfxReplaySave = true;
    simConfig.createRenderer = false;

    auto sim = Simulator::create_unique(simConfig);

    // add and pose objects
    {
      auto objAttrMgr = sim->getObjectAttributesManager();
      objAttrMgr->loadAllJSONConfigsFromPath(
          Cr::Utility::Path::join(TEST_ASSETS, "objects/donut"), true);

      auto handles = objAttrMgr->getObjectHandlesBySubstring("donut");
      CORRADE_VERIFY(!handles.empty());

      auto rigidObj0 =
          sim->getRigidObjectManager()->addBulletObjectByHandle(handles[0]);
      rigidObj0->setTranslation(
          Mn::Vector3(1.5f, 1.0f + envIndex * 0.2f, 0.7f));
      rigidObj0->setRotation(
          Mn::Quaternion::rotation(Mn::Deg(45.f - envIndex * 5.f),
                                   Mn::Vector3(1.f, 0.f, 0.f).normalized()));

      auto rigidObj1 =
          sim->getRigidObjectManager()->addBulletObjectByHandle(handles[0]);
      rigidObj1->setTranslation(
          Mn::Vector3(1.5f, 1.2f + envIndex * 0.2f, -0.7f));
      rigidObj1->setRotation(
          Mn::Quaternion::rotation(Mn::Deg(30.f + envIndex * 5.f),
                                   Mn::Vector3(0.f, 0.f, 1.f).normalized()));
    }

    auto& recorder = *sim->getGfxReplayManager()->getRecorder();
    recorder.addUserTransformToKeyframe(
        userPrefix + sensorName, Mn::Vector3(3.3f, 1.3f + envIndex * 0.1f, 0.f),
        Mn::Quaternion::rotation(Mn::Deg(80.f + envIndex * 5.f),
                                 Mn::Vector3(0.f, 1.f, 0.f)));

    std::string serKeyframe = esp::gfx::replay::Recorder::keyframeToString(
        recorder.extractKeyframe());
    serKeyframes.emplace_back(std::move(serKeyframe));
  }

  ReplayRendererConfiguration batchRendererConfig;
  batchRendererConfig.sensorSpecifications = getDefaultSensorSpecs(sensorName);
  batchRendererConfig.numEnvironments = numEnvs;
  {
    Cr::Containers::Pointer<esp::sim::AbstractReplayRenderer> renderer =
        data.create(batchRendererConfig);

    // Check that the context is properly created
    CORRADE_VERIFY(Mn::GL::Context::hasCurrent());

    std::vector<std::vector<char>> buffers(numEnvs);
    std::vector<Mn::MutableImageView2D> imageViews;

    for (int envIndex = 0; envIndex < numEnvs; envIndex++) {
      // TODO pass size as a Vector2i; use an Image instead of a std::vector
      //  once there's Iterable<MutableImageView2D> that can be implicitly
      //  converted from a list of Image2D.
      imageViews.emplace_back(getRGBView(renderer->sensorSize(envIndex).x(),
                                         renderer->sensorSize(envIndex).y(),
                                         buffers[envIndex]));
    }

    for (int envIndex = 0; envIndex < numEnvs; envIndex++) {
      renderer->setEnvironmentKeyframe(envIndex, serKeyframes[envIndex]);
      renderer->setSensorTransformsFromKeyframe(envIndex, userPrefix);
    }

    renderer->render(imageViews);

    for (int envIndex = 0; envIndex < numEnvs; envIndex++) {
      CORRADE_ITERATION(envIndex);
      std::string groundTruthImageFile =
          screenshotPrefix + std::to_string(envIndex) + screenshotExtension;
      CORRADE_COMPARE_WITH(
          Mn::ImageView2D{imageViews[envIndex]},
          Cr::Utility::Path::join(screenshotDir, groundTruthImageFile),
          (Mn::DebugTools::CompareImageToFile{maxThreshold, meanThreshold}));
    }

    const auto colorPtr = renderer->getCudaColorBufferDevicePointer();
    const auto depthPtr = renderer->getCudaDepthBufferDevicePointer();
    bool isBatchRenderer =
        dynamic_cast<esp::sim::BatchReplayRenderer*>(renderer.get());
#ifdef ESP_BUILD_WITH_CUDA
    if (isBatchRenderer) {
      CORRADE_VERIFY(colorPtr);
      CORRADE_VERIFY(depthPtr);
    } else {
      // Not implemented in ClassicReplayRenderer
      CORRADE_VERIFY(!colorPtr);
      CORRADE_VERIFY(!depthPtr);
    }
#else
    CORRADE_VERIFY(!colorPtr);
    CORRADE_VERIFY(!depthPtr);
#endif
  }
  // Check that the context is properly deleted
  CORRADE_VERIFY(!Mn::GL::Context::hasCurrent());
}

}  // namespace

CORRADE_TEST_MAIN(BatchReplayRendererTest)
