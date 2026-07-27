#include "kv_test/kv_test_config_helpers.h"
#include "kv_test/payload_buffer_runtime.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>

namespace UC::KVTest {
namespace {

class ScopedDeviceEnv {
public:
    ScopedDeviceEnv()
    {
        const char* value = std::getenv("UMC_ASU_DEVICE_ID");
        if (value != nullptr) { saved_ = value; }
        (void)unsetenv("UMC_ASU_DEVICE_ID");
    }

    ~ScopedDeviceEnv()
    {
        if (saved_.has_value()) {
            (void)setenv("UMC_ASU_DEVICE_ID", saved_->c_str(), 1);
        } else {
            (void)unsetenv("UMC_ASU_DEVICE_ID");
        }
    }

private:
    std::optional<std::string> saved_;
};

TEST(KvTestConfigHelpersTest, AivProviderDoesNotEnableFakeBackend)
{
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::AIV;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_FALSE(HasFakeProvider(config));
    MaybePrepareFakeBackend(config);
    EXPECT_EQ(config.asuClientConfig.transportConfigs.front().providerType,
              UC::ASU::TransProviderType::AIV);
    EXPECT_TRUE(config.asuClientConfig.transportConfigs.front().attrs.empty());
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, AicpuProviderUsesRegisterableDevicePayloads)
{
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::AICPU;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_FALSE(HasFakeProvider(config));
    EXPECT_FALSE(IsAivProviderMode(config));
    EXPECT_TRUE(IsAicpuProviderMode(config));
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, AicpuPayloadDeviceUsesExplicitLocalDevice)
{
    ScopedDeviceEnv deviceEnv;
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::AICPU;
    transportConfig.attrs["device_id"] = "5";
    transportConfig.endpoints.push_back(UC::ASU::AsuEndpoint{});
    transportConfig.endpoints.back().deviceId = 6;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_EQ(ResolvePayloadDeviceId(config), 5);
}

TEST(KvTestConfigHelpersTest, AicpuPayloadDeviceIgnoresRemoteEndpointDevice)
{
    ScopedDeviceEnv deviceEnv;
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::AICPU;
    transportConfig.endpoints.push_back(UC::ASU::AsuEndpoint{});
    transportConfig.endpoints.back().deviceId = 6;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_EQ(ResolvePayloadDeviceId(config), 0);
}

TEST(KvTestConfigHelpersTest, AivPayloadDeviceKeepsEndpointFallback)
{
    ScopedDeviceEnv deviceEnv;
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::AIV;
    transportConfig.endpoints.push_back(UC::ASU::AsuEndpoint{});
    transportConfig.endpoints.back().deviceId = 6;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_EQ(ResolvePayloadDeviceId(config), 6);
}

TEST(KvTestConfigHelpersTest, FakeDefaultsDoNotModifyAivTransport)
{
    KvTestConfig config;
    UC::ASU::TransportConfig fakeConfig;
    fakeConfig.asuId = 1;
    fakeConfig.providerType = UC::ASU::TransProviderType::FAKE;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(fakeConfig));

    UC::ASU::TransportConfig aivConfig;
    aivConfig.asuId = 2;
    aivConfig.providerType = UC::ASU::TransProviderType::AIV;
    aivConfig.attrs["sentinel"] = "unchanged";
    config.asuClientConfig.transportConfigs.emplace_back(std::move(aivConfig));

    ASSERT_TRUE(HasFakeProvider(config));
    MaybePrepareFakeBackend(config);

    const auto& patchedFake = config.asuClientConfig.transportConfigs[0];
    EXPECT_EQ(patchedFake.providerType, UC::ASU::TransProviderType::FAKE);
    EXPECT_EQ(patchedFake.attrs.at("fake_backend.path"), "./kv-test-fake-backend-store");

    const auto& unchangedAiv = config.asuClientConfig.transportConfigs[1];
    EXPECT_EQ(unchangedAiv.providerType, UC::ASU::TransProviderType::AIV);
    EXPECT_EQ(unchangedAiv.attrs.size(), 1);
    EXPECT_EQ(unchangedAiv.attrs.at("sentinel"), "unchanged");
    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::AIV_REGISTERABLE);
}

TEST(KvTestConfigHelpersTest, FakeProviderUsesDefaultDeviceAllocation)
{
    KvTestConfig config;
    UC::ASU::TransportConfig transportConfig;
    transportConfig.providerType = UC::ASU::TransProviderType::FAKE;
    config.asuClientConfig.transportConfigs.emplace_back(std::move(transportConfig));

    EXPECT_EQ(AllocationPolicyForConfig(config), DeviceAllocationPolicy::DEFAULT);
}

}  // namespace
}  // namespace UC::KVTest
