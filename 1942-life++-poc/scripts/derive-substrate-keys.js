// ============================================================================
// scripts/derive-substrate-keys.js - Derive Substrate keys from mnemonic phrases
// ============================================================================
const { Keyring } = require('@polkadot/keyring');
const { cryptoWaitReady } = require('@polkadot/util-crypto');

async function deriveSubstrateKeys() {
  console.log("🔑 Deriving Substrate keys from mnemonic phrases...\n");

  // 等待加密库准备就绪
  await cryptoWaitReady();

  // 助记词
  const mnemonics = {
    deployer: "gym prize december digital hover churn exile pledge path hub safe dolphin",
    agent: "alarm clerk hungry shield collect tattoo ten devote truth chaos zebra together", 
    validator: "shrimp muscle aunt escape dirt ancient shove reopen orange prefer another tail"
  };

  // 期望的地址
  const expectedAddresses = {
    deployer: "5EEe7y4NAUnAnnbQApDBnBuaFeEvuFDe9EexmwXqwdNZjGnA",
    agent: "5G4mF1uQ3R4Sf12XwH5Bu7frQUgAkKwPmyGxMrzAFWu7MYCb",
    validator: "5CvkxaxqUEeS56o6uWXSomMqQ1dwi55oLoDW9VSz1dnQcuaP"
  };

  console.log("📋 Deriving Substrate keys for:");
  console.log("   Deployer: 5EEe7y4NAUnAnnbQApDBnBuaFeEvuFDe9EexmwXqwdNZjGnA");
  console.log("   Agent:    5G4mF1uQ3R4Sf12XwH5Bu7frQUgAkKwPmyGxMrzAFWu7MYCb");
  console.log("   Validator: 5CvkxaxqUEeS56o6uWXSomMqQ1dwi55oLoDW9VSz1dnQcuaP\n");

  const derivedKeys = {};

  for (const [role, mnemonic] of Object.entries(mnemonics)) {
    try {
      // 创建 Keyring
      const keyring = new Keyring({ type: 'sr25519' });
      
      // 从助记词创建密钥对
      const keyPair = keyring.addFromMnemonic(mnemonic);
      
      // 获取地址和私钥
      const address = keyPair.address;
      const privateKey = keyPair.privateKey;
      
      derivedKeys[role] = {
        privateKey: privateKey,
        address: address,
        mnemonic: mnemonic
      };

      console.log(`✅ ${role.toUpperCase()}:`);
      console.log(`   Address: ${address}`);
      console.log(`   Private Key: ${privateKey}`);
      console.log(`   Expected: ${expectedAddresses[role]}`);
      console.log(`   Match: ${address === expectedAddresses[role] ? '✅' : '❌'}\n`);

    } catch (error) {
      console.log(`❌ Failed to derive key for ${role}:`, error.message);
    }
  }

  // 生成环境配置文件
  const envContent = `# ============================================================================
# Paseo Testnet Environment Configuration
# Generated from Substrate mnemonic phrases
# ============================================================================

# Network Configuration (Paseo Testnet)
PASEO_RPC=https://paseo-rpc.polkadot.io
RPC_URL=https://paseo-rpc.polkadot.io

# Private Keys (derived from Substrate mnemonic phrases)
PRIVATE_KEY=${derivedKeys.deployer?.privateKey || 'your-deployer-private-key-here'}
VALIDATOR_PRIVATE_KEY=${derivedKeys.validator?.privateKey || 'your-validator-private-key-here'}
AGENT_PRIVATE_KEY=${derivedKeys.agent?.privateKey || 'your-agent-private-key-here'}

# Contract Addresses (部署后会自动更新)
LEDGER_ADDRESS=
REGISTRY_ADDRESS=
CATK_ADDRESS=
ANFT_ADDRESS=
LEGAL_WRAPPER_ADDRESS=

# IPFS Configuration
IPFS_URL=https://ipfs.io

# Service Configuration
PORT=3000
CHECK_INTERVAL=10000

# Deployment Configuration
DEPLOYMENT_NETWORK=passetHub
GAS_LIMIT=8000000
GAS_PRICE=auto

# Hackathon Specific
HACKATHON_TRACK=track1
PROJECT_NAME=lifeplusplus-poc

# Note: Addresses are automatically derived from private keys
# No need to configure addresses manually
`;

  // 保存到文件
  const fs = require('fs');
  fs.writeFileSync('.env.passetHub', envContent);
  console.log("💾 Environment file saved to: .env.passetHub");

  console.log("\n" + "=".repeat(60));
  console.log("🎉 SUBSTRATE KEY DERIVATION COMPLETED!");
  console.log("=".repeat(60));
  console.log("\n📋 Next Steps:");
  console.log("1. Review the generated .env.passetHub file");
  console.log("2. Run deployment: npm run deploy:passethub");
  console.log("3. Test deployment: npm run test:passethub");
  console.log("\n⚠️  Security Note:");
  console.log("- Keep your mnemonic phrases secure");
  console.log("- Do not commit .env.passetHub to version control");
  console.log("- These are testnet keys only");
}

deriveSubstrateKeys()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("\n❌ Substrate key derivation failed:", error);
    process.exit(1);
  });
