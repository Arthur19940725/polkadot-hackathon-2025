# Hardhat 本地网络 + MetaMask 配置指南

本指南将帮助你在 MetaMask 中配置 Hardhat 本地网络并导入测试账户。

## ✅ 已完成的配置

前端已经配置好支持 Hardhat 本地网络：

- ✅ `src/config/chains.ts` - 添加了 Hardhat 链配置
- ✅ `src/config/wagmi.ts` - 更新了 wagmi 配置以支持本地网络

## 📝 步骤四：在 MetaMask 中添加本地网络和账户

### 1. 添加 Hardhat 本地网络到 MetaMask

#### 方式一：手动添加（推荐）

1. **打开 MetaMask 扩展**
2. **点击顶部的网络下拉菜单**（默认显示 "Ethereum Mainnet"）
3. **点击底部的 "添加网络" 或 "Add network"**
4. **点击 "手动添加网络" 或 "Add a network manually"**
5. **填写以下信息：**
   ```
   网络名称（Network Name）: Hardhat Local
   新的 RPC URL（New RPC URL）: http://127.0.0.1:8545
   链 ID（Chain ID）: 31337
   货币符号（Currency Symbol）: ETH
   区块浏览器 URL（可选）: (留空)
   ```
6. **点击 "保存" 或 "Save"**
7. **切换到 "Hardhat Local" 网络**

#### 方式二：通过前端应用添加

1. 启动你的前端应用：`pnpm dev`
2. 打开 http://localhost:3000
3. 点击钱包连接按钮
4. 当应用请求切换到 Hardhat 网络时，MetaMask 会自动提示你添加该网络
5. 点击 "批准" 或 "Approve"

### 2. 导入测试账户到 MetaMask

Hardhat 为你提供了 20 个预充值的测试账户，每个账户都有 10000 ETH。

#### 导入 Account #0（推荐用于开发）

1. **打开 MetaMask**
2. **点击右上角的账户图标**
3. **选择 "导入账户" 或 "Import Account"**
4. **选择导入类型：私钥（Private Key）**
5. **粘贴以下私钥：**
   ```
   0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
   ```
6. **点击 "导入" 或 "Import"**
7. **建议重命名账户为 "Hardhat #0"**（点击账户名称可编辑）

#### 导入更多测试账户（可选）

你可以导入更多账户用于测试多用户场景：

**Account #1**

- 地址：`0x70997970C51812dc3A010C7d01b50e0d17dc79C8`
- 私钥：`0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d`

**Account #2**

- 地址：`0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC`
- 私钥：`0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a`

**Account #3**

- 地址：`0x90F79bf6EB2c4f870365E785982E1f101E93b906`
- 私钥：`0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6`

重复上述导入步骤，使用不同的私钥即可。

### 3. 验证配置

1. **确保 MetaMask 已切换到 "Hardhat Local" 网络**
2. **检查账户余额应该显示 10000 ETH**
3. **打开前端应用并尝试连接钱包**
4. **你应该能看到 "Hardhat Local" 出现在网络选择中**

## ⚠️ 重要提示

### 安全警告

- ⚠️ **这些私钥是公开的，仅用于本地开发**
- ⚠️ **切勿在任何真实网络（主网、测试网）中使用这些私钥**
- ⚠️ **切勿向这些地址在真实网络上发送任何有价值的代币**

### 重启 Hardhat 节点后

每次重启 `npx hardhat node` 时：

- ✅ 所有账户余额会重置为 10000 ETH
- ✅ 所有已部署的合约会被清除
- ✅ 所有交易历史会被清除
- ⚠️ 你需要重新部署所有合约

### MetaMask 显示问题

如果遇到以下问题：

**问题 1：余额显示为 0 或不正确**

- 解决方案：打开 MetaMask 设置 → 高级 → 清除活动标签数据

**问题 2：交易 nonce 错误**

- 解决方案：打开 MetaMask 设置 → 高级 → 重置账户

**问题 3：无法连接到本地节点**

- 确保 Hardhat 节点正在运行（`npx hardhat node`）
- 确保 RPC URL 是 `http://127.0.0.1:8545`
- 检查防火墙设置

## 🚀 下一步

配置完成后，你可以：

1. **启动前端开发服务器：**

   ```bash
   pnpm dev
   ```

2. **部署合约到本地网络：**

   ```bash
   cd contracts
   npx hardhat run scripts/deploy.js --network localhost
   ```

3. **在浏览器中测试：**
   - 打开 http://localhost:3000
   - 连接 MetaMask 钱包
   - 选择 Hardhat Local 网络
   - 开始与合约交互

## 📚 相关文档

- [Hardhat 官方文档](https://hardhat.org/docs)
- [MetaMask 开发者文档](https://docs.metamask.io/)
- [Wagmi 文档](https://wagmi.sh/)

## 🐛 故障排除

### MetaMask 无法连接

```bash
# 确保 Hardhat 节点正在运行
npx hardhat node

# 在另一个终端窗口部署合约
cd contracts
npx hardhat run scripts/deploy.js --network localhost
```

### 网络切换失败

如果无法切换到 Hardhat 网络：

1. 在 MetaMask 中手动切换到 "Hardhat Local"
2. 刷新浏览器页面
3. 重新连接钱包

### 合约地址无效

部署合约后，确保将合约地址保存到环境变量或配置文件中。

---

**配置完成！** 🎉

现在你可以在本地环境中愉快地开发和测试你的 Web3 投票应用了！
