# TFACC reserved DDR + 1 GiB HugePage 扩展池

第一次接触该驱动时，建议先阅读 [TFACC2 Linux 驱动源码导读](./DRIVER_CODE_GUIDE.md)，
其中包括 Linux 驱动基础、完整调用链和逐函数说明。

驱动默认仍只使用 ACPI/DT 提供的 reserved DDR，不运行注册工具时与原有
4 GiB 版本的 `TF_BUF_CREATE`/`mmap` ABI 一致。注册 1 GiB HugeTLB 页后，驱动会
先分配 reserved DDR，用完后再从 HugePage 扩展池分配，`tfnn` 无需修改。

每个 tgid/chip 的第一块内存必须来自 reserved DDR，用于保留旧版 `tfnn`
结果队列的高位地址语义；后续模型数据才会在 reserved DDR 耗尽后进入
HugePage。如果连这块 reserved DDR 都无法分配，驱动会明确返回 `ENOMEM`，
而不是让旧 `tfnn` 在错误的高位地址上启动。

## 1. 准备 1 GiB HugeTLB 页

注册工具默认自动检查指定 NUMA node 的 1 GiB HugeTLB 空闲页。页数不足时，
它会写入对应的 `nr_hugepages` 自动扩容，所以通常不需要用户提前操作 sysfs：

```bash
sudo tf_hugepage_register --size 8G --chip 0 --node 0
```

如果系统运行较久、物理内存已经碎片化，工具会报告实际创建的页数，并提示
需要使用内核启动参数。大容量生产环境仍建议在启动时预留，例如预留 16 GiB：

```text
hugepagesz=1G hugepages=16
```

高级用户如果只允许使用已经存在的 HugePage 池、不希望工具自动扩容，可以用：

```bash
sudo tf_hugepage_register --size 8G --chip 0 --node 0 --no-grow
```

## 2. 构建

```bash
make tf_hugepage_register
sudo make tfacc2
```

`make tfacc2` 也会先构建注册工具，并按原 Makefile 的逻辑执行
`modules_install` 和 `depmod`，因此需要 root 权限。Makefile 默认使用
`/lib/modules/$(uname -r)/build`；定制内核可以用 `KDIR=/path/to/kernel/build`
覆盖。

## 3. 按用户需求注册容量

为 chip 0 从 NUMA node 0 注册 8 GiB：

```bash
sudo ./tf_hugepage_register --size 8G --chip 0 --node 0
```

不指定 `--node` 时使用进程的默认 NUMA 策略。工具退出后页仍被驱动长期
pin 住，直到执行 `--clear` 或卸载驱动。

查看注册情况：

```bash
sudo ./tf_hugepage_register --list
```

`HIGH` 列就是 DMA 地址的高 32 位。驱动按 `chip -> HIGH -> DMA` 排序保存
注册页，所以相同 `HIGH` 的页会相邻显示，也会被旧版逐块分配路径优先连续使用。
例如一个模型池预计需要 3 GiB，就必须至少有一个 `HIGH` 组的剩余容量达到
3 GiB；不同 `HIGH` 的容量不能相加后交给同一个算子。

在没有 NPU 进程使用 HugePage 时释放扩展池：

```bash
sudo ./tf_hugepage_register --clear
```

只要还有其他进程打开 `/dev/thinkforce0`，或扩展池仍有分配记录，
`--clear` 就会返回 `EBUSY`。先停止 NPU 应用再释放。

`--clear` 会解除驱动注册和长期 pin，页面随后回到 Linux HugeTLB 空闲池；
它不会擅自缩小管理员配置的 `nr_hugepages`。如需把 HugeTLB 池本身归还给普通
System RAM，可在清理后由管理员调低相应 node 的 `nr_hugepages`，或重启系统。

## 约束

- 只接受显式 1 GiB HugeTLB 映射，不接受 Transparent HugePage。
- 每个注册区物理连续；不同 1 GiB 区域不需要彼此连续。
- 驱动通过 DMA API 返回 NPU 总线地址；开启 IOMMU 时它可能与工具显示的
  CPU physical address 不同，这是预期行为。
- 驱动不会返回跨越 4 GiB 高位边界的单个 `TF_BUF_CREATE` 结果。
- 注册需要 `CAP_SYS_ADMIN`；扩展池不会在普通用户未授权的情况下自动增长。
- 当前 HugePage DMA 注册使用 ACPI platform device；非 ACPI 构建会保留原 reserved
  DDR 路径，并对 HugePage 注册返回 `EOPNOTSUPP`。
- 这里的兼容指 `tfnn` 内存申请和设备锁 ABI 不改；实际运行 4 GiB 以上地址时，
  TFEngine/NPU40T 会在启动算子前通过新增驱动 ioctl 设置共享高位寄存器。
- HugePage 会从 Linux System RAM 中预留，注册多少就会减少多少可供普通进程
  使用的内存。

## NPU40T 高位地址与并发锁

### 模型构建期的同窗口内存池

默认配置 `UseModelMemoryPool=true`。TFExecutor 在生成 NPU 描述符之前计算：

- FrugalMode 开启时，按复用后的激活峰值统计；关闭时按全部激活统计；
- 对 WeightManager 中共享同一 CPU backing 的权重去重后统计；
- 再增加 25%（至少 128 MiB）供权重变换、LUT、对齐和描述符内部缓冲使用。

随后通过 `TF_MODEL_POOL_ALLOC` 让驱动选择一个容量足够的 `HIGH`。驱动把当前
TGID 在该 high 下剩余的 reserved DDR 与已注册 HugePage 一起统计，每次返回
一个可 `mmap` 的连续段；TFEngine 重复申请，直到预留完整容量。用户态 arena
再以 16 KiB 对齐做子分配，因此激活、变换后的权重和 LUT 都落在同一 4 GiB
窗口。第二个使用同一 chip 的 TFACC handler 会复用这份 arena，不会重复预留。
普通激活和描述符临时内存析构后会插回空闲范围并合并；跨模型缓存的变换权重仍会
由引用它的 NPU 命令描述符持有。

`paramDict` 只保存弱引用，Convolution 描述符保存强引用。`ReleaseCmd` 删除最后
一个引用相同变换权重/偏置的描述符时，缓存对象随即析构并把范围还给 arena，
不需要 TFDL 软件层显式按 CPU 指针清缓存。TFExecutor 释放硬件 handler 后，
若某个 arena 只剩 `CoreInfo` 自己持有，则像清理 cuDNN workspace 一样删除它。
Arena 会先 `munmap` 每个段，再通过 `TF_MODEL_POOL_FREE` 按 `mmap_id` 注销驱动
分配，所以进程无需退出即可把容量还给 model pool。

驱动以活动 `kbuf` 区间为准扫描空闲范围，而不是只依赖单调增长的 `offset`。
因此多个小模型的 arena 即使不按创建顺序销毁，留下的非尾部空洞也能被后续模型
复用；释放最后一个区间时还会同步重算该 reserved DDR block 或 HugePage region
的高水位和 owner 状态。

权重缓存键现在包含 `CPU weight pointer + chip + HIGH + bytes`。相同权重只有在
目标模型窗口也相同时才复用；若模型切到另一个 high，会在新 arena 中重新生成
并注册权重，避免 LUT 中的 32 位低地址被错误地拼接旧 high。

如果估算值已经超过单个 4 GiB 窗口，构建会直接说明需要图分区和显式跨窗口
activation copy；如果预留后仍耗尽，也会保留第一次分配错误，不再回退软件内存
并产生 `weight nullptr`、Transpose 参数错误等级联信息。仅为运行旧的小模型和
旧驱动做临时兼容时，可以显式设置 `UseModelMemoryPool=false`。

高位寄存器不在驱动中维护额外的租约协议，仍复用稳定版 `tfnn` 的
`TF_TFNN_LockDevice_SEM` ABI：TFEngine 获取一个 TFACC 时，会同时锁住与它
共享 L1 cache 的另一个 TFACC，例如使用 0 时跨进程同时占住 0、1。进程内
对同一组锁做引用计数，最后一个使用者才成对解锁。

每个 chip 在 TFEngine 进程内维护 01、23、45、67 四份高位状态。每份状态
包含当前 `address_high` 和正在执行的算子数：

- 计数为 0 时，任意高位地址可写入并开始执行；
- 计数非 0 且地址相同，允许同组另一 TFACC 并发，计数加一；
- 计数非 0 且地址不同，等待正在执行的算子结束；
- 同步执行返回或异步 `Wait` 完成后计数减一，减到 0 后释放该高位状态。

动态 full-core（`-1`）执行算子时先找正在使用且高位相同的 01/23 组，再找
计数为 0 的组；lite-core（`-2`）以同样方式在 45/67 中选择。选择限制在
内存所属的同一个 chip 内。

命令缓冲区只保存该算子的 `address_high` 软件元数据，不再伪造一条 NPU
寄存器命令。首个使用者获得进程内高位状态后，TFEngine 调用
`TF_SET_ADDRESS_HIGH`；驱动校验当前 TGID 已成对持有共享 L1 的两个 TFACC，
再按 chip/pair 选择 ACP 控制块，由 CPU MMIO 只更新共享 L1 cache 一侧
`0x90`、`0x94` 中的高位字段。`0x10` 属于 direct TFACC 通路，包含旧版
`tfnn` 结果队列使用的 reserved DDR 高位，只在驱动初始化时设置，运行时绝对
不能跟随算子切换。驱动使用掩码替换旧值并执行写屏障/读回，所以从较大的
高位切换到较小高位时不会残留旧 bit。驱动写完前，同高位的其他进程内线程
也会等待，之后才允许并发启动。

当前 ACP 字段实际提供 8 位地址扩展，即 NPU 总线地址范围为 40 位；请求的
`address_high` 大于 `0xff` 会被拒绝。

驱动的 `TF_APP_LOCK` UAPI 没有变化，只把已有锁的拥有者检查统一为 TGID，
保证同一进程的不同线程可以由首线程加锁、末线程解锁。新增高位 ioctl 不
维护另一套跨进程租约，跨进程隔离仍由这两个成对锁保证。

## 64 位 cache invalid

旧版 `tfnn` 的 `TF_TFNN_InvalidCache` 会把地址高 32 位固定写成 0。驱动新增
`TF_CACHE_INVALIDATE`，校验请求范围属于当前进程后，同时写 cache invalid
的地址低位和高位。驱动在返回新的高地址内存块前先 invalid 一次，TFEngine
在释放高地址 `MmapBuf` 前再调用一次，因此无需修改稳定版 `tfnn`。
