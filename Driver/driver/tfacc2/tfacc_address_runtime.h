/* Internal driver helpers; caller supplies kernel MMIO/types and tf_device.
 * Shared with the host MMIO regression. This is not a userspace ABI. */
#ifndef TFACC_ADDRESS_RUNTIME_H
#define TFACC_ADDRESS_RUNTIME_H

static void tf_unmap_address_registers(struct tf_device *dev)
{
    unsigned int i;

    for (i = 0; i < ARRAY_SIZE(dev->ioreg_acp); ++i) {
        if (dev->ioreg_acp[i]) {
            iounmap(dev->ioreg_acp[i]);
            dev->ioreg_acp[i] = NULL;
        }
    }
}

/* Called once on a zero-initialized device, before publishing the cdev.
 * Mapping grants no ownership and does not read or write any register. */
static int tf_map_address_registers(struct tf_device *dev, int chip_count,
                                    u64 chip_gap)
{
    static const u64 bases[] = {
        TFACC0_FULL_ACP_BASE, TFACC1_FULL_ACP_BASE,
        TFACC2_FULL_ACP_BASE, TFACC3_FULL_ACP_BASE,
    };
    int chip;
    unsigned int pair;

    if (chip_count <= 0 || chip_count > MAX_TFACC_CNT / TFACC_REG_CNT)
        return -EINVAL;
    for (chip = 0; chip < chip_count; ++chip) {
        for (pair = 0; pair < ARRAY_SIZE(bases); ++pair) {
            unsigned int index = chip * ARRAY_SIZE(bases) + pair;

            dev->ioreg_acp[index] = ioremap(chip_gap * chip + bases[pair],
                                           TFACC_CLK_LENGTH);
            if (!dev->ioreg_acp[index]) {
                tf_unmap_address_registers(dev);
                return -ENOMEM;
            }
        }
    }
    return 0;
}

/* Only cache-side address extensions change. Direct command/result traffic
 * at 0x10 keeps the reserved-DDR high address installed by initialization. */
static void tfacc_write_cache_address_high(void __iomem *buf, u32 high_addr)
{
    u8 __iomem *regs = buf;
    u32 value;

    value = readl(regs + 0x90);
    value &= ~((0xffU << 0) | (0xffU << 16));
    value |= (1U << 28) | (1U << 8) | (1U << 24) |
             (high_addr << 0) | (high_addr << 16);
    writel(value, regs + 0x90);

    value = readl(regs + 0x94);
    value &= ~((0xffU << 0) | (0xffU << 16));
    value |= (1U << 28) | (1U << 8) | (1U << 24) |
             (high_addr << 0) | (high_addr << 16);
    writel(value, regs + 0x94);

    wmb();
    readl(regs + 0x94);
}

/* app_mutex must cover this entire call, including ownership checks and
 * MMIO, so unlock cannot hand the pair to another process during an update.
 * Always write HIGH, including zero and repeated values: another owner or
 * a hardware reset may have changed it since the preceding task. */
static int tf_set_address_high_locked(struct tf_device *dev, int tfacc_id,
                                      u32 high_addr, int chip_count,
                                      pid_t owner_tgid)
{
    int first;
    void __iomem *regs;

    if (chip_count <= 0 || chip_count > MAX_TFACC_CNT / TFACC_REG_CNT ||
        tfacc_id < 0 || tfacc_id >= chip_count * TFACC_REG_CNT ||
        high_addr > 0xffU)
        return -EINVAL;
    first = (tfacc_id / 2) * 2;
    if (dev->holdTFACCTgid[first] != owner_tgid ||
        dev->holdTFACCTgid[first + 1] != owner_tgid)
        return -EPERM;
    regs = dev->ioreg_acp[tfacc_id / 2];
    if (!regs)
        return -ENODEV;
    tfacc_write_cache_address_high(regs, high_addr);
    return 0;
}

#endif
