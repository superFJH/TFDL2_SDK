/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef TFACC_HUGEPAGE_UAPI_H
#define TFACC_HUGEPAGE_UAPI_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <linux/types.h>
#include <sys/ioctl.h>
#endif

#ifndef TF_MAGIC
#define TF_MAGIC 'x'
#endif

#define TF_HUGEPAGE_API_VERSION 1U
#define TF_HUGEPAGE_BYTES (1ULL << 30)

/* Register exactly one 1 GiB HugeTLB mapping owned by the calling process. */
struct tf_hugepage_register {
    __u64 user_addr;
    __u64 length;
    __u64 physical_addr;
    __u64 dma_addr;
    __s32 chip_id;
    __u32 flags;
    __u32 region_id;
    __u32 api_version;
};

/* Query one registered region by its zero-based list index. */
struct tf_hugepage_info {
    __u64 physical_addr;
    __u64 dma_addr;
    __u64 length;
    __u64 allocated;
    __s32 owner_tgid;
    __s32 chip_id;
    __u32 region_id;
    __u32 index;
    __u32 flags;
    __u32 api_version;
};

struct tf_hugepage_clear {
    __u32 api_version;
    __u32 flags;
    __u32 removed;
    __u32 reserved;
};

#define TF_HUGEPAGE_REGISTER \
    _IOWR(TF_MAGIC, 9, struct tf_hugepage_register)
#define TF_HUGEPAGE_QUERY \
    _IOWR(TF_MAGIC, 10, struct tf_hugepage_info)
#define TF_HUGEPAGE_CLEAR \
    _IOWR(TF_MAGIC, 11, struct tf_hugepage_clear)

#endif /* TFACC_HUGEPAGE_UAPI_H */
