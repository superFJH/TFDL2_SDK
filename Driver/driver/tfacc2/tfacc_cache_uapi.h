/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef TFACC_CACHE_UAPI_H
#define TFACC_CACHE_UAPI_H

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

#define TF_CACHE_API_VERSION 1U

/*
 * Invalidate one NPU-visible DMA range in every L1 cache attached to a chip.
 * The driver validates that the range belongs to the calling tgid before
 * touching cache registers.
 */
struct tf_cache_invalidate {
    __u64 dma_addr;
    __u64 length;
    __s32 chip_id;
    __u32 flags;
    __u32 api_version;
    __u32 reserved;
};

#define TF_CACHE_INVALIDATE \
    _IOW(TF_MAGIC, 12, struct tf_cache_invalidate)

#endif /* TFACC_CACHE_UAPI_H */
