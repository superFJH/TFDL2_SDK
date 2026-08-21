/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef TFACC_MODEL_POOL_UAPI_H
#define TFACC_MODEL_POOL_UAPI_H

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

#define TF_MODEL_POOL_API_VERSION 1U
#define TF_MODEL_POOL_AUTO_HIGH 0xffffffffU

/*
 * Allocate one mmap-able segment for a model arena.
 *
 * required_length is the total number of bytes which still have to be
 * reserved.  When requested_high is AUTO, the driver only selects a 4 GiB
 * window whose aggregate free reserved-DDR and HugePage capacity can satisfy
 * that value.
 * One call returns one physically/DMA-contiguous segment; userspace repeats
 * the request with the returned address_high until the complete arena has
 * been reserved.
 */
struct tf_model_pool_alloc {
    __u64 required_length;
    __u64 allocated_length;
    __u64 physical_addr;
    __u64 dma_addr;
    __s32 chip_id;
    __u32 requested_high;
    __u32 address_high;
    __u32 mmap_id;
    __u32 flags;
    __u32 api_version;
    __u32 reserved;
};

/* Release one segment previously returned by TF_MODEL_POOL_ALLOC.  Userspace
 * must unmap the segment before issuing this request. */
struct tf_model_pool_free {
    __u32 mmap_id;
    __u32 flags;
    __u32 api_version;
    __u32 reserved;
};

#define TF_MODEL_POOL_ALLOC \
    _IOWR(TF_MAGIC, 14, struct tf_model_pool_alloc)
#define TF_MODEL_POOL_FREE \
    _IOW(TF_MAGIC, 15, struct tf_model_pool_free)

#endif /* TFACC_MODEL_POOL_UAPI_H */
