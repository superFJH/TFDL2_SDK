/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
#ifndef TFACC_ADDRESS_UAPI_H
#define TFACC_ADDRESS_UAPI_H

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

#define TF_ADDRESS_API_VERSION 1U

/*
 * Install DMA address bits [39:32] for the L1 cache shared by one pair of
 * TFACC cores.  The caller must hold both TF_APP_LOCKs in that pair.
 */
struct tf_address_high {
    __s32 tfacc_id;
    __u32 address_high;
    __u32 flags;
    __u32 api_version;
};

#define TF_SET_ADDRESS_HIGH \
    _IOW(TF_MAGIC, 13, struct tf_address_high)

#endif /* TFACC_ADDRESS_UAPI_H */
