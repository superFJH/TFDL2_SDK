// SPDX-License-Identifier: GPL-2.0
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/memfd.h>
#include <linux/mempolicy.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "tfacc_hugepage_uapi.h"

#ifndef MFD_HUGE_SHIFT
#define MFD_HUGE_SHIFT 26
#endif
#ifndef MFD_HUGE_1GB
#define MFD_HUGE_1GB (30U << MFD_HUGE_SHIFT)
#endif

static const char *program_name;

#define TF_HUGEPAGE_SYSFS_GLOBAL \
    "/sys/kernel/mm/hugepages/hugepages-1048576kB"
#define TF_HUGEPAGE_SYSFS_NODE \
    "/sys/devices/system/node/node%d/hugepages/hugepages-1048576kB"

static void usage(FILE *stream)
{
    fprintf(stream,
            "Usage:\n"
            "  %s --size <N[G|T]> [--chip N] [--node N] [--no-grow] [--device PATH]\n"
            "  %s --list [--device PATH]\n"
            "  %s --clear [--device PATH]\n\n"
            "Examples:\n"
            "  %s --size 8G --chip 0 --node 0\n"
            "  %s --size 32G --chip 0\n"
            "  %s --list\n\n"
            "The requested size must be a multiple of 1 GiB.  By default the\n"
            "tool grows the kernel's 1 GiB HugeTLB pool when free pages are\n"
            "insufficient.  --no-grow only uses pages already in the pool.\n"
            "Registration and automatic pool growth normally require root.\n",
            program_name, program_name, program_name, program_name,
            program_name, program_name);
}

static int hugepage_pool_paths(int numa_node, char *nr_path,
                               size_t nr_path_size, char *free_path,
                               size_t free_path_size)
{
    char directory[PATH_MAX];
    int length;

    if (numa_node < 0) {
        length = snprintf(directory, sizeof(directory), "%s",
                          TF_HUGEPAGE_SYSFS_GLOBAL);
    } else {
        length = snprintf(directory, sizeof(directory),
                          TF_HUGEPAGE_SYSFS_NODE, numa_node);
    }
    if (length < 0 || (size_t)length >= sizeof(directory)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    length = snprintf(nr_path, nr_path_size, "%s/nr_hugepages", directory);
    if (length < 0 || (size_t)length >= nr_path_size) {
        errno = ENAMETOOLONG;
        return -1;
    }
    length = snprintf(free_path, free_path_size, "%s/free_hugepages",
                      directory);
    if (length < 0 || (size_t)length >= free_path_size) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return 0;
}

static int read_count_file(const char *path, uint64_t *value)
{
    FILE *stream;
    unsigned long long parsed;

    stream = fopen(path, "r");
    if (!stream) {
        return -1;
    }
    if (fscanf(stream, "%llu", &parsed) != 1) {
        int saved_errno = errno ? errno : EIO;

        fclose(stream);
        errno = saved_errno;
        return -1;
    }
    if (fclose(stream) != 0) {
        return -1;
    }
    *value = (uint64_t)parsed;
    return 0;
}

static int write_count_file(const char *path, uint64_t value)
{
    FILE *stream;
    int retval = 0;

    stream = fopen(path, "w");
    if (!stream) {
        return -1;
    }
    if (fprintf(stream, "%llu\n", (unsigned long long)value) < 0 ||
        fflush(stream) != 0) {
        retval = -1;
    }
    if (fclose(stream) != 0) {
        retval = -1;
    }
    return retval;
}

static int ensure_hugepage_pool(uint64_t required_pages, int numa_node,
                                bool grow_pool)
{
    char nr_path[PATH_MAX];
    char free_path[PATH_MAX];
    uint64_t nr_pages;
    uint64_t free_pages;
    uint64_t target_pages;
    char pool_name[64];

    if (numa_node < 0) {
        snprintf(pool_name, sizeof(pool_name), "global pool");
    } else {
        snprintf(pool_name, sizeof(pool_name), "NUMA node %d", numa_node);
    }

    if (hugepage_pool_paths(numa_node, nr_path, sizeof(nr_path), free_path,
                            sizeof(free_path)) < 0) {
        perror("build 1 GiB HugeTLB sysfs path");
        return -1;
    }
    if (read_count_file(nr_path, &nr_pages) < 0 ||
        read_count_file(free_path, &free_pages) < 0) {
        fprintf(stderr,
                "Cannot inspect the 1 GiB HugeTLB %s.\n", pool_name);
        fprintf(stderr, "Path: %s\n", nr_path);
        fprintf(stderr,
                "Check that this NUMA node exists and the kernel supports 1 GiB HugeTLB pages.\n");
        return -1;
    }
    if (free_pages >= required_pages) {
        printf("1 GiB HugeTLB pool ready: %llu free page(s).\n",
               (unsigned long long)free_pages);
        return 0;
    }
    if (!grow_pool) {
        fprintf(stderr,
                "Only %llu free 1 GiB HugeTLB page(s); %llu are required (--no-grow).\n",
                (unsigned long long)free_pages,
                (unsigned long long)required_pages);
        return -1;
    }
    if (nr_pages > UINT64_MAX - (required_pages - free_pages)) {
        errno = EOVERFLOW;
        perror("grow 1 GiB HugeTLB pool");
        return -1;
    }

    target_pages = nr_pages + required_pages - free_pages;
    printf("Growing 1 GiB HugeTLB %s: total %llu -> %llu page(s)...\n",
           pool_name,
           (unsigned long long)nr_pages,
           (unsigned long long)target_pages);
    if (write_count_file(nr_path, target_pages) < 0) {
        perror("grow 1 GiB HugeTLB pool");
        fprintf(stderr,
                "Run as root, or preallocate pages at boot with: hugepagesz=1G hugepages=%llu\n",
                (unsigned long long)required_pages);
        return -1;
    }
    if (read_count_file(nr_path, &nr_pages) < 0 ||
        read_count_file(free_path, &free_pages) < 0) {
        perror("recheck 1 GiB HugeTLB pool");
        return -1;
    }
    if (free_pages < required_pages) {
        fprintf(stderr,
                "The kernel created only %llu free 1 GiB page(s), but %llu are required.\n",
                (unsigned long long)free_pages,
                (unsigned long long)required_pages);
        fprintf(stderr,
                "Memory is probably fragmented. Add 'hugepagesz=1G hugepages=%llu' to the kernel command line and reboot.\n",
                (unsigned long long)target_pages);
        return -1;
    }

    printf("1 GiB HugeTLB pool ready: %llu total, %llu free page(s).\n",
           (unsigned long long)nr_pages, (unsigned long long)free_pages);
    return 0;
}

static int parse_size(const char *text, uint64_t *result)
{
    char *end = NULL;
    unsigned long long value;
    uint64_t multiplier = 1;

    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno || end == text) {
        return -1;
    }
    if (*end != '\0') {
        if (end[1] != '\0') {
            return -1;
        }
        switch (*end) {
        case 'g':
        case 'G':
            multiplier = 1ULL << 30;
            break;
        case 't':
        case 'T':
            multiplier = 1ULL << 40;
            break;
        default:
            return -1;
        }
    }
    if (value > UINT64_MAX / multiplier) {
        return -1;
    }
    *result = (uint64_t)value * multiplier;
    return *result && !(*result % TF_HUGEPAGE_BYTES) ? 0 : -1;
}

static int parse_nonnegative_int(const char *text, int *result)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || value < 0 ||
        value > INT32_MAX) {
        return -1;
    }
    *result = (int)value;
    return 0;
}

static int create_hugetlb_memfd(void)
{
    return syscall(SYS_memfd_create, "tfacc-hugepages",
                   MFD_CLOEXEC | MFD_HUGETLB | MFD_HUGE_1GB);
}

static int bind_mapping_to_node(void *address, size_t length, int node)
{
    unsigned long node_mask;

    if (node < 0) {
        return 0;
    }
    if ((unsigned int)node >= sizeof(node_mask) * 8U) {
        errno = ERANGE;
        return -1;
    }
    node_mask = 1UL << node;
    return syscall(SYS_mbind, address, length, MPOL_BIND, &node_mask,
                   sizeof(node_mask) * 8U, MPOL_MF_STRICT);
}

static int list_regions(int device_fd)
{
    struct tf_hugepage_info info;
    uint32_t index = 0;

    printf("%-6s %-6s %-6s %-6s %-18s %-18s %-10s %-8s\n",
           "INDEX", "ID", "CHIP", "HIGH", "PHYSICAL", "DMA",
           "USED/GiB", "OWNER");
    for (;;) {
        memset(&info, 0, sizeof(info));
        info.api_version = TF_HUGEPAGE_API_VERSION;
        info.index = index;
        if (ioctl(device_fd, TF_HUGEPAGE_QUERY, &info) < 0) {
            if (errno == ENOENT) {
                break;
            }
            perror("TF_HUGEPAGE_QUERY");
            return -1;
        }
        printf("%-6u %-6u %-6d 0x%-4llx 0x%016llx 0x%016llx %4.2f/%-4.2f %-8d\n",
               index, info.region_id, info.chip_id,
               (unsigned long long)(info.dma_addr >> 32),
               (unsigned long long)info.physical_addr,
               (unsigned long long)info.dma_addr,
               (double)info.allocated / (double)TF_HUGEPAGE_BYTES,
               (double)info.length / (double)TF_HUGEPAGE_BYTES,
               info.owner_tgid);
        ++index;
    }
    printf("Total: %u GiB registered\n", index);
    return 0;
}

static int clear_regions(int device_fd)
{
    struct tf_hugepage_clear request;

    memset(&request, 0, sizeof(request));
    request.api_version = TF_HUGEPAGE_API_VERSION;
    if (ioctl(device_fd, TF_HUGEPAGE_CLEAR, &request) < 0) {
        if (errno == EBUSY) {
            fprintf(stderr,
                    "HugePage memory is in use; stop NPU applications before clearing it.\n");
        } else {
            perror("TF_HUGEPAGE_CLEAR");
        }
        return -1;
    }
    printf("Released %u registered 1 GiB HugePage(s).\n", request.removed);
    return 0;
}

static int register_regions(int device_fd, uint64_t size, int chip_id,
                            int numa_node)
{
    uint64_t page_count = size / TF_HUGEPAGE_BYTES;
    uint64_t i;
    int memfd;

    memfd = create_hugetlb_memfd();
    if (memfd < 0) {
        perror("memfd_create(MFD_HUGETLB|MFD_HUGE_1GB)");
        return -1;
    }
    if (ftruncate(memfd, (off_t)size) < 0) {
        perror("ftruncate HugeTLB file");
        fprintf(stderr,
                "Check /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages.\n");
        close(memfd);
        return -1;
    }

    for (i = 0; i < page_count; ++i) {
        struct tf_hugepage_register request;
        void *mapping;
        off_t offset = (off_t)(i * TF_HUGEPAGE_BYTES);

        mapping = mmap(NULL, TF_HUGEPAGE_BYTES, PROT_READ | PROT_WRITE,
                       MAP_SHARED, memfd, offset);
        if (mapping == MAP_FAILED) {
            perror("mmap 1 GiB HugeTLB page");
            break;
        }
        if (bind_mapping_to_node(mapping, TF_HUGEPAGE_BYTES, numa_node) < 0) {
            perror("mbind HugeTLB page");
            munmap(mapping, TF_HUGEPAGE_BYTES);
            break;
        }

        memset(&request, 0, sizeof(request));
        request.api_version = TF_HUGEPAGE_API_VERSION;
        request.user_addr = (uintptr_t)mapping;
        request.length = TF_HUGEPAGE_BYTES;
        request.chip_id = chip_id;
        if (ioctl(device_fd, TF_HUGEPAGE_REGISTER, &request) < 0) {
            perror("TF_HUGEPAGE_REGISTER");
            munmap(mapping, TF_HUGEPAGE_BYTES);
            break;
        }

        printf("Registered %llu/%llu: id=%u chip=%d phys=0x%016llx dma=0x%016llx\n",
               (unsigned long long)(i + 1),
               (unsigned long long)page_count, request.region_id, chip_id,
               (unsigned long long)request.physical_addr,
               (unsigned long long)request.dma_addr);
        munmap(mapping, TF_HUGEPAGE_BYTES);
    }

    close(memfd);
    if (i != page_count) {
        fprintf(stderr,
                "Only %llu of %llu GiB were registered; already registered pages remain usable.\n",
                (unsigned long long)i, (unsigned long long)page_count);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    static const struct option options[] = {
        {"size", required_argument, NULL, 's'},
        {"chip", required_argument, NULL, 'c'},
        {"node", required_argument, NULL, 'n'},
        {"device", required_argument, NULL, 'd'},
        {"list", no_argument, NULL, 'l'},
        {"clear", no_argument, NULL, 'C'},
        {"no-grow", no_argument, NULL, 'N'},
        {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0},
    };
    const char *device_path = "/dev/thinkforce0";
    uint64_t size = 0;
    int chip_id = 0;
    int numa_node = -1;
    bool do_list = false;
    bool do_clear = false;
    bool grow_pool = true;
    int option;
    int device_fd;
    int retval;

    program_name = argv[0];
    while ((option = getopt_long(argc, argv, "s:c:n:d:lCNh", options,
                                 NULL)) != -1) {
        switch (option) {
        case 's':
            if (parse_size(optarg, &size)) {
                fprintf(stderr, "Invalid size: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'c':
            if (parse_nonnegative_int(optarg, &chip_id)) {
                fprintf(stderr, "Invalid chip id: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'n':
            if (parse_nonnegative_int(optarg, &numa_node)) {
                fprintf(stderr, "Invalid NUMA node: %s\n", optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'd':
            device_path = optarg;
            break;
        case 'l':
            do_list = true;
            break;
        case 'C':
            do_clear = true;
            break;
        case 'N':
            grow_pool = false;
            break;
        case 'h':
            usage(stdout);
            return EXIT_SUCCESS;
        default:
            usage(stderr);
            return EXIT_FAILURE;
        }
    }

    if ((do_list ? 1 : 0) + (do_clear ? 1 : 0) + (size ? 1 : 0) != 1 ||
        chip_id < 0 || numa_node < -1 || (!grow_pool && !size)) {
        usage(stderr);
        return EXIT_FAILURE;
    }

    device_fd = open(device_path, O_RDWR | O_CLOEXEC);
    if (device_fd < 0) {
        perror(device_path);
        return EXIT_FAILURE;
    }
    if (do_list) {
        retval = list_regions(device_fd);
    } else if (do_clear) {
        retval = clear_regions(device_fd);
    } else {
        retval = ensure_hugepage_pool(size / TF_HUGEPAGE_BYTES, numa_node,
                                      grow_pool);
        if (!retval) {
            retval = register_regions(device_fd, size, chip_id, numa_node);
        }
        if (!retval) {
            retval = list_regions(device_fd);
        }
    }
    close(device_fd);
    return retval ? EXIT_FAILURE : EXIT_SUCCESS;
}
