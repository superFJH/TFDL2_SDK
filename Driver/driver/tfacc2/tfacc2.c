#include "tfacc2.h"
#include "linux/of.h"
#include <asm/pgtable.h>
#include <linux/capability.h>
#include <linux/hugetlb.h>
#include <linux/io.h>
#include <linux/limits.h>
#include <linux/mm.h>
#include <linux/scatterlist.h>
#include <linux/sizes.h>
#include <linux/time.h>
#include <linux/jiffies.h>

static struct tf_device *tf_dev = NULL;
static bool versionIsRight = false; // 表明version是否check过，如果check成功则置为true
static int isNextUncache = 0;
static int chips = 1;
static int efuse = 0;
static int chipInit = 0;
static int cacheInit = 0;
static int chipInitCnt = 0;
static int cacheInitCnt = 0;
static int skipFullSwRst = 0;
static unsigned long long chipGap = 0; // 相邻两个chip之间的寄存器地址差多少
static unsigned long long ddrStart = 0x100000000LL;
static resource_size_t ddrSize = 0x100000000ULL;

struct tf_huge_region {
    struct list_head list;
    struct scatterlist sg;
    struct device *dma_device;
    struct page **pages;
    unsigned long start_pfn;
    unsigned long nr_pages;
    phys_addr_t physical_addr;
    dma_addr_t dma_addr;
    u64 length;
    u64 offset;
    pid_t owner_tgid;
    int chip_id;
    u32 region_id;
    u32 flags;
    int mapped_nents;
};

static LIST_HEAD(tf_huge_regions);
static DEFINE_MUTEX(tf_memory_mutex);
static u32 tf_next_huge_region_id = 1;

static void tf_rollback_kbuf(struct kbuf *kbuf_p);

static int needReset = 0;

static void cpu_write(unsigned long long base, int value) {
    volatile unsigned int *clk = ioremap(base, 4);
    clk[0] = value;
    iounmap(clk);
}

static void hardware_tfacc_reset(void) {
    if (efuse) {
        return;
    }
    // 下面这段代码是硬件复位
    cpu_write(0xfe110000 + 0x84 * 4, 0x2);
    cpu_write(0xfc200000 + 0xe0, 0x0);
    cpu_write(0xfc200000 + 0xf8, 0x0);
    cpu_write(0xfc200000 + 0xfc, 0x0);
    cpu_write(0xfc600000 + 0xe0, 0x0);
    cpu_write(0xfc600000 + 0xf8, 0x0);
    cpu_write(0xfc600000 + 0xfc, 0x0);
    cpu_write(0xfe110000 + 0x82 * 4, 0x1);
    cpu_write(0xfe110000 + 0x82 * 4, 0x3);
    cpu_write(0xfe110000 + 0x84 * 4, 0x0);

    cpu_write(0xed400000 + 0x84 * 4, 0x3);
    cpu_write(0xec200000 + 0xe0, 0x0);
    cpu_write(0xec200000 + 0xf8, 0x0);
    cpu_write(0xec200000 + 0xfc, 0x0);
    cpu_write(0xec600000 + 0xe0, 0x0);
    cpu_write(0xec600000 + 0xf8, 0x0);
    cpu_write(0xec600000 + 0xfc, 0x0);
    cpu_write(0xed400000 + 0x82 * 4, 0x1);
    cpu_write(0xed400000 + 0x82 * 4, 0x3);
    cpu_write(0xed400000 + 0x84 * 4, 0x1);
}

/* reserve mem */
struct ReserveDDRBlock reserveDDRBlocks[1005];
int reserveDDRBlcokCnt = 0;

static bool tf_ranges_overlap(u64 a, u64 a_len, u64 b, u64 b_len)
{
    return a < b + b_len && b < a + a_len;
}

/* Keep equal chip/high windows adjacent.  The legacy allocator walks this
 * list in order, so sorting also makes its HugePage fallback consume all
 * pages from one 4 GiB window before moving to another one. */
static void tf_insert_huge_region_sorted_locked(struct tf_huge_region *region)
{
    struct tf_huge_region *cursor;
    u64 region_high = region->dma_addr >> 32;

    list_for_each_entry(cursor, &tf_huge_regions, list) {
        u64 cursor_high = cursor->dma_addr >> 32;

        if (region->chip_id < cursor->chip_id ||
            (region->chip_id == cursor->chip_id &&
             (region_high < cursor_high ||
              (region_high == cursor_high &&
               region->dma_addr < cursor->dma_addr)))) {
            list_add_tail(&region->list, &cursor->list);
            return;
        }
    }
    list_add_tail(&region->list, &tf_huge_regions);
}

static void tf_unpin_huge_region(struct tf_huge_region *region)
{
    if (region->mapped_nents > 0) {
        dma_unmap_sg(region->dma_device, &region->sg, 1,
                     DMA_BIDIRECTIONAL);
    }

    unpin_user_pages_dirty_lock(region->pages, region->nr_pages, true);
    kvfree(region->pages);
    kfree(region);
}

static void tf_release_all_huge_regions(void)
{
    struct tf_huge_region *region, *tmp;

    mutex_lock(&tf_memory_mutex);
    list_for_each_entry_safe(region, tmp, &tf_huge_regions, list) {
        list_del(&region->list);
        tf_unpin_huge_region(region);
    }
    mutex_unlock(&tf_memory_mutex);
}

static int tf_validate_hugetlb_mapping(unsigned long start, u64 length)
{
    struct vm_area_struct *vma;
    int retval = 0;

    if (!current->mm || length != TF_HUGEPAGE_BYTES ||
        !IS_ALIGNED(start, TF_HUGEPAGE_BYTES) ||
        start > ULONG_MAX - length) {
        return -EINVAL;
    }

    mmap_read_lock(current->mm);
    vma = find_vma(current->mm, start);
    if (!vma || start < vma->vm_start || start + length > vma->vm_end ||
        !is_vm_hugetlb_page(vma) ||
        vma_kernel_pagesize(vma) != TF_HUGEPAGE_BYTES) {
        retval = -EINVAL;
    }
    mmap_read_unlock(current->mm);
    return retval;
}

static int tf_ioctl_register_hugepage(struct tf_device *dev, void __user *arg)
{
    struct tf_hugepage_register request;
    struct tf_huge_region *region = NULL;
    struct tf_huge_region *cursor;
    struct page **pages = NULL;
    unsigned long start;
    int nr_pages;
    int pinned = 0;
    int retval = 0;
    int i;

    if (!capable(CAP_SYS_ADMIN)) {
        return -EPERM;
    }
    if (copy_from_user(&request, arg, sizeof(request))) {
        return -EFAULT;
    }
    if (request.api_version != TF_HUGEPAGE_API_VERSION || request.flags ||
        request.chip_id < 0 || request.chip_id >= chips) {
        return -EINVAL;
    }

    start = (unsigned long)request.user_addr;
    if ((u64)start != request.user_addr) {
        return -EINVAL;
    }
    retval = tf_validate_hugetlb_mapping(start, request.length);
    if (retval) {
        return retval;
    }
    if (!dev->dma_device) {
        return -EOPNOTSUPP;
    }

    nr_pages = request.length >> PAGE_SHIFT;
    pages = kvmalloc_array(nr_pages, sizeof(*pages), GFP_KERNEL);
    if (!pages) {
        return -ENOMEM;
    }

    pinned = pin_user_pages_fast(start, nr_pages,
                                 FOLL_WRITE | FOLL_LONGTERM, pages);
    if (pinned != nr_pages) {
        retval = pinned < 0 ? (int)pinned : -EFAULT;
        goto out_unpin_partial;
    }

    for (i = 1; i < nr_pages; ++i) {
        if (page_to_pfn(pages[i]) != page_to_pfn(pages[0]) + i) {
            retval = -EINVAL;
            goto out_unpin_partial;
        }
    }

    region = kzalloc(sizeof(*region), GFP_KERNEL);
    if (!region) {
        retval = -ENOMEM;
        goto out_unpin_partial;
    }

    region->dma_device = dev->dma_device;
    region->start_pfn = page_to_pfn(pages[0]);
    region->nr_pages = nr_pages;
    region->pages = pages;
    region->physical_addr = page_to_phys(pages[0]);
    region->length = request.length;
    region->owner_tgid = -1;
    region->chip_id = request.chip_id;
    region->flags = request.flags;

    sg_init_table(&region->sg, 1);
    sg_set_page(&region->sg, pages[0], (unsigned int)request.length, 0);
    region->mapped_nents = dma_map_sg(region->dma_device, &region->sg, 1,
                                      DMA_BIDIRECTIONAL);
    if (region->mapped_nents != 1 ||
        sg_dma_len(&region->sg) < request.length) {
        retval = -EIO;
        if (region->mapped_nents > 0) {
            dma_unmap_sg(region->dma_device, &region->sg, 1,
                         DMA_BIDIRECTIONAL);
            region->mapped_nents = 0;
        }
        goto out_free_region;
    }
    region->dma_addr = sg_dma_address(&region->sg);

    mutex_lock(&tf_memory_mutex);
    list_for_each_entry(cursor, &tf_huge_regions, list) {
        if (tf_ranges_overlap(region->physical_addr, region->length,
                              cursor->physical_addr, cursor->length)) {
            retval = -EEXIST;
            break;
        }
    }
    if (!retval) {
        region->region_id = tf_next_huge_region_id++;
        tf_insert_huge_region_sorted_locked(region);
    }
    mutex_unlock(&tf_memory_mutex);
    if (retval) {
        goto out_unmap_region;
    }

    request.physical_addr = region->physical_addr;
    request.dma_addr = region->dma_addr;
    request.region_id = region->region_id;
    if (copy_to_user(arg, &request, sizeof(request))) {
        mutex_lock(&tf_memory_mutex);
        list_del(&region->list);
        mutex_unlock(&tf_memory_mutex);
        retval = -EFAULT;
        goto out_unmap_region;
    }

    printk(KERN_INFO "tfacc: registered 1G HugeTLB region %u, chip %d, phys 0x%llx, dma 0x%llx\n",
           region->region_id, region->chip_id,
           (unsigned long long)region->physical_addr,
           (unsigned long long)region->dma_addr);
    return 0;

out_unmap_region:
    dma_unmap_sg(region->dma_device, &region->sg, 1, DMA_BIDIRECTIONAL);
    region->mapped_nents = 0;
out_free_region:
    region->pages = NULL;
    kfree(region);
out_unpin_partial:
    if (pinned > 0) {
        unpin_user_pages(pages, pinned);
    }
    kvfree(pages);
    return retval;
}

static int tf_ioctl_query_hugepage(void __user *arg)
{
    struct tf_hugepage_info info;
    struct tf_huge_region *region;
    u32 index = 0;
    int retval = -ENOENT;

    if (copy_from_user(&info, arg, sizeof(info))) {
        return -EFAULT;
    }
    if (info.api_version != TF_HUGEPAGE_API_VERSION) {
        return -EINVAL;
    }

    mutex_lock(&tf_memory_mutex);
    list_for_each_entry(region, &tf_huge_regions, list) {
        if (index++ != info.index) {
            continue;
        }
        info.physical_addr = region->physical_addr;
        info.dma_addr = region->dma_addr;
        info.length = region->length;
        info.allocated = region->offset;
        info.owner_tgid = region->owner_tgid;
        info.chip_id = region->chip_id;
        info.region_id = region->region_id;
        info.flags = region->flags;
        retval = 0;
        break;
    }
    mutex_unlock(&tf_memory_mutex);
    if (retval) {
        return retval;
    }
    return copy_to_user(arg, &info, sizeof(info)) ? -EFAULT : 0;
}

static int tf_ioctl_clear_hugepages(struct tf_device *dev, void __user *arg)
{
    struct tf_hugepage_clear request;
    struct tf_huge_region *region, *tmp;
    int retval = 0;

    if (!capable(CAP_SYS_ADMIN)) {
        return -EPERM;
    }
    if (copy_from_user(&request, arg, sizeof(request))) {
        return -EFAULT;
    }
    if (request.api_version != TF_HUGEPAGE_API_VERSION || request.flags) {
        return -EINVAL;
    }

    request.removed = 0;
    mutex_lock(&tf_memory_mutex);

    /* The helper itself owns one file descriptor.  Any additional opener
     * may still submit work or mmap a buffer from this pool. */
    spin_lock(&dev->lock);
    if (dev->isBusy != 1) {
        retval = -EBUSY;
    }
    spin_unlock(&dev->lock);
    if (retval) {
        goto out_unlock;
    }

    list_for_each_entry(region, &tf_huge_regions, list) {
        if (region->owner_tgid != -1) {
            retval = -EBUSY;
            goto out_unlock;
        }
    }
    list_for_each_entry_safe(region, tmp, &tf_huge_regions, list) {
        list_del(&region->list);
        tf_unpin_huge_region(region);
        request.removed++;
    }

out_unlock:
    mutex_unlock(&tf_memory_mutex);
    if (retval) {
        return retval;
    }
    return copy_to_user(arg, &request, sizeof(request)) ? -EFAULT : 0;
}

#define TF_CACHE_INVALID_REQ_OFFSET       0x0058
#define TF_CACHE_INVALID_ADDR_OFFSET      0x005c
#define TF_CACHE_INVALID_ADDR_HI_OFFSET   0x0060
#define TF_CACHE_INVALID_LEN_OFFSET       0x0064
#define TF_CACHE_INVALID_ACK_OFFSET       0x0068
#define TF_CACHE_MAKE_INVALID_RANGE       1U
#define TF_CACHE_INVALID_TIMEOUT_US       1000000U

static int tf_cache_invalidate_dma_range(struct tf_device *dev, int chip_id,
                                         u64 dma_addr, u64 length)
{
    int first_core;
    int core;

    if (chip_id < 0 || chip_id >= chips || !length || length > U32_MAX ||
        dma_addr > U64_MAX - length) {
        return -EINVAL;
    }

    first_core = chip_id * TFACC_REG_CNT;
    for (core = 0; core < TFACC_REG_CNT; ++core) {
        u8 __iomem *cache = dev->ioreg_cache[first_core + core];
        unsigned int waited;

        if (!cache) {
            return -ENODEV;
        }

        writel(lower_32_bits(dma_addr),
               cache + TF_CACHE_INVALID_ADDR_OFFSET);
        writel(upper_32_bits(dma_addr),
               cache + TF_CACHE_INVALID_ADDR_HI_OFFSET);
        writel((u32)length, cache + TF_CACHE_INVALID_LEN_OFFSET);
        writel(TF_CACHE_MAKE_INVALID_RANGE,
               cache + TF_CACHE_INVALID_REQ_OFFSET);

        for (waited = 0; waited < TF_CACHE_INVALID_TIMEOUT_US; ++waited) {
            if (readl(cache + TF_CACHE_INVALID_ACK_OFFSET) == 1U) {
                break;
            }
            udelay(1);
        }
        if (waited == TF_CACHE_INVALID_TIMEOUT_US) {
            printk(KERN_ERR
                   "tfacc: cache invalidation timed out on chip %d core %d, dma 0x%llx, len 0x%llx\n",
                   chip_id, core, (unsigned long long)dma_addr,
                   (unsigned long long)length);
            return -ETIMEDOUT;
        }
    }

    return 0;
}

static int tf_kbuf_chip_id(const struct kbuf *kbuf_p)
{
    if (kbuf_p->backend == TF_MEMORY_HUGEPAGE && kbuf_p->huge_region) {
        return kbuf_p->huge_region->chip_id;
    }
    if (kbuf_p->backend == TF_MEMORY_RESERVED && kbuf_p->pool_index >= 0 &&
        kbuf_p->pool_index < reserveDDRBlcokCnt) {
        return reserveDDRBlocks[kbuf_p->pool_index].chipId;
    }
    return -1;
}

static bool tf_dma_range_owned_locked(struct tf_device *dev, pid_t tgid,
                                      int chip_id, u64 dma_addr, u64 length)
{
    struct kbuf *kbuf_p;
    int bucket;

    if (!length || dma_addr > U64_MAX - length) {
        return false;
    }

    hash_for_each(dev->buf_list, bucket, kbuf_p, list) {
        u64 buffer_start = kbuf_p->phy_addr;
        u64 buffer_length = kbuf_p->len;

        if (kbuf_p->owner_tgid != tgid ||
            tf_kbuf_chip_id(kbuf_p) != chip_id ||
            buffer_start > U64_MAX - buffer_length) {
            continue;
        }
        if (dma_addr >= buffer_start &&
            dma_addr + length <= buffer_start + buffer_length) {
            return true;
        }
    }
    return false;
}

static int tf_ioctl_cache_invalidate(struct tf_device *dev, void __user *arg)
{
    struct tf_cache_invalidate request;
    int retval;

    if (copy_from_user(&request, arg, sizeof(request))) {
        return -EFAULT;
    }
    if (request.api_version != TF_CACHE_API_VERSION || request.flags ||
        request.reserved || request.length > U32_MAX) {
        return -EINVAL;
    }

    mutex_lock(&tf_memory_mutex);
    if (!tf_dma_range_owned_locked(dev, current->tgid, request.chip_id,
                                   request.dma_addr, request.length)) {
        retval = -EPERM;
    } else {
        retval = tf_cache_invalidate_dma_range(dev, request.chip_id,
                                               request.dma_addr,
                                               request.length);
    }
    mutex_unlock(&tf_memory_mutex);
    return retval;
}

#define TF_MODEL_POOL_ALIGNMENT SZ_64K

static bool tf_has_reserved_anchor_locked(int chip_id, pid_t tgid)
{
    int i;

    for (i = 0; i < reserveDDRBlcokCnt; ++i) {
        if (reserveDDRBlocks[i].chipId == chip_id &&
            reserveDDRBlocks[i].isMalloc &&
            reserveDDRBlocks[i].tgid == tgid) {
            return true;
        }
    }
    return false;
}

/* Return the still usable part of one registered region in one address-high
 * window.  A region may have an IOVA which is not 1 GiB aligned, so clip it
 * against both the region end and the 4 GiB window end. */
static bool tf_model_address_window_space(u64 dma_base, u64 pool_length,
                                          u64 current_offset,
                                          u32 address_high,
                                          u64 *pool_offset, u64 *available)
{
    u64 window_start = (u64)address_high << 32;
    u64 window_end = ((u64)address_high + 1) << 32;
    u64 candidate = ALIGN(current_offset, (u64)TF_MODEL_POOL_ALIGNMENT);
    u64 dma_addr;
    u64 bytes;

    if (candidate >= pool_length || dma_base > U64_MAX - candidate) {
        return false;
    }
    dma_addr = dma_base + candidate;
    if (dma_addr < window_start) {
        u64 skip = window_start - dma_addr;

        if (skip >= pool_length - candidate) {
            return false;
        }
        candidate = ALIGN(candidate + skip,
                          (u64)TF_MODEL_POOL_ALIGNMENT);
        if (candidate >= pool_length || dma_base > U64_MAX - candidate) {
            return false;
        }
        dma_addr = dma_base + candidate;
    }
    if (dma_addr < window_start || dma_addr >= window_end) {
        return false;
    }

    /* Keep the legacy low-address-zero guard used by TFNN's allocator. */
    if (!(dma_addr & 0xffffffffULL)) {
        if (candidate > U64_MAX - SZ_1M ||
            candidate + SZ_1M >= pool_length) {
            return false;
        }
        candidate = ALIGN(candidate + SZ_1M,
                          (u64)TF_MODEL_POOL_ALIGNMENT);
        dma_addr = dma_base + candidate;
        if (dma_addr >= window_end) {
            return false;
        }
    }

    bytes = min_t(u64, pool_length - candidate, window_end - dma_addr);
    bytes &= ~((u64)PAGE_SIZE - 1);
    if (!bytes) {
        return false;
    }

    *pool_offset = candidate;
    *available = bytes;
    return true;
}

static bool tf_kbuf_matches_pool(const struct kbuf *kbuf_p,
                                 enum tf_memory_backend backend,
                                 int pool_index,
                                 const struct tf_huge_region *region)
{
    if (kbuf_p->backend != backend) {
        return false;
    }
    if (backend == TF_MEMORY_RESERVED) {
        return kbuf_p->pool_index == pool_index;
    }
    return kbuf_p->huge_region == region;
}

/* Find all free intervals in one physical pool and one 4 GiB address window.
 * buf_list is the allocation source of truth.  This is intentionally not a
 * bump-pointer-only calculation: model arenas may be destroyed in any order,
 * and their interior holes must become reusable without closing the process.
 * The number of model segments is small, so an allocation-free O(n^2) scan is
 * preferable to maintaining a second kernel free-list which could diverge. */
static bool tf_model_find_pool_gap_locked(
    struct tf_device *dev, enum tf_memory_backend backend, int pool_index,
    struct tf_huge_region *region, u64 dma_base, u64 pool_length,
    u32 address_high, u64 *largest_offset, u64 *largest_available,
    u64 *total_capacity)
{
    struct kbuf *kbuf_p;
    u64 cursor;
    u64 window_bytes;
    u64 window_end;
    u64 largest = 0;
    u64 largest_start = 0;
    u64 total = 0;
    int bucket;

    if (!tf_model_address_window_space(dma_base, pool_length, 0,
                                       address_high, &cursor,
                                       &window_bytes) ||
        cursor > U64_MAX - window_bytes) {
        return false;
    }
    window_end = cursor + window_bytes;

    while (cursor < window_end) {
        u64 next_start = window_end;
        u64 next_end = window_end;
        bool cursor_covered = false;

        hash_for_each(dev->buf_list, bucket, kbuf_p, list) {
            u64 allocation_start;
            u64 allocation_end;

            if (!tf_kbuf_matches_pool(kbuf_p, backend, pool_index,
                                      region) ||
                kbuf_p->len <= 0) {
                continue;
            }
            allocation_start = kbuf_p->pool_offset;
            if (allocation_start > U64_MAX - (u64)kbuf_p->len) {
                continue;
            }
            allocation_end = allocation_start + (u64)kbuf_p->len;
            if (allocation_end <= cursor || allocation_start >= window_end) {
                continue;
            }
            if (allocation_start <= cursor) {
                cursor = min_t(u64, ALIGN(allocation_end,
                                          (u64)TF_MODEL_POOL_ALIGNMENT),
                               window_end);
                cursor_covered = true;
                break;
            }
            if (allocation_start < next_start) {
                next_start = allocation_start;
                next_end = allocation_end;
            } else if (allocation_start == next_start &&
                       allocation_end > next_end) {
                next_end = allocation_end;
            }
        }
        if (cursor_covered) {
            continue;
        }

        if (next_start > cursor) {
            u64 available = next_start - cursor;

            available &= ~((u64)PAGE_SIZE - 1);
            if (available) {
                if (total > U64_MAX - available) {
                    total = U64_MAX;
                } else {
                    total += available;
                }
                if (available > largest) {
                    largest = available;
                    largest_start = cursor;
                }
            }
        }
        if (next_start == window_end) {
            break;
        }
        cursor = min_t(u64, ALIGN(next_end,
                                  (u64)TF_MODEL_POOL_ALIGNMENT),
                       window_end);
    }

    if (largest_offset) {
        *largest_offset = largest_start;
    }
    if (largest_available) {
        *largest_available = largest;
    }
    if (total_capacity) {
        *total_capacity = total;
    }
    return largest != 0;
}

static u64 tf_model_window_capacity_locked(struct tf_device *dev, int chip_id,
                                           pid_t tgid, u32 address_high)
{
    struct tf_huge_region *region;
    u64 capacity = 0;
    int i;

    for (i = 0; i < reserveDDRBlcokCnt; ++i) {
        u64 pool_capacity;

        if (reserveDDRBlocks[i].chipId != chip_id ||
            (reserveDDRBlocks[i].isMalloc &&
             reserveDDRBlocks[i].tgid != tgid) ||
            !tf_model_find_pool_gap_locked(
                dev, TF_MEMORY_RESERVED, i, NULL,
                reserveDDRBlocks[i].startPos, reserveDDRBlocks[i].len,
                address_high, NULL, NULL, &pool_capacity)) {
            continue;
        }
        if (capacity > U64_MAX - pool_capacity) {
            return U64_MAX;
        }
        capacity += pool_capacity;
    }

    list_for_each_entry(region, &tf_huge_regions, list) {
        u64 pool_capacity;

        if (region->chip_id != chip_id ||
            (region->owner_tgid != -1 && region->owner_tgid != tgid) ||
            !tf_model_find_pool_gap_locked(
                dev, TF_MEMORY_HUGEPAGE, -1, region, region->dma_addr,
                region->length, address_high, NULL, NULL,
                &pool_capacity)) {
            continue;
        }
        if (capacity > U64_MAX - pool_capacity) {
            return U64_MAX;
        }
        capacity += pool_capacity;
    }
    return capacity;
}

static int tf_ioctl_model_pool_alloc(struct tf_device *dev, void __user *arg)
{
    struct tf_model_pool_alloc request;
    struct tf_huge_region *region;
    struct tf_huge_region *best_region = NULL;
    int best_block_index = -1;
    struct kbuf *kbuf_p = NULL;
    u64 required;
    u64 best_offset = 0;
    u64 best_available = 0;
    u64 allocated;
    u64 capacity;
    u32 selected_high;
    u32 high;
    int i;
    int retval = 0;

    if (copy_from_user(&request, arg, sizeof(request))) {
        return -EFAULT;
    }
    if (request.api_version != TF_MODEL_POOL_API_VERSION || request.flags ||
        request.reserved || request.chip_id < 0 || request.chip_id >= chips ||
        !request.required_length ||
        (request.requested_high != TF_MODEL_POOL_AUTO_HIGH &&
         request.requested_high > 0xffU) ||
        request.required_length > U64_MAX - (PAGE_SIZE - 1)) {
        return -EINVAL;
    }
    required = ALIGN(request.required_length, (u64)PAGE_SIZE);

    mutex_lock(&tf_memory_mutex);
    if (!tf_has_reserved_anchor_locked(request.chip_id, current->tgid)) {
        retval = -ENXIO;
        goto out_unlock;
    }

    if (request.requested_high == TF_MODEL_POOL_AUTO_HIGH) {
        u64 best_capacity = 0;

        selected_high = TF_MODEL_POOL_AUTO_HIGH;
        for (high = 0; high <= 0xffU; ++high) {
            capacity = tf_model_window_capacity_locked(
                dev, request.chip_id, current->tgid, high);
            if (capacity >= required && capacity > best_capacity) {
                best_capacity = capacity;
                selected_high = high;
            }
        }
        if (selected_high == TF_MODEL_POOL_AUTO_HIGH) {
            retval = -ENOSPC;
            goto out_unlock;
        }
    } else {
        selected_high = request.requested_high;
        capacity = tf_model_window_capacity_locked(
            dev, request.chip_id, current->tgid, selected_high);
        if (capacity < required) {
            retval = -ENOSPC;
            goto out_unlock;
        }
    }

    /* Return the largest remaining segment first.  This leaves fewer tiny
     * tails for the userspace arena and makes large weights more likely to
     * fit without crossing a HugePage boundary. */
    for (i = 0; i < reserveDDRBlcokCnt; ++i) {
        u64 offset;
        u64 available;

        if (reserveDDRBlocks[i].chipId != request.chip_id ||
            (reserveDDRBlocks[i].isMalloc &&
             reserveDDRBlocks[i].tgid != current->tgid) ||
            !tf_model_find_pool_gap_locked(
                dev, TF_MEMORY_RESERVED, i, NULL,
                reserveDDRBlocks[i].startPos, reserveDDRBlocks[i].len,
                selected_high, &offset, &available, NULL)) {
            continue;
        }
        if (available > best_available) {
            best_block_index = i;
            best_region = NULL;
            best_offset = offset;
            best_available = available;
        }
    }
    list_for_each_entry(region, &tf_huge_regions, list) {
        u64 offset;
        u64 available;

        if (region->chip_id != request.chip_id ||
            (region->owner_tgid != -1 &&
             region->owner_tgid != current->tgid) ||
            !tf_model_find_pool_gap_locked(
                dev, TF_MEMORY_HUGEPAGE, -1, region, region->dma_addr,
                region->length, selected_high, &offset, &available,
                NULL)) {
            continue;
        }
        if (available > best_available) {
            best_region = region;
            best_block_index = -1;
            best_offset = offset;
            best_available = available;
        }
    }
    if (!best_region && best_block_index < 0) {
        retval = -ENOSPC;
        goto out_unlock;
    }

    allocated = min_t(u64, required, best_available);
    allocated = min_t(u64, allocated, (u64)INT_MAX);
    allocated &= ~((u64)PAGE_SIZE - 1);
    if (!allocated) {
        retval = -ENOSPC;
        goto out_unlock;
    }

    kbuf_p = kzalloc(sizeof(*kbuf_p), GFP_KERNEL);
    if (!kbuf_p) {
        retval = -ENOMEM;
        goto out_unlock;
    }
    kbuf_p->len = (int)allocated;
    kbuf_p->owner_tgid = current->tgid;
    kbuf_p->pool_offset = best_offset;
    kbuf_p->is_model_pool = true;
    if (best_region) {
        kbuf_p->phy_addr = best_region->dma_addr + best_offset;
        kbuf_p->cpu_phys_addr = best_region->physical_addr + best_offset;
        kbuf_p->backend = TF_MEMORY_HUGEPAGE;
        kbuf_p->huge_region = best_region;
        kbuf_p->pool_previous_offset = best_region->offset;
        kbuf_p->pool_index = -1;
        best_region->offset = max_t(u64, best_region->offset,
                                    best_offset + allocated);
        best_region->owner_tgid = current->tgid;
    } else {
        struct ReserveDDRBlock *block = &reserveDDRBlocks[best_block_index];

        kbuf_p->phy_addr = block->startPos + best_offset;
        kbuf_p->cpu_phys_addr = kbuf_p->phy_addr;
        kbuf_p->backend = TF_MEMORY_RESERVED;
        kbuf_p->huge_region = NULL;
        kbuf_p->pool_previous_offset = block->offset;
        kbuf_p->pool_index = best_block_index;
        block->offset = max_t(u64, block->offset,
                              best_offset + allocated);
        block->isMalloc = true;
        block->tgid = current->tgid;
    }
    kbuf_p->mmap_id = dev->mmap_id_counter++;
    if (dev->mmap_id_counter == 9000) {
        dev->mmap_id_counter = 11000;
    }

    retval = tf_cache_invalidate_dma_range(dev, request.chip_id,
                                           kbuf_p->phy_addr, allocated);
    if (retval) {
        tf_rollback_kbuf(kbuf_p);
        kfree(kbuf_p);
        goto out_unlock;
    }
    hash_add(dev->buf_list, &kbuf_p->list,
             kbuf_p->mmap_id & TF_BUF_HASHMASK);

    request.allocated_length = allocated;
    request.physical_addr = kbuf_p->cpu_phys_addr;
    request.dma_addr = kbuf_p->phy_addr;
    request.address_high = selected_high;
    request.mmap_id = kbuf_p->mmap_id;
    if (copy_to_user(arg, &request, sizeof(request))) {
        hash_del(&kbuf_p->list);
        tf_rollback_kbuf(kbuf_p);
        kfree(kbuf_p);
        retval = -EFAULT;
        goto out_unlock;
    }

    printk(KERN_INFO
           "tfacc: model pool segment chip %d high 0x%x dma 0x%llx len 0x%llx mmap %u\n",
           request.chip_id, selected_high,
           (unsigned long long)request.dma_addr,
           (unsigned long long)request.allocated_length, request.mmap_id);

out_unlock:
    mutex_unlock(&tf_memory_mutex);
    return retval;
}

/* Rebuild the legacy high-water mark and owner flags after removing an
 * arbitrary model segment.  The bump allocator still uses offset, while the
 * model allocator scans buf_list and can also reuse holes below this mark. */
static void tf_recompute_pool_state_locked(
    struct tf_device *dev, enum tf_memory_backend backend, int pool_index,
    struct tf_huge_region *region)
{
    struct kbuf *kbuf_p;
    u64 high_water = 0;
    pid_t owner_tgid = -1;
    bool has_allocations = false;
    int bucket;

    hash_for_each(dev->buf_list, bucket, kbuf_p, list) {
        u64 allocation_end;

        if (!tf_kbuf_matches_pool(kbuf_p, backend, pool_index, region) ||
            kbuf_p->len <= 0 ||
            kbuf_p->pool_offset > U64_MAX - (u64)kbuf_p->len) {
            continue;
        }
        allocation_end = kbuf_p->pool_offset + (u64)kbuf_p->len;
        high_water = max_t(u64, high_water, allocation_end);
        if (!has_allocations) {
            owner_tgid = kbuf_p->owner_tgid;
        }
        has_allocations = true;
    }

    if (backend == TF_MEMORY_RESERVED && pool_index >= 0 &&
        pool_index < reserveDDRBlcokCnt) {
        struct ReserveDDRBlock *block = &reserveDDRBlocks[pool_index];

        block->offset = high_water;
        block->isMalloc = has_allocations;
        block->tgid = has_allocations ? owner_tgid : -1;
    } else if (backend == TF_MEMORY_HUGEPAGE && region) {
        region->offset = high_water;
        region->owner_tgid = has_allocations ? owner_tgid : -1;
    }
}

static int tf_ioctl_model_pool_free(struct tf_device *dev, void __user *arg)
{
    struct tf_model_pool_free request;
    struct tf_huge_region *region;
    struct kbuf *kbuf_p;
    enum tf_memory_backend backend;
    int pool_index;
    int retval = -ENOENT;

    if (copy_from_user(&request, arg, sizeof(request))) {
        return -EFAULT;
    }
    if (request.api_version != TF_MODEL_POOL_API_VERSION || request.flags ||
        request.reserved || !request.mmap_id || request.mmap_id > INT_MAX) {
        return -EINVAL;
    }

    mutex_lock(&tf_memory_mutex);
    hash_for_each_possible(dev->buf_list, kbuf_p, list,
                           request.mmap_id & TF_BUF_HASHMASK) {
        if (kbuf_p->mmap_id != request.mmap_id) {
            continue;
        }
        if (kbuf_p->owner_tgid != current->tgid) {
            retval = -EPERM;
            goto out_unlock;
        }
        if (!kbuf_p->is_model_pool) {
            retval = -EINVAL;
            goto out_unlock;
        }

        backend = kbuf_p->backend;
        pool_index = kbuf_p->pool_index;
        region = kbuf_p->huge_region;
        hash_del(&kbuf_p->list);
        kfree(kbuf_p);
        tf_recompute_pool_state_locked(dev, backend, pool_index, region);
        retval = 0;
        goto out_unlock;
    }

out_unlock:
    mutex_unlock(&tf_memory_mutex);
    return retval;
}

static int tf_get_reserve_ddr_blocks(struct ReserveDDRBlock* p) {
    if (efuse) {
        return -1;
    }
    int i;
    for (i = 0; i < reserveDDRBlcokCnt; ++i) {
        copy_to_user((void *) p, &reserveDDRBlocks[i], sizeof(struct ReserveDDRBlock));
        p++;
    }
    return 0;
}

/* app lock */
static struct mutex app_mutex;
static spinlock_t app_spin_lock;

#define MAX_APP_LOCK_RECORDS 500
// Parameter 'size' for an ioctl code is limited with (16K -1).
static struct tf_lock_record app_lock_records[MAX_APP_LOCK_RECORDS];
static int curRecordOffset = 0;
static int tfaccRecordOffsets[MAX_TFACC_CNT];

// 记录一个TFACC的使用情况
// 分为两部分
// 第一部分是一个循环队列，记录过去最多300秒之内，每一秒的占用率
// 第二部分是还没有进入队列的信息
// 这里面事件记录均使用MS为单位
#define MAX_RECORD_SECONDS 300
#define RECORD_QUEUE_LEN 305
struct tf_use_record {
    // 这部分是循环队列, startTime[x]往后一秒内的使用率为percent[x] / 1000.0
    unsigned int percent[MAX_RECORD_SECONDS + 10];
    unsigned int startTime[MAX_RECORD_SECONDS + 10];
    int head, len;

    unsigned int lastTimeInQueue; // 即将计入队列的最早时间，这个数必须是1000的倍数。如果lastTimeInQueue = 2000，代表<=1999的数据都已经被计入队列了
    unsigned int lastRecordTime; // 最后一次记录的时间
    unsigned int lastUseTime; // 在[lastTimeInQueue, lastRecordTime]这段区间内有多久被锁定
    int isHolding;            // 是否还在持有
};
struct tf_use_record useRecords[MAX_TFACC_CNT];

static void initRecords(void) {
    if (efuse) {
        return;
    }
    int i = 0;
    for (i = 0; i < MAX_APP_LOCK_RECORDS; ++i) {
        app_lock_records[i].pid = 0;
        app_lock_records[i].tgid = 0;
        app_lock_records[i].isHolding = 0;
        app_lock_records[i].lockTime = 0;
        app_lock_records[i].unlockTime = 0;
        app_lock_records[i].tfaccID = -1;
    }

    for (i = 0; i < MAX_TFACC_CNT; ++i) {
        tfaccRecordOffsets[i] = 0;
    }

    for (i = 0; i < MAX_TFACC_CNT; i++) {
        useRecords[i].head = 0;
        useRecords[i].len = 0;
        useRecords[i].isHolding = 0;
        useRecords[i].lastTimeInQueue = 0;
        useRecords[i].lastRecordTime = 0;
        useRecords[i].lastUseTime = 0;
    }

    curRecordOffset = 0;
}

static void updateUseRecord(int tfaccID, int lastHolding, unsigned long curMs) {
    // 处理使用记录
    // 1. 先把超过300秒的记录都删掉
    struct tf_use_record *record = &useRecords[tfaccID];
    while (record->len > 0 && (curMs - record->startTime[record->head]) / 1000 + 1 > MAX_RECORD_SECONDS) {
        record->head = (record->head + 1) % RECORD_QUEUE_LEN;
        record->len--;
    }

    // 2. 处理LastRecordTime到当前时间
    if (record->isHolding) {
        // 如果最后处于锁定状态
        if ((curMs - record->lastTimeInQueue) / 1000 + 1 > MAX_RECORD_SECONDS) {
            // 上次记录已经过去很久了，调整到需要计入队列的最晚事件
            int x = (curMs / 1000 * 1000 - MAX_RECORD_SECONDS * 1000 + 1000);
            record->lastTimeInQueue = (x < 0) ? 0 : x;
            record->lastRecordTime = record->lastTimeInQueue;
            record->lastUseTime = 0;
        }

        while (record->lastTimeInQueue < curMs / 1000 * 1000) {
            int pos = (record->head + record->len) % RECORD_QUEUE_LEN;
            record->startTime[pos] = record->lastTimeInQueue;
            record->percent[pos] = (record->lastUseTime + (1000 - record->lastRecordTime % 1000));
            record->len++;

            record->lastTimeInQueue += 1000;
            record->lastRecordTime = record->lastTimeInQueue;
            record->lastUseTime = 0;
        }

        record->lastUseTime += (curMs - record->lastRecordTime);
        record->lastRecordTime = curMs;
    } else {
        // 如果最后处于非锁定状态
        if ((curMs - record->lastTimeInQueue) / 1000 + 1 > MAX_RECORD_SECONDS) {
            // 上次记录已经过去很久了.. 忽略
            record->lastTimeInQueue = curMs / 1000 * 1000;
            record->lastRecordTime = curMs;
            record->lastUseTime = 0;
        } else {
            // 上次记录的时间在300秒之内
            if (curMs / 1000 * 1000 == record->lastTimeInQueue) {
                // 上次记录还在1秒内
                record->lastRecordTime = curMs;
            } else {
                // 上次记录的时间超过1秒了，放入队列
                int pos = (record->head + record->len) % RECORD_QUEUE_LEN;
                record->startTime[pos] = record->lastTimeInQueue;
                record->percent[pos] = record->lastUseTime;
                record->len++;

                record->lastTimeInQueue = curMs / 1000 * 1000;
                record->lastRecordTime = curMs;
                record->lastUseTime = 0;
            }
        }
    }

    // 更新最终状态
    record->isHolding = lastHolding;
}

static void insertAppLockRecord(struct tf_device* dev, int tfaccID) {
    if (efuse) {
        return;
    }
    // 更新app 使用tfacc记录
    app_lock_records[curRecordOffset].pid = current->pid;
    app_lock_records[curRecordOffset].tgid = current->tgid;
    app_lock_records[curRecordOffset].lockTime = jiffies;
    app_lock_records[curRecordOffset].unlockTime = jiffies;
    app_lock_records[curRecordOffset].isHolding = 1;
    app_lock_records[curRecordOffset].tfaccID = tfaccID;

    dev->holdTFACCPid[tfaccID] = current->pid;
    dev->holdTFACCTgid[tfaccID] = current->tgid;

    tfaccRecordOffsets[tfaccID] = curRecordOffset;
    curRecordOffset = (curRecordOffset + 1) % MAX_APP_LOCK_RECORDS;

    updateUseRecord(tfaccID, 1, jiffies_to_msecs(jiffies));
}

static void finishAppLockRecord(struct tf_device* dev, int tfaccID) {
    if (efuse) {
        return;
    }
    int offset = tfaccRecordOffsets[tfaccID];

    dev->holdTFACCPid[tfaccID] = -1;
    dev->holdTFACCTgid[tfaccID] = -1;
    app_lock_records[offset].unlockTime = jiffies;
    app_lock_records[offset].isHolding = 0;

    // curRecordOffset = (curRecordOffset + 1) % MAX_APP_LOCK_RECORDS;
    updateUseRecord(tfaccID, 0, jiffies_to_msecs(jiffies));
}

static int tf_app_try_lock(struct tf_device* dev, int* p) {
    if (efuse) {
        return -1;
    }
    int sleepUs = 0;
    int tfaccID = -1;
    int lock_pid = -1;
    __get_user(sleepUs, p);
    p++;
    __get_user(tfaccID, p);

    if (tfaccID < 0 || tfaccID >= MAX_TFACC_CNT) {
        DPRINTK("TFACCID is not valid: %d\n", tfaccID);
        return -EBADMSG;
    }

    long uWait = (long) sleepUs;
    // DPRINTK("Try lock, pid: %d, tgid: %d, for tfacc: %d \n", current->pid, current->tgid, tfaccID);
    // DPRINTK("Current lock holder pid: %d, tgid: %d, tfacc: %d \n",
    // dev->holdTFACCPid[tfaccID], dev->holdTFACCTgid[tfaccID], tfaccID);

    // 如果已经持有锁，直接返回成功
    if (dev->holdTFACCTgid[tfaccID] == current->tgid) {
        DPRINTK("ALREADY HOLD TFACC, %d, tfacc: %d\n", current->pid, tfaccID);
        return 0;
    }

    while (true) {
        // 尝试进入临界区
        // DPRINTK("[ENTERING ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        mutex_lock(&app_mutex);
        // DPRINTK("[ENTERED ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        if (dev->holdTFACCTgid[tfaccID] == current->tgid) {
            lock_pid = current->pid;
        } else if (dev->holdTFACCPid[tfaccID] == -1) {
            lock_pid = current->pid;
            insertAppLockRecord(dev, tfaccID);
        }
        mutex_unlock(&app_mutex);
        // DPRINTK("[QUIT ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);

        // 检查是否拿到TFACC
        if (lock_pid == current->pid) {
            // DPRINTK("GOT TFACC SUCC: %d, tfacc: %d\n", current->pid, tfaccID);
            return 0;
        }

        // 如果只是尝试一次，那么失败立即返回
        if (sleepUs < 0) {
            DPRINTK("FAIL to get tfacc once: %d\n", current->pid);
            return -EBADMSG;
        }

        // 如果 = 0 ，表示不断重试
        if (sleepUs == 0) {
            // udelay(10);
            DPRINTK("Not supported wait until: %d\n", current->pid);
            return -EBADMSG;
        } else { // 不断重试, 直到时间
            udelay(10);
            uWait -= 10;
            if (uWait < 0) {
                DPRINTK("FAIL to get tfacc for useconds: %d pid: %d, tgid: %d, tfacc: %d\n",
                        sleepUs, current->pid, current->tgid, tfaccID);
                return -EBADMSG;
            }
        }
    }
    return 0;
}

static int tf_app_try_unlock(struct tf_device* dev, int* p) {
    if (efuse) {
        return -1;
    }
    int tfaccID = -1;
    __get_user(tfaccID, (int*) (p) );

    if (tfaccID < 0 || tfaccID >= MAX_TFACC_CNT) {
        DPRINTK("TFACCID is not valid: %d\n", tfaccID);
        return -EBADMSG;
    }

    // DPRINTK("Try unlock, pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
    // DPRINTK("Current lock holder pid: %d, tgid: %d, tfaccID: %d \n",
    // dev->holdTFACCPid[tfaccID], dev->holdTFACCTgid[tfaccID], tfaccID);

    {
        // 正常退出判断
        // DPRINTK("[ENTERING ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        mutex_lock(&app_mutex);
        // DPRINTK("[ENTERED ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
        if (dev->holdTFACCTgid[tfaccID] == current->tgid) {
            finishAppLockRecord(dev, tfaccID);
        }
        mutex_unlock(&app_mutex);
        // DPRINTK("[QUIT ZONE], pid: %d, tgid: %d, tfaccID: %d \n", current->pid, current->tgid, tfaccID);
    }

    // DPRINTK("Current lock holder pid: %d, tgid: %d, tfaccID: %d \n",
    // dev->holdTFACCPid[tfaccID], dev->holdTFACCTgid[tfaccID], tfaccID);
    return 0;
}

static int tf_app_release_tgid_lock(struct tf_device* dev) {
    if (efuse) {
        return -1;
    }
    int i;
    DPRINTK("Releasing, pid: %d, tgid: %d \n", current->pid, current->tgid);
    // DPRINTK("Current lock holder pid: %d, tgid: %d \n", dev->holdAppLockPid, dev->holdAppLockTgid);

    // 意味着进程组退出, 需要退出pid, 以及检查当前持有锁的tgid
    {
        // DPRINTK("TG [ENTERING ZONE], pid: %d, tgid: %d\n", current->pid, current->tgid);
        mutex_lock(&app_mutex);
        // DPRINTK("TG [ENTERED ZONE], pid: %d, tgid: %d\n", current->pid, current->tgid);

        for (i = 0; i < MAX_TFACC_CNT; ++i) {
            if (dev->holdTFACCPid[i] == current->pid) {
                finishAppLockRecord(dev, i);
            } else {
                if (dev->holdTFACCTgid[i] == current->tgid) {
                    finishAppLockRecord(dev, i);
                }
            }
        }
        mutex_unlock(&app_mutex);
        // DPRINTK("[QUIT ZONE], pid: %d, tgid: %d. \n", current->pid, current->tgid);
        // DPRINTK("TG [QUIT LOCK], pid: %d, tgid: %d\n", current->pid, current->tgid);
    }

    for (i = 0; i < MAX_TFACC_CNT; ++i) {
        DPRINTK("Current lock holder for tfacc: %d, pid: %d, tgid: %d \n",
                i, dev->holdTFACCPid[i], dev->holdTFACCTgid[i]);
    }
    // DPRINTK("Current lock holder pid: %d, tgid: %d \n", dev->holdAppLockPid, dev->holdAppLockTgid);
    return 0;
}

/// 获得所有锁获取记录
static int tf_get_app_lock_records(struct tf_lock_record* p) {
    if (efuse) {
        return -1;
    }
    int i;
    for (i = 0; i < MAX_APP_LOCK_RECORDS; ++i) {
        if (app_lock_records[i].isHolding) app_lock_records[i].unlockTime = jiffies;
        copy_to_user((void *) p, &app_lock_records[i], sizeof(struct tf_lock_record));
        p++;
    }
    return 0;
}

// 获取使用率状态
// p[0]: 15s之内的总使用率
// p[1]: 60s之内的总使用率
// p[2]: 300s之内的总使用率
// p[3 ~ 34]: p[i]代表(i - 3)号tfacc在15s内的使用率，如果为-1代表这个tfacc不存在，否则为0~1000之间的数代表利用率 * 1000
static int tf_get_app_usage(int *pp) {
    int i, j;
    int p[MAX_TFACC_CNT + 5];
    for (i = 0; i < MAX_TFACC_CNT + 3; i++) {
        __put_user(-1, pp + i);
        p[i] = -1;
    }
    if (efuse) {
        return -1;
    }
    p[0] = p[1] = p[2] = 0;

    for (i = 0; i < chips * 8; i++) {
        p[3 + i] = 0;
        int s1 = 0, s2 = 0;
        unsigned int curMs = jiffies_to_msecs(jiffies);
        updateUseRecord(i, useRecords[i].isHolding, curMs);
        struct tf_use_record *record = &useRecords[i];
        for (j = 0; j < record->len; j++) {
            int pos = (record->head + j) % RECORD_QUEUE_LEN;
            int gap = (curMs - record->startTime[pos]) / 1000 + 1;
            if (gap <= 15) {
                p[3 + i] += record->percent[pos];
            }
            if (gap <= 60) {
                s1 += record->percent[pos];
            }
            if (gap <= 300) {
                s2 += record->percent[pos];
            }
        }
        p[3 + i] += record->lastUseTime;
        p[3 + i] *= 1000;
        p[3 + i] /= (14000 + (record->lastRecordTime - record->lastTimeInQueue));
        p[0] += p[3 + i];

        s1 *= 1000;
        s1 /= (59000 + (record->lastRecordTime - record->lastTimeInQueue));

        s2 *= 1000;
        s2 /= (299000 + (record->lastRecordTime - record->lastTimeInQueue));

        p[1] += s1;
        p[2] += s2;
    }

    for (i = 0; i < MAX_TFACC_CNT + 3; i++) {
        __put_user(p[i], pp + i);
    }
    return 0;
}

/* pids using this driver, TODO use a struct to collect process info */
struct mutex pid_mutex;
#define MAXPINLIST 64
struct tf_app_info app_infos[MAXPINLIST];

void push_pid(void) {
    if (efuse) {
        return;
    }
    int pid = current->pid;
    int i = 0;

    mutex_lock(&pid_mutex);
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].pid == pid) {
            mutex_unlock(&pid_mutex);
            return;
        }
    }
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].pid == -1) {
            app_infos[i].pid = pid;
            app_infos[i].tgid = current->tgid;
            mutex_unlock(&pid_mutex);
            return;
        }
    }
    mutex_unlock(&pid_mutex);
    return;
}

void pop_pid(void) {
    if (efuse) {
        return;
    }
    int i = 0;
    int pid = current->pid;

    mutex_lock(&pid_mutex);
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].pid == pid) {
            app_infos[i].pid = -1;
            app_infos[i].tgid = -1;
        }
    }
    mutex_unlock(&pid_mutex);
    return;
}

void pop_tgid(void) {
    if (efuse) {
        return;
    }
    int i = 0;

    mutex_lock(&pid_mutex);
    for (i = 0; i < MAXPINLIST; ++i) {
        if (app_infos[i].tgid == current->tgid) {
            app_infos[i].pid = -1;
            app_infos[i].tgid = -1;
        }
    }
    mutex_unlock(&pid_mutex);
    return;
}

void get_pids(int* p) {
    if (efuse) {
        return;
    }
    int offset = 0;

    mutex_lock(&pid_mutex);
    for (offset = 0; offset < MAXPINLIST; ++offset) {
        if (app_infos[offset].pid >= 0) {
            __put_user(app_infos[offset].pid, p);
            p++;
        }
    }
    __put_user(-1, p);

    mutex_unlock(&pid_mutex);
    return;
}

static void init_pids(void) {
    if (efuse) {
        return;
    }
    // init pids
    int i = 0;
    for (i = 0; i < MAXPINLIST; ++i) {
        app_infos[i].pid = -1;
        app_infos[i].tgid = -1;
    }
    return;
}

static struct tf_device * tf_create_and_init_device(int device_id_counter) {
    int retval;
    struct tf_device * dev;
    int c;

    if (efuse) {
        return dev;
    }

    DPRINTK("ENTER\n");

    /* dev zeroed in alloc_etherdev */
    if (!(dev = (struct tf_device *) kzalloc(sizeof(struct tf_device), GFP_KERNEL))) {
        DPRINTK("failed to alloc tf_device\n");
        retval = -ENOMEM;
        goto fail_alloc_device;
    }

    spin_lock_init(&dev->lock);
    dev->isBusy = 0;
    // dev->holdAppLockPid = -1;
    for (c = 0; c < MAX_TFACC_CNT; ++c) {
        dev->holdTFACCPid[c] = -1;
        dev->holdTFACCTgid[c] = -1;
    }
    DPRINTK("dev tfacc inited\n");

    DPRINTK("isBusy = %d\n", dev->isBusy);
    tf_dev = dev;
    dev->mmap_id_counter = 1;
    hash_init(dev->buf_list);

    for (c = 0; c < chips; c++) {
        int i;
        unsigned long long gap = chipGap * c;
        unsigned int TFACC_BASE[TFACC_REG_CNT] = {TFACC0_BASE, TFACC1_BASE, TFACC2_BASE, TFACC3_BASE,
                                                  TFACCLITE0_BASE, TFACCLITE1_BASE, TFACCLITE2_BASE, TFACCLITE3_BASE};
        unsigned int TFACC_CACHE_BASE[TFACC_REG_CNT] = {
                TFACC0_CACHE_BASE, TFACC1_CACHE_BASE, TFACC2_CACHE_BASE, TFACC3_CACHE_BASE,
                TFACCLITE0_CACHE_BASE, TFACCLITE1_CACHE_BASE, TFACCLITE2_CACHE_BASE, TFACCLITE3_CACHE_BASE
        };
        for (i = 0; i < TFACC_REG_CNT; i++) {
            unsigned long long base = gap + TFACC_BASE[i];
            int index = i + c * TFACC_REG_CNT;
            dev->ioreg[index] = ioremap(base, DEVICE_IO_LENGTH);
            if (dev->ioreg[index] == NULL || (unsigned long long)dev->ioreg[index] == 0xFFFFFFFF) {
                DPRINTK("failed to map io reg!\n");
                goto fail_remap;
            }
            DPRINTK("version: 0x%x\n", *(volatile unsigned int *)(dev->ioreg[index]));
            dev->reg_buf[index].phy_addr = base;
            dev->reg_buf[index].kernel_addr = dev->ioreg[index];
            dev->reg_buf[index].len = DEVICE_IO_LENGTH;
            dev->reg_buf[index].mmap_id = (index == 0 ? 0 : REG2ID + (index - 1));

            base = gap + TFACC_CACHE_BASE[i];
            dev->ioreg_cache[index] = ioremap(base, DEVICE_IO_LENGTH);
            if (dev->ioreg_cache[index] == NULL || (unsigned long long)dev->ioreg_cache[index] == 0xFFFFFFFF) {
                DPRINTK("failed to map io reg_cache!\n");
                goto fail_remap;
            }
            dev->cache_reg_buf[index].phy_addr = base;
            dev->cache_reg_buf[index].kernel_addr = dev->ioreg_cache[index];
            dev->cache_reg_buf[index].len = DEVICE_IO_LENGTH;
            dev->cache_reg_buf[index].mmap_id = CACHEREGID + index;
        }
    }

    DPRINTK("EXIT, succeed\n");
    return dev;

    fail_remap:
    kfree(dev);
    dev = NULL;

    fail_alloc_device:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return NULL;
}

static void tf_remove_device_buf(struct tf_device * dev) {
    struct kbuf *kbuf_p;
    struct hlist_node *tmp;
    int bucket;

    if (efuse) {
        return;
    }

    DPRINTK("ENTER\n");
    hash_for_each_safe(dev->buf_list, bucket, tmp, kbuf_p, list) {
        hash_del(&kbuf_p->list);
        kfree(kbuf_p);
    }
    DPRINTK("EXIT, succeed\n");
}

static void tf_remove_tgid_buf(struct tf_device *dev, pid_t tgid)
{
    struct kbuf *kbuf_p;
    struct hlist_node *tmp;
    int bucket;

    hash_for_each_safe(dev->buf_list, bucket, tmp, kbuf_p, list) {
        if (kbuf_p->owner_tgid != tgid) {
            continue;
        }
        hash_del(&kbuf_p->list);
        kfree(kbuf_p);
    }
}

static void tf_reset_memory_owner(pid_t tgid, bool reset_all)
{
    struct tf_huge_region *region;
    int i;

    for (i = 0; i < reserveDDRBlcokCnt; ++i) {
        if (reset_all || reserveDDRBlocks[i].tgid == tgid) {
            reserveDDRBlocks[i].isMalloc = false;
            reserveDDRBlocks[i].offset = 0;
            reserveDDRBlocks[i].tgid = -1;
        }
    }
    list_for_each_entry(region, &tf_huge_regions, list) {
        if (reset_all || region->owner_tgid == tgid) {
            region->owner_tgid = -1;
            region->offset = 0;
        }
    }
}

static int tf_create_and_init_cdev(struct tf_device * dev, int device_id) {
    if (efuse) {
        return -1;
    }
    dev_t chrdev;
    int retval;

    assert(dev != NULL);

    // register a char device
    dev->minor = device_id;
    if ((retval = alloc_chrdev_region(&chrdev, dev->minor, 1, "thinkforce")) < 0) {
        DPRINTK("failed to alloc chrdev\n");
        goto origin;
    }

    dev->major = MAJOR(chrdev);
    DPRINTK("Major: %d, Minor:%d\n", dev->major, dev->minor);
    dev->device = device_create(thinkforce_class, dev->dma_device, chrdev,
                                NULL, "thinkforce" "%d", dev->minor);
    if (IS_ERR(dev->device)) {
        retval = PTR_ERR(dev->device);
        dev->device = NULL;
        DPRINTK("failed to create cdev\n");
        goto after_alloc;
    }

    cdev_init(&dev->cdev, &tf_device_ops);
    dev->cdev.owner = THIS_MODULE;
    if ((retval = cdev_add(&dev->cdev, chrdev, 1))) {
        DPRINTK("failed to add cdev %d\n", device_id);
        goto after_create;
    }
    dev_set_drvdata(dev->device, dev);

/*
    DPRINTK("dev->device->dma_mem: 0x%p\n", dev->device->dma_mem);
*/
    DPRINTK("EXIT, succeed\n");
    return 0;

    after_create:
    device_destroy(thinkforce_class, chrdev);

    after_alloc:
    unregister_chrdev_region(chrdev, 1);

    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);

    return retval;
}

static void tf_remove_cdev(struct tf_device *dev) {
    if (efuse) {
        return;
    }
    dev_t chrdev;
#if 0
    volatile unsigned int *cache0_apb = ioremap(TFACC0_CACHE_BASE, DEVICE_IO_LENGTH);
    int i;
    for (i = 0; i < 64; i++) {
        DPRINTK("cache0_apb[0x%x] = 0x%x\n", 0x100 + i * 4, cache0_apb[0x40 + i]);
    }
    iounmap(cache0_apb);

    volatile unsigned int *cache1_apb = ioremap(TFACC1_CACHE_BASE, DEVICE_IO_LENGTH);
    for (i = 0; i < 64; i++) {
        DPRINTK("cache1_apb[0x%x] = 0x%x\n", 0x100 + i * 4, cache1_apb[0x40 + i]);
    }
    iounmap(cache1_apb);

    volatile unsigned int *reg = ioremap(TFACC0_CACHE_BASE, DEVICE_IO_LENGTH);
    for (i = 0; i < 128; i++) {
        DPRINTK("reg[0x%x] = 0x%x\n", i * 4, reg[i]);
    }

    DPRINTK("reg[0x004] = 0x%x\n", reg[0x004 / 4]);
    DPRINTK("reg[0x198] = 0x%x\n", reg[0x198 / 4]);
    DPRINTK("reg[0x19C] = 0x%x\n", reg[0x19C / 4]);
    DPRINTK("reg[0x1A0] = 0x%x\n", reg[0x1A0 / 4]);
    DPRINTK("reg[0x1A4] = 0x%x\n", reg[0x1A4 / 4]);
    DPRINTK("reg[0x1A8] = 0x%x\n", reg[0x1A8 / 4]);
    DPRINTK("reg[0x1AC] = 0x%x\n", reg[0x1AC / 4]);
    DPRINTK("reg[0x1B0] = 0x%x\n", reg[0x1B0 / 4]);
    DPRINTK("reg[0x1B4] = 0x%x\n", reg[0x1B4 / 4]);
    DPRINTK("reg[0x1B8] = 0x%x\n", reg[0x1B8 / 4]);
    DPRINTK("reg[0x1BC] = 0x%x\n", reg[0x1BC / 4]);
    DPRINTK("reg[0x1C0] = 0x%x\n", reg[0x1C0 / 4]);
    DPRINTK("reg[0x1C4] = 0x%x\n", reg[0x1C4 / 4]);
#endif
    DPRINTK("ENTER\n");

    //remove cdev
    chrdev = MKDEV(dev->major, dev->minor);
    cdev_del(&dev->cdev);
    device_destroy(thinkforce_class, chrdev);
    unregister_chrdev_region(chrdev, 1);

    DPRINTK("EXIT, succeed\n");
}

void tfacc_full_swrst(unsigned long long, unsigned long long);
void tfacc_lite_swrst(unsigned long long);
void tfacc_enable_one_cache(unsigned long long);

static int tf_release(struct inode *inode, struct file *filp) {
    bool last_user;

    if (efuse) {
        return -1;
    }
    struct tf_device *dev = container_of(inode->i_cdev, struct tf_device, cdev);
    unsigned long long gap;

    DPRINTK("release ENTER\n");
    /* Serialize the last-close decision with open() and allocation. */
    mutex_lock(&tf_memory_mutex);
    spin_lock(&dev->lock);
    if (dev->isBusy > 0) {
        dev->isBusy--;
    }
    last_user = dev->isBusy == 0;
    DPRINTK("isBusy = %d\n", dev->isBusy);
    spin_unlock(&dev->lock);

    tf_reset_memory_owner(current->tgid, last_user);
    if (last_user) {
        tf_remove_device_buf(dev);
        dev->mmap_id_counter = 1;
    } else {
        tf_remove_tgid_buf(dev, current->tgid);
    }
    mutex_unlock(&tf_memory_mutex);

    // 进程退出的时候，把整个线程组的锁都释放掉
    // if (current->pid == current->tgid) {
    tf_app_release_tgid_lock(dev);
    pop_tgid();
    // } else {
    // tf_app_try_unlock(dev, NULL);
    // pop_pid();
    // }

    spin_lock(&dev->lock);
    if (last_user && dev->isBusy == 0) {
        int c;
        for (c = chips-1; c>=0; c--) {
            gap = chipGap * c;
            //tfacc_full_swrst(gap + TFACC0_BASE, gap + TFACC1_BASE);
            //tfacc_full_swrst(gap + TFACC2_BASE, gap + TFACC3_BASE);
            tfacc_lite_swrst(gap + TFACCLITE0_BASE);
            tfacc_lite_swrst(gap + TFACCLITE1_BASE);
            tfacc_lite_swrst(gap + TFACCLITE2_BASE);
            tfacc_lite_swrst(gap + TFACCLITE3_BASE);

            tfacc_enable_one_cache(TFACC0_CACHE_BASE);
            tfacc_enable_one_cache(TFACC1_CACHE_BASE);
            tfacc_enable_one_cache(TFACC2_CACHE_BASE);
            tfacc_enable_one_cache(TFACC3_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE0_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE1_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE2_CACHE_BASE);
            tfacc_enable_one_cache(TFACCLITE3_CACHE_BASE);

        }
    }
    spin_unlock(&dev->lock);

    DPRINTK("EXIT, succeed\n");
    return 0;
}

static int tf_mmap(struct file *filp, struct vm_area_struct *vma) {
    if (efuse) {
        return -1;
    }
    struct tf_device *dev = (struct tf_device *)filp->private_data;
    int size = vma->vm_end - vma->vm_start;
    int mmap_id = vma->vm_pgoff;
    dma_addr_t phy_addr = -1;
    phys_addr_t cpu_phys_addr = (phys_addr_t)-1;
    int max_size = 0;
    struct kbuf * kbuf_p;
    void *kernel_addr = NULL;


    DPRINTK("ENTER\n");
    DPRINTK("mapping typeid: %d\n", mmap_id);
/*
    if (!versionIsRight) {
        DPRINTK("SDK's version is wrong.\n");
        return -EINVAL;
    }
*/
    if (mmap_id == SRAM1ID) {
        //sram1
        DPRINTK("prepare mapping sram1\n");
        max_size = dev->sram1_buf.len;
        phy_addr = dev->sram1_buf.phy_addr;
        cpu_phys_addr = phy_addr;
        kernel_addr = dev->sram1_buf.kernel_addr;
    } else if (mmap_id == SRAM2ID) {
        //sram2
        DPRINTK("prepare mapping sram2\n");
        max_size = dev->sram2_buf.len;
        phy_addr = dev->sram2_buf.phy_addr;
        cpu_phys_addr = phy_addr;
        kernel_addr = dev->sram2_buf.kernel_addr;
    } else if (mmap_id == 0) {
        //reg
        DPRINTK("prepare mapping reg\n");
        max_size = dev->reg_buf[0].len;
        phy_addr = dev->reg_buf[0].phy_addr;
        cpu_phys_addr = phy_addr;
        kernel_addr = dev->reg_buf[0].kernel_addr;
    } else if (mmap_id >= REG2ID && mmap_id < REG2ID + TFACC_REG_CNT * chips - 1) {
        DPRINTK("prepare mapping reg2\n");
        max_size = dev->reg_buf[mmap_id - REG2ID + 1].len;
        phy_addr = dev->reg_buf[mmap_id - REG2ID + 1].phy_addr;
        cpu_phys_addr = phy_addr;
        kernel_addr = dev->reg_buf[mmap_id - REG2ID + 1].kernel_addr;
    } else if (mmap_id >= CACHEREGID && mmap_id < CACHEREGID + TFACC_REG_CNT * chips) {
        DPRINTK("prepare mapping cache reg\n");
        max_size = dev->cache_reg_buf[mmap_id - CACHEREGID].len;
        phy_addr = dev->cache_reg_buf[mmap_id - CACHEREGID].phy_addr;
        cpu_phys_addr = phy_addr;
        kernel_addr = dev->cache_reg_buf[mmap_id - CACHEREGID].kernel_addr;
    } else if (mmap_id == REGMAINID) {
        DPRINTK("prepare mapping regMain\n");
        max_size = dev->regMain_buf.len;
        phy_addr = dev->regMain_buf.phy_addr;
        cpu_phys_addr = phy_addr;
        kernel_addr = dev->regMain_buf.kernel_addr;
    } else if (mmap_id < dev->mmap_id_counter) {
        DPRINTK("prepare mapping buf %d\n", mmap_id);
        mutex_lock(&tf_memory_mutex);
        hash_for_each_possible(dev->buf_list, kbuf_p, list,
                               mmap_id & TF_BUF_HASHMASK) {
            if (kbuf_p->mmap_id == mmap_id) {
                max_size = kbuf_p->len;
                phy_addr = kbuf_p->phy_addr;
                cpu_phys_addr = kbuf_p->cpu_phys_addr;
                kernel_addr = kbuf_p->kernel_addr;
                break;
            }
        }
        mutex_unlock(&tf_memory_mutex);
        if (!max_size) {
            return -EINVAL;
        }
    }
    else {
        DPRINTK("invalid offset\n");
        return -EINVAL;
    }

    DPRINTK("size: %d, max_size: %d\n", size, max_size);

    if (size > max_size) {
        DPRINTK("require mmap size too large\n");
        return -EINVAL;
    }

    DPRINTK("mmap dma_addr: 0x%llx, cpu_phys_addr: 0x%llx, size: %d\n",
            (unsigned long long)phy_addr,
            (unsigned long long)cpu_phys_addr, size);
    vma->vm_flags |= VM_LOCKED | VM_DONTEXPAND | VM_DONTDUMP;

    DPRINTK("PAGE_SHIFT: %d\n", PAGE_SHIFT);
    if (!mmap_id || (mmap_id >= REG2ID && mmap_id < REG2ID + TFACC_REG_CNT * chips - 1) ||
        (mmap_id >= CACHEREGID && mmap_id < CACHEREGID + TFACC_REG_CNT * chips)) {
        int r;
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        r = remap_pfn_range(vma, vma->vm_start,
                            cpu_phys_addr >> PAGE_SHIFT, size,
                            vma->vm_page_prot);
        if (r != 0) {
            DPRINTK("remap page range failed: %d\n", r);
            return -ENXIO;
        }
    } else {
        int r;
        vma->vm_pgoff = 0;
        //DPRINTK("kernel_addr: 0x%p, phy_addr: 0x%p\n", kernel_addr, (void *)phy_addr);
        if (isNextUncache == 1) {
            //vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
            vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
            isNextUncache = 0;
        }

        //vma->vm_page_prot = pgprot_cached(vma->vm_page_prot);
        r = remap_pfn_range(vma, vma->vm_start,
                            cpu_phys_addr >> PAGE_SHIFT, size,
                            vma->vm_page_prot);
        if (r < 0) {
            DPRINTK("mmap coherent failed: %d\n", r);
            return -ENXIO;
        }
    }

    DPRINTK("EXIT, succeed\n");
    return 0;
}

void tfacc_full_clap(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int* clk = ioremap(base, TFACC_CLK_LENGTH);
    *(clk + 0x84) |= 0x3;
    iounmap(clk);
}
void tfacc_full_unclap(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int* clk = ioremap(base, TFACC_CLK_LENGTH);
    *(clk + 0x84) &= ~0x3;
    iounmap(clk);
}

/* Direct TFACC traffic includes the legacy tfnn result queues.  tfnn leaves
 * their HIGH registers at zero, so this value must stay in the reserved-DDR
 * window installed during driver initialization. */
static void tfacc_write_direct_address_high(void __iomem *buf, u32 high_addr)
{
    u8 __iomem *regs = buf;
    u32 value;

    value = readl(regs + 0x10);
    value &= ~((0xffU << 16) | (0xffU << 24));
    value |= (high_addr << 16) | (high_addr << 24);
    writel(value, regs + 0x10);
}

/* Operand addresses are extended inside the L1 shared by a TFACC pair.  Only
 * these cache-side fields may change between operators. */
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

    /* Complete the high-address update before the caller starts a command. */
    wmb();
    readl(regs + 0x94);
}

static int tfacc_set_cache_address_high(unsigned long long base, u32 high_addr)
{
    void __iomem *buf;

    if (high_addr > 0xffU) {
        return -ERANGE;
    }

    buf = ioremap(base, TFACC_CLK_LENGTH);
    if (!buf) {
        return -ENOMEM;
    }

    tfacc_write_cache_address_high(buf, high_addr);
    iounmap(buf);
    return 0;
}

void tfacc_full_acp(unsigned long long base, int highAddr) {
    void __iomem *buf;

    if (efuse) {
        return;
    }
    DPRINTK("base = 0x%llx, highAddr = 0x%x\n", base, highAddr);
    buf = ioremap(base, TFACC_CLK_LENGTH);
    if (buf != NULL) {
        u8 __iomem *regs = buf;
        unsigned int temp = readl(regs + 0x10);
        DPRINTK("prot = 0x%x\n", temp);
        temp = (temp & ~(7U << 8)) | (2U << 8);
        temp = (temp & ~(7U << 12)) | (2U << 12);
        writel(temp, regs + 0x10);

        if (base == TFACC3_FULL_ACP_BASE) {
            writel(readl(regs + 0xc4) | (0x3U << 2), regs + 0xc4);
            writel(readl(regs + 0xc8) | (0x7ffffU << 16), regs + 0xc8);
        } else {
            writel(readl(regs + 0x98) | 3U, regs + 0x98);
            writel(readl(regs + 0x9c) | 0x7ffffU, regs + 0x9c);
        }

        /* At initialization direct queues and cache operands both live in the
         * reserved-DDR window.  Runtime switching later changes cache only. */
        tfacc_write_direct_address_high(buf, (u32)highAddr);
        tfacc_write_cache_address_high(buf, (u32)highAddr);
        DPRINTK("prot = 0x%x\n", readl(regs + 0x10));
        iounmap(buf);
    }
}

static int tf_ioctl_set_address_high(struct tf_device *dev, void __user *arg)
{
    static const unsigned long long acp_bases[] = {
        TFACC0_FULL_ACP_BASE,
        TFACC1_FULL_ACP_BASE,
        TFACC2_FULL_ACP_BASE,
        TFACC3_FULL_ACP_BASE,
    };
    struct tf_address_high request;
    int first_tfacc;
    int chip_id;
    int pair_id;
    unsigned long long base;
    int retval;

    if (copy_from_user(&request, arg, sizeof(request))) {
        return -EFAULT;
    }
    if (request.api_version != TF_ADDRESS_API_VERSION || request.flags ||
        request.tfacc_id < 0 ||
        request.tfacc_id >= chips * TFACC_REG_CNT ||
        request.address_high > 0xffU) {
        return -EINVAL;
    }

    chip_id = request.tfacc_id / TFACC_REG_CNT;
    pair_id = (request.tfacc_id % TFACC_REG_CNT) / 2;
    first_tfacc = (request.tfacc_id / 2) * 2;
    base = chipGap * chip_id + acp_bases[pair_id];

    /* app_mutex keeps the ownership check and MMIO update atomic with
     * TF_APP_UNLOCK.  Both cores must be held because they share this L1. */
    mutex_lock(&app_mutex);
    if (dev->holdTFACCTgid[first_tfacc] != current->tgid ||
        dev->holdTFACCTgid[first_tfacc + 1] != current->tgid) {
        retval = -EPERM;
    } else {
        retval = tfacc_set_cache_address_high(base, request.address_high);
    }
    mutex_unlock(&app_mutex);

    return retval;
}

void tfacc_swrst_one_cache(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    //unsigned int cache_num=0;
    int data = 0;
    //volatile unsigned int *offset = apb + (cache_num * (1<<22) / sizeof(unsigned int));
    apb[0xe0 / 4] = 0x0;
    apb[0xf8 / 4] = 0x0;
    apb[0xfc / 4] = 0x0;

    iounmap(apb);
}
void tfacc_enable_one_cache(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    int data = 0;

    apb[0x50 / 4] = 1;
    __asm__ __volatile__ ("dmb sy");
    while (data != 1) {
        data = apb[0x54 / 4];
    }
    apb[0x04 / 4] = apb[0x04 / 4] | (1 << 6);


    iounmap(apb);
}
void tfacc_enable_one_uncache(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);

    apb[0x04 / 4] = apb[0x04 / 4] | (1 << 6);
    apb[0x10 / 4] = 0x100; // 31:0
    apb[0x14 / 4] = 0x0;   // 31:0
    apb[0x30 / 4] = 0x0;   // 63:32
    apb[0x34 / 4] = 0x0;   // 63:32
    apb[0x20 / 4] = 0x100; // 31:0
    apb[0x24 / 4] = 0x0;   // 31:0
    apb[0x40 / 4] = 0x0;   // 63:32
    apb[0x44 / 4] = 0x0;   // 63:32

    apb[0x18 / 4] = 0x0;          // 31:0
    apb[0x1C / 4] = 0xffffffff;   // 31:0
    apb[0x38 / 4] = 0x0;          // 63:32
    apb[0x3C / 4] = 0xffffffff;   // 63:32
    apb[0x28 / 4] = 0x0;          // 31:0
    apb[0x2C / 4] = 0xffffffff;   // 31:0
    apb[0x48 / 4] = 0x0;          // 63:32
    apb[0x4C / 4] = 0xffffffff;   // 63:32


    iounmap(apb);
}

void tfacc_full_enable_interleave(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int *apb = ioremap(base, DEVICE_IO_LENGTH);
    apb[0x04 / 4] = apb[0x04 / 4] | (1 << 12);
    DPRINTK("interleave = 0x%x\n", apb[0x04 / 4]);
    iounmap(apb);
}

void tfacc_lite_swrst(unsigned long long BASE) {
    if (efuse) {
        return;
    }
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    reg[0xff] = 0xc012 | (0x1<<31) | (0x1<<5);
    iounmap(reg);
}
void tfacc_full_swrst(unsigned long long BASE, unsigned long long BASE1) {
    if (efuse) {
        return;
    }
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    unsigned int *reg1 = ioremap(BASE1, DEVICE_IO_LENGTH);

    reg1[0xff] = (0x1<<5);
    reg1[0xfa]= (0x100);
    reg[0xfa] = (0x100);
    reg[0xff] = 0xc012 | (0x1 << 31) | (0x1<<5);
    __asm__ __volatile__ ("dmb sy");


    iounmap(reg);
    iounmap(reg1);
}

void tfacc_lite_enable_cache(unsigned long long BASE, unsigned long long CACHE_BASE) {
    if (efuse) {
        return;
    }
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);

    if (!cacheInit) {
        tfacc_swrst_one_cache(CACHE_BASE);
        reg[0xfa] = 0x1;
        reg[0xff] = 0xc012 | (0x1 << 31) | (0x1<<5);
        ++cacheInitCnt;
        if (cacheInitCnt == 6*chips)
            cacheInit = 1;
    }
    tfacc_enable_one_cache(CACHE_BASE);
    iounmap(reg);
}

void tfacc_full_enable(unsigned long long base);
void tfacc_full_disable(unsigned long long base);

void tfacc_full_enable_cache(unsigned long long BASE, unsigned long long BASE1, unsigned long long CACHE0_BASE, unsigned long long CACHE1_BASE,
                             unsigned long long CLKBASE, unsigned long long CFGBASE) {
    if (efuse) {
        return;
    }
    unsigned int *reg = ioremap(BASE, DEVICE_IO_LENGTH);
    unsigned int *reg1 = ioremap(BASE1, DEVICE_IO_LENGTH);
    unsigned int *cfgreg = ioremap(CFGBASE, TFACC_CLK_LENGTH);
    //unsigned int *clkreg = ioremap(CLKBASE, TFACC_CLK_LENGTH);

#if 1
    if (!cacheInit) {

//        tfacc_full_clap(CLKBASE);
        tfacc_swrst_one_cache(CACHE1_BASE);
//        tfacc_full_unclap(CLKBASE);

//        reg[0xfa] |= 0x1;
//        reg[0xff] = 0xc012 + (0x1 << 31);
//        __asm__ __volatile__ ("dmb sy");
//        reg[0xfa] = 0;
//        reg[0xff] = 0xc012;

//        tfacc_full_clap(CLKBASE);
//        tfacc_full_unclap(CLKBASE);
        tfacc_swrst_one_cache(CACHE0_BASE);

        //cfgreg[0x28/4] = 0x0;
        //cfgreg[0x20/4] = 0x1;

        //Reset Mau and set reset with Cache
        reg1[0xfa]|= (0x1|0x100);
        reg[0xfa] |= (0x1|0x100);

        //Manually reset bufaptr
        reg1[0xff] = (0x1<<5);
        reg[0xff] = (0x1<<5);

        reg1[0xff] |= (0x1<<5);
        reg1[0xff] &= ~(0x1<<5);

        reg[0xff] = 0xc012 | (0x1 << 31) | (0x1<<5);
        __asm__ __volatile__ ("dmb sy");

        reg[0xfa] = 0;
        reg1[0xfa] = 0;

        //if ((cfgreg[0x2c/4]&0x7)!=0) {
        //    DPRINTK("glitch at reset\n");
        //}
        //cfgreg[0x28/4] = 0xffffffff;
        //cfgreg[0x20/4] = 0x0;

//        tfacc_full_disable(CLKBASE);
//        tfacc_full_enable(CLKBASE);


        ++cacheInitCnt;
        if (cacheInitCnt == 6*chips)
            cacheInit = 1;

    }
    iounmap(reg);
    iounmap(reg1);
    iounmap(cfgreg);
    //iounmap(clkreg);
    tfacc_enable_one_cache(CACHE0_BASE);
    tfacc_enable_one_cache(CACHE1_BASE);
#else
    iounmap(reg);
    iounmap(reg1);
        ++cacheInitCnt;
        tfacc_enable_one_cache(CACHE0_BASE);
        tfacc_enable_one_cache(CACHE1_BASE);
#endif


    //tfacc_full_enable_interleave(CACHE0_BASE);
    //tfacc_full_enable_interleave(CACHE1_BASE);
}

void tfacc_full_enable(unsigned long long base) {
    if (efuse) {
        return;
    }
    unsigned int pllstatus;
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);

    if (clk != NULL) {
        *(clk + 0x24) = 0;          // unlock
        if (!chipInit) {
            DPRINTK("Reset tfacc%d pll\n", chipInitCnt);
            *(clk + 0x21) = 1;
            *(clk + 0x22) &= ~(1<<24);
            udelay(400);
            *(clk + 0x22) |= (1<<24);
            udelay(100);
            pllstatus = *(clk+0xf0);
            while ((pllstatus&0x3) != 0x3)
                pllstatus = *(clk+0xf0);
            chipInitCnt++;
            if (chipInitCnt == 2*chips)
                chipInit = 1;
        }
        *(clk + 0x86) |= 6;
        *(clk + 0x82) |= 2;
        *(clk + 0x84) &= ~0x3;
        iounmap(clk);
    }
}

void tfacc_full_checkrstcond(void) {
    if (efuse) {
        return;
    }
    unsigned int* periscfg = ioremap(TFACC0_FULL_ACP_BASE, TFACC_CLK_LENGTH);
    if (needReset == 1) periscfg[0x34/4] = 1;
    int retval;
    retval = periscfg[0x34/4];
    DPRINTK("Flag = %llx\n", retval);
    if (retval == 1) {
        periscfg[0x34/4] = 2;
        skipFullSwRst = 2;
    }
    iounmap(periscfg);
}

void tfacc_full_disable(unsigned long long base) {
    if (efuse) {
        return;
    }
    DPRINTK("ioremap %llx\n", base);
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    DPRINTK("ioremap %llx finish\n", base);

    if (clk != NULL) {
        *(clk + 0x24) = 0;          // unlock
        *(clk + 0x84) |= 0x3;
        if (skipFullSwRst != 2) {
            *(clk + 0x82) &= 0xFFFFFFFD;
            udelay(1);
            *(clk + 0x82) |= 2;
            udelay(10);
        } else {
            DPRINTK("Skip HW Reset\n");
        }
        *(clk + 0x86) &= 0xFFFFFFF9;

        //*(clk + 0x22) &= ~(1<<24);
        iounmap(clk);
    }
}

void tfacc_lite_enable(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    if (clk != NULL) {
        *(clk + 0x24) = 0;
        *(clk + 0x86) |= 0x3C0000;
        *(clk + 0x82) |= 0x600;
        *(clk + 0x84) &= ~(0x3<<9);
        iounmap(clk);
    }
}

void tfacc_lite_disable(unsigned long long base) {
    if (efuse) {
        return;
    }
    volatile unsigned int *clk = ioremap(base, TFACC_CLK_LENGTH);
    if (clk != NULL) {
        *(clk + 0x24) = 0;
        *(clk + 0x84) |= (0x3<<9);
        if (skipFullSwRst !=2) {
            *(clk + 0x82) &= 0xFFFFF9FF;
        } else {
            DPRINTK("Skip HW Reset\n");
        }
        *(clk + 0x86) &= 0xFFC3FFFF;
        iounmap(clk);
    }
}
static void tf_ioctl_clear(struct tf_device * dev) {
    if (efuse) {
        return;
    }
    DPRINTK("ENTER\n");

    mutex_lock(&tf_memory_mutex);
    tf_remove_device_buf(dev);
    tf_reset_memory_owner(0, true);
    dev->mmap_id_counter = 1;
    mutex_unlock(&tf_memory_mutex);

    DPRINTK("EXIT, succeed\n");
}

static void tf_ioctl_reset(struct tf_device * dev) {
    if (efuse) {
        return;
    }
    DPRINTK("ENTER\n");

    hardware_tfacc_reset();

    tfacc_full_enable(TFACC_BL_CLK_BASE);
    tfacc_full_enable(TFACC_BR_CLK_BASE);
    tfacc_lite_enable(TFACC_L_CLK_BASE);
    tfacc_lite_enable(TFACC_R_CLK_BASE);

    tfacc_full_acp(TFACC0_FULL_ACP_BASE, ddrStart >> 32);
    tfacc_full_acp(TFACC1_FULL_ACP_BASE, ddrStart >> 32);
    tfacc_full_acp(TFACC2_FULL_ACP_BASE, ddrStart >> 32);
    tfacc_full_acp(TFACC3_FULL_ACP_BASE, ddrStart >> 32);

    tfacc_full_enable_cache(TFACC0_BASE, TFACC1_BASE, TFACC0_CACHE_BASE, TFACC1_CACHE_BASE, TFACC_BL_CLK_BASE, TFACC0_FULL_ACP_BASE);
    tfacc_full_enable_cache(TFACC2_BASE, TFACC3_BASE, TFACC2_CACHE_BASE, TFACC3_CACHE_BASE, TFACC_BR_CLK_BASE, TFACC1_FULL_ACP_BASE);

    tfacc_lite_enable_cache(TFACCLITE0_BASE, TFACCLITE0_CACHE_BASE);
    tfacc_lite_enable_cache(TFACCLITE1_BASE, TFACCLITE1_CACHE_BASE);
    tfacc_lite_enable_cache(TFACCLITE2_BASE, TFACCLITE2_CACHE_BASE);
    tfacc_lite_enable_cache(TFACCLITE3_BASE, TFACCLITE3_CACHE_BASE);

    DPRINTK("EXIT, succeed\n");
}


static int tf_ioctl_check_version(struct tf_device * dev, void * arg) {
    if (efuse) {
        return -1;
    }
    struct tf_version *io_param;
    int retval;
    int minVersion = 1840;

    DPRINTK("ENTER\n");
    if (!(io_param = (struct tf_version*) kmalloc(sizeof(struct tf_version), GFP_KERNEL))) {
        DPRINTK("fail to alloc io_param\n");
        retval = -ENOMEM;
        goto origin;
    }
    if ((retval = copy_from_user(io_param, (void *) arg, sizeof(struct tf_version)))) {
        DPRINTK("fail to copy io_param from user\n");
        goto after_alloc_io_param;
    }

    //check version
    if (io_param->sdk_version >= minVersion) {
        versionIsRight = true;
        io_param->kernel_version = 20190605;
    } else {
        io_param->kernel_version = -1;
        io_param->excepted_sdk_version = minVersion;
    }

    if ((retval = copy_to_user((void *) arg, io_param, sizeof(struct tf_version)))) {
        DPRINTK("fail to copy io_param to user\n");
        goto after_alloc_io_param;
    }

    kfree(io_param);
    DPRINTK("EXIT, succeed\n");
    return 0;

    after_alloc_io_param:
    kfree(io_param);

    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return retval;
}

static int tf_init_kbuf(struct tf_device * dev,struct kbuf * kbuf_p) {
    struct tf_huge_region *region;
    u64 candidate;
    u64 dma_start;
    u64 dma_end;
    bool has_reserved_anchor = false;
    int i;

    if (efuse) {
        return -1;
    }

    DPRINTK("ENTER\n");

    if (kbuf_p->len <= 0) {
        return -EINVAL;
    }

    for (i = 0; i < reserveDDRBlcokCnt; i++) {
        if (reserveDDRBlocks[i].chipId != dev->useDDR2) {
            continue;
        }
        if (reserveDDRBlocks[i].isMalloc &&
            reserveDDRBlocks[i].tgid == current->tgid) {
            has_reserved_anchor = true;
        }
        if (reserveDDRBlocks[i].offset + kbuf_p->len >
            reserveDDRBlocks[i].len ||
            (reserveDDRBlocks[i].isMalloc &&
             reserveDDRBlocks[i].tgid != current->tgid)) {
            continue;
        }

        kbuf_p->phy_addr = reserveDDRBlocks[i].startPos +
                           reserveDDRBlocks[i].offset;
        kbuf_p->cpu_phys_addr = kbuf_p->phy_addr;
        kbuf_p->pool_offset = reserveDDRBlocks[i].offset;
        kbuf_p->pool_index = i;
        kbuf_p->backend = TF_MEMORY_RESERVED;
        kbuf_p->owner_tgid = current->tgid;
        reserveDDRBlocks[i].offset += kbuf_p->len;
        reserveDDRBlocks[i].isMalloc = true;
        reserveDDRBlocks[i].tgid = current->tgid;
        goto allocated;
    }

    /* tfnn places result queues in its first allocation and programs their
     * legacy high fields as zero.  Keep that per-chip anchor in reserved DDR;
     * only payload allocations after it may fall back to HugeTLB memory. */
    if (!has_reserved_anchor) {
        return -ENOMEM;
    }

    list_for_each_entry(region, &tf_huge_regions, list) {
        if (region->chip_id != dev->useDDR2 ||
            (region->owner_tgid != -1 &&
             region->owner_tgid != current->tgid)) {
            continue;
        }

        candidate = region->offset;
        while (candidate + kbuf_p->len <= region->length) {
            dma_start = region->dma_addr + candidate;
            if (!(dma_start & 0xffffffffULL)) {
                candidate += 1ULL << 20;
                continue;
            }
            dma_end = dma_start + kbuf_p->len - 1;
            if ((dma_start >> 32) != (dma_end >> 32)) {
                candidate += (1ULL << 32) -
                             (dma_start & 0xffffffffULL);
                continue;
            }

            kbuf_p->phy_addr = dma_start;
        kbuf_p->cpu_phys_addr = region->physical_addr + candidate;
        kbuf_p->pool_offset = candidate;
        kbuf_p->pool_previous_offset = region->offset;
        kbuf_p->pool_index = -1;
            kbuf_p->backend = TF_MEMORY_HUGEPAGE;
            kbuf_p->huge_region = region;
            kbuf_p->owner_tgid = current->tgid;
            region->offset = candidate + kbuf_p->len;
            region->owner_tgid = current->tgid;
            goto allocated;
        }
    }

    kbuf_p->phy_addr = 0;
    kbuf_p->cpu_phys_addr = 0;
    kbuf_p->mmap_id = -1;
    return -ENOMEM;

allocated:
    kbuf_p->mmap_id = dev->mmap_id_counter++;
    if (dev->mmap_id_counter == 9000) {
        //中间这一段编号保留，另做他用
        dev->mmap_id_counter = 11000;
    }

    DPRINTK("EXIT, succeed\n");
    return 0;
}

static void tf_rollback_kbuf(struct kbuf *kbuf_p)
{
    if (kbuf_p->backend == TF_MEMORY_RESERVED &&
        kbuf_p->pool_index >= 0 &&
        kbuf_p->pool_index < reserveDDRBlcokCnt) {
        struct ReserveDDRBlock *block =
            &reserveDDRBlocks[kbuf_p->pool_index];

        if (block->tgid == kbuf_p->owner_tgid &&
            block->offset == kbuf_p->pool_offset + kbuf_p->len) {
            block->offset = kbuf_p->is_model_pool ?
                            kbuf_p->pool_previous_offset :
                            kbuf_p->pool_offset;
            if (!block->offset) {
                block->isMalloc = false;
                block->tgid = -1;
            }
        }
    } else if (kbuf_p->backend == TF_MEMORY_HUGEPAGE &&
               kbuf_p->huge_region &&
               kbuf_p->huge_region->owner_tgid == kbuf_p->owner_tgid &&
               kbuf_p->huge_region->offset ==
                   kbuf_p->pool_offset + kbuf_p->len) {
        kbuf_p->huge_region->offset = kbuf_p->pool_previous_offset;
        if (!kbuf_p->huge_region->offset) {
            kbuf_p->huge_region->owner_tgid = -1;
        }
    }
}

static struct kbuf * tf_create_and_init_kbuf(struct tf_device * dev, int len) {
    if (efuse) {
        return NULL;
    }
    int retval;
    struct kbuf * kbuf_p;

    DPRINTK("ENTER, succeed\n");
    if (!(kbuf_p = kzalloc(sizeof(struct kbuf), GFP_KERNEL))) {
        retval = -ENOMEM;
        goto origin;
    }
    kbuf_p->len = len;
    DPRINTK("kbuf_p->len: 0x%08x\n", len);
    if ((retval = tf_init_kbuf(dev, kbuf_p))) {
        DPRINTK("failed to init kbuf\n");
        goto after_alloc_kbuf;
    }
    DPRINTK("buf phy_addr: %llx, kernel_addr: %p, len: 0x%08x, mmap_id: 0x%08x\n",
            kbuf_p->phy_addr, kbuf_p->kernel_addr, kbuf_p->len, kbuf_p->mmap_id);
    DPRINTK("EXIT, succeed\n");
    return kbuf_p;

    after_alloc_kbuf:
    kfree(kbuf_p);
    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return NULL;
}


static int tf_ioctl_create(struct tf_device * dev, void * arg) {
    if (efuse) {
        return -1;
    }
    struct tf_buf_io_param *io_param;
    struct kbuf * kbuf_p;
    int retval;

    DPRINTK("ENTER\n");
    if (!(io_param = (struct tf_buf_io_param*) kmalloc(sizeof(struct tf_buf_io_param), GFP_KERNEL))) {
        DPRINTK("fail to alloc io_param\n");
        retval = -ENOMEM;
        goto origin;
    }
    if ((retval = copy_from_user(io_param, (void *) arg, sizeof(struct tf_buf_io_param)))) {
        DPRINTK("fail to copy io_param from user\n");
        goto after_alloc_io_param;
    }

    dev->useDDR2 = io_param->useDDR2;

    mutex_lock(&tf_memory_mutex);

    //create kbuf
    if (!(kbuf_p = tf_create_and_init_kbuf(dev, io_param->len))) {
        DPRINTK("fail to create_and_init kbuf\n");
        retval = -ENOMEM;
        goto after_unlock_memory;
    }

    /* Stable tfnn writes zero to CACHE_INVALID_ADDR_HI.  Invalidate every
     * newly allocated high-address driver chunk here with its complete DMA
     * address before tfnn subdivides or reuses it. */
    if (upper_32_bits(kbuf_p->phy_addr)) {
        retval = tf_cache_invalidate_dma_range(dev, dev->useDDR2,
                                               kbuf_p->phy_addr,
                                               kbuf_p->len);
        if (retval) {
            goto after_init_kbuf;
        }
    }

    if (io_param->uncache) {
        isNextUncache = 1;
    }

    //add to hlist
    hash_add(dev->buf_list, &kbuf_p->list,
             kbuf_p->mmap_id & TF_BUF_HASHMASK);
    mutex_unlock(&tf_memory_mutex);
    //copy to user
    io_param->phy_addr = kbuf_p->phy_addr;
    io_param->mmap_id = kbuf_p->mmap_id;
    if ((retval = copy_to_user((void *) arg, io_param, sizeof(struct tf_buf_io_param)))) {
        DPRINTK("fail to copy io_param to user\n");
        goto after_add_to_hlist;
    }

    kfree(io_param);
    DPRINTK("EXIT, succeed\n");
    return 0;

    after_add_to_hlist:
    mutex_lock(&tf_memory_mutex);
    hash_del(&kbuf_p->list);

    after_init_kbuf:
    tf_rollback_kbuf(kbuf_p);
    kfree(kbuf_p);
    mutex_unlock(&tf_memory_mutex);
    goto after_alloc_io_param;

    after_unlock_memory:
    mutex_unlock(&tf_memory_mutex);
    after_alloc_io_param:
    kfree(io_param);
    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);
    return retval;
}

static int tf_ioctl_get_app_infos(struct tf_device* dev, int* p) {
    DPRINTK("GET PID LIST\n");

    get_pids(p);
    return 0;
}

static long tf_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    if (efuse) {
        return -1;
    }
    struct tf_device *dev = (struct tf_device *) filp->private_data;
    long retval;

    // DPRINTK("ENTER: %u, type: %d NR: %d, %u \n", cmd, _IOC_TYPE(cmd), _IOC_NR(cmd), TF_READ_PIDS);
    if (_IOC_TYPE(cmd) != TF_MAGIC) return -EINVAL;
    if (_IOC_NR(cmd) >= TF_MAX_NR) return -EINVAL;
    switch (cmd) {
        case TF_VERSION_CHECK:
            retval = tf_ioctl_check_version(dev, (void*) arg);
            break;
        case TF_BUF_RESET:
            tf_ioctl_reset(dev);
            retval = 0;
            break;
        case TF_BUF_CLEAR:
            tf_ioctl_clear(dev);
            retval = 0;
            break;
        case TF_BUF_CREATE:
            retval = tf_ioctl_create(dev, (void *) arg);
            break;
        case TF_READ_PIDS:
            retval = tf_ioctl_get_app_infos(dev, (int*) arg);
            break;

        case TF_APP_LOCK:
            retval = tf_app_try_lock(dev, (int*) arg);
            break;

        case TF_APP_UNLOCK:
            retval = tf_app_try_unlock(dev, (int*) arg);
            break;

        case TF_READ_APP_LOCK_RECORD:
            retval = tf_get_app_lock_records((struct tf_lock_record*) arg);
            break;

        case TF_READ_APP_USAGE:
            retval = tf_get_app_usage((int*)arg);
            break;

        case TF_READ_RESERVE_MEM_RECORD:
            retval = tf_get_reserve_ddr_blocks((struct ReserveDDRBlock*) arg);
            break;
        case TF_HUGEPAGE_REGISTER:
            retval = tf_ioctl_register_hugepage(dev, (void __user *)arg);
            break;
        case TF_HUGEPAGE_QUERY:
            retval = tf_ioctl_query_hugepage((void __user *)arg);
            break;
        case TF_HUGEPAGE_CLEAR:
            retval = tf_ioctl_clear_hugepages(dev, (void __user *)arg);
            break;
        case TF_CACHE_INVALIDATE:
            retval = tf_ioctl_cache_invalidate(dev, (void __user *)arg);
            break;
        case TF_SET_ADDRESS_HIGH:
            retval = tf_ioctl_set_address_high(dev, (void __user *)arg);
            break;
        case TF_MODEL_POOL_ALLOC:
            retval = tf_ioctl_model_pool_alloc(dev, (void __user *)arg);
            break;
        case TF_MODEL_POOL_FREE:
            retval = tf_ioctl_model_pool_free(dev, (void __user *)arg);
            break;
        default:
            DPRINTK("error cmd\n");
            retval = -EINVAL;
            break;
    }

    if (!retval) {
        // DPRINTK("EXIT, succeed\n");
        return 0;
    }
    DPRINTK("EXIT, failed with code %ld\n", retval);
    DPRINTK("FAIL CMD: %u, type: %d NR: %d \n", cmd, _IOC_TYPE(cmd), _IOC_NR(cmd));
    return retval;
}

void tf_remove(void) {
    if (efuse) {
        return;
    }
    DPRINTK("ENTER\n");

    dev_set_drvdata(tf_dev->device, NULL);
    tf_remove_cdev(tf_dev);
    mutex_lock(&tf_memory_mutex);
    tf_remove_device_buf(tf_dev);
    tf_reset_memory_owner(0, true);
    mutex_unlock(&tf_memory_mutex);
    tf_release_all_huge_regions();
    kfree(tf_dev);

    DPRINTK("EXIT, succeed\n");
}

//reserve reserveSize M bytes.
//normal: 256
//1MFace: 512
//5MFace: 1536
//10MFace: 2560
long long reserveSize = 0;

static ssize_t show_kernel_version(struct device *dev,
                                   struct device_attribute *attr, char *buf)
{
    if (efuse) {
        return -1;
    }
    int ret;
    if (reserveSize == 256) {
        ret = sprintf(buf, "Normal\n");
    } else if (reserveSize == 512) {
        ret = sprintf(buf, "1M Face\n");
    } else if (reserveSize == 800) {
        ret = sprintf(buf, "3M Face\n");
    } else if (reserveSize == 1536) {
        ret = sprintf(buf, "5M Face\n");
    } else if (reserveSize == 2560) {
        ret = sprintf(buf, "10M Face\n");
    } else {
        ret = sprintf(buf, "Unknown\n");
    }

    return ret;
}

static ssize_t set_my_kernel(struct device *dev,
                             struct device_attribute *attr,
                             const char *buf, size_t len)
{
    if (efuse) {
        return -1;
    }
    return len;
}

static DEVICE_ATTR(kernel_version, S_IWUSR|S_IRUSR, show_kernel_version, set_my_kernel);


struct file_operations mytest_ops={
        .owner  = THIS_MODULE,
};

static int major;
//static struct class *cls;
//static struct class *profileCls;

void output_tfacc_id(unsigned long long base) {
    if (efuse) {
        return;
    }
    unsigned int *reg = ioremap(base, DEVICE_IO_LENGTH);
    if (reg) {
        printk("version: 0x%08X\n", *reg);
        printk("ID: 0x%08X\n", *(reg + 1));
        iounmap(reg);
    }
}

static void readSocketInfo(void) {
    unsigned int isDualSocket, isDualDie;
    unsigned int *configBase = ioremap(0xFE170000, 0x100000);
    long long perChip, perBlock;
    int c, i;
    unsigned long long gap;
    unsigned int *efuseAddr = ioremap(EFUSE_BASE, 0x100);

    if (efuseAddr[0] & 2) {
        efuse = 1;
    }

    // 读取chip信息
    isDualSocket = (*(volatile unsigned int *)(configBase + 0x30 / 4) & 0x40) >> 6;
    isDualDie = *(volatile unsigned int *)(configBase + 0x3046C / 4) & 0x1;
    iounmap(configBase);

    if (isDualSocket) chips = 4;
    else if (isDualDie) chips = 2;
    else chips = 1;

    if (chips == 2) {
        chipGap = 0x8000000000LL;
    } else if (chips == 4) {
        chipGap = 0x4000000000LL;
    }

    reserveDDRBlcokCnt = 0;
    perChip = ddrSize;
    perBlock = 256 * 1024 * 1024;

    for (c = 0; c < chips; c++) {
        for (i = 0; (long long)i * perBlock < perChip; i++) {
            long long blockOffset = (long long)i * perBlock;
            long long blockLen = min_t(long long, perBlock,
                                       perChip - blockOffset);

            if (reserveDDRBlcokCnt >= ARRAY_SIZE(reserveDDRBlocks)) {
                printk(KERN_WARNING
                       "tfacc: reserved DDR block table is full; remaining memory is ignored\n");
                return;
            }
            gap = chipGap * c;
            reserveDDRBlocks[reserveDDRBlcokCnt].chipId = c;
            reserveDDRBlocks[reserveDDRBlcokCnt].len = blockLen;
            reserveDDRBlocks[reserveDDRBlcokCnt].isMalloc = false;
            reserveDDRBlocks[reserveDDRBlcokCnt].offset = 0;
            reserveDDRBlocks[reserveDDRBlcokCnt].startPos =
                ddrStart + gap + blockOffset;
            reserveDDRBlocks[reserveDDRBlcokCnt].tgid = -1;

            if (reserveDDRBlocks[reserveDDRBlcokCnt].startPos % 0x100000000 == 0) {
                reserveDDRBlocks[reserveDDRBlcokCnt].startPos += 1 * 1024 * 1024;
                reserveDDRBlocks[reserveDDRBlcokCnt].len -= 1 * 1024 * 1024;
            }
            reserveDDRBlcokCnt++;
        }
    }

    return;
}

static int tf_open(struct inode *inode, struct file *filp) {
    if (efuse) {
        return -1;
    }
    struct tf_device *dev = container_of(inode->i_cdev, struct tf_device, cdev);

    DPRINTK("ENTER: pid: %d, tgid: %d\n", current->pid, current->tgid);
    mutex_lock(&tf_memory_mutex);
    spin_lock(&dev->lock);
    DPRINTK("isBusy = %d\n", dev->isBusy);
    filp->private_data = dev;
    dev->isBusy++;
#if 0
    if (dev->isBusy == 1) {
        //if (chips == 1) {
        //    //hardware_tfacc_reset();
        //}

        tfacc_full_enable(TFACC_BL_CLK_BASE);
        tfacc_full_enable(TFACC_BR_CLK_BASE);
        tfacc_lite_enable(TFACC_L_CLK_BASE);
        tfacc_lite_enable(TFACC_R_CLK_BASE);

        tfacc_full_acp(TFACC0_FULL_ACP_BASE, ddrStart >> 32);
        tfacc_full_acp(TFACC1_FULL_ACP_BASE, ddrStart >> 32);
        tfacc_full_acp(TFACC2_FULL_ACP_BASE, ddrStart >> 32);
        tfacc_full_acp(TFACC3_FULL_ACP_BASE, ddrStart >> 32);

        tfacc_full_enable_cache(TFACC0_BASE, TFACC1_BASE, TFACC0_CACHE_BASE, TFACC1_CACHE_BASE, TFACC_BL_CLK_BASE);
        tfacc_full_enable_cache(TFACC2_BASE, TFACC1_BASE, TFACC2_CACHE_BASE, TFACC3_CACHE_BASE, TFACC_BR_CLK_BASE);

        tfacc_lite_enable_cache(TFACCLITE0_BASE, TFACCLITE0_CACHE_BASE);
        tfacc_lite_enable_cache(TFACCLITE1_BASE, TFACCLITE1_CACHE_BASE);
        tfacc_lite_enable_cache(TFACCLITE2_BASE, TFACCLITE2_CACHE_BASE);
        tfacc_lite_enable_cache(TFACCLITE3_BASE, TFACCLITE3_CACHE_BASE);
    }
#endif

    spin_unlock(&dev->lock);
    mutex_unlock(&tf_memory_mutex);

    push_pid();
    DPRINTK("current pid: %d\n", current->pid);

    DPRINTK("EXIT, succeed\n");
    return 0;
}

#ifdef CONFIG_ACPI
static int tf_init_module(struct platform_device *pdev)
#else
static int __init tf_init_module(void)
#endif
{
    if (efuse) {
        return -1;
    }
    int retval;
    struct tf_device *dev;
    int c;
    unsigned long long gap;

    //struct device *mydev;
    DPRINTK("page = %d\n", PAGE_SHIFT);
    DPRINTK("ENTER\n");

#ifdef CONFIG_ACPI
    struct resource *res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if (!res) {
        printk(KERN_ERR "tfacc: ACPI reserved DDR resource is missing\n");
        return -ENODEV;
    }
    ddrStart = res->start;
    ddrSize = resource_size(res);
#else
    // 读取reserveDDR的信息
    struct device_node *reserved0 = of_find_node_by_path("/reserved-memory/buffer@1");
    uint8_t *r0 = (uint8_t*)reserved0->properties[1].value;
    ddrStart = (((long long)r0[3] << 32) + ((long long)r0[4] << 24) + ((long long)r0[5] << 16) + ((long long)r0[6] << 8) + (long long)r0[7]);
#endif

    DPRINTK("ddrStart: 0x%llx\n", (unsigned long long)ddrStart);

    readSocketInfo();
    DPRINTK("EFUSE = %d\n", efuse);
    DPRINTK("chips = %d\n", chips);
    if (efuse) {
        return -1;
    }

    major = register_chrdev(0, "thinkforce_kernel", &mytest_ops);
    //cls = class_create(THIS_MODULE, "thinkforce_kernel_class");
    //mydev = device_create(cls, 0, MKDEV(major,0),NULL,"thinkforce_kernel_device");

    //if (sysfs_create_file(&(mydev->kobj), &dev_attr_kernel_version.attr)) {
    //    return -1;
    //}

    // init mutex for pid list
    mutex_init(&pid_mutex);
    mutex_init(&app_mutex);
    spin_lock_init(&app_spin_lock);
    init_pids();

    // init app lock records
    initRecords();

    {
        unsigned int* periscfg = ioremap(TFACC0_FULL_ACP_BASE, TFACC_CLK_LENGTH);
        retval = periscfg[0x34/4];
        DPRINTK("Init Flag = %llx\n", periscfg[0x34/4]);
        if (retval == 0) {
            periscfg[0x34/4] = 1;
            DPRINTK("Flag = %llx\n", periscfg[0x34/4]);
            skipFullSwRst = 0;
        } else {
            DPRINTK("Skip Full Cache rst\n");
            skipFullSwRst = retval;
        }
        iounmap(periscfg);
    }

    for (c = 0; c < chips; c++) {
        DPRINTK("init chip %d\n", c);
        gap = chipGap * c;

        tfacc_full_enable(gap + TFACC_BL_CLK_BASE);
        tfacc_full_enable(gap + TFACC_BR_CLK_BASE);
        tfacc_lite_enable(gap + TFACC_L_CLK_BASE);
        tfacc_lite_enable(gap + TFACC_R_CLK_BASE);

        tfacc_full_acp(gap + TFACC0_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));
        tfacc_full_acp(gap + TFACC1_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));
        tfacc_full_acp(gap + TFACC2_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));
        tfacc_full_acp(gap + TFACC3_FULL_ACP_BASE, 0x80 / (chips / 2) * c + (ddrStart >> 32));

        if (!skipFullSwRst) {
            tfacc_full_enable_cache(gap + TFACC0_BASE, gap + TFACC1_BASE, gap + TFACC0_CACHE_BASE, gap + TFACC1_CACHE_BASE, gap + TFACC_BL_CLK_BASE, gap+TFACC0_FULL_ACP_BASE);
            tfacc_full_enable_cache(gap + TFACC2_BASE, gap + TFACC3_BASE, gap + TFACC2_CACHE_BASE, gap + TFACC3_CACHE_BASE, gap + TFACC_BR_CLK_BASE, gap+TFACC1_FULL_ACP_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE0_BASE, gap + TFACCLITE0_CACHE_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE1_BASE, gap + TFACCLITE1_CACHE_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE2_BASE, gap + TFACCLITE2_CACHE_BASE);
            tfacc_lite_enable_cache(gap + TFACCLITE3_BASE, gap + TFACCLITE3_CACHE_BASE);
        }


        output_tfacc_id(gap + 0xFC000000);
        output_tfacc_id(gap + 0xFC100000);
        output_tfacc_id(gap + 0xEC000000);
        output_tfacc_id(gap + 0xEC100000);
        output_tfacc_id(gap + 0xF9800000);
        output_tfacc_id(gap + 0xF9900000);
        output_tfacc_id(gap + 0xE9800000);
        output_tfacc_id(gap + 0xE9900000);
    }

    if (IS_ERR(thinkforce_class = class_create(THIS_MODULE, "thinkforce_class"))) {
        DPRINTK("failed to device register class\n");
        retval = -ENOMEM;
        goto origin;
    }

    dev = tf_create_and_init_device(0);
    if (dev == NULL) {
        goto origin;
    }

#ifdef CONFIG_ACPI
    dev->dma_device = &pdev->dev;
    retval = dma_set_mask_and_coherent(dev->dma_device, DMA_BIT_MASK(64));
    if (retval) {
        printk(KERN_WARNING
               "tfacc: 64-bit DMA is unavailable; HugeTLB extension is disabled (%d)\n",
               retval);
        dev->dma_device = NULL;
    }
#endif

    if (tf_create_and_init_cdev(dev, 0) < 0) {
        goto create_error;
    }

    DPRINTK("EXIT, succeed\n");
    return 0;

    create_error:
#ifdef CONFIG_ACPI
    tf_cleanup_module(pdev);
#else
    tf_cleanup_module();
#endif

    origin:
    DPRINTK("EXIT, failed with code %d\n", retval);

    return retval;
}

void tfacc_cache_debug(unsigned long long BASE) {
    if (efuse) {
        return;
    }
    volatile unsigned int *apb = ioremap(BASE, DEVICE_IO_LENGTH);
    int i=0;
    while (i<0x200/4) {
        DPRINTK("%08x = %llx\n", 4*i, *(apb+i));
        i++;
    }
    iounmap(apb);
}

#ifdef CONFIG_ACPI
static int tf_cleanup_module(struct platform_device *pdev) {
	if (efuse) {
		return -1;
	}
#else
static void __exit tf_cleanup_module(void) {
    if (efuse) {
        return;
    }
#endif
    int c;
    unsigned long long gap;

    DPRINTK("ENTER\n");

    tf_remove();
    tf_dev = NULL;
    class_destroy(thinkforce_class);

//    device_destroy(cls, MKDEV(major,0));
//    class_destroy(cls);
    unregister_chrdev(major, "mytest");

    for (c = 0; c < chips; c++) {
        DPRINTK("disable chip %d tfacc\n", c);
        gap = chipGap * c;

//        tfacc_cache_debug(gap + TFACC1_CACHE_BASE);
//        tfacc_cache_debug(gap + TFACC3_CACHE_BASE);

        tfacc_full_disable(gap + TFACC_BL_CLK_BASE);
        tfacc_full_disable(gap + TFACC_BR_CLK_BASE);
        tfacc_lite_disable(gap + TFACC_L_CLK_BASE);
        tfacc_lite_disable(gap + TFACC_R_CLK_BASE);
    }
    tfacc_full_checkrstcond();

    DPRINTK("EXIT, succeed\n");
#ifdef CONFIG_ACPI
    return 0;
#endif
}

module_param(needReset, int, S_IWUSR|S_IRUSR);
MODULE_PARM_DESC(needReset, "Enforce TFACC hardware reset");
