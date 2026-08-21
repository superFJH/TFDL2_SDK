obj-m += tfacc2.o
PWD := $(shell pwd)
HOSTCC ?= cc
HOSTCFLAGS ?= -O2 -Wall -Wextra

KERNEL_VERSION := $(shell uname -r)
KDIR ?= /lib/modules/$(KERNEL_VERSION)/build
KERNEL_RELEASE = $(shell cat $(KDIR)/include/config/kernel.release)


install:
	bash ./build_driver.sh
	
tfacc2: tf_hugepage_register
	$(MAKE) -C $(KDIR) M=$(PWD) modules
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -A $(KERNEL_RELEASE)

tf_hugepage_register: tf_hugepage_register.c tfacc_hugepage_uapi.h
	$(HOSTCC) $(HOSTCFLAGS) -o $@ tf_hugepage_register.c

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f tf_hugepage_register
