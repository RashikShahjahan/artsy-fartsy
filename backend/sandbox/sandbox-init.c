#define _GNU_SOURCE

#include <errno.h>
#include <linux/prctl.h>
#include <seccomp.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

static int deny_syscall(scmp_filter_ctx filter, const char *name) {
    int syscall_number = seccomp_syscall_resolve_name(name);
    if (syscall_number < 0) {
        return 0;
    }
    int result = seccomp_rule_add(filter, SCMP_ACT_ERRNO(EPERM), syscall_number, 0);
    return result == -EEXIST ? 0 : result;
}

int main(int argc, char **argv) {
    static const char *denied_syscalls[] = {
        "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
        "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg",
        "shutdown", "getsockname", "getpeername", "setsockopt", "getsockopt",
        "clone", "clone3", "fork", "vfork",
        "unshare", "setns", "mount", "umount", "umount2", "pivot_root", "chroot",
        "ptrace", "process_vm_readv", "process_vm_writev",
        "bpf", "perf_event_open", "keyctl", "add_key", "request_key", "userfaultfd",
        "io_uring_setup", "io_uring_register", "io_uring_enter", "memfd_create",
        "open_by_handle_at", "name_to_handle_at", "kexec_load", "kexec_file_load",
        "init_module", "finit_module", "delete_module", "reboot", "swapon", "swapoff",
    };

    if (argc < 2) {
        fprintf(stderr, "usage: artsy-sandbox-init command [args...]\n");
        return 64;
    }

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        perror("prctl(PR_SET_NO_NEW_PRIVS)");
        return 70;
    }

    scmp_filter_ctx filter = seccomp_init(SCMP_ACT_ALLOW);
    if (filter == NULL) {
        fprintf(stderr, "failed to initialize seccomp\n");
        return 70;
    }

    for (size_t index = 0; index < sizeof(denied_syscalls) / sizeof(denied_syscalls[0]); index++) {
        int result = deny_syscall(filter, denied_syscalls[index]);
        if (result < 0) {
            fprintf(stderr, "failed to deny %s: %s\n", denied_syscalls[index], strerror(-result));
            seccomp_release(filter);
            return 70;
        }
    }

    int result = seccomp_load(filter);
    seccomp_release(filter);
    if (result < 0) {
        fprintf(stderr, "failed to load seccomp filter: %s\n", strerror(-result));
        return 70;
    }

    execvp(argv[1], &argv[1]);
    perror("execvp");
    return 70;
}
