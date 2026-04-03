; FastDB x86-64 Assembly Hot Paths
; NASM syntax, System V AMD64 ABI
;
; Functions:
;   fastdb_hash_asm     — SSE4.2 CRC32-based fast hash with avalanche mixing
;   fastdb_memcmp_asm   — SSE2 vectorized memory compare
;   fastdb_scan_asm     — scan array of uint64 for target hash
;   fastdb_crc64_asm    — CRC64 checksum
;   fastdb_memcpy_nt    — non-temporal memcpy for large buffers

section .text

; ============================================================
; uint64_t fastdb_hash_asm(const void *data, uint32_t len)
;   rdi = data pointer
;   esi = length
;   returns rax = 64-bit hash
; ============================================================
global fastdb_hash_asm
fastdb_hash_asm:
    push    rbx
    push    r12

    mov     r12, rdi            ; save data ptr
    mov     ecx, esi            ; len

    ; Start with a good seed
    mov     rax, 0x736F6D6570736575  ; "somepseu"
    mov     rbx, 0x646F72616E646F6D  ; "dorandom"

    ; Process 8 bytes at a time using CRC32
    cmp     ecx, 8
    jb      .hash_tail

.hash_loop8:
    crc32   rax, qword [r12]
    xor     rax, rbx
    ror     rbx, 17
    xor     rbx, rax
    add     r12, 8
    sub     ecx, 8
    cmp     ecx, 8
    jae     .hash_loop8

.hash_tail:
    ; Process remaining bytes one at a time
    test    ecx, ecx
    jz      .hash_finalize

.hash_tail_loop:
    crc32   eax, byte [r12]
    inc     r12
    dec     ecx
    jnz     .hash_tail_loop

.hash_finalize:
    ; Avalanche mixing (based on splitmix64)
    xor     rax, rbx
    mov     rcx, rax
    shr     rcx, 30
    xor     rax, rcx
    mov     rcx, 0xbf58476d1ce4e5b9
    imul    rax, rcx
    mov     rcx, rax
    shr     rcx, 27
    xor     rax, rcx
    mov     rcx, 0x94d049bb133111eb
    imul    rax, rcx
    mov     rcx, rax
    shr     rcx, 31
    xor     rax, rcx

    pop     r12
    pop     rbx
    ret

; ============================================================
; int fastdb_memcmp_asm(const void *a, const void *b, uint32_t len)
;   rdi = a, rsi = b, edx = len
;   returns 0 if equal, nonzero otherwise
; ============================================================
global fastdb_memcmp_asm
fastdb_memcmp_asm:
    mov     ecx, edx

    ; For 16+ bytes, use SSE2
    cmp     ecx, 16
    jb      .cmp_byte

.cmp_sse_loop:
    movdqu  xmm0, [rdi]
    movdqu  xmm1, [rsi]
    pcmpeqb xmm0, xmm1
    pmovmskb eax, xmm0
    cmp     eax, 0xFFFF
    jne     .cmp_not_equal
    add     rdi, 16
    add     rsi, 16
    sub     ecx, 16
    cmp     ecx, 16
    jae     .cmp_sse_loop

.cmp_byte:
    test    ecx, ecx
    jz      .cmp_equal

.cmp_byte_loop:
    mov     al, [rdi]
    cmp     al, [rsi]
    jne     .cmp_not_equal
    inc     rdi
    inc     rsi
    dec     ecx
    jnz     .cmp_byte_loop

.cmp_equal:
    xor     eax, eax
    ret

.cmp_not_equal:
    mov     eax, 1
    ret

; ============================================================
; int64_t fastdb_scan_asm(const void *haystack, uint64_t hay_len,
;                         uint64_t target_hash)
;   rdi = haystack (array of uint64_t), rsi = count, rdx = target
;   returns index of match or -1
; ============================================================
global fastdb_scan_asm
fastdb_scan_asm:
    ; Broadcast target hash to xmm register
    movq    xmm2, rdx
    punpcklqdq xmm2, xmm2      ; xmm2 = [target, target]

    xor     rcx, rcx            ; index = 0

    ; Process 2 uint64s at a time (16 bytes)
    mov     rax, rsi
    shr     rax, 1              ; count / 2

    test    rax, rax
    jz      .scan_tail

.scan_sse_loop:
    movdqu  xmm0, [rdi + rcx*8]
    pcmpeqd xmm0, xmm2         ; compare 32-bit lanes
    ; For 64-bit equality, both 32-bit halves must match
    movdqa  xmm1, xmm0
    pshufd  xmm1, xmm0, 0xB1   ; swap pairs
    pand    xmm0, xmm1
    pmovmskb r8d, xmm0

    test    r8d, 0x0F
    jnz     .scan_found_first
    test    r8d, 0xF0
    jnz     .scan_found_second

    add     rcx, 2
    dec     rax
    jnz     .scan_sse_loop

.scan_tail:
    ; Check remaining element if odd count
    test    rsi, 1
    jz      .scan_not_found

    cmp     rdx, [rdi + rcx*8]
    je      .scan_found_exact

.scan_not_found:
    mov     rax, -1
    ret

.scan_found_first:
    mov     rax, rcx
    ret

.scan_found_second:
    lea     rax, [rcx + 1]
    ret

.scan_found_exact:
    mov     rax, rcx
    ret

; ============================================================
; uint64_t fastdb_crc64_asm(const void *data, uint32_t len)
;   Uses CRC32 instructions chained for a 64-bit result
; ============================================================
global fastdb_crc64_asm
fastdb_crc64_asm:
    push    rbx

    mov     rcx, rdi            ; data
    mov     edx, esi            ; len

    mov     rax, 0xFFFFFFFFFFFFFFFF  ; crc_hi init
    mov     rbx, 0xFFFFFFFF         ; crc_lo init

    ; Process 8 bytes at a time
    cmp     edx, 8
    jb      .crc_tail

.crc_loop8:
    crc32   rax, qword [rcx]
    mov     r8, qword [rcx]
    bswap   r8
    crc32   rbx, r8
    add     rcx, 8
    sub     edx, 8
    cmp     edx, 8
    jae     .crc_loop8

.crc_tail:
    test    edx, edx
    jz      .crc_done

.crc_tail_loop:
    crc32   eax, byte [rcx]
    inc     rcx
    dec     edx
    jnz     .crc_tail_loop

.crc_done:
    ; Combine hi and lo
    shl     rax, 32
    or      rax, rbx
    xor     rax, 0xFFFFFFFFFFFFFFFF

    pop     rbx
    ret

; ============================================================
; void fastdb_memcpy_nt(void *dst, const void *src, size_t len)
;   Non-temporal stores for large copies (bypass cache pollution)
; ============================================================
global fastdb_memcpy_nt
fastdb_memcpy_nt:
    ; rdi = dst, rsi = src, rdx = len

    ; For small copies, use regular rep movsb
    cmp     rdx, 256
    jb      .memcpy_small

    ; Align destination to 16 bytes
    mov     rcx, rdi
    and     rcx, 0xF
    jz      .memcpy_aligned
    mov     rax, 16
    sub     rax, rcx            ; bytes to align
    cmp     rax, rdx
    ja      .memcpy_small
    mov     rcx, rax
    sub     rdx, rax
    rep     movsb

.memcpy_aligned:
    ; Main loop: 64 bytes per iteration using movntdq
    mov     rcx, rdx
    shr     rcx, 6             ; count / 64

    test    rcx, rcx
    jz      .memcpy_tail

.memcpy_nt_loop:
    movdqu  xmm0, [rsi]
    movdqu  xmm1, [rsi + 16]
    movdqu  xmm2, [rsi + 32]
    movdqu  xmm3, [rsi + 48]
    movntdq [rdi],      xmm0
    movntdq [rdi + 16], xmm1
    movntdq [rdi + 32], xmm2
    movntdq [rdi + 48], xmm3
    add     rsi, 64
    add     rdi, 64
    dec     rcx
    jnz     .memcpy_nt_loop

    sfence                      ; ensure non-temporal stores complete

.memcpy_tail:
    and     edx, 63            ; remaining bytes
    mov     ecx, edx
    rep     movsb
    ret

.memcpy_small:
    mov     rcx, rdx
    rep     movsb
    ret
