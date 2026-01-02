#![no_std]

use core::slice;

#[no_mangle]
pub extern "C" fn matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: i32,
    k: i32,
    n: i32,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let k = k as usize;
    let n = n as usize;

    unsafe {
        let a = slice::from_raw_parts(a_ptr, m * k);
        let b = slice::from_raw_parts(b_ptr, k * n);
        let c = slice::from_raw_parts_mut(c_ptr, m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                let a_row = i * k;
                for p in 0..k {
                    sum += a[a_row + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
