#![no_std]

use core::slice;
use libm::{expf, logf, sqrtf, tanhf};

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

#[no_mangle]
pub extern "C" fn softmax(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let output = slice::from_raw_parts_mut(output_ptr, rows * cols);

        for r in 0..rows {
            let row_start = r * cols;
            let mut max_val = input[row_start];
            for c in 1..cols {
                let val = input[row_start + c];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum = 0.0f32;
            for c in 0..cols {
                let exp_val = expf(input[row_start + c] - max_val);
                output[row_start + c] = exp_val;
                sum += exp_val;
            }

            if sum != 0.0 {
                let inv = 1.0f32 / sum;
                for c in 0..cols {
                    output[row_start + c] *= inv;
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn logsoftmax(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let output = slice::from_raw_parts_mut(output_ptr, rows * cols);

        for r in 0..rows {
            let row_start = r * cols;
            let mut max_val = input[row_start];
            for c in 1..cols {
                let val = input[row_start + c];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += expf(input[row_start + c] - max_val);
            }
            let log_sum = logf(sum) + max_val;
            for c in 0..cols {
                output[row_start + c] = input[row_start + c] - log_sum;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn matmul_backward(
    a_ptr: *const f32,
    b_ptr: *const f32,
    grad_ptr: *const f32,
    grad_a_ptr: *mut f32,
    grad_b_ptr: *mut f32,
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
        let grad = slice::from_raw_parts(grad_ptr, m * n);
        let grad_a = slice::from_raw_parts_mut(grad_a_ptr, m * k);
        let grad_b = slice::from_raw_parts_mut(grad_b_ptr, k * n);

        for i in 0..m {
            for p in 0..k {
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += grad[i * n + j] * b[p * n + j];
                }
                grad_a[i * k + p] = sum;
            }
        }

        for p in 0..k {
            for j in 0..n {
                let mut sum = 0.0f32;
                for i in 0..m {
                    sum += a[i * k + p] * grad[i * n + j];
                }
                grad_b[p * n + j] = sum;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn reduce_sum(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let output = slice::from_raw_parts_mut(output_ptr, rows);

        for r in 0..rows {
            let row_start = r * cols;
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += input[row_start + c];
            }
            output[r] = sum;
        }
    }
}

#[no_mangle]
pub extern "C" fn reduce_mean(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;
    let inv = 1.0f32 / cols as f32;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let output = slice::from_raw_parts_mut(output_ptr, rows);

        for r in 0..rows {
            let row_start = r * cols;
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += input[row_start + c];
            }
            output[r] = sum * inv;
        }
    }
}

#[no_mangle]
pub extern "C" fn reduce_max(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let output = slice::from_raw_parts_mut(output_ptr, rows);

        for r in 0..rows {
            let row_start = r * cols;
            let mut max_val = input[row_start];
            for c in 1..cols {
                let val = input[row_start + c];
                if val > max_val {
                    max_val = val;
                }
            }
            output[r] = max_val;
        }
    }
}

#[no_mangle]
pub extern "C" fn layernorm(
    input_ptr: *const f32,
    gamma_ptr: *const f32,
    beta_ptr: *const f32,
    output_ptr: *mut f32,
    rows: i32,
    cols: i32,
    eps: f32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;
    let inv = 1.0f32 / cols as f32;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let gamma = slice::from_raw_parts(gamma_ptr, cols);
        let beta = slice::from_raw_parts(beta_ptr, cols);
        let output = slice::from_raw_parts_mut(output_ptr, rows * cols);

        for r in 0..rows {
            let row_start = r * cols;
            let mut mean = 0.0f32;
            for c in 0..cols {
                mean += input[row_start + c];
            }
            mean *= inv;

            let mut var = 0.0f32;
            for c in 0..cols {
                let diff = input[row_start + c] - mean;
                var += diff * diff;
            }
            var *= inv;
            let inv_std = 1.0f32 / sqrtf(var + eps);

            for c in 0..cols {
                let idx = row_start + c;
                let norm = (input[idx] - mean) * inv_std;
                output[idx] = norm * gamma[c] + beta[c];
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn layernorm_backward(
    input_ptr: *const f32,
    gamma_ptr: *const f32,
    grad_out_ptr: *const f32,
    grad_input_ptr: *mut f32,
    grad_gamma_ptr: *mut f32,
    grad_beta_ptr: *mut f32,
    rows: i32,
    cols: i32,
    eps: f32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;
    let inv = 1.0f32 / cols as f32;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, rows * cols);
        let gamma = slice::from_raw_parts(gamma_ptr, cols);
        let grad_out = slice::from_raw_parts(grad_out_ptr, rows * cols);
        let grad_input = slice::from_raw_parts_mut(grad_input_ptr, rows * cols);
        let grad_gamma = slice::from_raw_parts_mut(grad_gamma_ptr, cols);
        let grad_beta = slice::from_raw_parts_mut(grad_beta_ptr, cols);

        for c in 0..cols {
            grad_gamma[c] = 0.0f32;
            grad_beta[c] = 0.0f32;
        }

        for r in 0..rows {
            let row_start = r * cols;
            let mut mean = 0.0f32;
            for c in 0..cols {
                mean += input[row_start + c];
            }
            mean *= inv;

            let mut var = 0.0f32;
            for c in 0..cols {
                let diff = input[row_start + c] - mean;
                var += diff * diff;
            }
            var *= inv;
            let inv_std = 1.0f32 / sqrtf(var + eps);

            let mut sum_dy = 0.0f32;
            let mut sum_dy_y = 0.0f32;
            for c in 0..cols {
                let idx = row_start + c;
                let y = (input[idx] - mean) * inv_std;
                let dy = grad_out[idx] * gamma[c];
                sum_dy += dy;
                sum_dy_y += dy * y;
                grad_gamma[c] += grad_out[idx] * y;
                grad_beta[c] += grad_out[idx];
            }

            for c in 0..cols {
                let idx = row_start + c;
                let y = (input[idx] - mean) * inv_std;
                let dy = grad_out[idx] * gamma[c];
                grad_input[idx] =
                    (inv_std * (dy * cols as f32 - sum_dy - y * sum_dy_y)) * inv;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn mlp_fused(
    input_ptr: *const f32,
    weight_ptr: *const f32,
    bias_ptr: *const f32,
    output_ptr: *mut f32,
    m: i32,
    k: i32,
    n: i32,
    activation: i32,
) {
    if m <= 0 || k <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let k = k as usize;
    let n = n as usize;

    unsafe {
        let input = slice::from_raw_parts(input_ptr, m * k);
        let weight = slice::from_raw_parts(weight_ptr, k * n);
        let bias = slice::from_raw_parts(bias_ptr, n);
        let output = slice::from_raw_parts_mut(output_ptr, m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                let in_row = i * k;
                for p in 0..k {
                    sum += input[in_row + p] * weight[p * n + j];
                }
                sum += bias[j];

                let activated = match activation {
                    1 => if sum > 0.0 { sum } else { 0.0 },
                    2 => tanhf(sum),
                    _ => sum,
                };

                output[i * n + j] = activated;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn gelu(
    input_ptr: *const f32,
    output_ptr: *mut f32,
    len: i32,
) {
    if len <= 0 {
        return;
    }

    let len = len as usize;
    let a = 0.7978845608028654f32; // sqrt(2/pi)

    unsafe {
        let input = slice::from_raw_parts(input_ptr, len);
        let output = slice::from_raw_parts_mut(output_ptr, len);
        for i in 0..len {
            let x = input[i];
            let x3 = x * x * x;
            let inner = a * (x + 0.044715 * x3);
            output[i] = 0.5 * x * (1.0 + tanhf(inner));
        }
    }
}

#[no_mangle]
pub extern "C" fn softmax_backward(
    softmax_ptr: *const f32,
    grad_ptr: *const f32,
    grad_input_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        let softmax = slice::from_raw_parts(softmax_ptr, rows * cols);
        let grad = slice::from_raw_parts(grad_ptr, rows * cols);
        let grad_input = slice::from_raw_parts_mut(grad_input_ptr, rows * cols);

        for r in 0..rows {
            let row_start = r * cols;
            let mut dot = 0.0f32;
            for c in 0..cols {
                dot += grad[row_start + c] * softmax[row_start + c];
            }
            for c in 0..cols {
                let idx = row_start + c;
                grad_input[idx] = softmax[idx] * (grad[idx] - dot);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn logsoftmax_backward(
    logsoftmax_ptr: *const f32,
    grad_ptr: *const f32,
    grad_input_ptr: *mut f32,
    rows: i32,
    cols: i32,
) {
    if rows <= 0 || cols <= 0 {
        return;
    }

    let rows = rows as usize;
    let cols = cols as usize;

    unsafe {
        let logsoftmax = slice::from_raw_parts(logsoftmax_ptr, rows * cols);
        let grad = slice::from_raw_parts(grad_ptr, rows * cols);
        let grad_input = slice::from_raw_parts_mut(grad_input_ptr, rows * cols);

        for r in 0..rows {
            let row_start = r * cols;
            let mut sum_grad = 0.0f32;
            for c in 0..cols {
                sum_grad += grad[row_start + c];
            }
            for c in 0..cols {
                let idx = row_start + c;
                let softmax = expf(logsoftmax[idx]);
                grad_input[idx] = grad[idx] - softmax * sum_grad;
            }
        }
    }
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
