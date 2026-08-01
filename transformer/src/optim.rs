use ndarray::Array2;

/// Adam optimizer with warmup learning rate schedule.
///
/// Paper: Section 5.3
/// "We used the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98, eps = 1e-9."
///
/// Learning rate schedule (Equation 3):
///   lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

pub struct Adam {
    beta1: f64,
    beta2: f64,
    eps: f64,
    step: usize,
    warmup_steps: usize,
    d_model: f64,
    // First and second moment estimates for each parameter
    m: Vec<Array2<f64>>,
    v: Vec<Array2<f64>>,
}

unsafe impl Send for Adam {}
unsafe impl Sync for Adam {}

impl Adam {
    /// Create a new Adam optimizer.
    ///
    /// - `d_model`: model dimension (used in LR schedule)
    /// - `warmup_steps`: number of warmup steps (paper uses 4000)
    pub fn new(d_model: usize, warmup_steps: usize) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.98,
            eps: 1e-9,
            step: 0,
            warmup_steps,
            d_model: d_model as f64,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Compute learning rate for the current step (Equation 3).
    ///
    /// lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
    pub fn learning_rate(&self) -> f64 {
        let step = self.step as f64;
        if step == 0.0 {
            return self.d_model.powf(-0.5) * self.warmup_steps as f64;
        }
        self.d_model.powf(-0.5)
            * step
                .powf(-0.5)
                .min(step * (self.warmup_steps as f64).powf(-1.5))
    }

    /// Register a parameter tensor for optimization.
    #[allow(dead_code)]
    pub fn add_param(&mut self, param: &mut Array2<f64>) {
        self.m.push(Array2::zeros(param.dim()));
        self.v.push(Array2::zeros(param.dim()));
    }

    /// Perform a single optimization step.
    ///
    /// Call this after computing gradients. Currently uses dummy zero gradients
    /// as this is a structural implementation — in a real training loop you'd
    /// compute actual gradients via backpropagation.
    pub fn step(&mut self) -> f64 {
        self.step += 1;
        let lr = self.learning_rate();
        lr
    }

    /// Update a parameter using its gradient (for use with autograd).
    #[allow(dead_code)]
    pub fn update_param(&mut self, idx: usize, param: &mut Array2<f64>, grad: &Array2<f64>) {
        self.step += 1;
        let lr = self.learning_rate();
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        // m = beta1 * m + (1 - beta1) * grad
        self.m[idx] = &self.m[idx] * self.beta1 + grad * (1.0 - self.beta1);

        // v = beta2 * v + (1 - beta2) * grad^2
        let grad_sq = grad.mapv(|x| x * x);
        self.v[idx] = &self.v[idx] * self.beta2 + &grad_sq * (1.0 - self.beta2);

        // Bias-corrected moment estimates
        let m_hat = &self.m[idx] / bias_correction1;
        let v_hat = &self.v[idx] / bias_correction2;

        // Update: param -= lr * m_hat / (sqrt(v_hat) + eps)
        let denom = v_hat.mapv(f64::sqrt) + self.eps;
        *param = &*param - &(&m_hat / &denom) * lr;
    }
}

/// Generate the learning rate schedule curve for visualization.
///
/// Returns a Vec of (step, lr) pairs.
pub fn lr_schedule(d_model: usize, warmup_steps: usize, total_steps: usize) -> Vec<(usize, f64)> {
    let d = d_model as f64;
    let w = warmup_steps as f64;
    let mut schedule = Vec::with_capacity(total_steps);

    for step in 1..=total_steps {
        let s = step as f64;
        let lr = d.powf(-0.5) * s.powf(-0.5).min(s * w.powf(-1.5));
        schedule.push((step, lr));
    }
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_warmup() {
        let opt = Adam::new(512, 4000);
        // Step 0 should return warmup_lr
        let lr0 = opt.learning_rate();
        assert!(lr0 > 0.0);

        // At warmup_steps, lr should peak
        let mut opt2 = Adam::new(512, 4000);
        for _ in 0..4000 {
            opt2.step();
        }
        let lr_peak = opt2.learning_rate();
        assert!(lr_peak > 0.0);

        // After warmup, lr should decrease
        for _ in 0..1000 {
            opt2.step();
        }
        let lr_after = opt2.learning_rate();
        assert!(lr_after < lr_peak, "lr should decrease after warmup");
    }

    #[test]
    fn test_schedule_curve() {
        let schedule = lr_schedule(512, 4000, 10000);
        assert_eq!(schedule.len(), 10000);
        // lr should increase then decrease
        let lr_100 = schedule[99].1;
        let lr_4000 = schedule[3999].1;
        let lr_10000 = schedule[9999].1;
        assert!(lr_4000 > lr_100, "lr should increase during warmup");
        assert!(lr_4000 > lr_10000, "lr should decrease after warmup");
    }
}
