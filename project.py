import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HW_NONIDEALITY = True        # flip off for a clean finite-diff sanity check
NumInput, NumHidden, NumOutput = 64, 50, 10
BatchSize, NumBatch, LearningRate = 100, 600, 10.0
sig = lambda x: 1.0 / (1.0 + np.exp(-x))




G_MIN, G_MAX = 0.1, 150.0      # uS
READ_NOISE_SIGMA = 0.1         # uS
N_BITS = 8
N_LEVELS = 2 ** N_BITS
EPS = 1e-12

noise_rng = np.random.default_rng(1)   # separate stream, just for HW noise


def to_conductance(W, w_full_scale):
    return np.clip(W, -w_full_scale, w_full_scale) / w_full_scale * G_MAX


def from_conductance(G, w_full_scale):
    return G / G_MAX * w_full_scale


def quantize8(G, g_min=G_MIN, g_max=G_MAX, n_levels=N_LEVELS):
    step = (2 * g_max) / (n_levels - 1)
    Gq = np.round(G / step) * step
    Gq = np.clip(Gq, -g_max, g_max)
    Gq[np.abs(Gq) < g_min] = 0.0      # devices can't resolve below G_MIN
    return Gq


def apply_read_noise(G, sigma=READ_NOISE_SIGMA, rng=noise_rng):
    return G + rng.standard_normal(G.shape).astype(np.float32) * sigma


def effective_weight(W_shadow, hw=True, rng=noise_rng):
    """What the hardware actually returns when this weight is 'read'."""
    if not hw:
        return W_shadow
    w_full_scale = np.max(np.abs(W_shadow)) + EPS
    G = to_conductance(W_shadow, w_full_scale)
    Gq = quantize8(G)
    Gn = apply_read_noise(Gq, rng=rng)
    return from_conductance(Gn, w_full_scale)



number_raw = sio.loadmat('MNIST_TrainSet_0to1_8x8pixel.mat')['number']   # 64 x 60000
label_raw = sio.loadmat('MNIST_TrainSet_Label.mat')['label'].ravel()     # 60000

init_rng = np.random.default_rng(3)
layer1_init = init_rng.random((NumInput + 1, NumHidden)).astype(np.float32) / 1e2   # 65x50
layer2_init = init_rng.random((NumHidden + 1, NumOutput)).astype(np.float32) / 1e2  # 51x10


shuffle_rng = np.random.default_rng(10)
perm = shuffle_rng.permutation(number_raw.shape[1])
number = number_raw[:, perm]
label = label_raw[perm]


#BackProp

def forward_bp(s1, L1, L2, hw):
    L1_eff = effective_weight(L1, hw=hw)
    L2_eff = effective_weight(L2, hw=hw)
    o1 = np.concatenate([s1, [1.0]]) @ L1_eff
    a1 = sig(o1)
    o2 = np.concatenate([a1, [1.0]]) @ L2_eff
    f = sig(o2)
    return o1, a1, o2, f, L1_eff, L2_eff


def energy_bp(s1, L1, L2, T, hw):
    f = forward_bp(s1, L1, L2, hw)[3]
    return np.sum((f - T) ** 2) / 2.0


print('=' * 60)
print('BACKPROP')
print('=' * 60)

# BackProp gradient vs. brute-force perturbation, image #1
i = 0
s1 = number[:, i]
T = np.zeros(10, dtype=np.float32); T[label[i]] = 3.0

o1, a1, o2, f, L1_eff, L2_eff = forward_bp(s1, layer1_init, layer2_init, hw=False)
Eref = np.sum((f - T) ** 2) / 2.0

d1 = np.exp(-o1) / (1 + np.exp(-o1)) ** 2
d2 = np.exp(-o2) / (1 + np.exp(-o2)) ** 2
dEdf = (f - T) * d2
BackOutput_L2 = (dEdf @ L2_eff.T) * np.concatenate([d1, [1.0]])
G1_bp = np.outer(np.concatenate([s1, [1.0]]), BackOutput_L2[:NumHidden])
G2_bp = np.outer(np.concatenate([a1, [1.0]]), dEdf)

dW = 1e-6
G1_pert = np.zeros((NumInput, NumHidden), dtype=np.float32)
G2_pert = np.zeros((NumHidden, NumOutput), dtype=np.float32)
for r in range(NumInput):
    for c in range(NumHidden):
        L = layer1_init.copy(); L[r, c] += dW
        G1_pert[r, c] = (energy_bp(s1, L, layer2_init, T, hw=False) - Eref) / dW
for r in range(NumHidden):
    for c in range(NumOutput):
        L = layer2_init.copy(); L[r, c] += dW
        G2_pert[r, c] = (energy_bp(s1, layer1_init, L, T, hw=False) - Eref) / dW

print(f'E_ref = {Eref:.7f}')
print('Part 1 (BackProp vs weight-perturbation gradient, non-bias rows):')
for name, A, B in [('Layer1', G1_bp[:NumInput], G1_pert),
                    ('Layer2', G2_bp[:NumHidden], G2_pert)]:
    print(f'  {name}: max|G_bp|={np.abs(A).max():.3e}  '
          f'max|diff|={np.abs(A - B).max():.3e}')

fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
ax[0].imshow(s1.reshape(8, 8), cmap='viridis'); ax[0].axis('off')
ax[0].set_title(f'input (label {label[i]})')
M = NumInput + 1 + NumHidden + 1 + NumOutput
Gfull_bp = np.zeros((M, M))
Gfull_bp[:NumInput + 1, NumInput + 1:NumInput + 1 + NumHidden] = G1_bp
Gfull_bp[NumInput + 1:NumInput + 2 + NumHidden, NumInput + NumHidden + 2:] = G2_bp
ax[1].imshow(Gfull_bp, vmin=-1e-3, vmax=1e-3); ax[1].set_title('G_bp')
Gfull_p = np.zeros((M, M))
Gfull_p[:NumInput, NumInput + 1:NumInput + 1 + NumHidden] = G1_pert
Gfull_p[NumInput + 1:NumInput + 1 + NumHidden, NumInput + NumHidden + 2:] = G2_pert
ax[2].imshow(Gfull_p, vmin=-1e-3, vmax=1e-3); ax[2].set_title('G_pert')
plt.tight_layout(); plt.savefig('part1_gradient_compare.png', dpi=150)
plt.close(fig)

#one epoch, batch gradient descent, BackProp
layer1, layer2 = layer1_init.copy(), layer2_init.copy()
bp_accuracy = np.zeros(NumBatch)

for b in range(NumBatch):
    idx = np.arange(b * BatchSize, (b + 1) * BatchSize)
    X = np.concatenate([number[:, idx].T, np.ones((BatchSize, 1))], axis=1)   # 100x65
    Tb = np.zeros((BatchSize, NumOutput), dtype=np.float32)
    Tb[np.arange(BatchSize), label[idx]] = 3.0

    L1_eff = effective_weight(layer1, hw=HW_NONIDEALITY)
    L2_eff = effective_weight(layer2, hw=HW_NONIDEALITY)

    o1 = X @ L1_eff
    a1 = sig(o1)
    A1 = np.concatenate([a1, np.ones((BatchSize, 1))], axis=1)
    o2 = A1 @ L2_eff
    f = sig(o2)

    bp_accuracy[b] = np.mean(np.argmax(f, axis=1) == label[idx]) * 100

    d1 = a1 * (1 - a1)
    d2 = f * (1 - f)
    dEdf = (f - Tb) * d2
    # straight-through: gradient uses the clean shadow weight layer2,
    # not the noisy/quantized L2_eff, same as the single-image case above.
    BackOut = (dEdf @ layer2.T) * np.concatenate([d1, np.ones((BatchSize, 1))], axis=1)
    G1 = X.T @ BackOut[:, :NumHidden]
    G2 = A1.T @ dEdf

    layer1 -= LearningRate * G1 / BatchSize
    layer2 -= LearningRate * G2 / BatchSize

    if b % 50 == 0:
        print(f'  batch {b:3d}   accuracy {bp_accuracy[b]:5.1f}%')

print(f'  mean accuracy, last 50 batches = {bp_accuracy[-50:].mean():.2f}%\n')

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(bp_accuracy, lw=0.9)
ax2.set_xlabel('Batches'); ax2.set_ylabel('Training Accuracy (%)')
ax2.set_ylim(0, 100); ax2.set_xlim(0, NumBatch); ax2.grid(alpha=.3)
ax2.set_title('One epoch - BackProp - conductance-unit weights (8-bit, noise on)')
plt.tight_layout(); plt.savefig('part2_accuracy_backprop.png', dpi=150)
plt.close(fig2)


#weight perturbation
print('=' * 60)
print('WEIGHT PERTURBATION')
print('=' * 60)

PerturbationSize = 1e-4


def perturbation_gradient_batch(indices, L1, L2, epsilon, hw):
    """Same algorithm as weight_purturbation.m's local function: perturb
    every non-bias weight independently (vectorized here instead of
    looped), average (Enew-Eref)/epsilon over the batch."""
    idx = np.asarray(indices)
    B = idx.size
    X = np.concatenate([number[:, idx].T, np.ones((B, 1))], axis=1)   # B x 65
    T = np.zeros((B, NumOutput), dtype=np.float32)
    T[np.arange(B), label[idx]] = 3.0

    L1_eff = effective_weight(L1, hw=hw)
    L2_eff = effective_weight(L2, hw=hw)

    z1_ref = X @ L1_eff
    A_ref = sig(z1_ref)
    A_bias_ref = np.concatenate([A_ref, np.ones((B, 1))], axis=1)
    z2_ref = A_bias_ref @ L2_eff
    f_ref = sig(z2_ref)
    Eref_total = 0.5 * np.sum((f_ref - T) ** 2, axis=1)

    G2 = np.zeros_like(L2)
    for k in range(NumOutput):
        z2_new_k = z2_ref[:, k:k + 1] + epsilon * A_bias_ref
        f_new_k = sig(z2_new_k)
        Eref_k = 0.5 * (f_ref[:, k:k + 1] - T[:, k:k + 1]) ** 2
        Enew_k = 0.5 * (f_new_k - T[:, k:k + 1]) ** 2
        Enew_total = Eref_total[:, None] + Enew_k - Eref_k
        G2[:, k] = np.mean((Enew_total - Eref_total[:, None]) / epsilon, axis=0)

    G1 = np.zeros_like(L1)
    T3 = T[:, None, :]
    z2_ref_3 = z2_ref[:, None, :]
    for h in range(NumHidden):
        z1_new_h = z1_ref[:, h:h + 1] + epsilon * X
        a_new_h = sig(z1_new_h)
        deltaA = a_new_h - A_ref[:, h:h + 1]
        Wout = L2[h, :][None, None, :]
        z2_new = z2_ref_3 + deltaA[:, :, None] * Wout
        f_new = sig(z2_new)
        Enew = 0.5 * np.sum((f_new - T3) ** 2, axis=2)
        G1[:, h] = np.mean((Enew - Eref_total[:, None]) / epsilon, axis=0)

    return G1, G2


# ---- Part 1: perturbation gradient for one example (figure only) ----
G1_ex, G2_ex = perturbation_gradient_batch([i], layer1_init, layer2_init,
                                            PerturbationSize, hw=False)
gradient = np.zeros((M, M))
gradient[:NumInput + 1, NumInput + 1:NumInput + 1 + NumHidden] = G1_ex
gradient[NumInput + 1:NumInput + 2 + NumHidden, NumInput + NumHidden + 2:] = G2_ex

fig3, ax3 = plt.subplots(1, 2, figsize=(7, 3.2))
ax3[0].imshow(number[:, i].reshape(8, 8), cmap='viridis'); ax3[0].axis('off')
ax3[1].imshow(gradient, vmin=-1e-3, vmax=1e-3); ax3[1].set_title('perturbation gradient')
plt.tight_layout(); plt.savefig('part1_wp_gradient.png', dpi=150)
plt.close(fig3)

# ---- Part 2: one epoch, weight-perturbation training ----
layer1, layer2 = layer1_init.copy(), layer2_init.copy()
wp_accuracy = np.zeros(NumBatch)

for b in range(NumBatch):
    start, end = b * BatchSize, min((b + 1) * BatchSize, number.shape[1])
    idx = np.arange(start, end)

    X = np.concatenate([number[:, idx].T, np.ones((len(idx), 1))], axis=1)
    L1_eff = effective_weight(layer1, hw=HW_NONIDEALITY)
    L2_eff = effective_weight(layer2, hw=HW_NONIDEALITY)
    a1 = sig(X @ L1_eff)
    f = sig(np.concatenate([a1, np.ones((len(idx), 1))], axis=1) @ L2_eff)
    wp_accuracy[b] = np.mean(np.argmax(f, axis=1) == label[idx]) * 100

    G1, G2 = perturbation_gradient_batch(idx, layer1, layer2,
                                          PerturbationSize, hw=HW_NONIDEALITY)
    layer1 -= LearningRate * G1
    layer2 -= LearningRate * G2

    if b % 50 == 0:
        print(f'  batch {b:3d}/{NumBatch}  accuracy {wp_accuracy[b]:5.1f}%')

print(f'  mean accuracy, last 50 batches = {wp_accuracy[-50:].mean():.2f}%')

fig4, ax4 = plt.subplots(figsize=(7, 4.5))
ax4.plot(wp_accuracy, lw=0.9)
ax4.set_xlabel('Batches'); ax4.set_ylabel('Training Accuracy (%)')
ax4.set_ylim(0, 100); ax4.set_xlim(0, NumBatch); ax4.grid(alpha=.3)
ax4.set_title('One epoch - weight perturbation - conductance-unit weights')
plt.tight_layout(); plt.savefig('part2_accuracy_wp.png', dpi=150)
plt.close(fig4)