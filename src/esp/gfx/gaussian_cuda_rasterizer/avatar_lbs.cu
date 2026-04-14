#include "avatar_lbs.h"

#include <cuda_runtime.h>

#include <cmath>

namespace {
constexpr float kQuatEps = 1.0e-8f;

__device__ __forceinline__ void normalizeQuat(float* q) {
  const float norm =
      sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float inv = (norm > kQuatEps) ? (1.0f / norm) : 1.0f;
  q[0] *= inv;
  q[1] *= inv;
  q[2] *= inv;
  q[3] *= inv;
}

__device__ __forceinline__ void quatToMatrix(const float* q, float* m) {
  const float r = q[0];
  const float x = q[1];
  const float y = q[2];
  const float z = q[3];
  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float rx = r * x;
  const float ry = r * y;
  const float rz = r * z;

  m[0] = 1.0f - 2.0f * (yy + zz);
  m[1] = 2.0f * (xy - rz);
  m[2] = 2.0f * (xz + ry);
  m[3] = 2.0f * (xy + rz);
  m[4] = 1.0f - 2.0f * (xx + zz);
  m[5] = 2.0f * (yz - rx);
  m[6] = 2.0f * (xz - ry);
  m[7] = 2.0f * (yz + rx);
  m[8] = 1.0f - 2.0f * (xx + yy);
}

__device__ __forceinline__ void matrixToQuat(const float* m, float* q) {
  const float m00 = m[0];
  const float m01 = m[1];
  const float m02 = m[2];
  const float m10 = m[3];
  const float m11 = m[4];
  const float m12 = m[5];
  const float m20 = m[6];
  const float m21 = m[7];
  const float m22 = m[8];
  const float trace = m00 + m11 + m22;

  if (trace > 0.0f) {
    const float s = sqrtf(trace + 1.0f) * 2.0f;
    q[0] = 0.25f * s;
    q[1] = (m21 - m12) / s;
    q[2] = (m02 - m20) / s;
    q[3] = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const float s = sqrtf(1.0f + m00 - m11 - m22) * 2.0f;
    q[0] = (m21 - m12) / s;
    q[1] = 0.25f * s;
    q[2] = (m01 + m10) / s;
    q[3] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const float s = sqrtf(1.0f + m11 - m00 - m22) * 2.0f;
    q[0] = (m02 - m20) / s;
    q[1] = (m01 + m10) / s;
    q[2] = 0.25f * s;
    q[3] = (m12 + m21) / s;
  } else {
    const float s = sqrtf(1.0f + m22 - m00 - m11) * 2.0f;
    q[0] = (m10 - m01) / s;
    q[1] = (m02 + m20) / s;
    q[2] = (m12 + m21) / s;
    q[3] = 0.25f * s;
  }
  normalizeQuat(q);
}

__device__ __forceinline__ void orthoNormalize(float* m) {
  float c0x = m[0];
  float c0y = m[3];
  float c0z = m[6];
  float c0n = sqrtf(c0x * c0x + c0y * c0y + c0z * c0z);
  if (c0n > kQuatEps) {
    c0x /= c0n;
    c0y /= c0n;
    c0z /= c0n;
  }

  float c1x = m[1];
  float c1y = m[4];
  float c1z = m[7];
  const float dot01 = c0x * c1x + c0y * c1y + c0z * c1z;
  c1x -= dot01 * c0x;
  c1y -= dot01 * c0y;
  c1z -= dot01 * c0z;
  float c1n = sqrtf(c1x * c1x + c1y * c1y + c1z * c1z);
  if (c1n > kQuatEps) {
    c1x /= c1n;
    c1y /= c1n;
    c1z /= c1n;
  }

  const float c2x = c0y * c1z - c0z * c1y;
  const float c2y = c0z * c1x - c0x * c1z;
  const float c2z = c0x * c1y - c0y * c1x;

  m[0] = c0x;
  m[3] = c0y;
  m[6] = c0z;
  m[1] = c1x;
  m[4] = c1y;
  m[7] = c1z;
  m[2] = c2x;
  m[5] = c2y;
  m[8] = c2z;
}
__global__ void avatarLBSKernel(const float* canonicalPositions,
                                const float* offsets,
                                const float* skinningWeights,
                                const float* jointMatrices,
                                const float* invBindMatrices,
                                const float* canonicalRotations,
                                int jointCount,
                                int pointCount,
                                float* outPositions,
                                float* outRotations) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= pointCount) {
    return;
  }

  const float px =
      canonicalPositions[idx * 3 + 0] + offsets[idx * 3 + 0];
  const float py =
      canonicalPositions[idx * 3 + 1] + offsets[idx * 3 + 1];
  const float pz =
      canonicalPositions[idx * 3 + 2] + offsets[idx * 3 + 2];

  float outX = 0.0f;
  float outY = 0.0f;
  float outZ = 0.0f;
  float rotBlend[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float weightSum = 0.0f;

  const float* weightPtr = skinningWeights + idx * jointCount;
  for (int j = 0; j < jointCount; ++j) {
    const float w = weightPtr[j];
    if (w == 0.0f) {
      continue;
    }
    weightSum += w;
    const float* inv = invBindMatrices + j * 16;
    // Row-major 4x4 matrix.
    const float ix = inv[0] * px + inv[1] * py + inv[2] * pz + inv[3];
    const float iy = inv[4] * px + inv[5] * py + inv[6] * pz + inv[7];
    const float iz = inv[8] * px + inv[9] * py + inv[10] * pz + inv[11];

    const float* m = jointMatrices + j * 16;
    const float tx = m[0] * ix + m[1] * iy + m[2] * iz + m[3];
    const float ty = m[4] * ix + m[5] * iy + m[6] * iz + m[7];
    const float tz = m[8] * ix + m[9] * iy + m[10] * iz + m[11];

    outX += w * tx;
    outY += w * ty;
    outZ += w * tz;

    const float r00 = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8];
    const float r01 = m[0] * inv[1] + m[1] * inv[5] + m[2] * inv[9];
    const float r02 = m[0] * inv[2] + m[1] * inv[6] + m[2] * inv[10];
    const float r10 = m[4] * inv[0] + m[5] * inv[4] + m[6] * inv[8];
    const float r11 = m[4] * inv[1] + m[5] * inv[5] + m[6] * inv[9];
    const float r12 = m[4] * inv[2] + m[5] * inv[6] + m[6] * inv[10];
    const float r20 = m[8] * inv[0] + m[9] * inv[4] + m[10] * inv[8];
    const float r21 = m[8] * inv[1] + m[9] * inv[5] + m[10] * inv[9];
    const float r22 = m[8] * inv[2] + m[9] * inv[6] + m[10] * inv[10];

    rotBlend[0] += w * r00;
    rotBlend[1] += w * r01;
    rotBlend[2] += w * r02;
    rotBlend[3] += w * r10;
    rotBlend[4] += w * r11;
    rotBlend[5] += w * r12;
    rotBlend[6] += w * r20;
    rotBlend[7] += w * r21;
    rotBlend[8] += w * r22;
  }

  outPositions[idx * 3 + 0] = outX;
  outPositions[idx * 3 + 1] = outY;
  outPositions[idx * 3 + 2] = outZ;

  if (outRotations && canonicalRotations) {
    const float* q0 = canonicalRotations + idx * 4;
    float q0n[4] = {q0[0], q0[1], q0[2], q0[3]};
    normalizeQuat(q0n);

    float baseRot[9];
    quatToMatrix(q0n, baseRot);

    float blendedRot[9];
    if (weightSum > kQuatEps) {
      blendedRot[0] = rotBlend[0];
      blendedRot[1] = rotBlend[1];
      blendedRot[2] = rotBlend[2];
      blendedRot[3] = rotBlend[3];
      blendedRot[4] = rotBlend[4];
      blendedRot[5] = rotBlend[5];
      blendedRot[6] = rotBlend[6];
      blendedRot[7] = rotBlend[7];
      blendedRot[8] = rotBlend[8];
    } else {
      blendedRot[0] = 1.0f;
      blendedRot[1] = 0.0f;
      blendedRot[2] = 0.0f;
      blendedRot[3] = 0.0f;
      blendedRot[4] = 1.0f;
      blendedRot[5] = 0.0f;
      blendedRot[6] = 0.0f;
      blendedRot[7] = 0.0f;
      blendedRot[8] = 1.0f;
    }

    float outRotMat[9];
    outRotMat[0] = blendedRot[0] * baseRot[0] + blendedRot[1] * baseRot[3] +
                   blendedRot[2] * baseRot[6];
    outRotMat[1] = blendedRot[0] * baseRot[1] + blendedRot[1] * baseRot[4] +
                   blendedRot[2] * baseRot[7];
    outRotMat[2] = blendedRot[0] * baseRot[2] + blendedRot[1] * baseRot[5] +
                   blendedRot[2] * baseRot[8];
    outRotMat[3] = blendedRot[3] * baseRot[0] + blendedRot[4] * baseRot[3] +
                   blendedRot[5] * baseRot[6];
    outRotMat[4] = blendedRot[3] * baseRot[1] + blendedRot[4] * baseRot[4] +
                   blendedRot[5] * baseRot[7];
    outRotMat[5] = blendedRot[3] * baseRot[2] + blendedRot[4] * baseRot[5] +
                   blendedRot[5] * baseRot[8];
    outRotMat[6] = blendedRot[6] * baseRot[0] + blendedRot[7] * baseRot[3] +
                   blendedRot[8] * baseRot[6];
    outRotMat[7] = blendedRot[6] * baseRot[1] + blendedRot[7] * baseRot[4] +
                   blendedRot[8] * baseRot[7];
    outRotMat[8] = blendedRot[6] * baseRot[2] + blendedRot[7] * baseRot[5] +
                   blendedRot[8] * baseRot[8];

    orthoNormalize(outRotMat);
    float* qOut = outRotations + idx * 4;
    matrixToQuat(outRotMat, qOut);
  }
}
}  // namespace

void launchAvatarLBS(const float* canonicalPositions,
                     const float* offsets,
                     const float* skinningWeights,
                     const float* jointMatrices,
                     const float* invBindMatrices,
                     const float* canonicalRotations,
                     int jointCount,
                     int pointCount,
                     float* outPositions,
                     float* outRotations,
                     cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (pointCount + threads - 1) / threads;
  avatarLBSKernel<<<blocks, threads, 0, stream>>>(
      canonicalPositions, offsets, skinningWeights, jointMatrices,
      invBindMatrices, canonicalRotations, jointCount, pointCount, outPositions,
      outRotations);
}
