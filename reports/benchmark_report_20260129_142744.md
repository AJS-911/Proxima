# LRET vs Cirq Benchmark Report

**Generated:** 2026-01-29T14:27:44.919744

## Configuration

- Qubit Range: 4 - 14
- Circuit Depth: 20
- Noise Level: 0.01
- Shots: 1024

## Summary Statistics

- Average Speedup: **4.02x**
- Maximum Speedup: **6.96x**
- Minimum Speedup: 1.63x
- Average Fidelity: **0.9994**
- Minimum Fidelity: 0.9990

## Detailed Results

| Qubits | LRET (ms) | Cirq (ms) | Speedup | Fidelity |
|--------|-----------|-----------|---------|----------|
| 4 | 6.70 | 10.92 | 1.63x | 0.9999 |
| 6 | 6.76 | 16.99 | 2.51x | 0.9998 |
| 8 | 7.63 | 24.25 | 3.18x | 0.9996 |
| 10 | 8.77 | 42.50 | 4.84x | 0.9993 |
| 12 | 11.87 | 58.97 | 4.97x | 0.9991 |
| 14 | 13.48 | 93.82 | 6.96x | 0.9990 |

## Conclusions

LRET demonstrates significant performance improvements over standard Cirq simulation, with speedup increasing as qubit count grows. Fidelity remains high across all test cases.
