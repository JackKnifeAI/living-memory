
## scratch

| variant | test acc | updates | learn time (s) | DRAM (MiB) |
|---|---|---|---|---|
| no adapter | 0.100 ± 0.010 | - | - | - |
| logistic regression refit on the stream | 0.928 ± 0.013 | - | - | - |
| float rule (software) | 0.878 ± 0.010 | 861.6 | - | - |
| ideal binary cells (software) | 0.872 ± 0.026 | 909.8 | - | - |
| device, matched cells | 0.592 ± 0.040 | 2033.4 | 3020.7 | 1.2 |
| device, shuffled pre (control) | 0.122 ± 0.022 | 3734.4 | 5552.3 | 1.2 |
| device, any vulnerable cell | 0.450 ± 0.033 | 2664.0 | 1735.1 | 0.5 |
| device, no profile | 0.081 ± 0.035 | 3938.8 | 9075.8 | 1.8 |
| device, no consolidation | 0.064 ± 0.028 | 4169.0 | 6199.6 | 1.2 |
| device, half-select leak 0.0 | 0.835 ± 0.042 | 1008.0 | 1496.2 | 1.2 |
| device, half-select leak 0.1 | 0.783 ± 0.039 | 1303.8 | 1935.4 | 1.2 |
| device, half-select leak 0.5 | 0.106 ± 0.045 | 3775.8 | 5615.0 | 1.2 |
| device, half-select leak 1.0 | 0.008 ± 0.005 | 4160.0 | 6190.7 | 1.2 |
| device, 2 cells/group | 0.344 ± 0.058 | 3064.4 | 1136.2 | 0.3 |
| device, 4 cells/group | 0.457 ± 0.039 | 2509.8 | 1874.7 | 0.6 |
| device, hammer 0.5 x HC50 | 0.562 ± 0.027 | 2146.4 | 1330.3 | 1.0 |
| device, hammer 2.0 x HC50 | 0.627 ± 0.020 | 1858.0 | 16887.1 | 3.5 |
| device, rank 2 | 0.153 ± 0.041 | 4094.0 | 2076.3 | 0.2 |
| float rule, rank 2 | 0.294 ± 0.024 | 3972.0 | - | - |
| device, rank 4 | 0.227 ± 0.021 | 3654.8 | 4293.5 | 0.5 |
| float rule, rank 4 | 0.518 ± 0.060 | 2898.0 | - | - |
| device, rank 8 | 0.318 ± 0.058 | 3163.8 | 7698.1 | 0.9 |
| float rule, rank 8 | 0.739 ± 0.075 | 1583.6 | - | - |

## adapt

| variant | test acc | updates | learn time (s) | DRAM (MiB) |
|---|---|---|---|---|
| no adapter | 0.341 ± 0.035 | - | - | - |
| logistic regression refit on the stream | 0.923 ± 0.013 | - | - | - |
| float rule (software) | 0.862 ± 0.023 | 446.2 | - | - |
| ideal binary cells (software) | 0.827 ± 0.038 | 489.2 | - | - |
| device, matched cells | 0.623 ± 0.058 | 805.8 | 1197.5 | 1.2 |
| device, shuffled pre (control) | 0.181 ± 0.038 | 1327.2 | 1975.6 | 1.2 |
| device, any vulnerable cell | 0.489 ± 0.037 | 1050.0 | 685.1 | 0.5 |
| device, no profile | 0.207 ± 0.052 | 1277.2 | 2942.6 | 1.8 |
| device, no consolidation | 0.351 ± 0.032 | 1201.0 | 1783.0 | 1.2 |
| device, half-select leak 0.0 | 0.822 ± 0.017 | 491.4 | 729.3 | 1.2 |
| device, half-select leak 0.1 | 0.775 ± 0.038 | 582.8 | 864.8 | 1.2 |
| device, half-select leak 0.5 | 0.177 ± 0.048 | 1334.6 | 1986.6 | 1.2 |
| device, half-select leak 1.0 | 0.023 ± 0.015 | 1532.4 | 2281.3 | 1.2 |
| device, 2 cells/group | 0.413 ± 0.047 | 1073.0 | 398.3 | 0.3 |
| device, 4 cells/group | 0.513 ± 0.065 | 958.8 | 714.7 | 0.6 |
| device, hammer 0.5 x HC50 | 0.594 ± 0.051 | 847.6 | 525.6 | 1.0 |
| device, hammer 2.0 x HC50 | 0.628 ± 0.028 | 803.4 | 7299.9 | 3.5 |
| device, rank 2 | 0.354 ± 0.009 | 1194.6 | 745.4 | 0.2 |
| float rule, rank 2 | 0.503 ± 0.033 | 1047.8 | - | - |
| device, rank 4 | 0.331 ± 0.065 | 1180.2 | 1661.9 | 0.5 |
| float rule, rank 4 | 0.661 ± 0.051 | 808.6 | - | - |
| device, rank 8 | 0.417 ± 0.055 | 1108.4 | 2929.0 | 0.9 |
| float rule, rank 8 | 0.787 ± 0.028 | 589.2 | - | - |
