### tab1

| year | dem | rep | share_dem_last4 | trend_4v4 | slope_8w | share_dem_full | margin_dem | pv_hit | ec_hit | trend_hit | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2008 | Barack Obama | John McCain | 0.604 | 0.058 | 0.011 | 0.582 | 7.270 | True | True | True | 156 daily obs |
| 2012 | Barack Obama | Mitt Romney | 0.258 | 0.039 | 0.003 | 0.252 | 3.860 | False | False | True | 158 daily obs |
| 2016 | Hillary Clinton | Donald Trump | 0.450 | 0.029 | -0.019 | 0.430 | 2.090 | False | True | True | 160 daily obs |
| 2020 | Joe Biden | Donald Trump | 0.525 | 0.084 | 0.010 | 0.428 | 4.450 | True | True | True | 155 daily obs |
| 2024 | Kamala Harris | Donald Trump | 0.668 | 0.007 | -0.002 | 0.626 | -1.480 | False | False | False | 157 daily obs |

### tab2

| modifier | year | share_dem_last4 | trend_4v4 | share_dem_full | zero_day_share | margin_dem | pv_hit | ec_hit | note |
|---|---|---|---|---|---|---|---|---|---|
| vote_for | 2008 | 0.600 | 0.038 | 0.552 | 0.897 | 7.270 | True | True |  |
| vote_for | 2012 | 0.222 | 0.077 | 0.180 | 0.848 | 3.860 | False | False |  |
| vote_for | 2016 | 0.462 | -0.058 | 0.497 | 0.275 | 2.090 | False | True |  |
| vote_for | 2020 | 0.632 | -0.051 | 0.687 | 0.400 | 4.450 | True | True |  |
| vote_for | 2024 | 0.625 | 0.005 | 0.658 | 0.490 | -1.480 | False | False |  |
| rally | 2008 | 0.873 |  | 0.873 | 0.981 | 7.270 | True | True |  |
| rally | 2012 | 0.068 | 0.068 | 0.066 | 0.987 | 3.860 | False | False |  |
| rally | 2016 | 0.423 | 0.074 | 0.331 | 0.287 | 2.090 | False | True |  |
| rally | 2020 | 0.475 | 0.117 | 0.364 | 0.561 | 4.450 | False | False |  |
| rally | 2024 | 0.805 | 0.144 | 0.679 | 0.312 | -1.480 | False | False |  |
| policies | 2008 | 0.515 |  | 0.677 | 1.000 | 7.270 | True | True |  |
| policies | 2012 | 0.183 | 0.183 | 0.203 | 0.994 | 3.860 | False | False |  |
| policies | 2016 | 0.551 | 0.032 | 0.519 | 0.700 | 2.090 | True | False |  |
| policies | 2020 | 0.769 | 0.002 | 0.831 | 0.710 | 4.450 | True | True |  |
| policies | 2024 | 0.786 | -0.077 | 0.860 | 0.459 | -1.480 | False | False |  |
| for_president | 2008 | 0.295 | 0.085 | 0.229 | 0.667 | 7.270 | False | False |  |
| for_president | 2012 | 0.310 | 0.079 | 0.232 | 0.658 | 3.860 | False | False |  |
| for_president | 2016 | 0.485 | 0.043 | 0.472 | 0.025 | 2.090 | False | True |  |
| for_president | 2020 | 0.672 | 0.071 | 0.663 | 0.000 | 4.450 | True | True |  |
| for_president | 2024 | 0.640 | -0.023 | 0.726 | 0.178 | -1.480 | False | False |  |

### tab2s

| modifier | pv_hits | ec_hits | pearson_share_margin | spearman | mean_zero_day_share |
|---|---|---|---|---|---|
| raw name (step 1) | 2/5 | 3/5 | -0.214 | -0.100 |  |
| vote_for | 2/5 | 3/5 | -0.069 | 0.100 | 0.582 |
| rally | 1/5 | 2/5 | -0.035 | 0.300 | 0.626 |
| policies | 3/5 | 2/5 | -0.386 | -0.500 | 0.772 |
| for_president | 1/5 | 2/5 | -0.603 | -0.400 | 0.306 |

### tab3

| year | state | share_dem_last4 | rel_share | margin_dem | hit | rel_hit | rel_margin |
|---|---|---|---|---|---|---|---|
| 2016 | US-PA | 0.460 | 0.010 | -0.720 | True | False | -2.810 |
| 2016 | US-MI | 0.448 | -0.003 | -0.230 | True | True | -2.320 |
| 2016 | US-WI | 0.443 | -0.007 | -0.770 | True | True | -2.860 |
| 2016 | US-GA | 0.446 | -0.004 | -5.130 | True | True | -7.220 |
| 2016 | US-AZ | 0.468 | 0.017 | -3.550 | True | False | -5.640 |
| 2016 | US-NV | 0.451 | 0.001 | 2.420 | False | True | 0.330 |
| 2016 | US-NC | 0.456 | 0.005 | -3.660 | True | False | -5.750 |
| 2020 | US-PA | 0.541 | 0.016 | 1.170 | True | False | -3.280 |
| 2020 | US-MI | 0.528 | 0.003 | 2.780 | True | False | -1.670 |
| 2020 | US-WI | 0.532 | 0.006 | 0.630 | True | False | -3.820 |
| 2020 | US-GA | 0.532 | 0.007 | 0.230 | True | False | -4.220 |
| 2020 | US-AZ | 0.531 | 0.006 | 0.310 | True | False | -4.140 |
| 2020 | US-NV | 0.503 | -0.023 | 2.390 | True | True | -2.060 |
| 2020 | US-NC | 0.531 | 0.006 | -1.350 | False | False | -5.800 |
| 2024 | US-PA | 0.662 | -0.006 | -1.710 | False | True | -0.230 |
| 2024 | US-MI | 0.667 | -0.001 | -1.420 | False | False | 0.060 |
| 2024 | US-WI | 0.691 | 0.023 | -0.860 | False | True | 0.620 |
| 2024 | US-GA | 0.683 | 0.015 | -2.200 | False | False | -0.720 |
| 2024 | US-AZ | 0.669 | 0.001 | -5.530 | False | False | -4.050 |
| 2024 | US-NV | 0.655 | -0.013 | -3.100 | False | True | -1.620 |
| 2024 | US-NC | 0.661 | -0.007 | -3.240 | False | True | -1.760 |

### tab4

| pair | year | share_dem_last4 | trend_4v4 | share_dem_full | zero_day_share | margin_dem | hit | note |
|---|---|---|---|---|---|---|---|---|
| party_name | 2006 | 0.523 | -0.000 | 0.549 | 0.119 | 8.000 | True |  |
| party_name | 2010 | 0.449 | 0.004 | 0.441 | 0.006 | -6.800 | True |  |
| party_name | 2014 | 0.484 | 0.004 | 0.474 | 0.000 | -5.700 | True |  |
| party_name | 2018 | 0.497 | 0.003 | 0.507 | 0.000 | 8.600 | False |  |
| party_name | 2022 | 0.470 | 0.020 | 0.447 | 0.000 | -2.800 | True |  |
| vote_party | 2006 | 0.143 | -0.857 | 0.227 | 0.994 | 8.000 | False |  |
| vote_party | 2010 | 0.349 | -0.194 | 0.304 | 0.929 | -6.800 | True |  |
| vote_party | 2014 | 0.325 | 0.164 | 0.274 | 0.833 | -5.700 | True |  |
| vote_party | 2018 | 0.430 | 0.049 | 0.404 | 0.158 | 8.600 | False |  |
| vote_party | 2022 | 0.377 | 0.075 | 0.328 | 0.056 | -2.800 | True |  |

### tab5

| pair | share_dem_last4 | trend_4v4 | slope_8w | share_dem_full | last_obs | n_days | note |
|---|---|---|---|---|---|---|---|
| party_name | 0.456 | -0.114 | -0.019 | 0.521 | 2026-09-16 | 108 |  |
| vote_party | 0.340 | -0.030 | -0.010 | 0.346 | 2026-09-16 | 108 |  |